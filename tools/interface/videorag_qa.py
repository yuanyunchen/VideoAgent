"""
Temporal Q&A Interfaces
"""

import os
import sys
import logging
import warnings
import hashlib
import time
from typing import Dict, Any, Optional, ClassVar

import numpy as np
from dotenv import load_dotenv

from tools.interface_base import Interface, InterfaceCategory, Video

# Suppress warnings and verbose logging
warnings.filterwarnings("ignore")
logging.getLogger("httpx").setLevel(logging.WARNING)


class VideoRAGTemporalQA(Interface):
    """VideoRAG-based Temporal Q&A Interface."""
    
    CATEGORY = InterfaceCategory.SUB_QUESTION_ANSWERING
    FUNCTIONALITY = "Answer temporal questions using RAG-based retrieval."
    REFERENCE_PAPER = "VideoRAG (2024)"
    TOOL_SOURCES = ["VideoRAG"]
    
    # Full input schema (system use)
    INPUT_SCHEMA = {
        "query": {"type": "string", "required": True},
        "video": {"type": "Video", "required": True},
        "model": {"type": "string", "required": False, "default": "gpt-4o-mini"}
    }
    
    OUTPUT_SCHEMA = {
        "answer": {"type": "string"},
        "evidence": {"type": "string"}
    }
    
    # Agent-facing (simplified) - NO options parameter
    AGENT_NAME = "temporal_qa"
    AGENT_DESCRIPTION = "Answer abstract questions about events - timing, sequences, durations, what happened."
    
    AGENT_INPUT_SCHEMA = {
        "query": {
            "type": "string",
            "required": True,
            "description": "Question about timing or event sequences"
        }
    }
    
    AGENT_OUTPUT_FORMAT = "Text answer with evidence about temporal information"
    
    # Class-level cache for VideoRAG instances (shared across all instances)
    _videorag_cache: Dict[str, Any] = {}  # video_path -> VideoRAG instance
    
    # Instance variables
    _videorag_instance: Any = None
    _llm_config: Any = None
    _processed_videos: set = None
    _working_dir_base: str = None
    _model_name: str = "gpt-4o-mini"
    _initialized: bool = False
    
    @classmethod
    def format_output_for_agent(cls, output: Dict[str, Any]) -> str:
        """Format output with answer and evidence for the agent."""
        if "error" in output:
            return f"Error: {output['error']}"
        
        answer = output.get("answer", "Could not determine answer.")
        evidence = output.get("evidence", "")
        
        if evidence:
            return f"Answer: {answer}\nEvidence: {evidence}"
        return f"Answer: {answer}"
    
    def initialize(self) -> None:
        """Initialize VideoRAG with LLM configuration.
        
        Sets up the OpenAI/AIML API configuration for embeddings and LLM calls.
        The actual VideoRAG instance is created per-video to manage working directories.
        """
        if self._initialized:
            return
        
        # Load environment variables
        load_dotenv()
        
        # Set up API keys
        api_key = os.environ.get("OPENAI_API_KEY")
        base_url = os.environ.get("OPENAI_BASE_URL")
        
        # If no OpenAI key, try AIML API
        if not api_key:
            api_key = os.environ.get("AIML_API_KEY")
            base_url = os.environ.get("AIML_BASE_URL", "https://api.aimlapi.com/v1")
        
        if not api_key:
            raise ValueError("No API key found. Set OPENAI_API_KEY or AIML_API_KEY in .env")
        
        # Set environment variables for VideoRAG's internal client
        os.environ["OPENAI_API_KEY"] = api_key
        if base_url:
            os.environ["OPENAI_BASE_URL"] = base_url
        
        # Add VideoRAG to path
        videorag_path = os.path.join(
            os.path.dirname(__file__), 
            "..", "models", "VideoRAG", "VideoRAG-algorithm"
        )
        videorag_path = os.path.abspath(videorag_path)
        if videorag_path not in sys.path:
            sys.path.insert(0, videorag_path)
        
        # Import VideoRAG components
        from videorag._llm import LLMConfig, openai_embedding, gpt_4o_mini_complete
        
        # Create LLM config
        self._llm_config = LLMConfig(
            embedding_func_raw=openai_embedding,
            embedding_model_name="text-embedding-3-small",
            embedding_dim=1536,
            embedding_max_token_size=8192,
            embedding_batch_num=32,
            embedding_func_max_async=16,
            query_better_than_threshold=0.2,
            best_model_func_raw=gpt_4o_mini_complete,
            best_model_name=self._model_name,
            best_model_max_token_size=32768,
            best_model_max_async=16,
            cheap_model_func_raw=gpt_4o_mini_complete,
            cheap_model_name=self._model_name,
            cheap_model_max_token_size=32768,
            cheap_model_max_async=16
        )
        
        # Set base working directory
        self._working_dir_base = os.path.join(
            os.path.dirname(__file__),
            "..", "models", "VideoRAG", "workdir"
        )
        self._working_dir_base = os.path.abspath(self._working_dir_base)
        os.makedirs(self._working_dir_base, exist_ok=True)
        
        self._processed_videos = set()
        self._initialized = True
    
    def _get_video_working_dir(self, video_path: str) -> str:
        """Get working directory for a specific video."""
        # Create a unique directory name based on video path
        video_hash = hashlib.md5(video_path.encode()).hexdigest()[:8]
        video_name = os.path.basename(video_path).split('.')[0]
        return os.path.join(self._working_dir_base, f"{video_name}_{video_hash}")
    
    def _clear_gpu_memory(self) -> None:
        """Clear GPU memory cache."""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                import gc
                gc.collect()
        except Exception:
            pass
    
    def _get_or_create_videorag(self, video_path: str, load_caption_model: bool = True) -> Any:
        """Get or create VideoRAG instance for a video.
        
        Uses class-level cache to avoid re-initialization for the same video.
        
        Args:
            video_path: Path to the video file
            load_caption_model: Whether to load caption model (needed for query with visual context)
        
        Returns:
            VideoRAG instance
        """
        from videorag import VideoRAG, QueryParam
        
        # Normalize video path for consistent cache key
        video_path_abs = os.path.abspath(video_path)
        
        # Check if we have a cached instance
        if video_path_abs in VideoRAGTemporalQA._videorag_cache:
            logging.info(f"Using cached VideoRAG instance for: {video_path}")
            videorag = VideoRAGTemporalQA._videorag_cache[video_path_abs]
            # Load caption model if needed (might have been unloaded)
            videorag.load_caption_model(debug=not load_caption_model)
            return videorag
        
        working_dir = self._get_video_working_dir(video_path)
        
        # Clear GPU memory before loading
        self._clear_gpu_memory()
        
        # Create VideoRAG instance
        videorag = VideoRAG(
            llm=self._llm_config,
            working_dir=working_dir
        )
        
        # Check if video needs to be processed
        video_name = os.path.basename(video_path).split('.')[0]
        if video_name not in videorag.video_segments._data:
            # Process the video
            logging.info(f"Processing video: {video_path}")
            videorag.insert_video(video_path_list=[video_path])
            self._processed_videos.add(video_path)
            # Clear GPU after processing
            self._clear_gpu_memory()
        
        # Load caption model for query (debug=True sets caption_model to None but attribute still exists)
        videorag.load_caption_model(debug=not load_caption_model)
        
        # Cache the instance
        VideoRAGTemporalQA._videorag_cache[video_path_abs] = videorag
        logging.info(f"Cached VideoRAG instance for: {video_path}")
        
        return videorag
    
    def __call__(self, **kwargs) -> Dict[str, Any]:
        """Execute temporal Q&A on a video.
        
        Args:
            query: The question to answer
            video: Video object containing the video path
            model: (optional) Model name for LLM calls
            debug: (optional) Skip loading caption model for debugging
        
        Returns:
            Dict with 'answer' and 'evidence' keys
        """
        if not self._initialized:
            self.initialize()
        
        # Extract parameters
        query = kwargs.get("query")
        video = kwargs.get("video")
        model = kwargs.get("model", "gpt-4o-mini")
        debug = kwargs.get("debug", False)
        
        # Validate inputs
        if not query:
            return {"error": "Query is required"}
        
        if not video:
            return {"error": "Video is required"}
        
        # Handle Video object or path string
        if isinstance(video, Video):
            video_path = video.path
        elif isinstance(video, str):
            video_path = video
        else:
            return {"error": f"Invalid video type: {type(video)}"}
        
        if not os.path.exists(video_path):
            return {"error": f"Video file not found: {video_path}"}
        
        # Update model if different
        if model != self._model_name:
            self._model_name = model
            self._initialized = False
            self.initialize()
        
        try:
            # Import QueryParam
            from videorag import QueryParam
            
            # Get VideoRAG instance (skip caption model loading in debug mode)
            videorag = self._get_or_create_videorag(video_path, load_caption_model=not debug)
            
            # Set up query parameters
            param = QueryParam(mode="videorag")
            param.wo_reference = True  # Don't add video clip references in response
            
            # Execute query
            response = videorag.query(query=query, param=param)
            
            # Extract evidence from the response if possible
            # VideoRAG returns a text response, try to parse evidence from it
            evidence = self._extract_evidence(response, query)
            
            return {
                "answer": response,
                "evidence": evidence
            }
            
        except Exception as e:
            logging.error(f"VideoRAG query failed: {e}")
            return {"error": str(e)}
    
    def _extract_evidence(self, response: str, query: str) -> str:
        """Extract evidence from VideoRAG response.
        
        Args:
            response: The full response from VideoRAG
            query: The original query
        
        Returns:
            Evidence string describing what was observed
        """
        # For now, return a summary indicating this came from video analysis
        # In future, could parse the response for specific frame references
        if response and len(response) > 50:
            return "Based on analysis of video frames and temporal context."
        return ""


# ========= VideoRAG batch pre-processing (migrated from preprocess_videorag.py) =========

VIDEORAG_REQUIRED_CACHE_FILES = [
    "kv_store_video_segments.json",
    "kv_store_text_chunks.json",
    "graph_chunk_entity_relation.graphml",
]


def _videorag_workdir_base() -> str:
    """Default working directory base for VideoRAG caches."""
    base = os.path.join(os.path.dirname(__file__), "..", "models", "VideoRAG", "workdir")
    return os.path.abspath(base)


def _videorag_workdir_for_video(video_path: str, working_dir_base: Optional[str] = None) -> str:
    """Compute the working directory path for a specific video."""
    working_dir_base = working_dir_base or _videorag_workdir_base()
    video_hash = hashlib.md5(video_path.encode()).hexdigest()[:8]
    video_name = os.path.basename(video_path).split(".")[0]
    return os.path.join(working_dir_base, f"{video_name}_{video_hash}")


def videorag_is_processed(video_path: str, working_dir_base: Optional[str] = None) -> bool:
    """Check whether VideoRAG cache exists for the given video."""
    working_dir = _videorag_workdir_for_video(video_path, working_dir_base)
    if not os.path.exists(working_dir):
        return False
    for file_name in VIDEORAG_REQUIRED_CACHE_FILES:
        if not os.path.exists(os.path.join(working_dir, file_name)):
            return False
    return True


def preprocess_videos_with_videorag(
    video_dir: str,
    limit: int = 0,
    skip_existing: bool = True,
    load_caption_model: bool = False,
) -> Dict[str, Any]:
    """
    Batch pre-process videos to build VideoRAG caches.

    Args:
        video_dir: Directory containing videos
        limit: Max number of videos to process (0 = all)
        skip_existing: Skip videos that already have cache files
        load_caption_model: Whether to load caption model during preprocessing

    Returns:
        Dict with processing statistics
    """
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    qa = VideoRAGTemporalQA()
    qa.initialize()
    working_dir_base = qa._working_dir_base

    video_files = sorted(
        [
            os.path.join(video_dir, f)
            for f in os.listdir(video_dir)
            if f.endswith((".mp4", ".avi", ".mov", ".mkv"))
        ]
    )
    if limit > 0:
        video_files = video_files[:limit]

    logger.info("Found %d videos to process", len(video_files))
    logger.info("Working directory: %s", working_dir_base)

    stats = {
        "total": len(video_files),
        "processed": 0,
        "skipped": 0,
        "failed": 0,
        "failed_videos": [],
    }

    for i, video_path in enumerate(video_files):
        video_name = os.path.basename(video_path)
        if skip_existing and videorag_is_processed(video_path, working_dir_base):
            logger.info("[%d/%d] Skipping (cached): %s", i + 1, len(video_files), video_name)
            stats["skipped"] += 1
            continue

        logger.info("[%d/%d] Processing: %s", i + 1, len(video_files), video_name)
        try:
            start_time = time.time()
            qa._get_or_create_videorag(video_path, load_caption_model=load_caption_model)
            elapsed = time.time() - start_time
            logger.info("  Completed in %.1fs", elapsed)
            stats["processed"] += 1
        except Exception as e:
            logger.error("  Failed: %s", e)
            stats["failed"] += 1
            stats["failed_videos"].append(video_name)

    return stats


def test_videorag_cache(video_dir: str, num_tests: int = 3) -> bool:
    """Verify that VideoRAG caches load and answer queries."""
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    logger.info("=" * 60)
    logger.info("Testing VideoRAG Cache Loading")
    logger.info("=" * 60)

    qa = VideoRAGTemporalQA()
    qa.initialize()
    working_dir_base = qa._working_dir_base

    video_files = sorted(
        [
            os.path.join(video_dir, f)
            for f in os.listdir(video_dir)
            if f.endswith((".mp4", ".avi", ".mov", ".mkv"))
        ]
    )
    cached_videos = [v for v in video_files if videorag_is_processed(v, working_dir_base)]
    if not cached_videos:
        logger.error("No cached videos found!")
        return False

    logger.info("Found %d cached videos", len(cached_videos))
    test_videos = cached_videos[:num_tests]
    all_passed = True

    for video_path in test_videos:
        video_name = os.path.basename(video_path)
        logger.info("Testing: %s", video_name)
        try:
            start_time = time.time()
            videorag = qa._get_or_create_videorag(video_path, load_caption_model=True)
            load_time = time.time() - start_time
            from videorag import QueryParam

            if video_name.split(".")[0] in getattr(videorag.video_segments, "_data", {}):
                logger.info("  [PASS] Loaded from cache in %.2fs", load_time)
                videorag.load_caption_model(debug=False)
                param = QueryParam(mode="videorag")
                param.wo_reference = True
                query_start = time.time()
                response = videorag.query(query="What is in this video?", param=param)
                query_time = time.time() - query_start
                logger.info("  [PASS] Query completed in %.2fs", query_time)
                logger.info("  Response: %s...", str(response)[:100])
            else:
                logger.error("  [FAIL] Video segments not found in cache")
                all_passed = False
        except Exception as e:
            logger.error("  [FAIL] Error: %s", e)
            all_passed = False

    return all_passed


def _videorag_cli() -> int:
    """CLI entry point for VideoRAG preprocessing and cache testing."""
    import argparse
    import multiprocessing

    parser = argparse.ArgumentParser(description="Pre-process videos for VideoRAG caching")
    parser.add_argument("--video_dir", type=str, required=True, help="Directory containing videos")
    parser.add_argument("--limit", type=int, default=0, help="Max videos to process (0 = all)")
    parser.add_argument("--skip_existing", action="store_true", default=True, help="Skip already processed videos")
    parser.add_argument("--test_only", action="store_true", help="Only test cache loading, don't process")
    parser.add_argument("--num_tests", type=int, default=3, help="Number of videos to test")
    parser.add_argument(
        "--load_caption_model",
        action="store_true",
        help="Load caption model during preprocessing (disabled by default to save time)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.video_dir):
        logging.error("Video directory not found: %s", args.video_dir)
        return 1

    multiprocessing.set_start_method("spawn", force=True)

    if args.test_only:
        success = test_videorag_cache(args.video_dir, args.num_tests)
        return 0 if success else 1

    stats = preprocess_videos_with_videorag(
        video_dir=args.video_dir,
        limit=args.limit,
        skip_existing=args.skip_existing,
        load_caption_model=args.load_caption_model,
    )

    logging.info("=" * 60)
    logging.info("Pre-processing Summary")
    logging.info("=" * 60)
    logging.info("Total videos: %d", stats["total"])
    logging.info("Processed: %d", stats["processed"])
    logging.info("Skipped (cached): %d", stats["skipped"])
    logging.info("Failed: %d", stats["failed"])
    if stats["failed_videos"]:
        logging.info("Failed videos:")
        for v in stats["failed_videos"]:
            logging.info("  - %s", v)

    if stats["processed"] > 0 or stats["skipped"] > 0:
        success = test_videorag_cache(args.video_dir, args.num_tests)
        if success:
            logging.info("[SUCCESS] All cache loading tests passed!")
        else:
            logging.error("[FAILED] Some cache loading tests failed!")
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(_videorag_cli())
