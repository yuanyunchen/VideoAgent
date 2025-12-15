"""
Evaluation Pipeline for VideoAgent Tools

Implements batch video processing with:
- Progress bar with accuracy metrics
- Detailed logging
- Result aggregation and export
- Multiprocessing support for parallel video processing
- Centralized GPU tool management via ToolServer
"""

# MUST be set before any imports that might load numexpr (used by pandas, numpy, etc.)
import os
os.environ["NUMEXPR_MAX_THREADS"] = "64"
os.environ["NUMEXPR_NUM_THREADS"] = "8"

import json
import csv
import traceback
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from collections import Counter
import logging
from multiprocessing import Process, Queue, Event, Manager
import queue

from tqdm import tqdm

from video_agent_tools.graph import VideoAgent
from video_agent_tools.utils.logging import (
    setup_logger,
    save_video_result,
    save_video_frames,
    LogCollector,
    WorkerLogger,
    create_worker_logger,
    start_log_collector,
)
from video_agent_tools.resource_management import (
    ToolServer,
    ToolClient,
    run_tool_server,
    MessageType,
    ToolRequest,
)


class EvalStatistics:
    """Track evaluation statistics during experiment."""
    
    def __init__(self):
        self.total = 0
        self.correct = 0
        self.forced_answers = 0
        self.invalid_videos = 0
        self.total_tool_calls = 0
        self.total_frames = 0
        self.tool_usage = Counter()
        self.answer_distribution = Counter()
        self.errors = []
    
    def update(self, result: Dict[str, Any]):
        """Update statistics with a new result."""
        self.total += 1
        
        final_answer = result.get("final_answer")
        truth = result.get("truth")
        is_valid = result.get("is_valid", True)
        is_forced = result.get("is_forced", False)
        is_correct = result.get("is_correct", False)
        tool_count = result.get("tool_call_count", 0)
        frame_count = result.get("frame_count", 0)
        
        if not is_valid:
            self.invalid_videos += 1
        
        if is_forced:
            self.forced_answers += 1
        
        if is_correct:
            self.correct += 1
        
        self.total_tool_calls += tool_count
        self.total_frames += frame_count
        
        if final_answer is not None:
            self.answer_distribution[final_answer] += 1
        
        # Track tool usage
        for tool_call in result.get("tool_history", []):
            tool_name = tool_call.get("tool_name", "unknown")
            self.tool_usage[tool_name] += 1
        
        # Track errors
        if "error" in result:
            self.errors.append({
                "video_id": result.get("video_id"),
                "error": result.get("error"),
            })
    
    def merge(self, other: "EvalStatistics"):
        """Merge another statistics object into this one."""
        self.total += other.total
        self.correct += other.correct
        self.forced_answers += other.forced_answers
        self.invalid_videos += other.invalid_videos
        self.total_tool_calls += other.total_tool_calls
        self.total_frames += other.total_frames
        self.tool_usage.update(other.tool_usage)
        self.answer_distribution.update(other.answer_distribution)
        self.errors.extend(other.errors)
    
    def get_accuracy(self) -> float:
        valid = self.total - self.invalid_videos
        return self.correct / valid if valid > 0 else 0.0
    
    def get_avg_tool_calls(self) -> float:
        valid = self.total - self.invalid_videos
        return self.total_tool_calls / valid if valid > 0 else 0.0
    
    def get_avg_frames(self) -> float:
        valid = self.total - self.invalid_videos
        return self.total_frames / valid if valid > 0 else 0.0
    
    def get_progress_dict(self) -> Dict[str, str]:
        """Get progress bar postfix dict."""
        acc = self.get_accuracy()
        valid_rate = (self.total - self.invalid_videos) / self.total if self.total > 0 else 1.0
        return {
            'Acc': f'{acc:.1%}',
            'Tools': f'{self.get_avg_tool_calls():.1f}',
            'Valid': f'{valid_rate:.1%}',
            'Forced': f'{self.forced_answers}',
        }
    
    def to_dict(self) -> Dict[str, Any]:
        valid = self.total - self.invalid_videos
        return {
            "total": self.total,
            "valid": valid,
            "invalid": self.invalid_videos,
            "correct": self.correct,
            "accuracy": self.get_accuracy(),
            "forced_answers": self.forced_answers,
            "forced_rate": self.forced_answers / valid if valid > 0 else 0.0,
            "avg_tool_calls": self.get_avg_tool_calls(),
            "avg_frames": self.get_avg_frames(),
            "total_tool_calls": self.total_tool_calls,
            "total_frames": self.total_frames,
            "tool_usage": dict(self.tool_usage),
            "answer_distribution": dict(self.answer_distribution),
            "error_count": len(self.errors),
        }


def parse_video_annotation(
    annotation_file: str,
    video_list_file: str = None,
    max_videos: int = -1,
) -> List[Dict[str, Any]]:
    """
    Parse video annotations for evaluation.
    
    Args:
        annotation_file: Path to annotations JSON
        video_list_file: Optional path to file with video IDs to include
        max_videos: Maximum videos to process (-1 for all)
    
    Returns:
        List of video info dicts with video_id, question, choices, truth
    """
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)
    
    # Load video list if provided
    video_ids = None
    if video_list_file and os.path.exists(video_list_file):
        with open(video_list_file, 'r') as f:
            video_ids = [line.strip() for line in f if line.strip()]
    
    videos_info = []
    
    for video_id, data in annotations.items():
        if video_ids is not None and video_id not in video_ids:
            continue
        
        # Extract question and choices
        question = data.get("question", "")
        
        # Get choices (option 0-4)
        choices = []
        for i in range(5):
            key = f"option {i}"
            if key in data:
                choices.append(data[key])
        
        # Get ground truth
        truth = data.get("truth")
        if truth is not None:
            try:
                truth = int(truth)
            except (ValueError, TypeError):
                truth = None
        
        videos_info.append({
            "video_id": video_id,
            "question": question,
            "choices": choices,
            "truth": truth,
        })
        
        if max_videos > 0 and len(videos_info) >= max_videos:
            break
    
    return videos_info


# ==============================================================================
# Worker Process Function
# ==============================================================================

def worker_process(
    worker_id: int,
    video_tasks: List[Dict[str, Any]],
    video_dir: str,
    videos_output_dir: str,
    result_queue: Queue,
    log_queue: Queue,
    shutdown_event: Event,
    config: Dict[str, Any],
    tool_request_queue: Queue = None,
    tool_response_queue: Queue = None,
):
    """
    Worker process function for parallel video processing.
    
    Workers are lightweight processes that handle LLM calls and workflow logic.
    GPU tools are accessed via ToolClient -> ToolServer (centralized management).
    
    Args:
        worker_id: Unique identifier for this worker
        video_tasks: List of video info dicts to process
        video_dir: Directory containing video files
        videos_output_dir: Base directory for per-video outputs
        result_queue: Queue for sending results back to main process
        log_queue: Queue for sending logs to collector
        shutdown_event: Event to signal shutdown
        config: Configuration dict for VideoAgent
        tool_request_queue: Queue for sending tool requests to ToolServer
        tool_response_queue: Queue for receiving tool responses from ToolServer
    """
    # Ensure numexpr thread limits are set in child process
    import os
    os.environ["NUMEXPR_MAX_THREADS"] = "64"
    os.environ["NUMEXPR_NUM_THREADS"] = "8"
    
    # Create worker logger
    logger = create_worker_logger(worker_id, log_queue)
    logger.info(f"Worker {worker_id} starting with {len(video_tasks)} videos")
    
    try:
        # Create ToolClient for accessing centralized ToolServer
        tool_client = None
        if tool_request_queue and tool_response_queue:
            tool_client = ToolClient(
                worker_id=worker_id,
                request_queue=tool_request_queue,
                response_queue=tool_response_queue,
                enabled_tools=config["enabled_tools"],
                logger=logging.getLogger(f"ToolClient.Worker{worker_id}"),
                max_view_frames=config["max_view_frames"],
                default_sample_frames=config["default_sample_frames"],
                min_sample_frames=config["min_sample_frames"],
                max_sample_frames=config["max_sample_frames"],
            )
            tool_client.initialize()
            logger.info(f"Worker {worker_id} connected to ToolServer")
        
        # Create VideoAgent for this worker
        # Pass tool_client as tool_manager - it has the same interface
        agent = VideoAgent(
            model=config["model"],
            enabled_tools=config["enabled_tools"],
            max_tool_calls=config["max_tool_calls"],
            max_parallel_tools=config["max_parallel_tools"],
            initial_frames=config["initial_frames"],
            captioner=config["captioner"],
            logger=logging.getLogger(f"VideoAgent.Worker{worker_id}"),
            max_view_frames=config["max_view_frames"],
            default_sample_frames=config["default_sample_frames"],
            min_sample_frames=config["min_sample_frames"],
            max_sample_frames=config["max_sample_frames"],
            tool_manager=tool_client,  # Use ToolClient for centralized tool access
        )
        
        # Process assigned videos
        for video_info in video_tasks:
            if shutdown_event.is_set():
                logger.info(f"Worker {worker_id} received shutdown signal")
                break
            
            video_id = video_info["video_id"]
            video_path = os.path.join(video_dir, f"{video_id}.mp4")
            
            # Start video group for log grouping
            logger.start_video_group(video_id)
            logger.info(f"Processing video: {video_id}")
            
            # Create per-video output directory
            video_output_dir = os.path.join(videos_output_dir, video_id)
            os.makedirs(video_output_dir, exist_ok=True)
            
            try:
                # Check if video exists
                if not os.path.exists(video_path):
                    logger.warning(f"Video not found: {video_path}")
                    result = {
                        "video_id": video_id,
                        "question": video_info["question"],
                        "final_answer": None,
                        "truth": video_info.get("truth"),
                        "is_valid": False,
                        "error": "Video file not found",
                        "choices": video_info["choices"],
                        "tool_history": [],
                        "video_context_dict": {},
                        "messages": [],
                    }
                else:
                    # Process video
                    result = agent.process_video(
                        video_path=video_path,
                        video_id=video_id,
                        question=video_info["question"],
                        choices=video_info["choices"],
                        truth=video_info.get("truth"),
                        video_output_dir=video_output_dir,
                        detailed=config["detailed"],
                    )
                
                # Save per-video result.json
                save_video_result(
                    video_output_dir=video_output_dir,
                    video_id=video_id,
                    question=video_info["question"],
                    choices=video_info["choices"],
                    final_answer=result.get("final_answer"),
                    explanation=result.get("explanation", ""),
                    truth=video_info.get("truth"),
                    is_correct=result.get("is_correct", False),
                    is_forced=result.get("is_forced", False),
                    is_valid=result.get("is_valid", True),
                    tool_history=result.get("tool_history", []),
                    video_context_dict=result.get("video_context_dict", {}),
                    messages=result.get("messages", []),
                    duration_s=result.get("duration_s"),
                    tool_call_rounds=result.get("tool_call_rounds"),
                )
                
                # Save frames if available
                if "frames" in result and result["frames"]:
                    save_video_frames(
                        video_output_dir=video_output_dir,
                        frames=result["frames"],
                        video_id=video_id,
                        fps=result.get("fps", 30.0),
                    )
                
                # Build result summary for logging
                tool_history = result.get("tool_history", [])
                tool_counts = Counter(t.get("tool_name", "unknown") for t in tool_history)
                tool_summary = ", ".join(f"{name} x{cnt}" for name, cnt in tool_counts.items())
                is_correct = result.get("is_correct", False)
                final_answer = result.get("final_answer")
                truth = result.get("truth")
                
                # Log per-video summary with tools and correctness
                logger.info(
                    f"[Video: {video_id}] Tools: [{tool_summary}] | "
                    f"Answer: {final_answer} | Truth: {truth} | Correct: {is_correct}"
                )
                
                # Send result to main process
                result_to_send = {
                    "video_id": result.get("video_id"),
                    "question": result.get("question"),
                    "final_answer": result.get("final_answer"),
                    "truth": result.get("truth"),
                    "is_correct": result.get("is_correct"),
                    "is_forced": result.get("is_forced", False),
                    "is_valid": result.get("is_valid", True),
                    "tool_call_count": result.get("tool_call_count", 0),
                    "tool_history": result.get("tool_history", []),
                    "frame_count": result.get("frame_count", 0),
                    "duration_s": result.get("duration_s"),
                    "explanation": result.get("explanation", ""),
                    "worker_id": worker_id,
                }
                if "error" in result:
                    result_to_send["error"] = result["error"]
                
                result_queue.put(("result", result_to_send))
                
            except Exception as e:
                logger.error(f"Error processing {video_id}: {e}")
                logger.error(traceback.format_exc())
                
                # Send error result
                error_result = {
                    "video_id": video_id,
                    "question": video_info["question"],
                    "final_answer": None,
                    "truth": video_info.get("truth"),
                    "is_valid": False,
                    "error": str(e),
                    "tool_call_count": 0,
                    "tool_history": [],
                    "frame_count": 0,
                    "worker_id": worker_id,
                }
                result_queue.put(("result", error_result))
            
            finally:
                # End video group
                logger.end_video_group()
        
        logger.info(f"Worker {worker_id} completed all tasks")
        
    except Exception as e:
        logger.error(f"Worker {worker_id} fatal error: {e}")
        logger.error(traceback.format_exc())
    
    finally:
        # Cleanup tool client
        if tool_client:
            tool_client.cleanup()
        
        # Signal completion
        result_queue.put(("worker_done", worker_id))


# ==============================================================================
# Main Evaluator Class
# ==============================================================================

class VideoAgentEvaluator:
    """
    Batch evaluator for VideoAgent.
    
    Handles:
    - Loading annotations
    - Running agent on multiple videos
    - Collecting and saving results
    - Progress tracking
    - Multiprocessing support (num_workers > 1)
    """
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        enabled_tools: List[str] = None,
        max_tool_calls: int = 10,
        max_parallel_tools: int = 3,
        initial_frames: int = 5,
        captioner: str = "gpt-4o-mini",
        output_dir: str = "results",
        experiment_name: str = "evaluation",
        detailed: bool = True,
        num_workers: int = 1,
        # Frame control parameters
        max_view_frames: int = 8,
        default_sample_frames: int = 16,
        min_sample_frames: int = 1,
        max_sample_frames: int = 32,
    ):
        """
        Initialize evaluator.
        
        Args:
            model: LLM model for agent
            enabled_tools: List of tool keys to enable
            max_tool_calls: Max tool calls per video
            max_parallel_tools: Maximum tools agent can request in a single turn
            initial_frames: Initial frames to caption
            captioner: Captioner model - 'omni-captioner' for local OmniCaptioner,
                      or API model name (e.g., 'gpt-4o-mini', 'x-ai/grok-4-1-fast-reasoning')
            output_dir: Base output directory
            experiment_name: Name for this experiment
            detailed: If True, print detailed info to console. If False, only progress bar.
                      Note: Files always save full details regardless of this setting.
            num_workers: Number of worker processes (1 = single process, same as before)
            max_view_frames: Maximum frames for view_frame tool
            default_sample_frames: Default number of frames for sampling tools
            min_sample_frames: Minimum number of frames for sampling tools
            max_sample_frames: Maximum number of frames for sampling tools
        """
        self.model = model
        # Default enabled tools (fully implemented)
        # See tools/interface/__init__.py INTERFACE_MAPPING for all available tools
        self.enabled_tools = enabled_tools or [
            "view_frame",
            "temporal_sample_frames",
            "temporal_spatial_sample_frames",
            "detect_objects",
            "detect_all_objects",
            "describe_region",
        ]
        self.max_tool_calls = max_tool_calls
        self.max_parallel_tools = max_parallel_tools
        self.initial_frames = initial_frames
        
        # Parse captioner setting: 'omni-captioner' = local, otherwise = API model
        self.captioner = captioner
        if captioner == "omni-captioner":
            self.use_api_captioner = False
            self.captioner_model = "U4R/OmniCaptioner"
            self.api_captioner_model = "gpt-4o-mini"  # Default, not used
        else:
            self.use_api_captioner = True
            self.captioner_model = "U4R/OmniCaptioner"  # Default, not used
            self.api_captioner_model = captioner
        
        self.output_dir_base = output_dir
        self.experiment_name = experiment_name
        self.detailed = detailed
        self.num_workers = max(1, num_workers)
        
        # Frame control parameters
        self.max_view_frames = max_view_frames
        self.default_sample_frames = default_sample_frames
        self.min_sample_frames = min_sample_frames
        self.max_sample_frames = max_sample_frames
        
        # Will be set during run
        self.output_dir: Optional[str] = None
        self.videos_dir: Optional[str] = None
        self.logger: Optional[logging.Logger] = None
        self.agent: Optional[VideoAgent] = None
        self.stats: Optional[EvalStatistics] = None
    
    def _get_config_dict(self) -> Dict[str, Any]:
        """Get configuration as dict for passing to workers.
        
        Note: Workers are lightweight and don't directly use GPUs.
        GPU tools are managed centrally by ToolServer.
        """
        return {
            "model": self.model,
            "enabled_tools": self.enabled_tools,
            "max_tool_calls": self.max_tool_calls,
            "max_parallel_tools": self.max_parallel_tools,
            "initial_frames": self.initial_frames,
            "captioner": self.captioner,
            "detailed": self.detailed,
            "max_view_frames": self.max_view_frames,
            "default_sample_frames": self.default_sample_frames,
            "min_sample_frames": self.min_sample_frames,
            "max_sample_frames": self.max_sample_frames,
        }
    
    def run(
        self,
        annotation_file: str,
        video_dir: str,
        video_list_file: str = None,
        max_videos: int = -1,
        restore_path: str = None,
    ) -> str:
        """
        Run evaluation on videos.
        
        Args:
            annotation_file: Path to annotations JSON
            video_dir: Directory containing video files
            video_list_file: Optional file with video IDs to include
            max_videos: Maximum videos to process (-1 for all)
            restore_path: Optional path to previous result directory to restore from
        
        Returns:
            Output directory path
        """
        # Handle restore mode - set output directory first if restoring
        restored_results = []
        completed_video_ids = set()
        
        # Create output directory (must be done before logger initialization)
        if restore_path:
            # Use the restore_path as output directory (will update results in place)
            self.output_dir = restore_path
            self.videos_dir = os.path.join(self.output_dir, "videos")
            os.makedirs(self.videos_dir, exist_ok=True)
        else:
            # Create new output directory
            timestamp = datetime.now().strftime("%m%d")
            model_short = os.path.basename(self.model)
            count_str = str(max_videos) if max_videos > 0 else "all"
            
            base_dir_name = f"{self.experiment_name}__{model_short}_videos_{count_str}_{timestamp}"
            self.output_dir = os.path.join(self.output_dir_base, base_dir_name)
            
            # Check if directory exists, add HHMM suffix if needed
            if os.path.exists(self.output_dir):
                time_suffix = datetime.now().strftime("%H%M")
                dir_name = f"{base_dir_name}_{time_suffix}"
                self.output_dir = os.path.join(self.output_dir_base, dir_name)
            
            os.makedirs(self.output_dir, exist_ok=True)
            
            # Create videos subdirectory for per-video outputs
            self.videos_dir = os.path.join(self.output_dir, "videos")
            os.makedirs(self.videos_dir, exist_ok=True)
        
        # Setup logging (must be done after output_dir is set)
        # Main logger writes to logging.log only (no console output)
        # Console output is handled by simple_logger when detailed=True
        self.logger = setup_logger(
            name="VideoAgentTools",
            output_dir=self.output_dir,
            level="INFO",
            console_output=False,  # Never output to console from main logger
        )
        
        # Now load restore results (after logger is initialized)
        if restore_path:
            restored_results, completed_video_ids = self._load_restore_results(restore_path)
            print(f"[RESTORE] Loaded {len(restored_results)} completed videos from {restore_path}")
            print(f"[RESTORE] Completed video IDs: {len(completed_video_ids)}")
            self.logger.info(f"[RESTORE] Loading results from {restore_path}")
            self.logger.info(f"[RESTORE] Found {len(restored_results)} completed videos")
        
        # Log experiment info to file
        self.logger.info("=" * 70)
        if restore_path:
            self.logger.info("VideoAgent Tools Evaluation (RESTORE MODE)")
        else:
            self.logger.info("VideoAgent Tools Evaluation")
        self.logger.info("=" * 70)
        self.logger.info(f"Experiment: {self.experiment_name}")
        self.logger.info(f"Model: {self.model}")
        self.logger.info(f"Enabled tools: {self.enabled_tools}")
        self.logger.info(f"Max tool calls: {self.max_tool_calls}")
        self.logger.info(f"Max parallel tools: {self.max_parallel_tools}")
        self.logger.info(f"Initial frames: {self.initial_frames}")
        self.logger.info(f"Num workers: {self.num_workers}")
        self.logger.info(f"Output directory: {self.output_dir}")
        if restore_path:
            self.logger.info(f"Restore path: {restore_path}")
            self.logger.info(f"Restored videos: {len(restored_results)}")
        self.logger.info("=" * 70)
        
        # Print experiment info to console (always, regardless of detailed)
        print("=" * 60)
        if restore_path:
            print("VideoAgent Tools Evaluation (RESTORE MODE)")
        else:
            print("VideoAgent Tools Evaluation")
        print("=" * 60)
        print(f"Experiment: {self.experiment_name}")
        print(f"Model: {self.model}")
        print(f"Max tool calls: {self.max_tool_calls}")
        print(f"Num workers: {self.num_workers}")
        print(f"Output: {self.output_dir}")
        if restore_path:
            print(f"Restore path: {restore_path}")
            print(f"Restored videos: {len(restored_results)}")
        print(f"Detailed console output: {self.detailed}")
        print("=" * 60)
        
        # Save config
        self._save_config()
        
        # Load annotations
        all_videos_info = parse_video_annotation(
            annotation_file=annotation_file,
            video_list_file=video_list_file,
            max_videos=max_videos,
        )
        
        # Filter out completed videos if restoring
        if restore_path:
            videos_info = [
                v for v in all_videos_info 
                if v["video_id"] not in completed_video_ids
            ]
            self.logger.info(f"Total videos in dataset: {len(all_videos_info)}")
            self.logger.info(f"Completed videos: {len(completed_video_ids)}")
            self.logger.info(f"Remaining videos to process: {len(videos_info)}")
            print(f"[RESTORE] Total videos: {len(all_videos_info)}")
            print(f"[RESTORE] Completed: {len(completed_video_ids)}")
            print(f"[RESTORE] Remaining: {len(videos_info)}")
        else:
            videos_info = all_videos_info
        
        self.logger.info(f"Loaded {len(videos_info)} videos for evaluation")
        print(f"Loaded {len(videos_info)} videos for evaluation")
        
        # Initialize statistics with restored results if any
        self.stats = EvalStatistics()
        if restored_results:
            for result in restored_results:
                self.stats.update(result)
            self.logger.info(f"[RESTORE] Initialized stats with {len(restored_results)} restored videos")
            print(f"[RESTORE] Initialized stats: {self.stats.total} videos, {self.stats.correct} correct")
        
        # Run based on number of workers (only if there are videos to process)
        new_results = []
        if videos_info:
            if self.num_workers == 1:
                # Single process mode (original behavior)
                new_results = self._run_single_process(videos_info, video_dir)
            else:
                # Multi-process mode
                new_results = self._run_multi_process(videos_info, video_dir)
        else:
            self.logger.info("[RESTORE] No remaining videos to process")
            print("[RESTORE] No remaining videos to process")
        
        # Merge restored and new results
        all_results = restored_results + new_results
        
        # Save results (merged)
        self._save_results(all_results)
        
        # Print summary
        self._print_summary()
        
        self.logger.info(f"Evaluation complete. Results in: {self.output_dir}")
        print(f"\nEvaluation complete. Results in: {self.output_dir}")
        
        return self.output_dir
    
    def _run_single_process(
        self,
        videos_info: List[Dict[str, Any]],
        video_dir: str,
    ) -> List[Dict[str, Any]]:
        """
        Run evaluation in single process mode (original behavior).
        
        Args:
            videos_info: List of video info dicts
            video_dir: Directory containing video files
            
        Returns:
            List of result dicts
        """
        # Initialize agent
        self.agent = VideoAgent(
            model=self.model,
            enabled_tools=self.enabled_tools,
            max_tool_calls=self.max_tool_calls,
            max_parallel_tools=self.max_parallel_tools,
            initial_frames=self.initial_frames,
            captioner=self.captioner,
            logger=self.logger,
            # Frame control parameters
            max_view_frames=self.max_view_frames,
            default_sample_frames=self.default_sample_frames,
            min_sample_frames=self.min_sample_frames,
            max_sample_frames=self.max_sample_frames,
        )
        
        # Initialize statistics
        self.stats = EvalStatistics()
        total_videos = len(videos_info)
        stats_log_interval = max(1, total_videos // 20)  # Log stats every 5% progress
        
        # Process videos
        results = []
        
        pbar = tqdm(
            videos_info,
            desc="Processing",
            unit="video",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
        )
        
        for video_info in pbar:
            video_id = video_info["video_id"]
            video_path = os.path.join(video_dir, f"{video_id}.mp4")
            
            # Create per-video output directory
            video_output_dir = os.path.join(self.videos_dir, video_id)
            os.makedirs(video_output_dir, exist_ok=True)
            
            # Check if video exists
            if not os.path.exists(video_path):
                self.logger.warning(f"Video not found: {video_path}")
                result = {
                    "video_id": video_id,
                    "question": video_info["question"],
                    "final_answer": None,
                    "truth": video_info.get("truth"),
                    "is_valid": False,
                    "error": "Video file not found",
                    "choices": video_info["choices"],
                    "tool_history": [],
                    "video_context_dict": {},
                    "messages": [],
                }
            else:
                # Process video with per-video output directory
                result = self.agent.process_video(
                    video_path=video_path,
                    video_id=video_id,
                    question=video_info["question"],
                    choices=video_info["choices"],
                    truth=video_info.get("truth"),
                    video_output_dir=video_output_dir,
                    detailed=self.detailed,
                )
            
            # Save per-video result.json
            save_video_result(
                video_output_dir=video_output_dir,
                video_id=video_id,
                question=video_info["question"],
                choices=video_info["choices"],
                final_answer=result.get("final_answer"),
                explanation=result.get("explanation", ""),
                truth=video_info.get("truth"),
                is_correct=result.get("is_correct", False),
                is_forced=result.get("is_forced", False),
                is_valid=result.get("is_valid", True),
                tool_history=result.get("tool_history", []),
                video_context_dict=result.get("video_context_dict", {}),
                messages=result.get("messages", []),
                duration_s=result.get("duration_s"),
                tool_call_rounds=result.get("tool_call_rounds"),
            )
            
            # Save frames if available (with timestamp in filename)
            if "frames" in result and result["frames"]:
                save_video_frames(
                    video_output_dir=video_output_dir,
                    frames=result["frames"],
                    video_id=video_id,
                    fps=result.get("fps", 30.0),
                )
            
            results.append(result)
            self.stats.update(result)
            
            # Log per-video summary
            is_correct = result.get("is_correct", False)
            final_answer = result.get("final_answer")
            truth = result.get("truth")
            tool_history = result.get("tool_history", [])
            tool_counts = Counter(t.get("tool_name", "unknown") for t in tool_history)
            tool_summary = ", ".join(f"{name} x{cnt}" for name, cnt in tool_counts.items()) if tool_counts else "none"
            
            self.logger.info(
                f"[Video: {video_id}] Tools: [{tool_summary}] | "
                f"Answer: {final_answer} | Truth: {truth} | Correct: {is_correct}"
            )
            
            # Update progress bar
            pbar.set_postfix(self.stats.get_progress_dict())
            
            # Log running statistics periodically
            completed = self.stats.total
            if completed % stats_log_interval == 0 or completed == total_videos:
                progress_dict = self.stats.get_progress_dict()
                stats_line = (
                    f"[PROGRESS {completed}/{total_videos}] "
                    f"Acc={progress_dict['Acc']}, "
                    f"Tools={progress_dict['Tools']}, "
                    f"Valid={progress_dict['Valid']}, "
                    f"Forced={progress_dict['Forced']}"
                )
                self.logger.info(stats_line)
                print(stats_line)
        
        return results
    
    def _run_multi_process(
        self,
        videos_info: List[Dict[str, Any]],
        video_dir: str,
    ) -> List[Dict[str, Any]]:
        """
        Run evaluation in multi-process mode with centralized ToolServer.
        
        Architecture:
        - ToolServer: Single process managing all GPU tools across all GPUs
        - Workers: Lightweight processes doing LLM calls, access tools via ToolClient
        - LogCollector: Centralized logging from all workers
        
        Args:
            videos_info: List of video info dicts
            video_dir: Directory containing video files
            
        Returns:
            List of result dicts
        """
        print(f"\nStarting multiprocessing with {self.num_workers} workers and centralized ToolServer...")
        self.logger.info(f"Starting multiprocessing with {self.num_workers} workers")
        
        # Initialize statistics
        self.stats = EvalStatistics()
        
        # Create queues for IPC
        result_queue = Queue()
        log_queue = Queue()
        shutdown_event = Event()
        tool_server_shutdown = Event()
        
        # Create tool request queue (shared by all workers)
        tool_request_queue = Queue()
        
        # Create per-worker response queues
        tool_response_queues = {i: Queue() for i in range(self.num_workers)}
        
        # Start log collector
        log_collector = start_log_collector(
            log_queue=log_queue,
            output_dir=self.output_dir,
            num_workers=self.num_workers,
            console_output=self.detailed,
        )
        
        # Start ToolServer process
        print("Starting ToolServer for centralized GPU tool management...")
        self.logger.info("Starting ToolServer process")
        
        tool_server_process = Process(
            target=run_tool_server,
            args=(
                tool_request_queue,
                tool_response_queues,
                tool_server_shutdown,
                self.enabled_tools,
                self.captioner,
                log_queue,  # ToolServer can also log to collector
                self.output_dir,  # For resource_management.log
                self.max_view_frames,
                self.default_sample_frames,
                self.min_sample_frames,
                self.max_sample_frames,
            ),
            daemon=True,
        )
        tool_server_process.start()
        self.logger.info(f"ToolServer started (PID: {tool_server_process.pid})")
        print(f"ToolServer started (PID: {tool_server_process.pid})")
        
        # Give ToolServer time to initialize
        import time
        time.sleep(2.0)
        
        # Distribute videos to workers (round-robin)
        worker_tasks: List[List[Dict[str, Any]]] = [[] for _ in range(self.num_workers)]
        for i, video_info in enumerate(videos_info):
            worker_id = i % self.num_workers
            worker_tasks[worker_id].append(video_info)
        
        # Get config for workers
        config = self._get_config_dict()
        
        # Start worker processes
        workers = []
        for worker_id in range(self.num_workers):
            p = Process(
                target=worker_process,
                args=(
                    worker_id,
                    worker_tasks[worker_id],
                    video_dir,
                    self.videos_dir,
                    result_queue,
                    log_queue,
                    shutdown_event,
                    config,
                    tool_request_queue,  # Shared request queue
                    tool_response_queues[worker_id],  # Worker-specific response queue
                ),
                daemon=True,
            )
            p.start()
            workers.append(p)
            self.logger.info(f"Started worker {worker_id} (PID: {p.pid}) with {len(worker_tasks[worker_id])} videos")
        
        print(f"Started {self.num_workers} worker processes")
        
        # Collect results
        results = []
        workers_done = 0
        total_videos = len(videos_info)
        stats_log_interval = max(1, total_videos // 20)  # Log stats every 5% progress
        
        pbar = tqdm(
            total=total_videos,
            desc="Processing",
            unit="video",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
        )
        
        try:
            while workers_done < self.num_workers:
                try:
                    msg_type, msg_data = result_queue.get(timeout=1.0)
                    
                    if msg_type == "result":
                        results.append(msg_data)
                        self.stats.update(msg_data)
                        pbar.update(1)
                        pbar.set_postfix(self.stats.get_progress_dict())
                        
                        # Log per-video summary to main log
                        video_id = msg_data.get("video_id", "unknown")
                        is_correct = msg_data.get("is_correct", False)
                        final_answer = msg_data.get("final_answer")
                        truth = msg_data.get("truth")
                        tool_history = msg_data.get("tool_history", [])
                        tool_counts = Counter(t.get("tool_name", "unknown") for t in tool_history)
                        tool_summary = ", ".join(f"{name} x{cnt}" for name, cnt in tool_counts.items())
                        
                        self.logger.info(
                            f"[Video: {video_id}] Tools: [{tool_summary}] | "
                            f"Answer: {final_answer} | Truth: {truth} | Correct: {is_correct}"
                        )
                        
                        # Log running statistics periodically
                        completed = self.stats.total
                        if completed % stats_log_interval == 0 or completed == total_videos:
                            progress_dict = self.stats.get_progress_dict()
                            stats_line = (
                                f"[PROGRESS {completed}/{total_videos}] "
                                f"Acc={progress_dict['Acc']}, "
                                f"Tools={progress_dict['Tools']}, "
                                f"Valid={progress_dict['Valid']}, "
                                f"Forced={progress_dict['Forced']}"
                            )
                            self.logger.info(stats_line)
                            print(stats_line)
                        
                    elif msg_type == "worker_done":
                        workers_done += 1
                        self.logger.info(f"Worker {msg_data} finished ({workers_done}/{self.num_workers})")
                        
                except queue.Empty:
                    # Check if any workers crashed
                    for i, p in enumerate(workers):
                        if not p.is_alive() and i not in [r.get("worker_id") for r in results if "worker_id" in r]:
                            self.logger.warning(f"Worker {i} appears to have crashed")
                    
                    # Check if ToolServer crashed
                    if not tool_server_process.is_alive():
                        self.logger.error("ToolServer process died unexpectedly!")
                        break
                    continue
                    
        except KeyboardInterrupt:
            print("\nInterrupted! Shutting down...")
            shutdown_event.set()
            
        finally:
            pbar.close()
            
            # Signal shutdown to workers
            shutdown_event.set()
            
            # Wait for workers to finish
            for p in workers:
                p.join(timeout=10.0)
                if p.is_alive():
                    p.terminate()
            
            # Shutdown ToolServer
            self.logger.info("Shutting down ToolServer...")
            tool_server_shutdown.set()
            
            # Send shutdown message to ToolServer
            try:
                import uuid
                shutdown_request = ToolRequest(
                    msg_type=MessageType.SHUTDOWN,
                    request_id=str(uuid.uuid4()),
                    worker_id=-1,
                )
                tool_request_queue.put(shutdown_request)
            except:
                pass
            
            tool_server_process.join(timeout=10.0)
            if tool_server_process.is_alive():
                tool_server_process.terminate()
            
            # Shutdown log collector
            log_collector.shutdown()
        
        self.logger.info(f"All processes finished. Collected {len(results)} results.")
        
        return results
    
    def _load_restore_results(self, restore_path: str) -> Tuple[List[Dict[str, Any]], set]:
        """
        Load results from previous run for restoration.
        
        Args:
            restore_path: Path to previous result directory
        
        Returns:
            Tuple of (list of result dicts, set of completed video IDs)
        """
        result_file = os.path.join(restore_path, "result.json")
        if not os.path.exists(result_file):
            raise FileNotFoundError(f"Result file not found: {result_file}")
        
        with open(result_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        results = data.get("results", {})
        restored_results = []
        completed_video_ids = set()
        
        for video_id, video_result in results.items():
            # Consider a video completed if it has a valid final_answer
            final_answer = video_result.get("final_answer")
            is_valid = video_result.get("is_valid", True)
            
            if final_answer is not None and is_valid:
                # Convert back to result dict format expected by stats.update()
                result_dict = {
                    "video_id": video_id,
                    "question": video_result.get("question", ""),
                    "final_answer": final_answer,
                    "explanation": video_result.get("explanation", ""),
                    "truth": video_result.get("truth"),
                    "is_correct": video_result.get("is_correct", False),
                    "is_forced": video_result.get("is_forced", False),
                    "is_valid": is_valid,
                    "tool_call_count": video_result.get("tool_call_count", 0),
                    "tool_history": video_result.get("tool_history", []),
                    "frame_count": video_result.get("frame_count", 0),
                    "duration_s": video_result.get("duration_s"),
                }
                if "error" in video_result:
                    result_dict["error"] = video_result["error"]
                
                restored_results.append(result_dict)
                completed_video_ids.add(video_id)
        
        return restored_results, completed_video_ids
    
    def _save_config(self):
        """Save experiment configuration."""
        config = {
            "experiment_name": self.experiment_name,
            "model": self.model,
            "enabled_tools": self.enabled_tools,
            "max_tool_calls": self.max_tool_calls,
            "initial_frames": self.initial_frames,
            "captioner": self.captioner,
            "num_workers": self.num_workers,
            "timestamp": datetime.now().isoformat(),
        }
        
        config_file = os.path.join(self.output_dir, "experiment_config.yaml")
        import yaml
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    
    def _save_results(self, results: List[Dict[str, Any]]):
        """Save all result files."""
        stats_dict = self.stats.to_dict()
        
        # 1. Save result.json
        video_results = {}
        for result in results:
            video_id = result.get("video_id", "unknown")
            video_results[video_id] = {
                "question": result.get("question", ""),
                "final_answer": result.get("final_answer"),
                "explanation": result.get("explanation", ""),
                "truth": result.get("truth"),
                "is_correct": result.get("is_correct"),
                "is_forced": result.get("is_forced", False),
                "is_valid": result.get("is_valid", True),
                "tool_call_count": result.get("tool_call_count", 0),
                "tool_history": result.get("tool_history", []),
                "frame_count": result.get("frame_count", 0),
                "duration_s": result.get("duration_s"),
            }
            if "error" in result:
                video_results[video_id]["error"] = result["error"]
        
        result_output = {
            "_summary": stats_dict,
            "results": video_results,
        }
        
        result_file = os.path.join(self.output_dir, "result.json")
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result_output, f, indent=2, ensure_ascii=False)
        
        # 2. Save metrics.csv
        metrics_file = os.path.join(self.output_dir, "metrics.csv")
        with open(metrics_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            writer.writerow(["total_videos", stats_dict["total"]])
            writer.writerow(["valid_videos", stats_dict["valid"]])
            writer.writerow(["invalid_videos", stats_dict["invalid"]])
            writer.writerow(["correct_answers", stats_dict["correct"]])
            writer.writerow(["accuracy", f"{stats_dict['accuracy']:.4f}"])
            writer.writerow(["forced_answers", stats_dict["forced_answers"]])
            writer.writerow(["forced_rate", f"{stats_dict['forced_rate']:.4f}"])
            writer.writerow(["avg_tool_calls", f"{stats_dict['avg_tool_calls']:.2f}"])
            writer.writerow(["avg_frames", f"{stats_dict['avg_frames']:.2f}"])
            writer.writerow(["total_tool_calls", stats_dict["total_tool_calls"]])
            writer.writerow(["total_frames", stats_dict["total_frames"]])
            writer.writerow(["num_workers", self.num_workers])
        
        # 3. Save summary.txt
        summary_file = os.path.join(self.output_dir, "summary.txt")
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("VideoAgent Tools Evaluation Summary\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Output Directory: {self.output_dir}\n\n")
            
            f.write("-" * 70 + "\n")
            f.write("Configuration\n")
            f.write("-" * 70 + "\n")
            f.write(f"Model: {self.model}\n")
            f.write(f"Enabled Tools: {', '.join(self.enabled_tools)}\n")
            f.write(f"Max Tool Calls: {self.max_tool_calls}\n")
            f.write(f"Initial Frames: {self.initial_frames}\n")
            f.write(f"Num Workers: {self.num_workers}\n\n")
            
            f.write("-" * 70 + "\n")
            f.write("Overall Results\n")
            f.write("-" * 70 + "\n")
            f.write(f"Total Videos:       {stats_dict['total']}\n")
            f.write(f"Valid Videos:       {stats_dict['valid']}\n")
            f.write(f"Invalid Videos:     {stats_dict['invalid']}\n")
            f.write(f"Correct Answers:    {stats_dict['correct']}\n\n")
            
            f.write("-" * 70 + "\n")
            f.write("Accuracy Metrics\n")
            f.write("-" * 70 + "\n")
            f.write(f"Accuracy:           {stats_dict['accuracy']:.2%} ({stats_dict['correct']}/{stats_dict['valid']})\n")
            f.write(f"Forced Answers:     {stats_dict['forced_answers']} ({stats_dict['forced_rate']:.2%})\n\n")
            
            f.write("-" * 70 + "\n")
            f.write("Resource Usage\n")
            f.write("-" * 70 + "\n")
            f.write(f"Avg Tool Calls/Video: {stats_dict['avg_tool_calls']:.2f}\n")
            f.write(f"Avg Frames/Video:     {stats_dict['avg_frames']:.1f}\n")
            f.write(f"Total Tool Calls:     {stats_dict['total_tool_calls']}\n")
            f.write(f"Total Frames:         {stats_dict['total_frames']}\n\n")
            
            if stats_dict["tool_usage"]:
                f.write("-" * 70 + "\n")
                f.write("Tool Usage\n")
                f.write("-" * 70 + "\n")
                for tool, count in sorted(stats_dict["tool_usage"].items(), key=lambda x: -x[1]):
                    f.write(f"  {tool}: {count}\n")
                f.write("\n")
            
            f.write("=" * 70 + "\n")
        
        # 4. Save accuracy.txt (for compatibility)
        accuracy_file = os.path.join(self.output_dir, "accuracy.txt")
        with open(accuracy_file, 'w', encoding='utf-8') as f:
            f.write(f"number_videos: {stats_dict['total']}\n")
            f.write(f"mean_accuracy: {stats_dict['accuracy']}\n")
            f.write(f"mean_tool_calls: {stats_dict['avg_tool_calls']}\n")
            f.write(f"mean_frames: {stats_dict['avg_frames']}\n")
            f.write(f"forced_answers: {stats_dict['forced_answers']}\n")
            f.write(f"invalid_videos: {stats_dict['invalid']}\n")
            f.write(f"num_workers: {self.num_workers}\n")
    
    def _print_summary(self):
        """Print summary to logger and console."""
        stats_dict = self.stats.to_dict()
        
        # Log to file
        self.logger.info("")
        self.logger.info("=" * 70)
        self.logger.info("EVALUATION SUMMARY")
        self.logger.info("=" * 70)
        self.logger.info(f"Total Videos:    {stats_dict['total']}")
        self.logger.info(f"Valid Videos:    {stats_dict['valid']}")
        self.logger.info(f"Correct:         {stats_dict['correct']}")
        self.logger.info(f"Accuracy:        {stats_dict['accuracy']:.2%}")
        self.logger.info(f"Forced Answers:  {stats_dict['forced_answers']}")
        self.logger.info(f"Avg Tool Calls:  {stats_dict['avg_tool_calls']:.2f}")
        self.logger.info(f"Avg Frames:      {stats_dict['avg_frames']:.1f}")
        self.logger.info(f"Num Workers:     {self.num_workers}")
        self.logger.info("=" * 70)
        
        # Also print to console (always)
        print("")
        print("=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Total Videos:    {stats_dict['total']}")
        print(f"Valid Videos:    {stats_dict['valid']}")
        print(f"Correct:         {stats_dict['correct']}")
        print(f"Accuracy:        {stats_dict['accuracy']:.2%}")
        print(f"Forced Answers:  {stats_dict['forced_answers']}")
        print(f"Avg Tool Calls:  {stats_dict['avg_tool_calls']:.2f}")
        print(f"Avg Frames:      {stats_dict['avg_frames']:.1f}")
        print(f"Num Workers:     {self.num_workers}")
        print("=" * 60)
