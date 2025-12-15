"""
InternVideo2.5 Multi-Interface Module

Provides four interfaces (all prefixed with InternVideo for clarity):
1. InternVideoGeneralQA - General video Q&A with 128 frames
2. InternVideoTemporalQA - Temporal localization Q&A with precise time segments
3. InternVideoObjectTracking - Video object tracking with trajectory output
4. InternVideoDescription - Video summary with action timeline

All interfaces support long videos through chunked processing.
"""

import os
import sys
import re
import logging
import warnings
import gc
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image as PILImage
from decord import VideoReader, cpu
from torchvision.transforms.functional import InterpolationMode

from tools.interface_base import Interface, InterfaceCategory, Video, Image, BoundingBox

# Suppress warnings
warnings.filterwarnings("ignore")
logging.getLogger("httpx").setLevel(logging.WARNING)

# Constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# =============================================================================
# Data Classes for Structured Output
# =============================================================================

@dataclass
class TemporalSegment:
    """Represents a temporal segment with confidence."""
    start_sec: float
    end_sec: float
    confidence: float
    description: str = ""


@dataclass
class TrackingPoint:
    """Represents a point in object trajectory."""
    frame_id: int
    timestamp: float
    bbox: BoundingBox
    confidence: float


# =============================================================================
# InternVideo2.5 Base Class (Shared Model Wrapper)
# =============================================================================

class InternVideo2_5Base:
    """Base class providing InternVideo2.5 model utilities.
    
    Handles model loading, caching, video preprocessing, and chunked processing.
    """
    
    MODEL_PATH = "/root/autodl-tmp/VideoAgent/tools/models/InternVideo2_5_Chat_8B"
    DEFAULT_DEVICE = None  # Auto-select
    
    # Class-level model cache (shared across all instances)
    _model_cache: Dict[str, Any] = {}
    _tokenizer_cache: Dict[str, Any] = {}
    
    def __init__(self, device: str = None):
        self.device = self._resolve_device(device or self.DEFAULT_DEVICE)
        self._initialized = False
        self._model_dtype = None

    @staticmethod
    def _resolve_device(device: Optional[str]) -> str:
        """Choose a valid device string with graceful fallback."""
        if device:
            try:
                torch.device(device)
                return device
            except Exception:
                logging.warning(f"Invalid device '{device}', falling back to auto selection.")
        if torch.cuda.is_available():
            return "cuda:0"
        return "cpu"

    @staticmethod
    def _select_dtype(device: str) -> torch.dtype:
        """Select preferred dtype based on device capability."""
        if device.startswith("cuda") and torch.cuda.is_available():
            if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
                return torch.bfloat16
            return torch.float16
        return torch.float32
    
    def initialize(self) -> None:
        """Initialize InternVideo2.5 model with caching."""
        if self._initialized and self.device in InternVideo2_5Base._model_cache:
            return
        
        self.device = self._resolve_device(self.device)
        
        if self.device in InternVideo2_5Base._model_cache:
            logging.info(f"Using cached InternVideo2.5 model on {self.device}")
            cached_model = InternVideo2_5Base._model_cache[self.device]
            try:
                self._model_dtype = next(cached_model.parameters()).dtype
            except StopIteration:
                self._model_dtype = torch.bfloat16
            self._initialized = True
            return
        
        logging.info(f"Loading InternVideo2.5 model on {self.device}...")
        
        from transformers import AutoModel, AutoTokenizer
        
        tokenizer = InternVideo2_5Base._tokenizer_cache.get("shared")
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(
                self.MODEL_PATH,
                trust_remote_code=True
            )
        
        self._model_dtype = self._select_dtype(self.device)
        
        model = AutoModel.from_pretrained(
            self.MODEL_PATH,
            trust_remote_code=True,
            torch_dtype=self._model_dtype
        ).to(self.device)
        model.eval()
        
        InternVideo2_5Base._model_cache[self.device] = model
        InternVideo2_5Base._tokenizer_cache["shared"] = tokenizer
        
        logging.info(f"InternVideo2.5 model loaded on {self.device}")
        self._initialized = True
    
    @property
    def model(self):
        return InternVideo2_5Base._model_cache.get(self.device)
    
    @property
    def tokenizer(self):
        return InternVideo2_5Base._tokenizer_cache.get("shared") or InternVideo2_5Base._tokenizer_cache.get(self.device)
    
    def clear_gpu_cache(self) -> None:
        """Clear GPU memory cache."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def cleanup(self) -> None:
        """Release model from GPU memory for resource management.
        
        This method is called by GPUResourceManager when the tool needs to be
        unloaded to free memory for other tools.
        """
        if self.device in InternVideo2_5Base._model_cache:
            logging.info(f"Releasing InternVideo2.5 model from {self.device}")
            del InternVideo2_5Base._model_cache[self.device]
        
        self._initialized = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    @classmethod
    def cleanup_all(cls) -> None:
        """Release all cached models (for complete cleanup)."""
        for device in list(cls._model_cache.keys()):
            logging.info(f"Releasing InternVideo2.5 model from {device}")
            del cls._model_cache[device]
        cls._tokenizer_cache.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    @staticmethod
    def build_transform(input_size: int = 448):
        """Build image transform for InternVideo2.5."""
        return T.Compose([
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    
    @staticmethod
    def dynamic_preprocess(image: PILImage.Image, min_num=1, max_num=1, 
                           image_size=448, use_thumbnail=True) -> List[PILImage.Image]:
        """Dynamically preprocess image with aspect ratio handling."""
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height
        
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if i * j <= max_num and i * j >= min_num
        )
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
        
        # Find closest aspect ratio
        best_ratio = (1, 1)
        best_ratio_diff = float("inf")
        for ratio in target_ratios:
            target_aspect = ratio[0] / ratio[1]
            diff = abs(aspect_ratio - target_aspect)
            if diff < best_ratio_diff:
                best_ratio_diff = diff
                best_ratio = ratio
        
        target_width = image_size * best_ratio[0]
        target_height = image_size * best_ratio[1]
        blocks = best_ratio[0] * best_ratio[1]
        
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            processed_images.append(resized_img.crop(box))
        
        if use_thumbnail and len(processed_images) != 1:
            processed_images.append(image.resize((image_size, image_size)))
        
        return processed_images
    
    def load_video_segment(
        self, 
        video_path: str, 
        start_sec: float = 0, 
        end_sec: float = None,
        fps: float = 1.0,
        max_frames: int = 128
    ) -> Tuple[torch.Tensor, List[int], List[float]]:
        """Load a segment of video with specified fps.
        
        Args:
            video_path: Path to video file
            start_sec: Start time in seconds
            end_sec: End time in seconds (None = end of video)
            fps: Frames per second to sample
            max_frames: Maximum number of frames to return
            
        Returns:
            Tuple of (pixel_values, num_patches_list, timestamps)
        """
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        video_fps = float(vr.get_avg_fps())
        total_frames = len(vr)
        duration = total_frames / video_fps
        
        if end_sec is None:
            end_sec = duration
        
        # Calculate frame indices
        start_frame = int(start_sec * video_fps)
        end_frame = min(int(end_sec * video_fps), total_frames - 1)
        
        # Sample frames at specified fps
        frame_interval = max(1, int(video_fps / fps))
        frame_indices = list(range(start_frame, end_frame + 1, frame_interval))
        
        # Limit frames
        if len(frame_indices) > max_frames:
            step = len(frame_indices) / max_frames
            frame_indices = [frame_indices[int(i * step)] for i in range(max_frames)]
        
        # Ensure frame count is multiple of 4 (required by InternVideo2.5)
        LOCAL_NUM_FRAMES = 4
        remainder = len(frame_indices) % LOCAL_NUM_FRAMES
        if remainder != 0:
            # Trim to nearest multiple of 4
            frame_indices = frame_indices[:len(frame_indices) - remainder]
        
        # Ensure at least 4 frames
        if len(frame_indices) < LOCAL_NUM_FRAMES:
            # Pad by repeating last frame indices
            while len(frame_indices) < LOCAL_NUM_FRAMES:
                frame_indices.append(frame_indices[-1] if frame_indices else 0)
        
        # Load and process frames
        transform = self.build_transform()
        pixel_values_list = []
        num_patches_list = []
        timestamps = []
        
        for frame_idx in frame_indices:
            img = PILImage.fromarray(vr[frame_idx].asnumpy()).convert("RGB")
            img_list = self.dynamic_preprocess(img)
            pixel_values = torch.stack([transform(tile) for tile in img_list])
            pixel_values_list.append(pixel_values)
            num_patches_list.append(pixel_values.shape[0])
            timestamps.append(frame_idx / video_fps)
        
        # Ensure frame count is multiple of 4 after potential early stop
        remainder_frames = len(pixel_values_list) % LOCAL_NUM_FRAMES
        if remainder_frames:
            pixel_values_list = pixel_values_list[:-remainder_frames]
            num_patches_list = num_patches_list[:-remainder_frames]
            timestamps = timestamps[:-remainder_frames]
        
        # Guarantee at least 4 frames by padding duplicates if needed
        while len(pixel_values_list) < LOCAL_NUM_FRAMES and len(pixel_values_list) > 0:
            pixel_values_list.append(pixel_values_list[-1])
            num_patches_list.append(num_patches_list[-1])
            timestamps.append(timestamps[-1])
        
        if not pixel_values_list:
            raise ValueError("Failed to load any frames from video segment.")
        
        pixel_values = torch.cat(pixel_values_list)
        return pixel_values, num_patches_list, timestamps
    
    def get_video_info(self, video_path: str) -> Dict[str, Any]:
        """Get video metadata."""
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        fps = float(vr.get_avg_fps())
        total_frames = len(vr)
        duration = total_frames / fps
        return {
            "fps": fps,
            "total_frames": total_frames,
            "duration": duration,
            "width": vr[0].shape[1],
            "height": vr[0].shape[0]
        }
    
    def query_video(
        self,
        pixel_values: torch.Tensor,
        num_patches_list: List[int],
        question: str,
        max_new_tokens: int = 512
    ) -> str:
        """Query the model about video content.
        
        Args:
            pixel_values: Preprocessed video frames
            num_patches_list: Number of patches per frame
            question: Question to ask
            max_new_tokens: Maximum tokens in response
            
        Returns:
            Model response string
        """
        # Check if model needs (re-)initialization
        # Model could be None if evicted from cache or not yet loaded
        if not self._initialized or self.model is None:
            self._initialized = False  # Reset to force full initialization
            self.initialize()
        
        if self.model is None:
            raise RuntimeError(f"Failed to initialize InternVideo2.5 model on {self.device}")
        
        with torch.no_grad():
            target_dtype = self._model_dtype or torch.bfloat16
            pixel_values = pixel_values.to(device=self.device, dtype=target_dtype)
            
            # Create frame prefix
            video_prefix = "".join([
                f"Frame{i+1}: <image>\n"
                for i in range(len(num_patches_list))
            ])
            
            full_question = video_prefix + question
            
            generation_config = {
                "do_sample": False,
                "temperature": 0.0,
                "max_new_tokens": max_new_tokens,
                "top_p": 0.1,
                "num_beams": 1
            }
            
            response, _ = self.model.chat(
                self.tokenizer,
                pixel_values,
                full_question,
                generation_config,
                num_patches_list=num_patches_list,
                history=None,
                return_history=True
            )
        
        return response.strip()


# =============================================================================
# Time Parsing Utilities
# =============================================================================

def parse_time_from_text(text: str, video_duration: float = None, allow_end_fallback: bool = False) -> List[TemporalSegment]:
    """Parse time references from natural language text.
    
    Handles formats:
    - "at 2:30" or "at 2 minutes 30 seconds"
    - "around 45 seconds"  
    - "between 1:00 and 1:30"
    - "from the beginning to 0:30"
    - "at the start/end"
    
    Returns list of TemporalSegment with parsed times.
    """
    segments = []
    
    # Pattern for MM:SS format
    time_pattern = r'(\d+):(\d+)'
    
    # Pattern for "X seconds" or "X minutes"
    sec_pattern = r'(\d+(?:\.\d+)?)\s*(?:second|sec|s\b)'
    min_pattern = r'(\d+(?:\.\d+)?)\s*(?:minute|min|m\b)'
    
    # Pattern for ranges "between X and Y" or "from X to Y"
    range_pattern = r'(?:between|from)\s+(.+?)\s+(?:and|to)\s+(.+?)(?:\.|,|$)'
    
    def time_str_to_seconds(time_str: str) -> Optional[float]:
        """Convert time string to seconds."""
        time_str = time_str.strip().lower()
        
        # Handle MM:SS
        mm_ss = re.search(r'(\d+):(\d+)', time_str)
        if mm_ss:
            return int(mm_ss.group(1)) * 60 + int(mm_ss.group(2))
        
        # Handle "X seconds"
        sec_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:second|sec|s\b)', time_str)
        if sec_match:
            return float(sec_match.group(1))
        
        # Handle "X minutes"
        min_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:minute|min|m\b)', time_str)
        if min_match:
            return float(min_match.group(1)) * 60
        
        # Handle combined "X minutes Y seconds"
        combined = re.search(r'(\d+)\s*(?:minute|min)\s*(?:and\s+)?(\d+)\s*(?:second|sec)', time_str)
        if combined:
            return int(combined.group(1)) * 60 + int(combined.group(2))
        
        # Try parsing as plain number (assume seconds)
        plain_num = re.search(r'^(\d+(?:\.\d+)?)$', time_str.strip())
        if plain_num:
            return float(plain_num.group(1))
        
        return None
    
    text_lower = text.lower()
    
    # Look for range patterns first
    range_matches = re.finditer(range_pattern, text_lower, re.IGNORECASE)
    for match in range_matches:
        start_time = time_str_to_seconds(match.group(1))
        end_time = time_str_to_seconds(match.group(2))
        if start_time is not None and end_time is not None:
            segments.append(TemporalSegment(
                start_sec=start_time,
                end_sec=end_time,
                confidence=0.8,
                description=match.group(0)
            ))
    
    # Look for "at X" or "around X" patterns
    at_pattern = r'(?:at|around|approximately|about)\s+(\d+:\d+|\d+\s*(?:second|sec|minute|min)[s]?)'
    at_matches = re.finditer(at_pattern, text_lower)
    for match in at_matches:
        time_sec = time_str_to_seconds(match.group(1))
        if time_sec is not None:
            # Create a small window around the time
            window = 3.0  # 3 second window
            segments.append(TemporalSegment(
                start_sec=max(0, time_sec - window),
                end_sec=time_sec + window,
                confidence=0.7,
                description=match.group(0)
            ))
    
    # Handle "beginning/start" and "end"
    if "beginning" in text_lower or "start" in text_lower:
        if not any(s.start_sec < 5 for s in segments):
            segments.append(TemporalSegment(
                start_sec=0,
                end_sec=10,
                confidence=0.6,
                description="at the beginning"
            ))
    
    if allow_end_fallback and "end" in text_lower and video_duration:
        if not any(s.end_sec > video_duration - 10 for s in segments):
            segments.append(TemporalSegment(
                start_sec=max(0, video_duration - 10),
                end_sec=video_duration,
                confidence=0.6,
                description="at the end"
            ))
    
    # Look for any remaining time mentions
    all_times = re.findall(r'(\d+):(\d+)', text)
    for match in all_times:
        time_sec = int(match[0]) * 60 + int(match[1])
        if not any(abs(s.start_sec - time_sec) < 5 or abs(s.end_sec - time_sec) < 5 for s in segments):
            segments.append(TemporalSegment(
                start_sec=max(0, time_sec - 2),
                end_sec=time_sec + 2,
                confidence=0.5,
                description=f"{match[0]}:{match[1]}"
            ))
    
    return segments


# =============================================================================
# Interface 1: TemporalQAInterface
# =============================================================================

# =============================================================================
# General Video Q&A Interface (Basic, Fast)
# =============================================================================

class InternVideoGeneralQA(Interface, InternVideo2_5Base):
    """General InternVideo2.5 Q&A Interface.
    
    General-purpose video question answering using 128 frames.
    Fast and effective for most video understanding tasks.
    """
    
    CATEGORY = InterfaceCategory.SUB_QUESTION_ANSWERING
    FUNCTIONALITY = "Answer questions about video content using InternVideo2.5."
    REFERENCE_PAPER = "InternVideo2.5 (2025)"
    TOOL_SOURCES = ["InternVideo2.5"]
    
    INPUT_SCHEMA = {
        "query": {"type": "string", "required": True},
        "video": {"type": "Video", "required": True},
        "num_segments": {"type": "integer", "required": False, "default": 128},
        "start_frame": {"type": "integer", "required": False, "default": None},
        "end_frame": {"type": "integer", "required": False, "default": None}
    }
    
    OUTPUT_SCHEMA = {
        "answer": {"type": "string"},
        "frame_range": {"type": "string"}
    }
    
    AGENT_NAME = "internvideo_general_qa"
    AGENT_DESCRIPTION = "Answer general questions about video content - what, who, where, why, how. Optionally specify a frame range to focus on a specific segment."
    
    AGENT_INPUT_SCHEMA = {
        "query": {
            "type": "string",
            "required": True,
            "description": "Question about the video content"
        },
        "start_frame": {
            "type": "integer",
            "required": False,
            "description": "Start frame index (0-based, optional). Default: 0 (video start)"
        },
        "end_frame": {
            "type": "integer",
            "required": False,
            "description": "End frame index (0-based, inclusive, optional). Default: last frame"
        }
    }
    
    AGENT_OUTPUT_FORMAT = "Text answer based on video understanding"
    
    DEFAULT_NUM_SEGMENTS = 128
    MAX_NUM_SEGMENTS = 256
    
    def __init__(self, device: str = None, num_segments: int = None, **kwargs):
        Interface.__init__(self)
        InternVideo2_5Base.__init__(self, device)
        requested = num_segments or self.DEFAULT_NUM_SEGMENTS
        self._num_segments = self._effective_num_segments(requested)

    def _effective_num_segments(self, requested: int) -> int:
        """Clamp and align segment count to multiples of 4."""
        segments = max(4, min(requested, self.MAX_NUM_SEGMENTS))
        remainder = segments % 4
        if remainder:
            segments -= remainder
            if segments < 4:
                segments = 4
        return segments
    
    def initialize(self) -> None:
        """Initialize the InternVideo2.5 model."""
        InternVideo2_5Base.initialize(self)
    
    @classmethod
    def format_output_for_agent(cls, output: Dict[str, Any]) -> str:
        if "error" in output:
            return f"Error: {output['error']}"
        
        answer = output.get("answer", "Could not determine answer.")
        frame_range = output.get("frame_range", "")
        
        if frame_range and "full video" not in frame_range:
            return f"[Frames {frame_range}] Answer: {answer}"
        return f"Answer: {answer}"
    
    def _load_video_uniform(self, video_path: str, start_frame: int = None, 
                            end_frame: int = None) -> Tuple[torch.Tensor, List[int], Dict]:
        """Load video with uniform sampling, optionally within a frame range.
        
        Args:
            video_path: Path to video file
            start_frame: Start frame index (0-based, inclusive). None = start from 0
            end_frame: End frame index (0-based, inclusive). None = go to last frame
            
        Returns:
            Tuple of (pixel_values, num_patches_list, segment_info)
        """
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        total_frames = len(vr)
        video_fps = float(vr.get_avg_fps())
        
        # Determine frame range
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = total_frames - 1
        
        # Clamp to valid range
        start_frame = max(0, min(start_frame, total_frames - 1))
        end_frame = max(start_frame, min(end_frame, total_frames - 1))
        
        frame_range = end_frame - start_frame + 1
        segment_duration = frame_range / video_fps if video_fps > 0 else 0
        
        num_segments = self._effective_num_segments(self._num_segments)
        
        # Uniform sampling within the specified range
        seg_size = float(frame_range) / num_segments
        frame_indices = [
            start_frame + int((seg_size / 2) + np.round(seg_size * idx))
            for idx in range(num_segments)
        ]
        # Clamp indices to valid range
        frame_indices = [min(idx, end_frame) for idx in frame_indices]
        
        transform = self.build_transform()
        pixel_values_list = []
        num_patches_list = []
        
        for frame_idx in frame_indices:
            img = PILImage.fromarray(vr[frame_idx].asnumpy()).convert("RGB")
            img_list = self.dynamic_preprocess(img)
            pixel_values = torch.stack([transform(tile) for tile in img_list])
            frame_patches = pixel_values.shape[0]
            
            pixel_values_list.append(pixel_values)
            num_patches_list.append(frame_patches)
        
        # Ensure frame count is multiple of 4
        remainder = len(pixel_values_list) % 4
        if remainder:
            pixel_values_list = pixel_values_list[:-remainder]
            num_patches_list = num_patches_list[:-remainder]
        
        # Guarantee minimum frames
        while len(pixel_values_list) < 4 and len(pixel_values_list) > 0:
            pixel_values_list.append(pixel_values_list[-1])
            num_patches_list.append(num_patches_list[-1])

        if not pixel_values_list:
            raise ValueError("No frames loaded for uniform sampling.")

        pixel_values = torch.cat(pixel_values_list)
        
        segment_info = {
            "start_frame": start_frame,
            "end_frame": end_frame,
            "total_frames": total_frames,
            "segment_duration": segment_duration,
            "video_fps": video_fps,
        }
        
        return pixel_values, num_patches_list, segment_info
    
    def __call__(self, **kwargs) -> Dict[str, Any]:
        """Execute general video Q&A, optionally within a frame range."""
        if not self._initialized:
            self.initialize()
        
        query = kwargs.get("query")
        video = kwargs.get("video")
        start_frame = kwargs.get("start_frame", None)
        end_frame = kwargs.get("end_frame", None)
        
        if not query:
            return {"error": "Query is required"}
        if not video:
            return {"error": "Video is required"}
        
        video_path = video.path if isinstance(video, Video) else video
        if not os.path.exists(video_path):
            return {"error": f"Video not found: {video_path}"}
        
        try:
            pixel_values, num_patches_list, segment_info = self._load_video_uniform(
                video_path, start_frame, end_frame
            )
            response = self.query_video(pixel_values, num_patches_list, query)
            
            # Build frame range string for output
            is_segment = (segment_info["start_frame"] > 0 or 
                         segment_info["end_frame"] < segment_info["total_frames"] - 1)
            if is_segment:
                frame_range = f"{segment_info['start_frame']}-{segment_info['end_frame']}"
            else:
                frame_range = f"0-{segment_info['total_frames'] - 1} (full video)"
            
            return {
                "answer": response,
                "frame_range": frame_range
            }
            
        except Exception as e:
            logging.error(f"InternVideoGeneralQA failed: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}


# (Temporal QA and Object Tracking interfaces removed per request)


# =============================================================================
# Interface 3: VideoDescriptionInterface
# =============================================================================

class InternVideoDescription(Interface, InternVideo2_5Base):
    """Video description interface - generates comprehensive video summary.
    
    Uses the same configuration as GeneralQA (128 uniform frames) to provide
    detailed descriptions of video content including all events and actions.
    Supports optional start_frame and end_frame to describe a specific segment.
    """
    
    CATEGORY = InterfaceCategory.SUB_QUESTION_ANSWERING
    FUNCTIONALITY = "Generate comprehensive video description including overview, setting, main subjects, events sequence, and objects."
    REFERENCE_PAPER = "InternVideo2.5 (2025)"
    TOOL_SOURCES = ["InternVideo2.5"]
    
    INPUT_SCHEMA = {
        "video": {"type": "Video", "required": True},
        "start_frame": {"type": "integer", "required": False, "default": None},
        "end_frame": {"type": "integer", "required": False, "default": None}
    }
    
    OUTPUT_SCHEMA = {
        "description": {"type": "string"},
        "frame_range": {"type": "string"},
        "segment_duration": {"type": "float"}
    }
    
    AGENT_NAME = "internvideo_description"
    AGENT_DESCRIPTION = "Generate detailed description for a VIDEO CLIP (specific segment). REQUIRES start_frame and end_frame. Use this tool when you need to analyze a specific segment in more detail. Output includes: overview, setting, main subjects, events, and secondary objects."
    
    AGENT_INPUT_SCHEMA = {
        "start_frame": {
            "type": "integer",
            "required": True,
            "description": "Start frame index (0-based, required). Check video total_frames in context."
        },
        "end_frame": {
            "type": "integer",
            "required": True,
            "description": "End frame index (0-based, inclusive, required). Must be > start_frame."
        }
    }
    
    AGENT_OUTPUT_FORMAT = "Structured description: OVERVIEW, SETTING, MAIN SUBJECTS, EVENTS (beginning/middle/end), SECONDARY OBJECTS"
    
    # Configuration - same as GeneralQA for consistency
    DEFAULT_NUM_SEGMENTS = 128
    MAX_NUM_SEGMENTS = 256
    
    def __init__(self, device: str = None, num_segments: int = None, **kwargs):
        Interface.__init__(self)
        InternVideo2_5Base.__init__(self, device)
        requested = num_segments or self.DEFAULT_NUM_SEGMENTS
        self._num_segments = self._effective_num_segments(requested)
    
    def _effective_num_segments(self, requested: int) -> int:
        """Clamp and align segment count to multiples of 4."""
        segments = max(4, min(requested, self.MAX_NUM_SEGMENTS))
        remainder = segments % 4
        if remainder:
            segments -= remainder
            if segments < 4:
                segments = 4
        return segments
    
    def initialize(self) -> None:
        """Initialize the InternVideo2.5 model."""
        InternVideo2_5Base.initialize(self)
    
    @classmethod
    def format_output_for_agent(cls, output: Dict[str, Any]) -> str:
        if "error" in output:
            return f"Error: {output['error']}"
        
        description = output.get("description", "No description available.")
        frame_range = output.get("frame_range", "")
        segment_duration = output.get("segment_duration", 0)
        
        if frame_range:
            header = f"[Frame range: {frame_range}, Duration: {segment_duration:.1f}s]\n\n"
            return header + description
        return description
    
    def _load_video_uniform(self, video_path: str, start_frame: int = None, 
                            end_frame: int = None) -> Tuple[torch.Tensor, List[int], Dict]:
        """Load video with uniform sampling, optionally within a frame range.
        
        Args:
            video_path: Path to video file
            start_frame: Start frame index (0-based, inclusive). None = start from 0
            end_frame: End frame index (0-based, inclusive). None = go to last frame
            
        Returns:
            Tuple of (pixel_values, num_patches_list, segment_info)
        """
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        total_frames = len(vr)
        video_fps = float(vr.get_avg_fps())
        
        # Determine frame range
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = total_frames - 1
        
        # Clamp to valid range
        start_frame = max(0, min(start_frame, total_frames - 1))
        end_frame = max(start_frame, min(end_frame, total_frames - 1))
        
        frame_range = end_frame - start_frame + 1
        segment_duration = frame_range / video_fps
        
        num_segments = self._effective_num_segments(self._num_segments)
        
        # Uniform sampling within the specified range
        seg_size = float(frame_range) / num_segments
        frame_indices = [
            start_frame + int((seg_size / 2) + np.round(seg_size * idx))
            for idx in range(num_segments)
        ]
        # Clamp indices to valid range
        frame_indices = [min(idx, end_frame) for idx in frame_indices]
        
        transform = self.build_transform()
        pixel_values_list = []
        num_patches_list = []
        
        for frame_idx in frame_indices:
            img = PILImage.fromarray(vr[frame_idx].asnumpy()).convert("RGB")
            img_list = self.dynamic_preprocess(img)
            pixel_values = torch.stack([transform(tile) for tile in img_list])
            frame_patches = pixel_values.shape[0]
            
            pixel_values_list.append(pixel_values)
            num_patches_list.append(frame_patches)
        
        # Ensure frame count is multiple of 4
        remainder = len(pixel_values_list) % 4
        if remainder:
            pixel_values_list = pixel_values_list[:-remainder]
            num_patches_list = num_patches_list[:-remainder]
        
        # Guarantee minimum frames
        while len(pixel_values_list) < 4 and len(pixel_values_list) > 0:
            pixel_values_list.append(pixel_values_list[-1])
            num_patches_list.append(num_patches_list[-1])

        if not pixel_values_list:
            raise ValueError("No frames loaded for uniform sampling.")

        pixel_values = torch.cat(pixel_values_list)
        
        segment_info = {
            "start_frame": start_frame,
            "end_frame": end_frame,
            "total_frames": total_frames,
            "segment_duration": segment_duration,
            "video_fps": video_fps,
            "start_sec": start_frame / video_fps,
            "end_sec": (end_frame + 1) / video_fps
        }
        
        return pixel_values, num_patches_list, segment_info
    
    def __call__(self, **kwargs) -> Dict[str, Any]:
        """Generate comprehensive video description for full video or a frame range."""
        if not self._initialized:
            self.initialize()
        
        video = kwargs.get("video")
        start_frame = kwargs.get("start_frame", None)
        end_frame = kwargs.get("end_frame", None)
        
        if not video:
            return {"error": "Video is required"}
        
        video_path = video.path if isinstance(video, Video) else video
        if not os.path.exists(video_path):
            return {"error": f"Video not found: {video_path}"}
        
        try:
            # Load video with optional frame range
            pixel_values, num_patches_list, segment_info = self._load_video_uniform(
                video_path, start_frame, end_frame
            )
            
            # Use segment duration for the prompt
            duration = segment_info["segment_duration"]
            start_sec = segment_info["start_sec"]
            end_sec = segment_info["end_sec"]
            is_segment = (segment_info["start_frame"] > 0 or 
                         segment_info["end_frame"] < segment_info["total_frames"] - 1)
            
            # Build description prompt
            num_frames = len(num_patches_list)
            
            if is_segment:
                # Describing a specific segment
                segment_desc = f"video segment from {start_sec:.1f}s to {end_sec:.1f}s (frames {segment_info['start_frame']}-{segment_info['end_frame']})"
                time_prefix = f"{start_sec:.0f}"
            else:
                segment_desc = f"{duration:.0f}-second video"
                time_prefix = "0"
            
            description_prompt = f"""Describe this {segment_desc} in complete detail. You are analyzing {num_frames} frames.

Write a comprehensive description with ALL of the following sections:

**OVERVIEW:** Provide a brief summary of what this video is about in 2-3 sentences. What is the main activity or theme?

**SETTING:** Describe the environment in detail - location type, colors, lighting, layout, and visible items in the background. (2-4 sentences)

**MAIN SUBJECTS:** Identify the primary subjects (person, animal, object, or machine) in the video. Describe their appearance and their relationship to each other or to the environment. (2-3 sentences)

**EVENTS:**

- BEGINNING: Describe what happens in the first part of the video. Focus on the sequence of events and interactions. Write naturally without mentioning specific timestamps. (4-6 sentences)

- EVENTS:  MIDDLE:** Describe what happens in the middle part of the video. Continue the narrative flow of events and activities. (4-6 sentences)

- END:  Describe what happens in the final part of the video. Include the conclusion or outcome of the activities. (4-6 sentences)

**OBJECTS:** List other objects visible in the scene that are used or present in the environment. Briefly note their role or location. (2-4 sentences)

RULES:
- Cover the FULL video content
- Write naturally without specific timestamps
- NEVER repeat phrases
- Target 400-600 words total"""

            description = self.query_video(pixel_values, num_patches_list, description_prompt, max_new_tokens=800)
            
            # Post-process to remove repetitions
            description = self._clean_repetitions(description)
            
            # Build frame range string
            if is_segment:
                frame_range_str = f"{segment_info['start_frame']}-{segment_info['end_frame']} (of {segment_info['total_frames']} total)"
            else:
                frame_range_str = f"0-{segment_info['total_frames']-1} (full video)"
            
            return {
                "description": description,
                "frame_range": frame_range_str,
                "segment_duration": duration,
                "num_frames_analyzed": len(num_patches_list),
                "start_frame": segment_info["start_frame"],
                "end_frame": segment_info["end_frame"]
            }
        
        except Exception as e:
            logging.error(f"InternVideoDescription failed: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}
    
    def _clean_repetitions(self, text: str) -> str:
        """Remove repeated phrases from model output."""
        # Find and truncate obvious repetition patterns
        # Pattern: same phrase repeated many times with commas
        
        # Split by lines and process each
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Only process lines with many commas (potential repetition)
            comma_count = line.count(', ')
            if comma_count > 15:  # Suspicious number of commas
                parts = line.split(', ')
                seen = set()
                unique_parts = []
                for p in parts:
                    p_normalized = p.strip().lower()
                    if p_normalized not in seen and len(p_normalized) > 0:
                        seen.add(p_normalized)
                        unique_parts.append(p.strip())
                # Only take first 12 unique items
                if len(unique_parts) > 12:
                    unique_parts = unique_parts[:12]
                line = ', '.join(unique_parts)
            
            cleaned_lines.append(line)
        
        result = '\n'.join(cleaned_lines)
        
        # Truncate after OUTCOME section (remove anything after 3 lines of OUTCOME)
        if '**OUTCOME:**' in result:
            idx = result.find('**OUTCOME:**')
            outcome_section = result[idx:]
            outcome_lines = outcome_section.split('\n')[:4]  # Keep header + 3 lines
            result = result[:idx] + '\n'.join(outcome_lines)
        
        return result.strip()


