"""
Video Processing Utilities

Functions for loading videos, extracting frames, and sampling.
"""

import os
import cv2
import json
import hashlib
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

from video_agent_tools.state import VideoContext


# Default cache directory for OmniCaptioner
DEFAULT_CAPTION_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "tools", "cache", "omnicaptioner"
)


def load_video_context(video_path: str, video_id: str = None) -> VideoContext:
    """
    Load video and create VideoContext with metadata.
    
    Args:
        video_path: Path to video file
        video_id: Optional video identifier (defaults to filename)
    
    Returns:
        VideoContext with video metadata
    
    Raises:
        FileNotFoundError: If video file doesn't exist
        ValueError: If video cannot be opened
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0
        
        if video_id is None:
            video_id = os.path.splitext(os.path.basename(video_path))[0]
        
        return VideoContext(
            video_id=video_id,
            video_path=video_path,
            total_frames=total_frames,
            fps=fps,
            duration=duration,
            width=width,
            height=height,
        )
    finally:
        cap.release()


def extract_frame(video_path: str, frame_index: int) -> Optional[np.ndarray]:
    """
    Extract a single frame from video.
    
    Args:
        video_path: Path to video file
        frame_index: 0-based frame index
    
    Returns:
        Frame as numpy array (RGB format) or None if failed
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        
        if not ret:
            return None
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame_rgb
    finally:
        cap.release()


def extract_frames(video_path: str, frame_indices: List[int]) -> Dict[int, np.ndarray]:
    """
    Extract multiple frames from video efficiently.
    
    Args:
        video_path: Path to video file
        frame_indices: List of 0-based frame indices
    
    Returns:
        Dict mapping frame index to numpy array (RGB format)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {}
    
    frames = {}
    
    try:
        # Sort indices for efficient sequential access
        sorted_indices = sorted(frame_indices)
        
        for idx in sorted_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames[idx] = frame_rgb
    finally:
        cap.release()
    
    return frames


def sample_uniform_indices(total_frames: int, num_samples: int) -> List[int]:
    """
    Sample frame indices uniformly across the video.
    
    Args:
        total_frames: Total number of frames in video
        num_samples: Number of frames to sample
    
    Returns:
        List of frame indices
    """
    if num_samples <= 0:
        return []
    
    if num_samples >= total_frames:
        return list(range(total_frames))
    
    # Calculate step size
    step = (total_frames - 1) / (num_samples - 1) if num_samples > 1 else 0
    
    indices = []
    for i in range(num_samples):
        idx = int(round(i * step))
        idx = min(idx, total_frames - 1)
        if idx not in indices:
            indices.append(idx)
    
    return indices


def sample_indices_in_range(
    total_frames: int,
    num_samples: int,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    exclude_indices: List[int] = None,
) -> List[int]:
    """
    Sample frame indices within a specific range, excluding already sampled frames.
    
    Args:
        total_frames: Total number of frames in video
        num_samples: Number of frames to sample
        start_frame: Start of range (inclusive)
        end_frame: End of range (inclusive), defaults to last frame
        exclude_indices: Indices to exclude from sampling
    
    Returns:
        List of new frame indices
    """
    if end_frame is None:
        end_frame = total_frames - 1
    
    # Clamp range
    start_frame = max(0, min(start_frame, total_frames - 1))
    end_frame = max(0, min(end_frame, total_frames - 1))
    
    if start_frame > end_frame:
        start_frame, end_frame = end_frame, start_frame
    
    exclude_set = set(exclude_indices) if exclude_indices else set()
    
    # Get available indices in range
    available = [i for i in range(start_frame, end_frame + 1) if i not in exclude_set]
    
    if not available:
        return []
    
    if len(available) <= num_samples:
        return available
    
    # Uniform sampling from available indices
    step = len(available) / num_samples
    indices = []
    
    for i in range(num_samples):
        idx = int(i * step)
        idx = min(idx, len(available) - 1)
        if available[idx] not in indices:
            indices.append(available[idx])
    
    return sorted(indices)


def get_video_info(video_path: str) -> Dict[str, Any]:
    """
    Get video metadata without loading frames.
    
    Args:
        video_path: Path to video file
    
    Returns:
        Dict with video properties
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": f"Cannot open video: {video_path}"}
    
    try:
        return {
            "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "duration": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 0,
        }
    finally:
        cap.release()


def timestamp_to_frame(timestamp: float, fps: float) -> int:
    """Convert timestamp (seconds) to frame index."""
    return int(timestamp * fps)


def frame_to_timestamp(frame_index: int, fps: float) -> float:
    """Convert frame index to timestamp (seconds)."""
    return frame_index / fps if fps > 0 else 0.0


def get_cache_path(video_path: str, cache_dir: str = None) -> str:
    """Get cache file path for a video."""
    if cache_dir is None:
        cache_dir = DEFAULT_CAPTION_CACHE_DIR
    
    video_hash = hashlib.md5(os.path.abspath(video_path).encode()).hexdigest()[:8]
    video_name = os.path.basename(video_path).split('.')[0]
    return os.path.join(cache_dir, f"{video_name}_{video_hash}.json")


def load_cached_captions(
    video_path: str,
    cache_dir: str = None,
    num_frames: int = None,
) -> Optional[Dict[int, str]]:
    """
    Load cached captions for a video.
    
    Args:
        video_path: Path to video file
        cache_dir: Cache directory (defaults to omnicaptioner cache)
        num_frames: Expected number of frames (optional validation)
    
    Returns:
        Dict mapping frame index (int) to caption string, or None if not cached
    """
    if cache_dir is None:
        cache_dir = DEFAULT_CAPTION_CACHE_DIR
    
    cache_path = get_cache_path(video_path, cache_dir)
    
    if not os.path.exists(cache_path):
        return None
    
    try:
        with open(cache_path, 'r') as f:
            cache_data = json.load(f)
        
        captions_raw = cache_data.get("captions", {})
        
        # Validate frame count if requested
        if num_frames is not None and len(captions_raw) < num_frames:
            return None
        
        # Convert string keys to integers (JSON stores keys as strings)
        captions = {int(k): v for k, v in captions_raw.items()}
        
        return captions
        
    except Exception:
        return None


def is_video_cached(
    video_path: str,
    cache_dir: str = None,
    num_frames: int = None,
) -> bool:
    """Check if captions are cached for a video."""
    captions = load_cached_captions(video_path, cache_dir, num_frames)
    return captions is not None


def load_video_context_with_captions(
    video_path: str,
    video_id: str = None,
    cache_dir: str = None,
    num_frames: int = None,
) -> Tuple[VideoContext, bool]:
    """
    Load video context and pre-populate with cached captions if available.
    
    Args:
        video_path: Path to video file
        video_id: Optional video identifier
        cache_dir: Cache directory for captions
        num_frames: Expected number of frames in cache
    
    Returns:
        Tuple of (VideoContext, cache_hit: bool)
    """
    # Load video context
    video_context = load_video_context(video_path, video_id)
    
    # Try to load cached captions
    cached_captions = load_cached_captions(video_path, cache_dir, num_frames)
    
    if cached_captions:
        video_context.frame_captions = cached_captions
        video_context.sampled_indices = sorted(cached_captions.keys())
        return video_context, True
    
    return video_context, False



