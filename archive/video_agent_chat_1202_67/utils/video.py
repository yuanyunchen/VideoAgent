"""
Video processing utilities for VideoAgent.
"""

import os
import cv2
import numpy as np
from typing import List, Optional


def get_video_frames(video_path: str, interval: int = 30) -> Optional[List[np.ndarray]]:
    """
    Load video frames from file.
    
    Args:
        video_path: Path to video file
        interval: Frame sampling interval
        
    Returns:
        List of frames or None if video cannot be opened
    """
    if not os.path.exists(video_path):
        return None
    
    frames = []
    cap = cv2.VideoCapture(video_path)
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx % interval == 0:
            frames.append(frame)
        
        frame_idx += 1
    
    cap.release()
    return frames if frames else None


def get_video_info(video_path: str) -> Optional[dict]:
    """
    Get video information.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Dictionary with video info or None if video cannot be opened
    """
    if not os.path.exists(video_path):
        return None
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    info = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    
    cap.release()
    return info


def sample_frame_indices(total_frames: int, num_samples: int) -> List[int]:
    """
    Sample frame indices uniformly from video.
    
    Args:
        total_frames: Total number of frames
        num_samples: Number of samples to take
        
    Returns:
        List of frame indices
    """
    if total_frames <= num_samples:
        return list(range(total_frames))
    
    indices = np.linspace(0, total_frames - 1, num=num_samples, dtype=int).tolist()
    return indices


def sample_frame_indices_in_range(
    total_frames: int,
    num_samples: int,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    sampled_indices: Optional[set] = None
) -> List[int]:
    """
    Sample frame indices uniformly within a specified range, excluding already sampled frames.
    
    Args:
        total_frames: Total number of frames in video
        num_samples: Number of samples to take
        start_frame: Starting frame index (inclusive, default: 0)
        end_frame: Ending frame index (inclusive, default: last frame)
        sampled_indices: Set of already sampled frame indices to exclude
        
    Returns:
        List of frame indices within the range
    """
    # Validate and adjust range
    start_frame = max(0, start_frame)
    if end_frame is None:
        end_frame = total_frames - 1
    end_frame = min(end_frame, total_frames - 1)
    
    if start_frame > end_frame:
        return []
    
    # Get available indices in range
    if sampled_indices is None:
        sampled_indices = set()
    
    available_in_range = [
        i for i in range(start_frame, end_frame + 1)
        if i not in sampled_indices
    ]
    
    if not available_in_range:
        return []
    
    # If we need more samples than available, return all available
    if len(available_in_range) <= num_samples:
        return sorted(available_in_range)
    
    # Uniformly sample from available frames in range
    indices = np.linspace(0, len(available_in_range) - 1, num=num_samples, dtype=int)
    result = [available_in_range[i] for i in indices]
    
    return sorted(result)


def save_frame(frame: np.ndarray, output_path: str) -> bool:
    """
    Save a frame to file.
    
    Args:
        frame: Frame as numpy array
        output_path: Output file path
        
    Returns:
        True if successful, False otherwise
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, frame)
        return True
    except Exception:
        return False

