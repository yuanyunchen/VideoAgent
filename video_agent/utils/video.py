"""
Video processing utilities for VideoAgent.
"""

import os
import cv2
from typing import List, Optional


def get_video_frames(video_path: str, interval: int = 30) -> Optional[List]:
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

