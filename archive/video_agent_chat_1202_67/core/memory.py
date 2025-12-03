"""
Memory module for VideoAgent.
Manages video frame captions and provides formatted output for Solver.
"""

import os
import cv2
import json
import numpy as np
from typing import List, Dict, Set, Optional, TypedDict


class MemoryUnit(TypedDict):
    """Single memory unit containing frame information."""
    frame_id: int
    caption: str


class Memory:
    """
    Manages video frame captions for the multi-agent video understanding system.
    
    This class stores frame captions and provides formatted text for the Solver agent.
    """
    
    def __init__(self, video_frames: List[np.ndarray], video_id: str = ""):
        """
        Initialize Memory with video frames.
        
        Args:
            video_frames: List of video frames as numpy arrays
            video_id: Identifier for the video
        """
        self.video_id = video_id
        self.video_frames = video_frames
        self.n_frames = len(video_frames)
        
        # Memory units storage
        self.units: List[MemoryUnit] = []
        
        # Track which frames have been sampled
        self.sampled_indices: Set[int] = set()
    
    def add_frame(self, frame_id: int, caption: str) -> None:
        """
        Add a single frame caption to memory.
        
        Args:
            frame_id: Frame index
            caption: Caption text for the frame
        """
        if frame_id not in self.sampled_indices:
            self.units.append(MemoryUnit(frame_id=frame_id, caption=caption))
            self.sampled_indices.add(frame_id)
            # Keep units sorted by frame_id
            self.units.sort(key=lambda x: x["frame_id"])
    
    def add_frames(self, frame_ids: List[int], captions: List[str]) -> str:
        """
        Add multiple frame captions and return formatted notification string.
        
        Args:
            frame_ids: List of frame indices
            captions: List of caption texts
            
        Returns:
            Formatted notification string for Solver
        """
        new_units = []
        for frame_id, caption in zip(frame_ids, captions):
            if frame_id not in self.sampled_indices:
                unit = MemoryUnit(frame_id=frame_id, caption=caption)
                self.units.append(unit)
                self.sampled_indices.add(frame_id)
                new_units.append(unit)
        
        # Keep units sorted by frame_id
        self.units.sort(key=lambda x: x["frame_id"])
        
        # Format notification
        if new_units:
            lines = [f"[Frame {u['frame_id']}]: {u['caption']}" for u in new_units]
            return f"System Notification: Retrieved {len(new_units)} new frames:\n" + "\n".join(lines)
        return "System Notification: No new frames were added (all requested frames already in memory)."
    
    def get_frame(self, frame_id: int) -> Optional[np.ndarray]:
        """
        Get a specific video frame by index.
        
        Args:
            frame_id: Frame index
            
        Returns:
            Frame as numpy array, or None if index is out of range
        """
        if 0 <= frame_id < self.n_frames:
            return self.video_frames[frame_id]
        return None
    
    def get_frames(self, frame_ids: List[int]) -> List[np.ndarray]:
        """
        Get multiple video frames.
        
        Args:
            frame_ids: List of frame indices
            
        Returns:
            List of frames as numpy arrays
        """
        frames = []
        for frame_id in frame_ids:
            if 0 <= frame_id < self.n_frames:
                frames.append(self.video_frames[frame_id])
        return frames
    
    def get_available_frame_indices(self) -> List[int]:
        """
        Get indices of frames that haven't been sampled yet.
        
        Returns:
            List of available frame indices
        """
        return [i for i in range(self.n_frames) if i not in self.sampled_indices]
    
    def format_for_solver(self) -> str:
        """
        Format memory content as text for Solver input.
        
        Returns:
            Formatted string containing all frame captions
        """
        if not self.units:
            return "Memory is empty. No frame captions available."
        
        lines = ["Current Memory (Frame Captions):"]
        lines.append("-" * 40)
        for unit in self.units:
            lines.append(f"[Frame {unit['frame_id']}]: {unit['caption']}")
        lines.append("-" * 40)
        lines.append(f"Total frames in video: {self.n_frames}")
        lines.append(f"Frames in memory: {len(self.units)}")
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict:
        """
        Convert memory to dictionary for JSON serialization.
        
        Returns:
            Dictionary representation of memory
        """
        return {
            "video_id": self.video_id,
            "n_frames": self.n_frames,
            "sampled_count": len(self.units),
            "units": [dict(u) for u in self.units]
        }
    
    def save_frames(self, output_dir: str) -> None:
        """
        Save sampled frames to directory.
        
        Args:
            output_dir: Output directory path
        """
        frames_dir = os.path.join(output_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        
        for unit in self.units:
            frame_id = unit["frame_id"]
            if 0 <= frame_id < self.n_frames:
                frame_path = os.path.join(frames_dir, f"{frame_id}.png")
                cv2.imwrite(frame_path, self.video_frames[frame_id])
    
    def save_to_json(self, output_path: str) -> None:
        """
        Save memory content to JSON file.
        
        Args:
            output_path: Output file path
        """
        with open(output_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def __len__(self) -> int:
        """Return number of memory units."""
        return len(self.units)
    
    def __str__(self) -> str:
        """String representation."""
        return self.format_for_solver()

