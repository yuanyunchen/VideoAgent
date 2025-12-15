"""
View Frame Interface

Extract specific frames from video using OpenCV.
Supports viewing single or multiple frames with visual data returned to agent.
"""

import cv2
import numpy as np
from typing import Dict, Any, Optional, Union, List
from tools.interface_base import Interface, InterfaceCategory, Video, Image


class ViewFrame(Interface):
    """Extract and view frames from video.
    
    This tool allows the agent to directly view frames from the video.
    The frames are returned as images that the agent can analyze visually.
    """
    
    CATEGORY = InterfaceCategory.TOOL
    FUNCTIONALITY = "Extract and view frames from video. Returns actual images for visual analysis."
    REFERENCE_PAPER = "N/A"
    TOOL_SOURCES = ["OpenCV/FFmpeg"]
    
    INPUT_SCHEMA = {
        "video": {"type": "Video", "required": True},
        "frame_indices": {"type": "List[int]", "required": False},
        "frame_index": {"type": "int", "required": False},
        "timestamp": {"type": "float", "required": False}
    }
    
    OUTPUT_SCHEMA = {
        "frames": {"type": "List[Image]"},
        "frame_indices": {"type": "List[int]"},
        "timestamps": {"type": "List[float]"}
    }
    
    # Agent-facing
    AGENT_NAME = "view_frame"
    AGENT_DESCRIPTION = (
        "View one or more frames from the video. "
        "You will SEE the actual images and can analyze visual details directly. "
        "Images accumulate in your visual memory."
    )
    
    AGENT_INPUT_SCHEMA = {
        "frame_indices": {
            "type": "List[int]",
            "required": False,
            "description": "List of frame numbers to view (0-based). Preferred for multiple frames."
        },
        "frame_index": {
            "type": "int",
            "required": False,
            "description": "Single frame number to view (0-based)"
        },
        "timestamp": {
            "type": "float",
            "required": False,
            "description": "Time in seconds for single frame (e.g., 30.5)"
        }
    }
    
    AGENT_OUTPUT_FORMAT = "The requested frame(s) displayed for your visual analysis"
    
    def __init__(self):
        """Initialize ViewFrame interface."""
        self._initialized = False
    
    @classmethod
    def format_output_for_agent(cls, output: Dict[str, Any]) -> str:
        """Format output text (images are handled separately by the agent graph)."""
        if "error" in output:
            return f"Error: {output['error']}"
        
        frames = output.get("frames", [])
        frame_indices = output.get("frame_indices", [])
        timestamps = output.get("timestamps", [])
        
        if not frames:
            return "No frames extracted."
        
        if len(frames) == 1:
            idx = frame_indices[0] if frame_indices else 0
            ts = timestamps[0] if timestamps else 0
            return f"Viewing Frame {idx} @ {ts:.1f}s - analyze the image above."
        else:
            lines = [f"Viewing {len(frames)} frames:"]
            for i, (idx, ts) in enumerate(zip(frame_indices, timestamps)):
                lines.append(f"  Image {i+1}: Frame {idx} @ {ts:.1f}s")
            lines.append("Analyze the images above to answer the question.")
            return "\n".join(lines)
    
    def initialize(self) -> None:
        """Initialize the interface."""
        self._initialized = True
    
    def __call__(
        self,
        video: Union[Video, str],
        frame_indices: Optional[List[int]] = None,
        frame_index: Optional[int] = None,
        timestamp: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Extract frames from video.
        
        Args:
            video: Video object or path to video file
            frame_indices: List of frame numbers to extract (0-based)
            frame_index: Single frame number (0-based)
            timestamp: Time in seconds (for single frame)
        
        Returns:
            Dict with 'frames' (list of Image objects), 'frame_indices', 'timestamps'
        """
        if not self._initialized:
            self.initialize()
        
        # Get video path
        if isinstance(video, Video):
            video_path = video.path
        elif isinstance(video, str):
            video_path = video
        else:
            return {"error": f"Invalid video type: {type(video)}. Expected Video or str."}
        
        # Determine which frames to extract
        indices_to_extract = []
        
        if frame_indices is not None and len(frame_indices) > 0:
            indices_to_extract = list(frame_indices)
        elif frame_index is not None:
            indices_to_extract = [frame_index]
        elif timestamp is not None:
            # Will calculate frame index after opening video
            indices_to_extract = None  # Sentinel value
        else:
            return {"error": "Provide frame_indices, frame_index, or timestamp."}
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"error": f"Failed to open video: {video_path}"}
        
        try:
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Handle timestamp -> frame index conversion
            if indices_to_extract is None:
                frame_idx = int(timestamp * fps)
                indices_to_extract = [frame_idx]
            
            # Validate and filter frame indices
            valid_indices = []
            for idx in indices_to_extract:
                if 0 <= idx < total_frames:
                    valid_indices.append(idx)
            
            if not valid_indices:
                return {
                    "error": f"No valid frame indices. Valid range: 0-{total_frames - 1}"
                }
            
            # Limit number of frames to prevent memory issues
            MAX_FRAMES = 8
            if len(valid_indices) > MAX_FRAMES:
                valid_indices = valid_indices[:MAX_FRAMES]
            
            # Extract frames
            frames = []
            extracted_indices = []
            extracted_timestamps = []
            
            for idx in valid_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                
                if ret:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Calculate timestamp
                    ts = idx / fps if fps > 0 else 0.0
                    
                    # Create Image object
                    image = Image(
                        path=None,
                        data=frame_rgb,
                        width=width,
                        height=height,
                        frame_index=idx,
                        timestamp=ts
                    )
                    
                    frames.append(image)
                    extracted_indices.append(idx)
                    extracted_timestamps.append(ts)
            
            if not frames:
                return {"error": "Failed to extract any frames"}
            
            return {
                "frames": frames,
                "frame_indices": extracted_indices,
                "timestamps": extracted_timestamps,
            }
            
        finally:
            cap.release()
    
    def get_video_info(self, video: Union[Video, str]) -> Dict[str, Any]:
        """Get video metadata."""
        if isinstance(video, Video):
            video_path = video.path
        elif isinstance(video, str):
            video_path = video
        else:
            return {"error": f"Invalid video type: {type(video)}"}
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"error": f"Failed to open video: {video_path}"}
        
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            return {
                "fps": fps,
                "duration": duration,
                "frame_count": total_frames,
                "width": width,
                "height": height
            }
        finally:
            cap.release()
