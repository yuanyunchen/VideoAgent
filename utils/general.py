"""
General utilities for VideoAgent.
"""

import json
import logging
import os
import pickle
import hashlib
from typing import List, Dict, Any, Optional
import numpy as np

class CacheManager:
    """Simple cache manager for LLM and CLIP responses."""
    
    def __init__(self, llm_cache_file: str, clip_cache_file: str, use_cache: bool = True):
        """Initialize cache manager."""
        self.llm_cache_file = llm_cache_file
        self.clip_cache_file = clip_cache_file
        self.use_cache = use_cache
        
        # Load caches
        self.llm_cache = self._load_cache(llm_cache_file) if use_cache else {}
        self.clip_cache = self._load_cache(clip_cache_file) if use_cache else {}
    
    def _load_cache(self, cache_file: str) -> Dict:
        """Load cache from file."""
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except (pickle.PickleError, EOFError):
                pass
        return {}
    
    def save_caches(self):
        """Save caches to files."""
        if not self.use_cache:
            return
            
        os.makedirs(os.path.dirname(self.llm_cache_file), exist_ok=True)
        os.makedirs(os.path.dirname(self.clip_cache_file), exist_ok=True)
        
        with open(self.llm_cache_file, 'wb') as f:
            pickle.dump(self.llm_cache, f)
        
        with open(self.clip_cache_file, 'wb') as f:
            pickle.dump(self.clip_cache, f)

def setup_logger(name: str, output_dir: str, level: str = "INFO", 
                enable_llm_logging: bool = False) -> logging.Logger:
    """Setup logger for experiment."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # File handler only - no console output
    os.makedirs(output_dir, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(output_dir, "logging.log"))
    file_handler.setLevel(getattr(logging, level.upper()))
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # LLM logger if enabled
    if enable_llm_logging:
        llm_handler = logging.FileHandler(os.path.join(output_dir, "llm.log"))
        llm_handler.setLevel(logging.DEBUG)
        llm_handler.setFormatter(formatter)
        logger.addHandler(llm_handler)
    
    return logger

def parse_video_annotation(annotation_file: str, video_list_file: Optional[str] = None, 
                         max_videos: int = -1) -> List[Dict[str, Any]]:
    """
    Parse video annotations and return list of video info.
    
    Args:
        annotation_file: Path to annotation JSON file
        video_list_file: Optional file containing video IDs to process
        max_videos: Maximum number of videos to return (-1 for all)
        
    Returns:
        List of video information dictionaries
    """
    # Load annotations
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)
    
    # Load video list if provided
    target_videos = None
    if video_list_file and os.path.exists(video_list_file):
        with open(video_list_file, 'r') as f:
            target_videos = set(line.strip() for line in f if line.strip())
    
    # Process annotations
    videos_info = []
    for video_id, data in annotations.items():
        if target_videos and video_id not in target_videos:
            continue
            
        video_info = {
            "video_id": video_id,
            "question": data.get("question", ""),
            "answer": data.get("answer", "")
        }
        
        # Add multiple choice options if available
        for i in range(5):
            option_key = f"option {i}"
            if option_key in data:
                video_info[option_key] = data[option_key]
        
        if "truth" in data:
            video_info["truth"] = data["truth"]
        
        videos_info.append(video_info)
        
        if max_videos > 0 and len(videos_info) >= max_videos:
            break
    
    return videos_info

# ============================================================================
# BACKWARD COMPATIBILITY FUNCTIONS
# ============================================================================

def create_logger(log_file: str) -> logging.Logger:
    """Create logger (backward compatibility)."""
    logger = logging.getLogger(f"VideoAgent_{os.path.basename(log_file)}")
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # File handler
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    handler = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger

def get_video_frames(video_path: str, interval: int = 30) -> Optional[List]:
    """Load video frames (backward compatibility)."""
    import cv2
    
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

def get_tasks(video_list_file: Optional[str], max_videos: int):
    """Get tasks (backward compatibility)."""
    # This is a placeholder for backward compatibility
    # In the new system, use parse_video_annotation instead
    return [], {}


def parse_text_find_number(text: str, key: str) -> int:
    """Parse text to find number (backward compatibility)."""
    import re
    
    # Handle JSON response wrapped in code blocks
    text_content = text.strip()
    if '```json' in text_content:
        # Extract JSON content between ```json and ```
        start_idx = text_content.find('```json') + 7
        end_idx = text_content.rfind('```')
        if end_idx > start_idx:
            json_content = text_content[start_idx:end_idx].strip()
        else:
            json_content = text_content
    else:
        json_content = text_content
    
    # Try to parse as JSON first
    try:
        # Handle both single and double quotes in JSON
        json_content_fixed = json_content.replace("'", '"')
        import json
        data = json.loads(json_content_fixed)
        if key in data:
            return int(data[key])
    except (json.JSONDecodeError, ValueError, KeyError):
        pass
    
    # Fallback to regex patterns
    patterns = [
        rf'"{key}":\s*(\d+)',  # "key": number
        rf"'{key}':\s*(\d+)",  # 'key': number  
        rf'{key}.*?(\d+)',     # key followed by number
        r'(\d+)',              # any number
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text_content)
        if match:
            return int(match.group(1))
    
    return -1

def header_line(title: str) -> str:
    """Format header line (backward compatibility)."""
    return f"\n=== {title} ===\n"

def line() -> str:
    """Return line separator (backward compatibility)."""
    return "\n" + "="*50 + "\n"

def parse_analysis_and_json(text: str):
    """Parse analysis and JSON (backward compatibility)."""
    import re
    
    # Handle JSON response wrapped in code blocks
    text_content = text.strip()
    if '```json' in text_content:
        # Extract JSON content between ```json and ```
        start_idx = text_content.find('```json') + 7
        end_idx = text_content.rfind('```')
        if end_idx > start_idx:
            json_content = text_content[start_idx:end_idx].strip()
            analysis = text_content[:start_idx-7].strip()
        else:
            json_content = text_content
            analysis = text_content
    else:
        # Try to find JSON pattern without code blocks
        json_pattern = r'\{[^{}]*\}'
        match = re.search(json_pattern, text_content)
        if match:
            json_content = match.group(0)
            analysis = text_content[:match.start()].strip()
        else:
            return text_content, {}
    
    try:
        # Handle both single and double quotes in JSON
        json_content_fixed = json_content.replace("'", '"')
        json_data = json.loads(json_content_fixed)
        return analysis, json_data
    except json.JSONDecodeError:
        pass
    
    return text_content, {}

def get_llm_response(system_prompt: str, user_prompt: str, images=None, model: str = None, 
                    json_format: bool = False, logger=None) -> str:
    """Get LLM response (backward compatibility)."""
    from utils.AIML_API import get_llm_response as api_get_llm_response
    return api_get_llm_response(model or "gpt-4o-mini-2024-07-18", user_prompt, images)

def retrieve_frames_by_section(section_predictions: Dict[int, int], 
                             segment_descriptions: Dict[int, str]) -> List[int]:
    """
    Retrieve frame indices based on section predictions.
    
    Args:
        section_predictions: Dictionary mapping section ID to number of frames
        segment_descriptions: Dictionary mapping section ID to frame range descriptions
        
    Returns:
        List of frame indices to sample
    """
    new_indices = []
    
    for section_id, frame_count in section_predictions.items():
        if section_id in segment_descriptions:
            # Parse frame range from description
            frame_range = segment_descriptions[section_id]
            try:
                # Extract frame range (e.g., "Frame 10-20" -> [10, 20])
                parts = frame_range.replace("Frame ", "").split("-")
                start_frame = int(parts[0])
                end_frame = int(parts[1]) if len(parts) > 1 else start_frame + 10
                
                # Sample frames evenly within the range (avoiding endpoints)
                if frame_count > 0:
                    step = (end_frame - start_frame) / (frame_count + 1)
                    indices = [start_frame + int(step * (i + 1)) for i in range(frame_count)]
                    new_indices.extend(indices)
                    
            except (ValueError, IndexError):
                # Fallback: just add some frames around the start
                for i in range(frame_count):
                    new_indices.append(section_id * 10 + i)
    
    return sorted(list(set(new_indices)))
