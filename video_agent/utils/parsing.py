"""
Parsing utilities for VideoAgent.
"""

import os
import re
import json
from typing import List, Dict, Any, Optional, Tuple


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


def parse_text_find_number(text: str, key: str) -> int:
    """
    Parse text to find number for a given key.
    
    Args:
        text: Text to parse
        key: Key to find (e.g., "final_answer", "confidence")
        
    Returns:
        Extracted integer or -1 if not found
    """
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


def parse_analysis_and_json(text: str) -> Tuple[str, Dict]:
    """
    Parse analysis text and extract JSON.
    
    Args:
        text: Text containing analysis and JSON
        
    Returns:
        Tuple of (analysis_text, json_dict)
    """
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


def header_line(title: str) -> str:
    """Format header line."""
    return f"\n=== {title} ===\n"


def line() -> str:
    """Return line separator."""
    return "\n" + "="*50 + "\n"

