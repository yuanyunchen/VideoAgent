"""
Parsing utilities for VideoAgent.
"""

import os
import re
import json
from typing import List, Dict, Any, Optional, Tuple


def _clean_backticks(text: str) -> str:
    """
    Clean up stray triple backticks from LLM output.
    
    Some models (e.g., grok-4-1-fast-non-reasoning) wrap JSON code blocks
    in extra backticks like:
        ```
        ```json
        {'key': value}
        ```
        ```
    
    This function removes isolated ``` lines to normalize the format.
    
    Args:
        text: Raw text from LLM response
        
    Returns:
        Cleaned text with stray backticks removed
    """
    lines = text.split('\n')
    cleaned_lines = []
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        # Keep the line if it's not just ```
        # But keep ```json, ```python, etc.
        if stripped == '```':
            # Check if this is a stray backtick (not part of a code block)
            # A stray backtick is one that appears right before ```json or right after a closing ```
            # We remove it to normalize the format
            
            # Look ahead: if next non-empty line starts with ```json, this is stray
            is_stray = False
            for j in range(i + 1, min(i + 3, len(lines))):
                next_line = lines[j].strip()
                if next_line.startswith('```') and next_line != '```':
                    # This is like ```json, so current ``` is stray
                    is_stray = True
                    break
                elif next_line:
                    # Non-empty line that's not a code fence
                    break
            
            # Look behind: if previous content ended with ```, this might be stray
            if not is_stray:
                for j in range(i - 1, max(i - 3, -1), -1):
                    prev_line = lines[j].strip()
                    if prev_line == '```':
                        # Previous was also ```, so this is likely a stray closing
                        is_stray = True
                        break
                    elif prev_line:
                        # Non-empty line
                        break
            
            if not is_stray:
                cleaned_lines.append(line)
        else:
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)


def parse_video_annotation(annotation_file: str, video_list_file: Optional[str] = None, 
                         max_videos: int = -1) -> List[Dict[str, Any]]:
    """
    Parse video annotations and return list of video info.
    
    Args:
        annotation_file: Path to annotation JSON file
        video_list_file: Optional file containing video IDs to process (preserves order and duplicates)
        max_videos: Maximum number of videos to return (-1 for all)
        
    Returns:
        List of video information dictionaries
    """
    # Load annotations
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)
    
    # Load video list if provided (preserve order and allow duplicates)
    target_video_list = None
    if video_list_file and os.path.exists(video_list_file):
        with open(video_list_file, 'r') as f:
            target_video_list = [line.strip() for line in f if line.strip()]
    
    # Process annotations
    videos_info = []
    
    if target_video_list:
        # Follow the order in video list (allows duplicates for testing)
        for video_id in target_video_list:
            if video_id not in annotations:
                continue  # Skip if video not in annotations
                
            data = annotations[video_id]
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
    else:
        # No video list, use all annotations
        for video_id, data in annotations.items():
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
    # Clean up stray backticks first (handles double-wrapped JSON from some models)
    text_content = _clean_backticks(text.strip())
    
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
        # Handle Python None vs JSON null (some models output Python syntax)
        json_content_fixed = re.sub(r'\bNone\b', 'null', json_content_fixed)
        data = json.loads(json_content_fixed)
        if key in data and data[key] is not None:
            return int(data[key])
    except (json.JSONDecodeError, ValueError, KeyError, TypeError):
        pass
    
    # Fallback to regex patterns
    patterns = [
        rf'"{key}":\s*(-?\d+)',  # "key": number (including negative)
        rf"'{key}':\s*(-?\d+)",  # 'key': number (including negative)
        rf'{key}.*?(-?\d+)',     # key followed by number
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
    # Clean up stray backticks first (handles double-wrapped JSON from some models)
    text_content = _clean_backticks(text.strip())
    
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
        # Handle Python None vs JSON null (some models output Python syntax)
        json_content_fixed = re.sub(r'\bNone\b', 'null', json_content_fixed)
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

