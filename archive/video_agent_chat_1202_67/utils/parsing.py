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
    """
    lines = text.split('\n')
    cleaned_lines = []
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped == '```':
            is_stray = False
            for j in range(i + 1, min(i + 3, len(lines))):
                next_line = lines[j].strip()
                if next_line.startswith('```') and next_line != '```':
                    is_stray = True
                    break
                elif next_line:
                    break
            
            if not is_stray:
                for j in range(i - 1, max(i - 3, -1), -1):
                    prev_line = lines[j].strip()
                    if prev_line == '```':
                        is_stray = True
                        break
                    elif prev_line:
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
    """
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)
    
    target_video_list = None
    if video_list_file and os.path.exists(video_list_file):
        with open(video_list_file, 'r') as f:
            target_video_list = [line.strip() for line in f if line.strip()]
    
    videos_info = []
    
    if target_video_list:
        for video_id in target_video_list:
            if video_id not in annotations:
                continue
                
            data = annotations[video_id]
            video_info = {
                "video_id": video_id,
                "question": data.get("question", ""),
                "answer": data.get("answer", "")
            }
            
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
        for video_id, data in annotations.items():
            video_info = {
                "video_id": video_id,
                "question": data.get("question", ""),
                "answer": data.get("answer", "")
            }
            
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
    """
    text_content = _clean_backticks(text.strip())
    
    if '```json' in text_content:
        start_idx = text_content.find('```json') + 7
        end_idx = text_content.rfind('```')
        if end_idx > start_idx:
            json_content = text_content[start_idx:end_idx].strip()
        else:
            json_content = text_content
    else:
        json_content = text_content
    
    try:
        json_content_fixed = json_content.replace("'", '"')
        json_content_fixed = re.sub(r'\bNone\b', 'null', json_content_fixed)
        data = json.loads(json_content_fixed)
        if key in data and data[key] is not None:
            return int(data[key])
    except (json.JSONDecodeError, ValueError, KeyError, TypeError):
        pass
    
    patterns = [
        rf'"{key}":\s*(-?\d+)',
        rf"'{key}':\s*(-?\d+)",
        rf'{key}.*?(-?\d+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text_content)
        if match:
            return int(match.group(1))
    
    return -1


def parse_analysis_and_json(text: str) -> Tuple[str, Dict]:
    """
    Parse analysis text and extract JSON.
    """
    text_content = _clean_backticks(text.strip())
    
    if '```json' in text_content:
        start_idx = text_content.find('```json') + 7
        end_idx = text_content.rfind('```')
        if end_idx > start_idx:
            json_content = text_content[start_idx:end_idx].strip()
            analysis = text_content[:start_idx-7].strip()
        else:
            json_content = text_content
            analysis = text_content
    else:
        json_pattern = r'\{[^{}]*\}'
        match = re.search(json_pattern, text_content)
        if match:
            json_content = match.group(0)
            analysis = text_content[:match.start()].strip()
        else:
            return text_content, {}
    
    try:
        json_content_fixed = re.sub(r'\bNone\b', 'null', json_content)
        json_content_fixed = re.sub(r'\bTrue\b', 'true', json_content_fixed)
        json_content_fixed = re.sub(r'\bFalse\b', 'false', json_content_fixed)
        json_data = json.loads(json_content_fixed)
        return analysis, json_data
    except json.JSONDecodeError:
        # Try fixing single-quoted keys only
        try:
            fixed_str = re.sub(r"'([^']+)'(\s*:)", r'"\1"\2', json_content)
            json_data = json.loads(fixed_str)
            return analysis, json_data
        except json.JSONDecodeError:
            pass
    
    return text_content, {}


def extract_json_from_response(text: str) -> Tuple[str, Optional[Dict]]:
    """
    Extract JSON block and thinking text from LLM response.
    
    Returns:
        Tuple of (thinking_text, json_dict or None if parse failed)
    """
    text_content = _clean_backticks(text.strip())
    
    # Try to find ```json block
    json_match = re.search(r'```json\s*([\s\S]*?)\s*```', text_content)
    
    if json_match:
        json_str = json_match.group(1).strip()
        thinking = text_content[:json_match.start()].strip()
        
        try:
            # Fix common Python-style issues (but NOT single quotes in values)
            json_str = re.sub(r'\bNone\b', 'null', json_str)
            json_str = re.sub(r'\bTrue\b', 'true', json_str)
            json_str = re.sub(r'\bFalse\b', 'false', json_str)
            json_data = json.loads(json_str)
            return thinking, json_data
        except json.JSONDecodeError:
            # Try fixing single-quoted keys only (not values)
            try:
                # Replace single-quoted keys: 'key': -> "key":
                fixed_str = re.sub(r"'([^']+)'(\s*:)", r'"\1"\2', json_str)
                json_data = json.loads(fixed_str)
                return thinking, json_data
            except json.JSONDecodeError:
                pass
    
    # Fallback: try to find any JSON object
    json_pattern = r'\{[\s\S]*\}'
    match = re.search(json_pattern, text_content)
    if match:
        try:
            json_str = match.group(0)
            json_str = re.sub(r'\bNone\b', 'null', json_str)
            json_str = re.sub(r'\bTrue\b', 'true', json_str)
            json_str = re.sub(r'\bFalse\b', 'false', json_str)
            json_data = json.loads(json_str)
            thinking = text_content[:match.start()].strip()
            return thinking, json_data
        except json.JSONDecodeError:
            # Try fixing single-quoted keys only
            try:
                fixed_str = re.sub(r"'([^']+)'(\s*:)", r'"\1"\2', json_str)
                json_data = json.loads(fixed_str)
                thinking = text_content[:match.start()].strip()
                return thinking, json_data
            except json.JSONDecodeError:
                pass
    
    return text_content, None


def retrieve_frames_by_section(section_predictions: Dict[int, int], 
                             segment_descriptions: Dict[int, str]) -> List[int]:
    """
    Retrieve frame indices based on section predictions.
    """
    new_indices = []
    
    for section_id, frame_count in section_predictions.items():
        if section_id in segment_descriptions:
            frame_range = segment_descriptions[section_id]
            try:
                parts = frame_range.replace("Frame ", "").split("-")
                start_frame = int(parts[0])
                end_frame = int(parts[1]) if len(parts) > 1 else start_frame + 10
                
                if frame_count > 0:
                    step = (end_frame - start_frame) / (frame_count + 1)
                    indices = [start_frame + int(step * (i + 1)) for i in range(frame_count)]
                    new_indices.extend(indices)
                    
            except (ValueError, IndexError):
                for i in range(frame_count):
                    new_indices.append(section_id * 10 + i)
    
    return sorted(list(set(new_indices)))


def header_line(title: str) -> str:
    """Format header line."""
    return f"\n=== {title} ===\n"


def line() -> str:
    """Return line separator."""
    return "\n" + "="*50 + "\n"

