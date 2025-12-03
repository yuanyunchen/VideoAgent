"""
Caption Generator for VideoAgent.
Generates captions for video frames using vision LLM.
"""

import logging
import numpy as np
from typing import List, Dict, Optional

from video_agent.utils.api import get_llm_response


# System prompt for caption generation
CAPTION_SYSTEM_PROMPT = """You are an expert video analyst tasked with generating detailed captions for video frames.

For each frame, describe the scene in a clear, concise caption (approximately 50 words). Include:
- Main objects or people present
- Use #C to indicate actions by the camera wearer (first-person perspective)
- Use #O to indicate actions by someone other than the camera wearer
- Spatial relationships between objects
- Key visual elements and context

Focus on what is visually prominent and avoid speculation beyond what is shown.
"""


class CaptionGenerator:
    """
    Generates captions for video frames using a vision-capable LLM.
    """
    
    def __init__(self, model: str, logger: Optional[logging.Logger] = None):
        """
        Initialize CaptionGenerator.
        
        Args:
            model: Vision-capable model name for LLM calls
            logger: Optional logger for debugging
        """
        self.model = model
        self.logger = logger
        self.reference_length = 50  # Target caption length in words
    
    def generate_captions(self, frames: List[np.ndarray], frame_ids: List[int]) -> List[str]:
        """
        Generate captions for multiple frames.
        
        Args:
            frames: List of frames as numpy arrays
            frame_ids: List of frame indices
            
        Returns:
            List of caption strings
        """
        if not frames:
            return []
        
        # Build prompt
        frames_list = ", ".join([f"Frame {idx}" for idx in frame_ids])
        user_prompt = f"""Here are {len(frames)} video frames to caption: {frames_list}.

For each frame, provide a detailed caption describing:
- What is happening in the frame
- Key objects, people, and their actions (use #C for camera wearer, #O for others)
- The setting and context

Output format (one line per frame):
Frame {frame_ids[0]}: [your caption here]
Frame {frame_ids[1]}: [your caption here]
...

Make sure to caption ALL frames in the list!
"""
        
        if self.logger:
            self.logger.info(f"=== Caption Generation ===")
            self.logger.info(f"Generating captions for frames: {frame_ids}")
        
        try:
            response = get_llm_response(
                model=self.model,
                query=user_prompt,
                images=frames,
                system_prompt=CAPTION_SYSTEM_PROMPT,
                logger=self.logger
            )
            
            # Parse response to extract captions
            captions = self._parse_captions(response, frame_ids)
            
            if self.logger:
                self.logger.info(f"Generated {len(captions)} captions")
            
            return captions
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Caption generation failed: {e}")
            # Return placeholder captions on error
            return [f"[Caption generation failed for frame {idx}]" for idx in frame_ids]
    
    def generate_single_caption(self, frame: np.ndarray, frame_id: int) -> str:
        """
        Generate caption for a single frame.
        
        Args:
            frame: Frame as numpy array
            frame_id: Frame index
            
        Returns:
            Caption string
        """
        captions = self.generate_captions([frame], [frame_id])
        return captions[0] if captions else f"[Caption generation failed for frame {frame_id}]"
    
    def _parse_captions(self, response: str, frame_ids: List[int]) -> List[str]:
        """
        Parse LLM response to extract captions.
        
        Args:
            response: Raw LLM response
            frame_ids: Expected frame indices
            
        Returns:
            List of caption strings
        """
        captions = {}
        lines = response.strip().split('\n')
        
        # Track which frame IDs we've seen in order
        count = 0
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Try to parse "Frame X: caption" format
            if line.startswith("Frame "):
                try:
                    colon_idx = line.find(':')
                    if colon_idx == -1:
                        continue
                    
                    frame_part = line[:colon_idx].replace("Frame ", "").strip()
                    caption = line[colon_idx + 1:].strip()
                    
                    # Try to parse frame number
                    try:
                        frame_idx = int(frame_part)
                        if frame_idx in frame_ids:
                            captions[frame_idx] = caption
                        elif count < len(frame_ids):
                            # Use positional mapping if frame number doesn't match
                            captions[frame_ids[count]] = caption
                            count += 1
                    except ValueError:
                        # If frame number can't be parsed, use positional
                        if count < len(frame_ids):
                            captions[frame_ids[count]] = caption
                            count += 1
                        
                except Exception:
                    continue
        
        # Build result list, using placeholders for missing captions
        result = []
        for frame_id in frame_ids:
            if frame_id in captions:
                result.append(captions[frame_id])
            else:
                result.append(f"[No caption generated for frame {frame_id}]")
        
        return result

