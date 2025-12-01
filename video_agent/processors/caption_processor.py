"""
Caption processor for video frame analysis.
"""

import json
import time
from typing import Dict, Any, List

from video_agent.core.video_memory import VideoMemory
from video_agent.utils.api import get_llm_response


class CaptionProcessor:
    """Handles caption generation for video frames."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize with configuration."""
        self.config = config
        self.g_reference_length = config.get("reference_length", 50)
        self.g_use_cache = config.get("use_cache", True)
    
    def generate_captions(self, video_memory: VideoMemory, logger=None) -> None:
        """
        Generate captions for video frames.
        
        Args:
            video_memory: VideoMemory instance to update with captions
            logger: Optional logger for detailed LLM logging
        """
        # Get new frames that need captions (only frames that don't have captions yet)
        new_idx = [idx for idx in video_memory.sampled_idx if idx not in video_memory.captions]
        
        if not new_idx:
            return
            
        # Use multi-level caption generation by default
        self._write_multi_level_captions(video_memory, new_idx, logger)
            
    def _get_detailed_caption_prompt(self, reference_length: int) -> str:
        """Generate detailed caption prompt."""
        return f"""describe the scene in a clear, concise caption. Include key details such as:
        - Main objects or people present (tag with #C if the action is done by the camera wearer, #O if done by someone else)
        - Their spatial relationships.
        - other visual elements. 
        Focus on what is visually prominent and avoid speculation beyond what is shown. Do not provide any prediction of events happening. 
        Keep the caption length close to {reference_length} words to control detail level."""
    
    def _write_group_detailed_captions(self, memory: VideoMemory, new_idx: List[int], viewer_model: str, logger=None) -> str:
        """
        Generate detailed captions for frame groups.
        
        Args:
            memory: VideoMemory instance
            new_idx: List of new frame indices to caption
            viewer_model: Model name for caption generation
            logger: Optional logger
            
        Returns:
            Response from LLM
        """
        # Sort the new indices to maintain chronological order
        new_idx.sort()

        # Construct the user prompt with the list of frames
        frames_list = ", ".join([f"Frame {idx}" for idx in new_idx])
        detailed_caption_prompt = self._get_detailed_caption_prompt(self.g_reference_length)
        
        user_prompt = f"""{time.time() if not self.g_use_cache else ""}
        Here are the frames to caption: {frames_list}.
        For each frame, {detailed_caption_prompt}
        Output the captions in this format:
        Frame {new_idx[0]}: [caption]
        Frame {new_idx[1]}: [caption]
        
        Make sure that you caption all the frames in list!!
        ...
        """
        
        # Collect all images into a list
        images = [memory.video_frames[idx] for idx in new_idx]

        # Send all images to the LLM together
        try:
            response = get_llm_response(
                model=viewer_model,
                query=user_prompt,
                images=images,
                logger=logger
            )
            
            # Validate that the response contains the expected format
            if not response or len(response.strip()) == 0:
                if logger:
                    logger.error("Empty response from LLM in _write_group_detailed_captions")
                # Set error captions for all frames
                for idx in new_idx:
                    memory.captions[idx] = "Error: Failed to generate caption"
                return "Error: Empty LLM response"
            
        except Exception as e:
            if logger:
                logger.error(f"Error in _write_group_detailed_captions API call: {e}")
            # Set error captions for all frames
            for idx in new_idx:
                memory.captions[idx] = "Error: Failed to generate caption"
            return f"Error: LLM API error: {str(e)}"

        # Parse the response to extract captions
        try:
            captions = {}
            lines = response.split('\n')
            count = 0
            for line in lines:
                if line.startswith("Frame "):
                    parts = line.split(":", 1)
                    if len(parts) == 2:
                        frame_idx = int(parts[0][6:].strip())
                        if frame_idx in new_idx:
                            captions[frame_idx] = parts[1].strip()
                        else:
                            captions[new_idx[count]] = parts[1].strip()
                    count += 1

            # Store the captions
            for idx, caption in captions.items():
                memory.captions[idx] = caption
                
            # Check if we got captions for all frames
            missing_frames = [idx for idx in new_idx if idx not in captions]
            if missing_frames:
                if logger:
                    logger.warning(f"Missing captions for frames: {missing_frames}")
                # Set error captions for missing frames
                for idx in missing_frames:
                    memory.captions[idx] = "Error: Caption not generated"
                    
        except Exception as e:
            if logger:
                logger.error(f"Error parsing caption response: {e}")
            # Set error captions for all frames that don't have captions yet
            for idx in new_idx:
                if idx not in memory.captions:
                    memory.captions[idx] = "Error: Failed to parse caption"

        return response
    
    def _write_multi_level_captions(self, memory: VideoMemory, new_idx: List[int], logger=None) -> str:
        """
        Generate multi-level captions with visual and event descriptions.
        
        Args:
            memory: VideoMemory instance
            new_idx: List of new frame indices
            logger: Optional logger
            
        Returns:
            Response from LLM
        """
        viewer_model = self.config.get("viewer_model", "gpt-4o-mini-2024-07-18")
        
        # Step 1: Generate detailed captions for new frames (visual-level only)
        self._write_group_detailed_captions(memory, new_idx, viewer_model, logger)

        # Step 2: Use the current overview and events along with the new detailed captions
        # to generate an updated high-level overview and event descriptions
        new_frames_list = ", ".join([f"{idx}" for idx in new_idx])
        images = [memory.video_frames[idx] for idx in new_idx]
        
        user_prompt = f"""
        You are provided with information about video clips and corresponding frames. The video clip details use the following notation:
        - #C indicates an action performed by the camera wearer.
        - #O indicates an action performed by someone other than the camera wearer.
        
        Video clip information:
        {str(memory)}
        
        Corresponding frames:
        {new_frames_list}
        
        ------------------------------------------------------------------------
        Using the provided information, complete the following tasks:
        1. Write brief event descriptions for the new frames, using #C and #O notation. Focus on the actions occurring in each frame.
         You should provide an abstract understanding of the process and state, summarizing what #C is doing across the new frames. Avoid repeating detailed captions or preliminary visual elements.
        2. Update the overall event overview and descriptions of all events (old and new frames) based on the new information, ensuring consistency across all descriptions. Only include interpretations that are certain, without speculation.
        
        Output the results in valid JSON format with these keys:
        - "overview": a string summarizing the updated overall event,
        - "events": an object mapping frame indices (for both old and new frames) to their event descriptions.
        """
        
        # Get the updated overview and events from the LLM
        try:
            response = get_llm_response(
                model=viewer_model,
                query=user_prompt,
                images=images,
                logger=logger
            )
            
            # Validate that the response contains the expected format
            if not response or len(response.strip()) == 0:
                if logger:
                    logger.error("Empty response from LLM in _write_multi_level_captions")
                memory.event_descriptions = "Error: Failed to generate event descriptions"
                memory.overview = "Error: Failed to generate overview"
                return "Error: Empty LLM response"
            
        except Exception as e:
            if logger:
                logger.error(f"Error in _write_multi_level_captions API call: {e}")
            memory.event_descriptions = "Error: Failed to generate event descriptions"
            memory.overview = "Error: Failed to generate overview"
            return f"Error: LLM API error: {str(e)}"
        
        try:
            # Handle JSON response wrapped in code blocks
            response_text = response.strip()
            if response_text.startswith('```json'):
                # Extract JSON content between ```json and ```
                start_idx = response_text.find('```json') + 7
                end_idx = response_text.rfind('```')
                if end_idx > start_idx:
                    json_content = response_text[start_idx:end_idx].strip()
                else:
                    json_content = response_text
            else:
                json_content = response_text
            
            parsed = json.loads(json_content)
            updated_overview = parsed.get("overview", getattr(memory, 'overview', ''))
            updated_events_str = parsed.get("events", {})
            updated_events = {int(str(idx_str).strip()): event for idx_str, event in updated_events_str.items()}
            
            # Update memory with overview and events
            memory.event_descriptions = updated_overview
            memory.overview = updated_overview
            memory.events.update(updated_events)
            
        except Exception as e:
            if logger:
                logger.error(f"Error parsing multi-level caption response: {e}")
                logger.error(f"Response was: {response}")
            memory.event_descriptions = "Error: Failed to parse event descriptions"
            memory.overview = "Error: Failed to parse overview"
        
        # Finally, update the sampled frames in the memory using the correct method
        memory.add_sampled_frames(new_idx)
            
        return response
    
    # Backward compatibility methods
    def _generate_detailed_captions(self, video_memory: VideoMemory, logger=None) -> None:
        """Generate detailed captions for individual frames."""
        prompt = self._get_detailed_caption_prompt(self.g_reference_length)
        
        # Process frames individually
        for i, frame_idx in enumerate(video_memory.sampled_idx):
            if frame_idx in video_memory.captions:
                continue  # Skip if already has caption
                
            frame = video_memory.sampled_frames[i]
            
            response = get_llm_response(
                model=self.config.get("viewer_model", "gpt-4o-mini-2024-07-18"),
                query=prompt,
                images=[frame],
                logger=logger
            )
            
            video_memory.update_caption(frame_idx, response.strip())
    
    def _generate_multi_level_captions(self, video_memory: VideoMemory, logger=None) -> None:
        """Generate multi-level captions with visual and event descriptions."""
        # Get new frames that need captions
        new_idx = [idx for idx in video_memory.sampled_idx if idx not in video_memory.captions]
        
        if new_idx:
            self._write_multi_level_captions(video_memory, new_idx, logger)
    
    def _generate_group_captions(self, video_memory: VideoMemory, logger=None) -> None:
        """Generate captions for frame groups efficiently."""
        new_idx = [idx for idx in video_memory.sampled_idx if idx not in video_memory.captions]
        
        if new_idx:
            self._write_group_detailed_captions(video_memory, new_idx, 
                                               self.config.get("viewer_model", "gpt-4o-mini-2024-07-18"), logger)

