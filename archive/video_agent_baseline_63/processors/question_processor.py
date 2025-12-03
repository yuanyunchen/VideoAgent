"""
Question processor for video analysis.
"""

import json
import random
from typing import Dict, Any, List, Tuple

from video_agent.core.video_memory import VideoMemory
from video_agent.utils.api import get_llm_response
from video_agent.utils.parsing import parse_analysis_and_json, parse_text_find_number, retrieve_frames_by_section

# System prompts matching Base Project
SYSTEM_PROMPT_ANSWER = "You are a helpful video analysis assistant for answering questions related to video."
SYSTEM_PROMPT_CONFIDENCE = "You are a helpful and critical assistant designed to evaluate the trustworthiness of a video Q&A process."
SYSTEM_PROMPT_SEGMENT = "You are an expert video analyst tasked with identifying relevant video segments to answer a specific question"


class QuestionProcessor:
    """Handles question answering and confidence evaluation."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize with configuration."""
        self.config = config
        self.g_max_retrieved_frames = config.get("max_retrieved_frames", 5)
        self.g_min_retrieved_frames = config.get("min_retrieved_frames", 2)
    
    def answer_question(self, video_memory: VideoMemory, logger=None) -> str:
        """
        Answer question based on video memory.
        
        Args:
            video_memory: VideoMemory instance with captions and event descriptions
            logger: Optional logger for detailed LLM logging
            
        Returns:
            Answer response string
        """
        answer_format = {"final_answer": 1}
        
        # Format the question with multiple choice options
        formatted_question = video_memory.question
        if video_memory.choices:
            formatted_question = (
                f"Here is the question: \n{video_memory.question}\n\n"
                + "Here are the choices: \n"
                + "\n".join([f"{i}. {choice}" for i, choice in enumerate(video_memory.choices)])
            )
        
        user_prompt = f"""
        You are provided with a video containing {video_memory.n_frames} frames, decoded at 1 frame per second. The video description uses the following notation:
        - #C indicates an action performed by the camera wearer.
        - #O indicates an action performed by someone other than the camera wearer.
        
        Video description:
        {str(video_memory)}

        ------------------------------------------------------------------------
        Answer the following question:
        {formatted_question}
        
        IMPORTANT RULES:
        - You MUST select exactly ONE answer from the given choices (0, 1, 2, 3, or 4).
        - Do NOT output "none", "null", "N/A", or any non-numeric answer.
        - Even if the video content does not perfectly match any choice, you MUST select the option that is MOST LIKELY correct based on the available information.
        - If uncertain, make your best educated guess based on partial evidence or general reasoning.
        
        Format your response as follows:
        - First, write your step-by-step analysis as plain text.
        - Then, on a new line, include the JSON output enclosed in triple backticks (```json ... ```), like this:
        ```json
        {answer_format}
        ```
        
        Remember: final_answer MUST be an integer from 0 to 4. No exceptions.
        """
        
        try:
            response = get_llm_response(
                model=self.config.get("scheduler_model", "gpt-4o-mini-2024-07-18"),
                query=user_prompt,
                system_prompt=SYSTEM_PROMPT_ANSWER,
                logger=logger
            )
            
            # Validate that the response contains the expected format
            if not response or len(response.strip()) == 0:
                if logger:
                    logger.error("Empty response from LLM in answer_question")
                return self._generate_error_response("Empty LLM response", answer_format)
            
            # Try to parse the response to ensure it contains valid JSON
            try:
                _, json_part = parse_analysis_and_json(response)
                if "final_answer" not in json_part:
                    if logger:
                        logger.error("Missing 'final_answer' key in LLM response")
                    return self._generate_error_response("Missing final_answer key", answer_format)
            except Exception as parse_error:
                if logger:
                    logger.error(f"Failed to parse LLM response JSON: {parse_error}")
                return self._generate_error_response("Invalid JSON format", answer_format)
            
            return response
            
        except Exception as e:
            if logger:
                logger.error(f"Error in answer_question: {e}")
            return self._generate_error_response(f"LLM API error: {str(e)}", answer_format)
    
    def _generate_error_response(self, error_msg: str, format_template: dict) -> str:
        """Generate error response with -1 as final_answer."""
        error_json = format_template.copy()
        if "final_answer" in error_json:
            error_json["final_answer"] = -1
        elif "confidence" in error_json:
            error_json["confidence"] = -1
        
        return f"Error: {error_msg}\n\n```json\n{json.dumps(error_json)}\n```"
    
    def evaluate_confidence(self, video_memory: VideoMemory, question: str, answer_str: str, logger=None, not_confident: bool = True) -> str:
        """
        Evaluate confidence of answer.
        
        Args:
            video_memory: VideoMemory instance
            question: Original question
            answer_str: Answer response from answer_question
            logger: Optional logger for detailed LLM logging
            not_confident: Whether to include additional guidance for low confidence (default: True)
            
        Returns:
            Confidence evaluation response string
        """
        confidence_level_guidance = """When evaluating confidence, remain cautious and avoid overconfidence. If there's any uncertainty or need for more information, assign a confidence level of '2'. Prioritize accuracy by frequently using '2' when evidence is incomplete or unclear."""
        
        confidence_format = {"confidence": 2}
        
        user_prompt = f"""
        You are tasked with evaluating the trustworthiness of an answer based on a video's content, a question, and the decision-making process provided. Follow these steps:

        1. Review the video content summary:  
           {str(video_memory)}  
           -----------------------------------------------------
        
        2. Review the question:  
           {question}  
           -----------------------------------------------------
        
        3. Review the answer and decision-making process:  
           {answer_str}  
           -----------------------------------------------------
        
        4. Assess the confidence level of the answer's trustworthiness using the following criteria:  
           - **Confidence Level 1 (Insufficient Information):** The provided information is too limited to support a reasonable conclusion.  
           - **Confidence Level 2 (Partial Information):** The information partially supports an informed guess but is incomplete or unclear in key areas.  
           - **Confidence Level 3 (Sufficient Information):** The information is clear, complete, and fully supports a reliable, well-informed conclusion.  
        
            Focus your evaluation on:  
           - Relevance: Does the information directly address the question?  
           - Completeness: Is there enough detail to make a sound decision?  
           - Clarity: Is the information unambiguous and easy to understand?  
        
            If the confidence level is below 3, consider additional guidance:  
           {confidence_level_guidance if not_confident else ""}  
        
        5.Format your response as follows:
        - First, write your step-by-step analysis as plain text.
        - Then, on a new line, include the JSON output enclosed in triple backticks (```json ... ```), like this:
           {confidence_format}
        """
        
        try:
            response = get_llm_response(
                model=self.config.get("scheduler_model", "gpt-4o-mini-2024-07-18"),
                query=user_prompt,
                system_prompt=SYSTEM_PROMPT_CONFIDENCE,
                logger=logger
            )
            
            # Validate that the response contains the expected format
            if not response or len(response.strip()) == 0:
                if logger:
                    logger.error("Empty response from LLM in evaluate_confidence")
                return self._generate_error_response("Empty LLM response", confidence_format)
            
            # Try to parse the response to ensure it contains valid JSON
            try:
                _, json_part = parse_analysis_and_json(response)
                if "confidence" not in json_part:
                    if logger:
                        logger.error("Missing 'confidence' key in LLM response")
                    return self._generate_error_response("Missing confidence key", confidence_format)
            except Exception as parse_error:
                if logger:
                    logger.error(f"Failed to parse LLM confidence response JSON: {parse_error}")
                return self._generate_error_response("Invalid JSON format", confidence_format)
            
            return response
            
        except Exception as e:
            if logger:
                logger.error(f"Error in evaluate_confidence: {e}")
            return self._generate_error_response(f"LLM API error: {str(e)}", confidence_format)
    
    def generate_segment_steps(self, video_memory: VideoMemory, question: str, logger=None) -> Tuple[List[int], str]:
        """
        Generate new frame indices using LLM segment prediction.
        
        Args:
            video_memory: VideoMemory instance
            question: Original question
            logger: Optional logger for detailed LLM logging
            
        Returns:
            Tuple of (new_frame_indices, response_string)
        """
        # Step 1: Define segment ranges based on sampled_idx
        segment_des = {
            i + 1: f"Frame {video_memory.sampled_idx[i]}-{video_memory.sampled_idx[i + 1]}"
            for i in range(len(video_memory.sampled_idx) - 1)
        }
        
        # Format the current captions for the prompt
        example_format = {
            "1": 3,
            "4": 2
        }
        segments_list = "\n".join([f"{k}: {v}" for k, v in segment_des.items()])
        
        # Format the question with multiple choice options
        formatted_question = question
        if video_memory.choices:
            formatted_question = (
                f"Here is the question: \n{question}\n\n"
                + "Here are the choices: \n"
                + "\n".join([f"{i}. {choice}" for i, choice in enumerate(video_memory.choices)])
            )
        
        user_prompt = f"""
        You are tasked with answering the following question :
        {formatted_question}
        
        Provided information:
        {str(video_memory)}
        
        Available segments:
        {segments_list}
        
        ------------------------------------------------------------------------
        Using the provided information, perform the following:
        1. Identify which segments are most relevant to answering the question.
        2. For each relevant segment, determine how many frames to retrieve, ensuring the total number of frames across all segments is {self.g_min_retrieved_frames} to {self.g_max_retrieved_frames}.
        3. Only include segments that directly contribute to answering the question.
        
        Remind Again:
        Ensure the total number of new sampled frames across all segments is {self.g_min_retrieved_frames} to {self.g_max_retrieved_frames}. 
        
        Format your response as follows:
        - First, write your step-by-step analysis as plain text.
        - Then, on a new line, include the JSON output enclosed in triple backticks (```json ... ```), with this structure "segment_id (int)": number_of_frames_to_retrieve.:
        Example: 
        {example_format}
        """
        
        # Step 3: Get the model's prediction
        response = get_llm_response(
            model=self.config.get("scheduler_model", "gpt-4o-mini-2024-07-18"),
            query=user_prompt,
            system_prompt=SYSTEM_PROMPT_SEGMENT,
            logger=logger
        )
        
        # Step 4: Process the response and calculate new_idx
        try:
            _, segment_str_predictions = parse_analysis_and_json(response)
            segment_predictions = {int(k): v for k, v in segment_str_predictions.items()}
            
            # Use retrieve_frames_by_section to get new indices
            new_idx = retrieve_frames_by_section(segment_predictions, segment_des)
            
            # Ensure new_idx is not empty and does not exceed max frames
            if not new_idx:
                if logger:
                    logger.warning("Segment prediction returned no new frames. Using fallback.")
                # Fallback: Sample randomly if no segments are returned
                new_idx = self._fallback_sampling(video_memory)

            # Cap the number of new frames to avoid excessive processing
            if len(new_idx) > self.g_max_retrieved_frames:
                if logger:
                    logger.warning(f"Too many frames requested ({len(new_idx)}). Capping at {self.g_max_retrieved_frames}.")
                new_idx = new_idx[:self.g_max_retrieved_frames]
                
        except Exception as e:
            if logger:
                logger.error(f"Error parsing segment prediction response: {e}")
                logger.warning("Falling back to random sampling.")
            new_idx = self._fallback_sampling(video_memory)

        return new_idx, response

    def _fallback_sampling(self, memory: VideoMemory) -> List[int]:
        """Fallback to random sampling if segment prediction fails."""
        available_frames = list(set(range(memory.n_frames)) - set(memory.sampled_idx))
        if not available_frames:
            return []
        
        sample_count = min(len(available_frames), self.g_min_retrieved_frames)
        return random.sample(available_frames, sample_count)
    
    def process_question(self, video_memory: VideoMemory, logger=None) -> Dict[str, Any]:
        """
        Process question with separated answer and confidence evaluation.
        
        Args:
            video_memory: VideoMemory instance
            logger: Optional logger for detailed LLM logging
            
        Returns:
            Dictionary with answer, confidence, and analysis
        """
        # Step 1: Answer the question
        answer_str = self.answer_question(video_memory, logger)
        answer = parse_text_find_number(answer_str, "final_answer")
        
        # Step 2: Evaluate confidence
        confidence_str = self.evaluate_confidence(video_memory, video_memory.question, answer_str, logger)
        confidence = parse_text_find_number(confidence_str, "confidence")
        
        return {
            "answer": answer,
            "confidence": confidence,
            "answer_analysis": answer_str,
            "confidence_analysis": confidence_str
        }
    
    def resample_frames(self, video_memory: VideoMemory, logger=None) -> List[int]:
        """
        Resample frames using segment prediction when confidence is low.
        
        Args:
            video_memory: VideoMemory instance
            logger: Optional logger for detailed LLM logging
            
        Returns:
            List of new frame indices
        """
        new_idx, response = self.generate_segment_steps(video_memory, video_memory.question, logger)
        
        # Add new frames to video memory
        if new_idx:
            video_memory.add_sampled_frames(new_idx)
        
        return new_idx

