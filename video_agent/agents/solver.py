"""
Solver Agent for VideoAgent.
Stateful agent that maintains conversation history and decides actions.
"""

import logging
from typing import Dict, List, Tuple, Optional, Any

from video_agent.utils.api import get_chat_response
from video_agent.utils.parsing import extract_json_from_response


# System prompt for Solver agent
SOLVER_SYSTEM_PROMPT = """You are a video understanding expert. Your task is to answer questions about video content based on frame captions stored in your memory.

## Your Capabilities
- You receive frame captions that describe individual video frames
- You can request more frames from specific time segments of the video
- You must provide a final answer when you have enough information

## Decision Process
For each turn, you must:
1. First, think step-by-step about the question and available information (write your reasoning in natural language)
2. Then, output your decision as a JSON block

## Available Actions

### Action 1: Retrieve More Frames
Use this when you need more information to answer the question confidently.
You can specify a time range to focus on specific parts of the video.
```json
{
  "thought_summary": "Brief summary of why more frames are needed",
  "action_type": "retrieve_more_frames",
  "action_payload": {
    "count": 5,
    "start_frame": 0,
    "end_frame": 100,
    "focus": "What specific content to look for in new frames"
  }
}
```
Parameters:
- count: Number of frames to retrieve (required)
- start_frame: Starting frame index for the range (optional, default: 0)
- end_frame: Ending frame index for the range (optional, default: last frame)
- focus: Description of what to look for (optional)

If you want frames from the entire video, omit start_frame and end_frame.
If you want to focus on a specific segment (e.g., frames 30-60), specify both.

### Action 2: Answer the Question
Use this when you have enough information to provide a confident answer.
```json
{
  "thought_summary": "Brief summary of reasoning",
  "action_type": "answer_question",
  "action_payload": {
    "final_answer": "Your answer (for multiple choice, use the option number 0-4)",
    "confidence": 8,
    "explanation": "Detailed explanation supporting your answer"
  }
}
```
Parameters:
- final_answer: Your answer (integer 0-4 for multiple choice)
- confidence: Self-assessed confidence level (1-10, where 10 is absolutely certain)
- explanation: Detailed reasoning for your answer

## Important Rules
1. Always output your thinking BEFORE the JSON block
2. The JSON block must be enclosed in ```json and ``` markers
3. For multiple choice questions, final_answer must be an integer (0, 1, 2, 3, or 4)
4. Be decisive - if you have reasonable evidence, provide an answer rather than requesting excessive frames
5. Consider the trade-off: more frames give more information but cost resources
6. Use frame ranges strategically: if you need details about the beginning, middle, or end of the video, specify the appropriate range
"""


class Solver:
    """
    Stateful Solver agent that maintains conversation history.
    
    The Solver decides whether to retrieve more frames or answer the question
    based on the current memory state and conversation context.
    """
    
    def __init__(self, model: str, logger: Optional[logging.Logger] = None):
        """
        Initialize Solver with model and optional logger.
        
        Args:
            model: Model name for LLM calls
            logger: Optional logger for debugging
        """
        self.model = model
        self.logger = logger
        self.chat_history: List[Dict[str, str]] = []
        self.parse_error_count = 0
        self.max_parse_errors = 3
        
        # Initialize with system prompt
        self._init_system_prompt()
    
    def _init_system_prompt(self) -> None:
        """Initialize conversation with system prompt."""
        self.chat_history.append({
            "role": "system",
            "content": SOLVER_SYSTEM_PROMPT
        })
    
    def send_message(self, content: str) -> Tuple[str, Optional[Dict]]:
        """
        Send a message to the Solver and get parsed response.
        
        This method maintains the conversation history by:
        1. Appending the user message to history
        2. Calling the LLM with full history
        3. Appending the assistant response to history
        4. Parsing and returning the response
        
        Args:
            content: Message content to send (user role)
            
        Returns:
            Tuple of (thinking_text, action_dict or None if parse failed)
        """
        # Add user message to history
        self.chat_history.append({
            "role": "user",
            "content": content
        })
        
        if self.logger:
            self.logger.info(f"=== Solver Input ===")
            self.logger.info(f"Message: {content[:500]}..." if len(content) > 500 else f"Message: {content}")
        
        # Call LLM with full chat history
        response = self._call_llm()
        
        # Add assistant response to history
        self.chat_history.append({
            "role": "assistant",
            "content": response
        })
        
        if self.logger:
            self.logger.info(f"=== Solver Output ===")
            self.logger.info(f"Response: {response}")
        
        # Parse response
        thinking, action_json = self._parse_response(response)
        
        return thinking, action_json
    
    def send_error_message(self, error_msg: str) -> Tuple[str, Optional[Dict]]:
        """
        Send an error message to the Solver for retry.
        
        Args:
            error_msg: Error message to send
            
        Returns:
            Tuple of (thinking_text, action_dict or None if parse failed)
        """
        self.parse_error_count += 1
        
        if self.parse_error_count >= self.max_parse_errors:
            if self.logger:
                self.logger.warning(f"Max parse errors ({self.max_parse_errors}) reached")
            return "", None
        
        return self.send_message(error_msg)
    
    def send_feedback(self, feedback: str) -> Tuple[str, Optional[Dict]]:
        """
        Send checker feedback to the Solver.
        
        Args:
            feedback: Formatted feedback message
            
        Returns:
            Tuple of (thinking_text, action_dict or None if parse failed)
        """
        # Reset parse error count on new feedback cycle
        self.parse_error_count = 0
        return self.send_message(feedback)
    
    def send_frame_notification(self, notification: str) -> Tuple[str, Optional[Dict]]:
        """
        Send frame retrieval notification to the Solver.
        
        Args:
            notification: Formatted notification with new frame captions
            
        Returns:
            Tuple of (thinking_text, action_dict or None if parse failed)
        """
        # Reset parse error count on new frame notification
        self.parse_error_count = 0
        return self.send_message(notification)
    
    def _call_llm(self) -> str:
        """
        Call LLM with full chat history.
        
        Returns:
            LLM response string
        """
        try:
            response = get_chat_response(
                model=self.model,
                messages=self.chat_history,
                logger=self.logger
            )
            return response
        except Exception as e:
            if self.logger:
                self.logger.error(f"LLM call failed: {e}")
            raise
    
    def _parse_response(self, response: str) -> Tuple[str, Optional[Dict]]:
        """
        Parse LLM response to extract thinking and action JSON.
        
        Args:
            response: Raw LLM response string
            
        Returns:
            Tuple of (thinking_text, action_dict or None if parse failed)
        """
        thinking, action_json = extract_json_from_response(response)
        
        if action_json is None:
            if self.logger:
                self.logger.warning("Failed to parse JSON from Solver response")
            return thinking, None
        
        # Validate required fields
        if "action_type" not in action_json:
            if self.logger:
                self.logger.warning("Missing 'action_type' in Solver response")
            return thinking, None
        
        action_type = action_json.get("action_type")
        if action_type not in ["retrieve_more_frames", "answer_question"]:
            if self.logger:
                self.logger.warning(f"Invalid action_type: {action_type}")
            return thinking, None
        
        # Validate payload
        if "action_payload" not in action_json:
            if self.logger:
                self.logger.warning("Missing 'action_payload' in Solver response")
            return thinking, None
        
        return thinking, action_json
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """
        Get the full conversation history.
        
        Returns:
            List of message dictionaries
        """
        return self.chat_history.copy()
    
    def get_history_summary(self) -> List[Dict[str, Any]]:
        """
        Get a summary of conversation history for logging.
        
        Returns:
            List of summarized messages
        """
        summary = []
        for msg in self.chat_history:
            if msg["role"] == "system":
                continue
            
            content = msg["content"]
            # Truncate long content
            if len(content) > 200:
                content = content[:200] + "..."
            
            summary.append({
                "role": msg["role"],
                "content": content
            })
        
        return summary

