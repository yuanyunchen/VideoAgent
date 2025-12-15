"""
LangGraph Agent Definition

Implements the ReAct-style agent using LangGraph with:
- Agent node (LLM decision making)
- Tool node (tool execution)
- Conditional routing (continue/end/force)
"""

import os
import json
import re
import logging
from typing import Dict, Any, List, Literal, Optional
from datetime import datetime

from langgraph.graph import StateGraph, END
from openai import OpenAI
from dotenv import load_dotenv

from video_agent_tools.state import (
    AgentState, 
    ToolCall, 
    VideoContext, 
    VideoMemory,
    ToolHistoryEntry,
    AgentHistory, 
    create_initial_state, 
    get_tool_type,
    is_frame_content_tool,
    is_qa_content_tool,
)
from video_agent_tools.prompts import (
    build_system_prompt,
    build_task_prompt,
    build_task_prompt_v2,
    build_memory_context,
    build_force_prompt,
    build_tool_error_prompt,
    PARSE_ERROR_PROMPT,
    NO_TOOL_SUBMIT_WARNING,
)
from video_agent_tools.tools import ToolManager
from tools.interface import Video as VideoInput
from video_agent_tools.utils.video import (
    load_video_context,
    sample_uniform_indices,
    extract_frames,
)
from video_agent_tools.utils.logging import (
    log_tool_call,
    log_llm_interaction,
    log_video_start,
    log_video_end,
    log_caption_generation,
    setup_video_logger,
    log_llm_interaction_full,
    log_tool_call_full,
    setup_simple_logger,
    log_tool_call_simple,
    log_agent_action_simple,
    log_session_start_simple,
    log_session_end_simple,
    log_initial_captions_simple,
)


# Load environment variables
load_dotenv()


class VideoAgent:
    """
    LangGraph-based Video Understanding Agent.
    
    Implements a ReAct-style agent that:
    1. Receives video with initial frame captions
    2. Iteratively calls tools or submits answers
    3. Forces answer when max tool calls reached
    """
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        enabled_tools: List[str] = None,
        max_tool_calls: int = 10,
        max_parallel_tools: int = 3,
        max_parse_errors: int = 3,
        initial_frames: int = 5,
        captioner: str = "gpt-4o-mini",
        logger: logging.Logger = None,
        # Frame control parameters
        max_view_frames: int = 8,
        default_sample_frames: int = 16,
        min_sample_frames: int = 1,
        max_sample_frames: int = 32,
        # Tool manager/client injection (for multiprocessing mode)
        tool_manager: Optional["ToolManager"] = None,
    ):
        """
        Initialize VideoAgent.
        
        Args:
            model: LLM model name for agent decisions
            enabled_tools: List of tool keys to enable
            max_tool_calls: Maximum number of tool calls before forced answer
            max_parallel_tools: Maximum tools agent can request per turn (all will be executed)
            max_parse_errors: Maximum parse errors before forcing answer
            initial_frames: Number of frames to caption at initialization
            captioner: Captioner model - 'omni-captioner' for local OmniCaptioner,
                      or API model name (e.g., 'gpt-4o-mini', 'x-ai/grok-4-1-fast-reasoning')
            logger: Logger instance
            max_view_frames: Maximum frames for view_frame tool
            default_sample_frames: Default number of frames for sampling tools
            min_sample_frames: Minimum number of frames for sampling tools
            max_sample_frames: Maximum number of frames for sampling tools
            tool_manager: Optional pre-initialized ToolManager/ToolClient (for multiprocessing)
        """
        self.model = model
        # Default enabled tools
        # Available tools in INTERFACE_MAPPING:
        #   Frame/Caption: view_frame, caption_image, detailed_captioning
        #   Sampling: temporal_sample_frames, temporal_spatial_sample_frames
        #   Detection: detect_objects, detect_all_objects
        #   Description: describe_region
        #   QA: temporal_qa (VideoRAG), temporal_spatial_qa (TStar)
        self.enabled_tools = enabled_tools or [
            # Frame viewing
            "view_frame",
            # Sampling (temporal/spatial)
            "temporal_sample_frames",
            "temporal_spatial_sample_frames",
            # Sub-question answering (for decomposition)
            "temporal_qa",
            "temporal_spatial_qa",
            # Object detection
            "detect_objects",
            "detect_all_objects",
            # Region description (DAM)
            "describe_region",
        ]
        self.max_tool_calls = max_tool_calls
        self.max_parallel_tools = max_parallel_tools
        self.max_parse_errors = max_parse_errors
        self.initial_frames = initial_frames
        
        # Parse captioner setting: 'omni-captioner' = local, otherwise = API model
        self.captioner = captioner
        if captioner == "omni-captioner":
            self.use_api_captioner = False
            self.captioner_model = "U4R/OmniCaptioner"
            self.api_captioner_model = "gpt-4o-mini"  # Default, not used
        else:
            self.use_api_captioner = True
            self.captioner_model = "U4R/OmniCaptioner"  # Default, not used
            self.api_captioner_model = captioner
        
        self.logger = logger or logging.getLogger(__name__)
        
        # Frame control parameters
        self.max_view_frames = max_view_frames
        self.default_sample_frames = default_sample_frames
        self.min_sample_frames = min_sample_frames
        self.max_sample_frames = max_sample_frames
        
        # Initialize OpenAI client
        self._init_llm_client()
        
        # Tool manager (can be injected for multiprocessing, otherwise lazy initialized)
        # When tool_manager is provided (e.g., ToolClient), it will be used instead of
        # creating a new ToolManager locally. This is used in multiprocessing mode
        # where tools are managed by a central ToolServer.
        self.tool_manager: Optional[ToolManager] = tool_manager
        
        # Build the graph
        self.graph = self._build_graph()
    
    def _init_llm_client(self):
        """Initialize the LLM client.
        
        Raises:
            ValueError: If no API key is found in environment variables
        """
        # Try AIML API first, then OpenAI
        aiml_key = os.environ.get("AIML_API_KEY")
        openai_key = os.environ.get("OPENAI_API_KEY")
        
        if not aiml_key and not openai_key:
            raise ValueError(
                "No API key found. Please set either AIML_API_KEY or OPENAI_API_KEY "
                "environment variable. You can set it in a .env file or export it directly."
            )
        
        if aiml_key:
            base_url = os.environ.get("AIML_BASE_URL", "https://api.aimlapi.com/v1")
            self.client = OpenAI(api_key=aiml_key, base_url=base_url)
        else:
            self.client = OpenAI(api_key=openai_key)
    
    def _generate_video_description(self, video_path: str, video_context: Any) -> tuple:
        """Generate video description using InternVideoDescription tool.
        
        This method uses ToolManager to execute internvideo_description if enabled,
        or creates a temporary instance that shares GPU resources properly.
        
        In multiprocessing mode (using ToolClient), only use tool_manager.execute()
        and skip local model loading to avoid loading models in worker processes.
        
        Args:
            video_path: Path to the video file
            video_context: VideoContext object for the video
            
        Returns:
            tuple: (description_text, cache_hit) where cache_hit is True if from cache
        """
        from tools.interface import InternVideoDescription
        
        try:
            # Check if internvideo_description is enabled - use tool_manager if so
            if "internvideo_description" in self.enabled_tools:
                # Use tool_manager's execute method for proper resource management
                # This works for both local ToolManager and remote ToolClient
                result = self.tool_manager.execute(
                    "internvideo_description",
                    {
                        "start_frame": 0,
                        "end_frame": max(0, video_context.total_frames - 1),
                    },
                    video_context,
                )
                if result.get("error"):
                    self.logger.warning(f"Video description failed: {result['error']}")
                    return "", False
                cache_hit = result.get("cache_hit", False)
                # Format output - result contains 'result' key with formatted text
                if "result" in result:
                    return result["result"], cache_hit
                return InternVideoDescription.format_output_for_agent(result), cache_hit
            
            # Check if we're using ToolClient (multiprocessing mode)
            # In multiprocessing mode, don't load models locally - skip if not enabled
            from video_agent_tools.resource_management import ToolClient
            if isinstance(self.tool_manager, ToolClient):
                # In multiprocessing mode, skip video description if not in enabled_tools
                self.logger.debug("Skipping video description (not in enabled_tools, multiprocessing mode)")
                return "", False
            
            # Single-process mode: create temporary instance with proper GPU management
            from video_agent_tools.resource_management import ToolPriority
            
            target_device = None
            if self.tool_manager and hasattr(self.tool_manager, '_gpu_manager') and self.tool_manager._gpu_manager:
                target_device = self.tool_manager._gpu_manager.select_device_for_tool("internvideo_description")
                self.logger.info(f"Using device {target_device} for video description")
            
            # Initialize InternVideoDescription
            if target_device:
                description_tool = InternVideoDescription(device=target_device)
            else:
                description_tool = InternVideoDescription()
            description_tool.initialize()
            
            # Register with GPU manager so it can be reused by internvideo_general_qa
            if self.tool_manager and hasattr(self.tool_manager, '_gpu_manager') and self.tool_manager._gpu_manager:
                self.tool_manager._gpu_manager.register_tool(
                    tool_name="internvideo_description",
                    instance=description_tool,
                    device=target_device or "cuda:0",
                    priority=ToolPriority.LOW,
                )
                # Also store in tool_manager for reuse
                self.tool_manager._tool_instances["internvideo_description"] = description_tool
            
            # Execute the description
            video_input = VideoInput(path=video_path)
            result = description_tool(video=video_input)
            
            if "error" in result:
                self.logger.warning(f"Video description failed: {result['error']}")
                return "", False
            
            # Format the output
            description = InternVideoDescription.format_output_for_agent(result)
            return description, False  # Single-process mode doesn't use cache currently
                
        except Exception as e:
            self.logger.warning(f"Video description generation error: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return "", False
    
    def _validate_tool_args(self, tool_name: str, args: Dict[str, Any]) -> Optional[str]:
        """Validate tool arguments using ToolManager's registry (includes frame_indices).
        
        Returns:
            Error message if validation fails, None if valid
        """
        if self.tool_manager is None:
            return f"Tool manager not initialized"
        
        registry = self.tool_manager.get_tool_registry()
        
        if tool_name not in registry:
            return f"Unknown tool: {tool_name}. Available: {list(registry.keys())}"
        
        schema = registry[tool_name].get("args_schema", {})
        
        # Check required arguments
        missing = []
        for arg_name, arg_info in schema.items():
            if arg_info.get("required", False) and arg_name not in args:
                missing.append(arg_name)
        
        if missing:
            return f"Missing required argument(s): {', '.join(missing)}"
        
        return None
    
    def _update_agent_history(self, state: AgentState, parsed: Dict[str, Any]) -> AgentHistory:
        """Update agent history with reasoning state from parsed LLM output.
        
        Extracts hypothesis, ruled_out, finding, and open_question from parsed output
        and updates the agent's reasoning history.
        
        Args:
            state: Current agent state
            parsed: Parsed LLM output containing optional reasoning fields
            
        Returns:
            Updated AgentHistory instance
        """
        # Get existing history or create new one
        history = state.get("agent_history")
        if history is None:
            history = AgentHistory()
        
        # Update hypothesis if provided
        hypothesis = parsed.get("hypothesis")
        if hypothesis and isinstance(hypothesis, dict):
            choice = hypothesis.get("choice")
            confidence = hypothesis.get("confidence", 0.5)
            evidence = hypothesis.get("evidence", [])
            
            if choice is not None:
                # Ensure evidence is a list
                if isinstance(evidence, str):
                    evidence = [evidence]
                # Ensure confidence is float between 0 and 1
                try:
                    confidence = float(confidence)
                    confidence = max(0.0, min(1.0, confidence))
                except (ValueError, TypeError):
                    confidence = 0.5
                
                # Handle choice being a list (e.g., [0] instead of 0)
                if isinstance(choice, list):
                    choice = choice[0] if choice else 0
                history.add_hypothesis(int(choice), confidence, evidence)
        
        # Update ruled out if provided
        ruled_out = parsed.get("ruled_out")
        if ruled_out and isinstance(ruled_out, dict):
            choice = ruled_out.get("choice")
            reason = ruled_out.get("reason", "")
            
            if choice is not None:
                # Handle choice being a list (e.g., [0] instead of 0)
                if isinstance(choice, list):
                    choice = choice[0] if choice else 0
                history.rule_out(int(choice), reason)
        
        # Add finding if provided
        finding = parsed.get("finding")
        if finding and isinstance(finding, str):
            history.add_finding(finding)
        
        # Add open question if provided
        open_question = parsed.get("open_question")
        if open_question and isinstance(open_question, str):
            history.add_open_question(open_question)
        
        return history
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state graph."""
        graph = StateGraph(AgentState)
        
        # Add nodes
        graph.add_node("agent", self._agent_node)
        graph.add_node("tools", self._tool_node)
        graph.add_node("force_answer", self._force_answer_node)
        
        # Set entry point
        graph.set_entry_point("agent")
        
        # Add conditional edges from agent
        graph.add_conditional_edges(
            "agent",
            self._should_continue,
            {
                "continue": "tools",      # Has pending tool to execute
                "retry": "agent",         # Needs to retry (errors, invalid input)
                "end": END,
                "force": "force_answer",
            }
        )
        
        # Tool -> Agent
        graph.add_edge("tools", "agent")
        
        # Force answer -> End
        graph.add_edge("force_answer", END)
        
        return graph.compile()
    
    def _extract_json_for_history(self, output: str) -> str:
        """Extract only the JSON block from LLM output for conversation history.
        
        This prevents verbose reasoning text from accumulating in conversation history,
        while preserving the actual decisions (tool calls, answers).
        
        Args:
            output: Full LLM response including reasoning and JSON
            
        Returns:
            Only the JSON portion (with markdown fence), or original if not found
        """
        import re
        # Match JSON block in markdown code fence
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', output)
        if json_match:
            return f"```json\n{json_match.group(1).strip()}\n```"
        # Fallback: try to find raw JSON object/array
        json_match = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', output)
        if json_match:
            return json_match.group(1)
        # Last resort: return original
        return output
    
    def _agent_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Agent node: Call LLM to decide next action.
        
        Returns updated state with new message and parsed action.
        """
        start_time = datetime.now()
        
        # Build messages for LLM
        messages = self._build_llm_messages(state)
        
        # Append pending feedback if exists (one-time error/warning, not accumulated)
        pending_feedback = state.get("_pending_feedback")
        if pending_feedback:
            messages.append({"role": "user", "content": pending_feedback})
        
        # Call LLM
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.1,
                max_tokens=8192,  # Increased from 4096 to prevent truncation
            )
            
            output = response.choices[0].message.content
            
        except Exception as e:
            self.logger.error(f"LLM call failed: {e}")
            # Return error state
            return {
                "messages": [{
                    "role": "assistant",
                    "content": f"LLM Error: {str(e)}",
                }],
                "parse_error_count": state["parse_error_count"] + 1,
            }
        
        duration_ms = (datetime.now() - start_time).total_seconds() * 1000
        
        # Parse LLM output
        parsed = self._parse_llm_output(output)
        
        # Log interaction to main logger (truncated)
        log_llm_interaction(
            logger=self.logger,
            model=self.model,
            input_messages=messages,
            output=output,
            is_tool_call=parsed.get("action") == "tool_call",
            tool_name=parsed.get("tool_name"),
            tool_args=parsed.get("tool_args"),
            answer=parsed.get("answer"),
            duration_ms=duration_ms,
        )
        
        # Log full interaction to per-video logger
        if hasattr(self, '_current_video_logger') and self._current_video_logger:
            # Get current memory state (frame captions)
            memory_state = None
            if state.get("video_context"):
                memory_state = state["video_context"].get_caption_summary()
            
            log_llm_interaction_full(
                logger=self._current_video_logger,
                model=self.model,
                input_messages=messages,
                output=output,
                is_tool_call=parsed.get("action") == "tool_call",
                tool_name=parsed.get("tool_name"),
                tool_args=parsed.get("tool_args"),
                answer=parsed.get("answer"),
                duration_ms=duration_ms,
                step_number=state.get("tool_call_count", 0) + 1,
                memory_state=memory_state,
            )
        
        # Log agent action to simple logger (with full LLM output)
        if hasattr(self, '_current_simple_logger') and self._current_simple_logger:
            is_retry = state.get("parse_error_count", 0) > 0
            log_agent_action_simple(
                logger=self._current_simple_logger,
                action=parsed.get("action", "unknown"),
                llm_output=output,  # Full LLM response (not truncated)
                tool_name=parsed.get("tool_name"),
                tool_args=parsed.get("tool_args"),
                tool_calls=parsed.get("tool_calls"),  # For multiple tool calls
                answer=parsed.get("answer"),
                explanation=parsed.get("explanation"),
                step_number=state.get("tool_call_count", 0) + 1,
                duration_ms=duration_ms,
                error_reason=parsed.get("error") if parsed.get("action") == "error" else None,
                is_retry=is_retry,
            )
        
        # Update agent history with reasoning state from parsed output
        updated_history = self._update_agent_history(state, parsed)
        
        # Handle parsed result
        if parsed.get("action") in ("tool_call", "tool_calls"):
            # Get list of tool calls (handle both single and multiple)
            if parsed.get("action") == "tool_calls":
                tool_calls = parsed.get("tool_calls", [])
            else:
                tool_calls = [{
                    "tool_name": parsed["tool_name"],
                    "tool_args": parsed.get("tool_args", {}),
                }]
            
            # Limit to max_parallel_tools
            if len(tool_calls) > self.max_parallel_tools:
                self.logger.info(f"Limiting tool calls from {len(tool_calls)} to {self.max_parallel_tools}")
                tool_calls = tool_calls[:self.max_parallel_tools]
            
            # Validate all tool calls
            validated_tools = []
            for tc in tool_calls:
                tool_name = tc["tool_name"]
                tool_args = tc.get("tool_args", {})
                
                # Check if tool is enabled
                if tool_name not in self.enabled_tools:
                    error_msg = build_tool_error_prompt(
                        "InvalidTool",
                        f"Tool '{tool_name}' is not available. Enabled tools: {self.enabled_tools}"
                    )
                    return {
                        "messages": [{"role": "assistant", "content": self._extract_json_for_history(output)}],
                        "_pending_feedback": error_msg,  # Will be sent on next LLM call
                        "agent_history": updated_history,
                        "tool_call_rounds": state.get("tool_call_rounds", 0) + 1,
                    }
                
                # Validate arguments
                validation_error = self._validate_tool_args(tool_name, tool_args)
                if validation_error:
                    error_msg = build_tool_error_prompt("InvalidArguments", f"[{tool_name}] {validation_error}")
                    return {
                        "messages": [{"role": "assistant", "content": self._extract_json_for_history(output)}],
                        "_pending_feedback": error_msg,  # Will be sent on next LLM call
                        "agent_history": updated_history,
                        "tool_call_rounds": state.get("tool_call_rounds", 0) + 1,
                    }
                
                validated_tools.append({
                    "tool_name": tool_name,
                    "tool_args": tool_args,
                })
            
            # Store pending tool calls in state
            return {
                "messages": [{"role": "assistant", "content": self._extract_json_for_history(output)}],
                "_pending_tools": validated_tools,
                "_pending_feedback": None,  # Clear any previous feedback
                "agent_history": updated_history,
                "tool_call_rounds": state.get("tool_call_rounds", 0) + 1,
            }
        
        elif parsed.get("action") == "submit_answer":
            answer = parsed.get("answer")
            explanation = parsed.get("explanation", "")
            
            # Validate answer based on actual number of choices
            num_choices = len(state.get("choices", []))
            max_answer = num_choices - 1 if num_choices > 0 else 4
            
            if answer is None or not isinstance(answer, int) or answer < 0 or answer > max_answer:
                error_msg = build_tool_error_prompt(
                    "InvalidAnswer",
                    f"Answer must be an integer from 0 to {max_answer}. Got: {answer}"
                )
                return {
                    "messages": [{"role": "assistant", "content": self._extract_json_for_history(output)}],
                    "_pending_feedback": error_msg,  # Will be sent on next LLM call
                    "agent_history": updated_history,
                    "tool_call_rounds": state.get("tool_call_rounds", 0) + 1,
                }
            
            # Check if agent is submitting answer without having used ANY tools
            # Only warn once - if already warned, allow submission
            tool_call_count = state.get("tool_call_count", 0)
            warned_no_tool = state.get("warned_no_tool_submit", False)
            
            if tool_call_count == 0 and not warned_no_tool:
                # First time submitting without tools - warn the agent
                self.logger.info("Agent attempting to submit answer without using any tools - sending warning")
                return {
                    "messages": [{"role": "assistant", "content": self._extract_json_for_history(output)}],
                    "_pending_feedback": NO_TOOL_SUBMIT_WARNING,  # Will be sent on next LLM call
                    "agent_history": updated_history,
                    "tool_call_rounds": state.get("tool_call_rounds", 0) + 1,
                    "warned_no_tool_submit": True,  # Mark as warned
                }
            
            return {
                "messages": [{"role": "assistant", "content": self._extract_json_for_history(output)}],
                "final_answer": answer,
                "explanation": explanation,
                "is_complete": True,
                "_pending_feedback": None,  # Clear any previous feedback
                "agent_history": updated_history,
                "tool_call_rounds": state.get("tool_call_rounds", 0) + 1,
            }
        
        else:
            # Parse error
            self.logger.warning(f"Parse error: {parsed.get('error', 'Unknown')}")
            return {
                "messages": [{"role": "assistant", "content": self._extract_json_for_history(output)}],
                "_pending_feedback": PARSE_ERROR_PROMPT,  # Will be sent on next LLM call
                "parse_error_count": state["parse_error_count"] + 1,
                "agent_history": updated_history,
                "tool_call_rounds": state.get("tool_call_rounds", 0) + 1,
            }
    
    def _tool_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Tool node: Execute all pending tool calls.
        Supports multiple parallel tool calls.
        
        NEW (12.9): Routes results to VideoMemory and/or ToolHistoryEntry
        - Frame-content tools: Update VideoMemory, add minimal history entry
        - Q&A tools: Add full Q&A to tool_history_qa
        """
        pending_tools = state.get("_pending_tools") or []
        if not pending_tools:
            return {
                "messages": [{"role": "user", "content": "No tool call to execute."}],
            }
        
        # Execute all pending tools
        all_results = []
        all_records = []
        all_images = []
        all_history_qa = []  # NEW: Q&A format history entries
        total_duration_ms = 0
        
        sampling_count = 0
        qa_count = 0
        detection_count = 0
        
        step_base = state.get("tool_call_count", 0) + 1
        
        # Get current video memory (will be updated)
        video_memory = state.get("video_memory")
        
        for i, pending in enumerate(pending_tools):
            tool_name = pending["tool_name"]
            tool_args = pending["tool_args"]
            
            start_time = datetime.now()
            
            # Execute tool
            result = self.tool_manager.execute(
                tool_name=tool_name,
                tool_args=tool_args,
                video_context=state["video_context"],
            )
            
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            total_duration_ms += duration_ms
            
            # Log tool call to main logger
            log_tool_call(
                logger=self.logger,
                tool_name=tool_name,
                tool_args=tool_args,
                result=result.get("result", ""),
                success=result.get("success", False),
                error=result.get("error"),
                duration_ms=duration_ms,
            )
            
            # Log full tool call to per-video logger
            if hasattr(self, '_current_video_logger') and self._current_video_logger:
                log_tool_call_full(
                    logger=self._current_video_logger,
                    tool_name=tool_name,
                    tool_args=tool_args,
                    result=result.get("result", ""),
                    success=result.get("success", False),
                    error=result.get("error"),
                    duration_ms=duration_ms,
                    step_number=step_base,
                    tool_index=i + 1,
                    total_tools=len(pending_tools),
                )
            
            # Log tool call to simple logger
            if hasattr(self, '_current_simple_logger') and self._current_simple_logger:
                log_tool_call_simple(
                    logger=self._current_simple_logger,
                    tool_name=tool_name,
                    tool_args=tool_args,
                    result=result.get("result", ""),
                    success=result.get("success", False),
                    error=result.get("error"),
                    duration_ms=duration_ms,
                    step_number=step_base,
                    tool_index=i + 1,
                    total_tools=len(pending_tools),
                )
            
            # Create tool call record (for backward compatibility)
            tool_call_record = ToolCall(
                tool_name=tool_name,
                tool_args=tool_args,
                result=result.get("result", ""),
                timestamp=datetime.now().isoformat(),
                success=result.get("success", False),
                error=result.get("error"),
            )
            all_records.append(tool_call_record)
            
            # Collect result for message (backward compatibility)
            all_results.append(f"[{tool_name}]:\n{result.get('result', 'No result')}")
            
            # Collect images
            if result.get("has_images") and result.get("images"):
                all_images.extend(result["images"])
            
            # ===== NEW: Route results to VideoMemory and/or ToolHistoryEntry =====
            frame_refs = []
            
            if is_frame_content_tool(tool_name):
                # Frame-content tools: update VideoMemory, minimal history entry
                frame_refs = self._update_video_memory_from_tool(
                    video_memory=video_memory,
                    tool_name=tool_name,
                    tool_args=tool_args,
                    result=result,
                )
                
                # Create minimal history entry (frames only, content in memory)
                query = self._format_tool_query(tool_name, tool_args)
                history_entry = ToolHistoryEntry(
                    tool_name=tool_name,
                    query=query,
                    answer=f"Sampled {len(frame_refs)} frames",
                    frame_refs=frame_refs,
                    success=result.get("success", True),
                )
                all_history_qa.append(history_entry)
            else:
                # Q&A tools: full Q&A in history
                query = self._format_tool_query(tool_name, tool_args)
                answer = result.get("result", "No result")
                
                history_entry = ToolHistoryEntry(
                    tool_name=tool_name,
                    query=query,
                    answer=answer,
                    frame_refs=[],
                    success=result.get("success", True),
                )
                all_history_qa.append(history_entry)
            
            # Track tool type
            tool_type = get_tool_type(tool_name)
            if tool_type == "sampling":
                sampling_count += 1
            elif tool_type == "qa":
                qa_count += 1
            elif tool_type == "detection":
                detection_count += 1
        
        # Build combined tool result message (for backward compatibility, still in messages)
        if len(all_results) == 1:
            first_tool_name = pending_tools[0]['tool_name']
            first_result_content = all_results[0].split(':\n', 1)[1] if ':\n' in all_results[0] else all_results[0]
            tool_result_msg = f"Tool Result ({first_tool_name}):\n{first_result_content}"
        else:
            tool_result_msg = f"Tool Results ({len(all_results)} tools executed):\n\n" + "\n\n".join(all_results)
        
        updates = {
            "messages": [{"role": "user", "content": tool_result_msg}],
            "tool_history": state["tool_history"] + all_records,
            "tool_history_qa": state.get("tool_history_qa", []) + all_history_qa,  # NEW
            "tool_call_count": state["tool_call_count"] + len(pending_tools),
            "_pending_tools": None,
        }
        
        # Update video_memory if it was modified
        if video_memory is not None:
            updates["video_memory"] = video_memory
        
        # Handle images - accumulate in visual memory
        MAX_ACCUMULATED_IMAGES = 16
        
        if all_images:
            existing_images = state.get("_pending_images") or []
            existing_indices = {img["frame_index"] for img in existing_images}
            
            for img in all_images:
                if img["frame_index"] not in existing_indices:
                    existing_images.append(img)
                    existing_indices.add(img["frame_index"])
            
            if len(existing_images) > MAX_ACCUMULATED_IMAGES:
                existing_images.sort(key=lambda x: x["timestamp"])
                existing_images = existing_images[-MAX_ACCUMULATED_IMAGES:]
                self.logger.info(f"Visual memory exceeded limit, keeping {MAX_ACCUMULATED_IMAGES} most recent frames")
            
            updates["_pending_images"] = existing_images
            self.logger.info(f"Agent visual memory: {len(existing_images)} frames (added {len(all_images)} new)")
        
        # Increment tool type counters
        if sampling_count > 0:
            updates["sampling_tool_count"] = state.get("sampling_tool_count", 0) + sampling_count
        if qa_count > 0:
            updates["qa_tool_count"] = state.get("qa_tool_count", 0) + qa_count
        if detection_count > 0:
            updates["detection_tool_count"] = state.get("detection_tool_count", 0) + detection_count
        
        return updates
    
    def _format_tool_query(self, tool_name: str, tool_args: Dict[str, Any]) -> str:
        """Format tool arguments into a human-readable query string."""
        # Helper to format optional frame range
        def _frame_range_suffix(args: Dict[str, Any]) -> str:
            start = args.get("start_frame")
            end = args.get("end_frame")
            if start is not None or end is not None:
                start_str = str(start) if start is not None else "0"
                end_str = str(end) if end is not None else "end"
                return f" [frames {start_str}-{end_str}]"
            return ""
        
        if tool_name == "internvideo_general_qa":
            query = tool_args.get("query", str(tool_args))
            return f"{query}{_frame_range_suffix(tool_args)}"
        elif tool_name == "general_vqa":
            return tool_args.get("query", str(tool_args))
        elif tool_name == "internvideo_description":
            start = tool_args.get("start_frame", 0)
            end = tool_args.get("end_frame", "?")
            return f"Describe video segment from frame {start} to {end}"
        elif tool_name in ("temporal_sample_frames", "temporal_spatial_sample_frames"):
            query = tool_args.get("query", "")
            num_frames = tool_args.get("num_frames", 5)
            return f"{query} (sample {num_frames} frames){_frame_range_suffix(tool_args)}"
        elif tool_name == "view_frame":
            indices = tool_args.get("frame_indices", [])
            if not indices:
                idx = tool_args.get("frame_index")
                if idx is not None:
                    indices = [idx]
            return f"View frames: {indices}"
        elif tool_name == "detailed_captioning":
            indices = tool_args.get("frame_indices", [])
            return f"Generate detailed captions for frames: {indices}"
        elif tool_name in ("detect_objects", "detect_all_objects"):
            categories = tool_args.get("categories", [])
            indices = tool_args.get("frame_indices", [])
            return f"Detect {categories} in frames {indices}"
        else:
            return str(tool_args)
    
    def _update_video_memory_from_tool(
        self, 
        video_memory: VideoMemory,
        tool_name: str, 
        tool_args: Dict[str, Any], 
        result: Dict[str, Any]
    ) -> List[int]:
        """Update VideoMemory with frames from frame-content tools.
        
        Returns:
            List of frame IDs that were added/updated
        """
        if video_memory is None:
            return []
        
        frame_refs = []
        result_text = result.get("result", "")
        
        if tool_name in ("temporal_sample_frames", "temporal_spatial_sample_frames"):
            # Parse frame captions from result
            # Format: "[Frame X @ Y.Zs]: caption text"
            frame_pattern = r'\[Frame (\d+) @ ([\d.]+)s\]:\s*(.+?)(?=\[Frame|\Z)'
            matches = re.findall(frame_pattern, result_text, re.DOTALL)
            
            for frame_id_str, timestamp_str, caption in matches:
                frame_id = int(frame_id_str)
                timestamp = float(timestamp_str)
                caption = caption.strip()
                
                video_memory.add_frame(
                    frame_id=frame_id,
                    timestamp=timestamp,
                    caption=caption,
                    source=tool_name,
                )
                frame_refs.append(frame_id)
        
        elif tool_name == "view_frame":
            # Extract frame indices from args
            indices = tool_args.get("frame_indices", [])
            if not indices:
                idx = tool_args.get("frame_index")
                if idx is not None:
                    indices = [idx]
            
            for frame_id in indices:
                video_memory.add_frame(
                    frame_id=frame_id,
                    viewed=True,
                    source="view_frame",
                )
                frame_refs.append(frame_id)
        
        elif tool_name == "detailed_captioning":
            # Parse detailed captions from result
            indices = tool_args.get("frame_indices", [])
            
            # Try to parse individual frame captions
            # Format may vary, but typically "[Frame X @ Y.Zs]: detailed caption"
            frame_pattern = r'\[Frame (\d+) @ ([\d.]+)s\]:\s*(.+?)(?=\[Frame|\Z)'
            matches = re.findall(frame_pattern, result_text, re.DOTALL)
            
            if matches:
                for frame_id_str, timestamp_str, caption in matches:
                    frame_id = int(frame_id_str)
                    timestamp = float(timestamp_str)
                    caption = caption.strip()
                    
                    video_memory.add_frame(
                        frame_id=frame_id,
                        timestamp=timestamp,
                        detailed_caption=caption,
                        source="detailed_captioning",
                    )
                    frame_refs.append(frame_id)
            else:
                # If no pattern match, just mark frames as having detailed caption
                # No truncation - store full result
                for frame_id in indices:
                    video_memory.add_frame(
                        frame_id=frame_id,
                        detailed_caption=result_text,
                        source="detailed_captioning",
                    )
                    frame_refs.append(frame_id)
        
        return frame_refs
    
    def _force_answer_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Force answer node: Force the agent to provide an answer.
        """
        self.logger.info("Forcing answer due to max tool calls or parse errors")
        
        force_msg = build_force_prompt(state["max_tool_calls"])
        
        # Get number of choices for validation
        num_choices = len(state.get("choices", []))
        max_answer = num_choices - 1 if num_choices > 0 else 4
        
        # Build messages
        messages = self._build_llm_messages(state)
        messages.append({"role": "user", "content": force_msg})
        
        # Call LLM for forced answer
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.1,
                max_tokens=2048,  # Increased from 1024 for complex reasoning
            )
            
            output = response.choices[0].message.content
            parsed = self._parse_llm_output(output)
            
            if parsed.get("action") == "submit_answer":
                answer = parsed.get("answer")
                if answer is not None and isinstance(answer, int) and 0 <= answer <= max_answer:
                    return {
                        "messages": [
                            {"role": "user", "content": force_msg},
                            {"role": "assistant", "content": self._extract_json_for_history(output)},
                        ],
                        "final_answer": answer,
                        "explanation": parsed.get("explanation", "Forced answer"),
                        "is_forced": True,
                        "is_complete": True,
                    }
            
            # If still invalid, try to extract any number
            answer = self._extract_any_answer(output, max_answer)
            if answer is not None:
                return {
                    "messages": [
                        {"role": "user", "content": force_msg},
                        {"role": "assistant", "content": self._extract_json_for_history(output)},
                    ],
                    "final_answer": answer,
                    "explanation": "Forced answer (extracted)",
                    "is_forced": True,
                    "is_complete": True,
                }
            
        except Exception as e:
            self.logger.error(f"Force answer LLM call failed: {e}")
        
        # Last resort: random answer
        import random
        fallback = random.randint(0, max_answer)
        self.logger.warning(f"Using random fallback answer: {fallback}")
        
        return {
            "final_answer": fallback,
            "explanation": "Random fallback due to errors",
            "is_forced": True,
            "is_complete": True,
        }
    
    def _should_continue(self, state: AgentState) -> Literal["continue", "retry", "end", "force"]:
        """
        Determine next step based on state.
        
        Returns:
            - "continue": Go to tools node (pending tool exists)
            - "retry": Go back to agent node (errors, need to re-prompt)
            - "end": Task complete
            - "force": Force answer due to limits
        """
        # Check if complete
        if state.get("is_complete"):
            return "end"
        
        # Check if final answer was set (shouldn't happen here, but safety check)
        if state.get("final_answer") is not None:
            return "end"
        
        # Check if max tool calls reached
        if state.get("tool_call_count", 0) >= state.get("max_tool_calls", 10):
            return "force"
        
        # Check if max parse errors reached
        if state.get("parse_error_count", 0) >= state.get("max_parse_errors", 3):
            return "force"
        
        # Check if pending tool calls - go to tools node
        if state.get("_pending_tools"):
            return "continue"
        
        # No pending tool but not complete - agent needs to retry
        # This handles: parse errors, validation errors, invalid answers
        return "retry"
    
    def _build_llm_messages(self, state: AgentState) -> List[Dict[str, Any]]:
        """Build messages list for LLM call (V2 - fresh prompt each round).
        
        NEW ARCHITECTURE (12.9):
        - Build fresh prompt each round instead of accumulating conversation history
        - Always include: System + Task + Video Memory + Tool History (Q&A)
        - No assistant reasoning history in messages (reduces noise)
        
        Message structure:
        1. System message: Static agent role and tool descriptions
        2. User message: Task + Video Memory + Tool History Q&A + Reasoning State
        
        Supports multimodal messages when _pending_images is set.
        Images are included as base64 data URLs in the user message.
        """
        messages = []
        
        # 1. SYSTEM MESSAGE (static - agent role + tools)
        system_prompt = build_system_prompt(
            enabled_tools=state["enabled_tools"],
            tool_registry=self.tool_manager.get_tool_registry(),
            max_parallel_tools=self.max_parallel_tools,
        )
        messages.append({"role": "system", "content": system_prompt})
        
        # 2. USER MESSAGE (fresh each round - includes all context)
        # Get reasoning history summary if available
        history_summary = ""
        if state.get("agent_history"):
            history_summary = state["agent_history"].get_summary()
        
        # Build task prompt using v2 format (Video Memory + Q&A Tool History)
        video_memory = state.get("video_memory")
        tool_history_qa = state.get("tool_history_qa", [])
        
        task_prompt = build_task_prompt_v2(
            question=state["question"],
            choices=state["choices"],
            video_memory=video_memory,
            tool_history_qa=tool_history_qa,
            reasoning_history=history_summary,
        )
        
        # Check if we have pending images to include
        pending_images = state.get("_pending_images")
        
        if pending_images:
            # Build multimodal content with images
            content = []
            
            # Add task prompt text first
            content.append({
                "type": "text",
                "text": task_prompt
            })
            
            # Add visual memory section with images
            frame_summary_parts = []
            for idx, img_data in enumerate(pending_images):
                frame_idx = img_data.get("frame_index", "?")
                timestamp = img_data.get("timestamp", 0)
                frame_summary_parts.append(f"Image {idx+1}: Frame {frame_idx} @ {timestamp:.1f}s")
            
            visual_memory_header = (
                f"\n---\n"
                f"## Visual Memory ({len(pending_images)} frames viewed)\n"
                f"The following are actual images from frames you have viewed:\n"
                f"{chr(10).join(frame_summary_parts)}\n"
            )
            content.append({
                "type": "text",
                "text": visual_memory_header
            })
            
            # Add images WITH identification labels
            for idx, img_data in enumerate(pending_images):
                frame_idx = img_data.get("frame_index", "?")
                timestamp = img_data.get("timestamp", 0)
                
                # Add text label before each image
                content.append({
                    "type": "text",
                    "text": f"[Image {idx+1}: Frame {frame_idx} @ {timestamp:.1f}s]"
                })
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_data['base64']}",
                        "detail": "high"  # Use high detail for better analysis
                    }
                })
            
            messages.append({"role": "user", "content": content})
        else:
            # Text-only message
            messages.append({"role": "user", "content": task_prompt})
        
        return messages
    
    def _extract_json_object(self, text: str) -> Optional[str]:
        """Extract JSON object from text using balanced brace matching.
        
        NOTE: This only extracts the first object. For multiple objects,
        use _extract_all_json_objects instead.
        """
        objects = self._extract_all_json_objects(text)
        return objects[0] if objects else None
    
    def _extract_all_json_objects(self, text: str) -> List[str]:
        """Extract ALL JSON objects from text using balanced brace matching.
        
        Handles cases like: {...}, {...} or {...}\n{...}
        Returns list of JSON strings.
        """
        objects = []
        pos = 0
        
        while pos < len(text):
            # Find next '{' that looks like an action JSON
            start = -1
            for i in range(pos, len(text)):
                if text[i] == '{':
                    # Check if this looks like the action JSON
                    remaining = text[i:i+50]
                    if '"action"' in remaining or "'action'" in remaining:
                        start = i
                        break
            
            if start == -1:
                # No more action JSON objects found
                break
            
            # Balance braces to find the end
            depth = 0
            in_string = False
            escape = False
            end = start
            
            for i, c in enumerate(text[start:], start):
                if escape:
                    escape = False
                    continue
                if c == '\\':
                    escape = True
                    continue
                if c == '"' and not escape:
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if c == '{':
                    depth += 1
                elif c == '}':
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            
            if depth == 0 and end > start:
                objects.append(text[start:end])
                pos = end
            else:
                # Unbalanced - extract what we have and stop
                if start < len(text):
                    objects.append(text[start:] + '}' * depth)
                break
        
        return objects
    
    def _repair_json(self, json_str: str) -> str:
        """Repair common JSON errors from LLM output."""
        # Remove trailing commas before } or ]
        json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
        
        # Fix unquoted keys (simple cases)
        json_str = re.sub(r'(\{|,)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', json_str)
        
        # Fix single quotes to double quotes (careful with apostrophes)
        # Only do this for key-value patterns
        json_str = re.sub(r"'([^']+)'(\s*:)", r'"\1"\2', json_str)
        json_str = re.sub(r":\s*'([^']*)'", r': "\1"', json_str)
        
        return json_str
    
    def _aggressive_json_repair(self, json_str: str) -> str:
        """Aggressive JSON repair for malformed LLM output.
        
        Handles cases like multiple objects after a key:
        "ruled_out": {"choice": 0}, {"choice": 1}, ...
        -> "ruled_out": [{"choice": 0}, {"choice": 1}, ...]
        
        Also handles invalid syntax like:
        "choice": 0: "reason": "text" -> "choice": 0, "reason": "text"
        "choice": 0, "reason": "text"; "choice": 1 -> with commas
        """
        # Fix semicolons used as separators instead of commas
        # Pattern: "value"; "key": or "value";\n"key":
        # Example: "No food prep/stirring/heating"; "choice": 1
        json_str = re.sub(r'";(\s*)"', r'", \1"', json_str)
        # Also handle: number; "key" -> number, "key"
        json_str = re.sub(r'(\d);(\s*)"', r'\1,\2"', json_str)
        
        # ========== FIX PREMATURE CLOSE BEFORE CONTEXT KEYS (must run early) ==========
        # Pattern: {..., "reason": "..."}, "hypothesis": {...}
        # The closing } after "reason" is premature - context keys should be inside
        # Only fix the FIRST }, "hypothesis": after "reason" (not all }, "context_key":)
        def fix_premature_close_after_reason(json_str: str) -> str:
            """Fix: "reason": "..."}, "hypothesis": -> "reason": "...", "hypothesis":"""
            pattern = r'("reason"\s*:\s*"[^"]*")\s*\}\s*,?\s*("hypothesis"\s*:)'
            json_str = re.sub(pattern, r'\1, \2', json_str)
            return json_str
        
        json_str = fix_premature_close_after_reason(json_str)
        
        # ========== MUST RUN BEFORE OTHER COLON FIXES ==========
        # Pattern A: "choice": N: "reason text" (inside ruled_out object)
        # LLM uses "choice": N as a key with colon separator and reason as value
        # Example: "ruled_out": { "choice": 0: "No painting", "choice": 1: "..." }
        # Should become: "ruled_out": [{"choice": 0, "reason": "No painting"}, ...]
        def fix_choice_colon_pattern(json_str: str) -> str:
            # Pattern: "choice": N: "string" where N is a digit
            # Match "choice": followed by a number, then : then a quoted string
            pattern = r'"choice"\s*:\s*(\d+)\s*:\s*"([^"]*)"'
            matches = list(re.finditer(pattern, json_str))
            if not matches:
                return json_str
            
            # Rebuild string with proper format
            result = []
            last_end = 0
            for m in matches:
                result.append(json_str[last_end:m.start()])
                choice_num = m.group(1)
                reason_text = m.group(2)
                result.append(f'{{"choice": {choice_num}, "reason": "{reason_text}"}}')
                last_end = m.end()
            result.append(json_str[last_end:])
            return ''.join(result)
        
        json_str = fix_choice_colon_pattern(json_str)
        
        # Fix invalid syntax: "key": value: "another_key": -> "key": value, "another_key":
        # Pattern: value followed by colon then quote (should be comma)
        # NOTE: This MUST run AFTER fix_choice_colon_pattern to not break "choice": N: "text" pattern
        # Examples: 
        #   "other": 0: "reason": -> "other": 0, "reason":
        json_str = re.sub(r'(":\s*\d+)\s*:\s*"', r'\1, "', json_str)
        
        # Fix: "key": "value": "another_key": -> "key": "value", "another_key":
        json_str = re.sub(r'(":\s*"[^"]*")\s*:\s*"', r'\1, "', json_str)
        
        # Fix invalid object key syntax inside braces: { "choice": 0: "reason": ...}
        # Pattern: number or closing quote followed by : " (should be , ")
        json_str = re.sub(r'(\d)\s*:\s*"(\w+)":', r'\1, "\2":', json_str)
        
        # Pattern B: Orphaned "choice": N, "reason": "..." at top level after ruled_out
        # Example: "ruled_out": {...}, "choice": 1, "reason": "..."}, "choice": 2, ...
        # The subsequent items are missing opening braces
        def fix_orphaned_choice_reason(json_str: str) -> str:
            # Pattern: }, "choice": N, "reason": "..." }  (missing opening brace)
            # Should be: }, {"choice": N, "reason": "..."}
            pattern = r'\}\s*,\s*"choice"\s*:\s*(\d+)\s*,\s*"reason"\s*:\s*"([^"]*)"\s*\}'
            
            def replace_orphan(m):
                choice = m.group(1)
                reason = m.group(2)
                return f'}}, {{"choice": {choice}, "reason": "{reason}"}}'
            
            # Apply multiple times as there might be chains
            prev = None
            while prev != json_str:
                prev = json_str
                json_str = re.sub(pattern, replace_orphan, json_str)
            return json_str
        
        json_str = fix_orphaned_choice_reason(json_str)
        
        # Pattern B2: Collect orphaned objects into ruled_out array
        # After fix_orphaned_choice_reason, we have: "ruled_out": {...}, {...}, {...}
        # Need to collect these into an array
        def collect_ruled_out_objects(json_str: str) -> str:
            match = re.search(r'"ruled_out"\s*:\s*(\{[^{}]*\})', json_str)
            if not match:
                return json_str
            
            first_obj = match.group(1)
            end_pos = match.end()
            
            objects = [first_obj]
            pos = end_pos
            
            while pos < len(json_str):
                # Skip whitespace and comma
                while pos < len(json_str) and json_str[pos] in ' \t\n\r,':
                    pos += 1
                
                if pos >= len(json_str) or json_str[pos] != '{':
                    break
                
                brace_count = 0
                obj_start = pos
                found_obj = False
                for i in range(pos, len(json_str)):
                    if json_str[i] == '{':
                        brace_count += 1
                    elif json_str[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            obj_end = i + 1
                            obj = json_str[obj_start:obj_end]
                            if '"choice"' in obj:
                                objects.append(obj)
                                pos = obj_end
                                found_obj = True
                            else:
                                pos = obj_start
                            break
                
                if not found_obj:
                    break
            
            if len(objects) > 1:
                array_str = '"ruled_out": [' + ', '.join(objects) + ']'
                # Check if we need to add a comma before the next key
                remaining = json_str[pos:].lstrip()
                if remaining and remaining[0] == '"':
                    array_str += ', '
                json_str = json_str[:match.start()] + array_str + json_str[pos:].lstrip()
            
            return json_str
        
        json_str = collect_ruled_out_objects(json_str)
        
        # Pattern C: Multiple action objects missing opening braces
        # Example: {...}, "action": "tool_call", ...}
        # Should have opening brace: {...}, {"action": "tool_call", ...}
        def fix_orphaned_action(json_str: str) -> str:
            # Pattern: }, "action": "..." (missing opening brace before "action")
            # But be careful not to match properly nested ones
            pattern = r'\}\s*,\s*"action"\s*:\s*"([^"]+)"'
            
            def replace_action(m):
                action_value = m.group(1)
                return f'}}, {{"action": "{action_value}"'
            
            prev = None
            while prev != json_str:
                prev = json_str
                json_str = re.sub(pattern, replace_action, json_str)
            return json_str
        
        json_str = fix_orphaned_action(json_str)
        
        # Pattern C2: Orphaned "tool_name": after }, (missing opening brace)
        # Example: {...}, "tool_name": "general_vqa", "tool_args": {...}
        # Should be: {...}, {"tool_name": "general_vqa", "tool_args": {...}}
        def fix_orphaned_tool_name(json_str: str) -> str:
            pattern = r'\}\s*,\s*"tool_name"\s*:\s*"([^"]+)"'
            
            def replace_tool_name(m):
                tool_value = m.group(1)
                return f'}}, {{"tool_name": "{tool_value}"'
            
            prev = None
            while prev != json_str:
                prev = json_str
                json_str = re.sub(pattern, replace_tool_name, json_str)
            return json_str
        
        json_str = fix_orphaned_tool_name(json_str)
        
        # Pattern C3: Newline-separated JSON objects (no commas between them)
        # Example: {"action": "tool_call", ...}\n{"action": "tool_call", ...}
        # Should be: [{"action": "tool_call", ...}, {"action": "tool_call", ...}]
        def fix_newline_separated_objects(json_str: str) -> str:
            # Pattern: } followed by newline(s) then { (two separate objects)
            # Add comma between them
            json_str = re.sub(r'\}\s*\n\s*\{', '}, {', json_str)
            return json_str
        
        json_str = fix_newline_separated_objects(json_str)
        
        # Pattern D: Convert {"key": {obj1}, {obj2}, ...} to {"key": [obj1, obj2, ...]}
        # This handles nested objects that should be in an array
        def convert_object_values_to_array(json_str: str) -> str:
            for key in ["ruled_out", "hypothesis", "evidence"]:
                start_pattern = rf'("{key}"\s*:\s*)\{{'
                match = re.search(start_pattern, json_str)
                if not match:
                    continue
                
                key_prefix = match.group(1)
                start_pos = match.end() - 1  # Position of opening {
                
                # Check if next non-space char is another {
                check_pos = start_pos + 1
                while check_pos < len(json_str) and json_str[check_pos] in ' \t\n\r':
                    check_pos += 1
                
                if check_pos < len(json_str) and json_str[check_pos] == '{':
                    # Nested braces: "key": { {obj1}, {obj2} }
                    start_pos = check_pos
                
                # Collect all objects
                objects = []
                pos = start_pos
                
                while pos < len(json_str):
                    if json_str[pos] != '{':
                        if json_str[pos] in ' \t\n\r,':
                            pos += 1
                            continue
                        break
                    
                    brace_count = 0
                    obj_start = pos
                    found_end = False
                    for i in range(pos, len(json_str)):
                        if json_str[i] == '{':
                            brace_count += 1
                        elif json_str[i] == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                obj_end = i + 1
                                objects.append(json_str[obj_start:obj_end])
                                pos = obj_end
                                found_end = True
                                break
                    if not found_end:
                        break
                
                if len(objects) > 1:
                    array_str = key_prefix + '[' + ', '.join(objects) + ']'
                    # Skip the extra closing } from the original outer brace if present
                    remaining_start = pos
                    while remaining_start < len(json_str) and json_str[remaining_start] in ' \t\n\r':
                        remaining_start += 1
                    if remaining_start < len(json_str) and json_str[remaining_start] == '}':
                        remaining_start += 1  # Skip the extra }
                    json_str = json_str[:match.start()] + array_str + json_str[remaining_start:]
            
            return json_str
        
        json_str = convert_object_values_to_array(json_str)
        
        # Pattern E: Wrap multiple top-level action objects in array
        # Example: {...}, {...} at top level -> [{...}, {...}]
        def wrap_multiple_actions(json_str: str) -> str:
            stripped = json_str.strip()
            if stripped.startswith('{') and '}, {' in stripped:
                # Try parsing first - if valid, don't modify
                try:
                    json.loads(stripped)
                    return json_str
                except:
                    pass
                # Wrap in array
                return '[' + stripped + ']'
            return json_str
        
        json_str = wrap_multiple_actions(json_str)
        
        # Fix multiple objects after a key (should be an array)
        # Pattern: "key": {...}, {...}  -> "key": [{...}, {...}]
        def fix_multiple_objects(match):
            key = match.group(1)
            objects = match.group(2)
            # Check if already an array
            if objects.strip().startswith('['):
                return match.group(0)
            # Wrap in array
            return f'"{key}": [{objects}]'
        
        # Look for patterns like "ruled_out": {...}, {...}
        json_str = re.sub(
            r'"(ruled_out|hypothesis|evidence)"\s*:\s*(\{[^}]+\}(?:\s*,\s*\{[^}]+\})+)',
            fix_multiple_objects,
            json_str
        )
        
        # Fix duplicate keys: "ruled_out": {...}, "ruled_out": {...} -> "ruled_out": [{...}, {...}]
        # This handles LLM outputting the same key multiple times
        def fix_duplicate_keys(json_str, key_name):
            # Pattern: "key": {...} repeated with comma/whitespace separation
            pattern = rf'"{key_name}"\s*:\s*(\{{[^{{}}]*\}})'
            matches = list(re.finditer(pattern, json_str))
            
            if len(matches) <= 1:
                return json_str
            
            # Collect all values for this key
            values = [m.group(1) for m in matches]
            
            # Replace first occurrence with array of all values
            first_match = matches[0]
            array_value = f'"{key_name}": [{", ".join(values)}]'
            
            # Build result by removing duplicate occurrences
            result = json_str[:first_match.start()] + array_value
            
            # Track position after first match
            last_end = first_match.end()
            
            for m in matches[1:]:
                # Add content between previous match and this one (excluding the duplicate key-value)
                between = json_str[last_end:m.start()]
                # Remove trailing comma before this duplicate
                between = re.sub(r',\s*$', '', between)
                result += between
                last_end = m.end()
                # Skip any comma after the duplicate
                remaining = json_str[last_end:]
                comma_match = re.match(r'\s*,\s*', remaining)
                if comma_match:
                    last_end += comma_match.end()
            
            # Add remaining content
            result += json_str[last_end:]
            
            return result
        
        # Fix common duplicate keys
        for key in ["ruled_out", "hypothesis", "evidence"]:
            json_str = fix_duplicate_keys(json_str, key)
        
        # Ensure the JSON ends properly - balance braces and brackets
        # Count braces
        open_braces = json_str.count('{')
        close_braces = json_str.count('}')
        if open_braces > close_braces:
            # Add missing closing braces
            json_str = json_str + '}' * (open_braces - close_braces)
        elif close_braces > open_braces:
            # Remove extra closing braces from the end
            excess = close_braces - open_braces
            # Only remove from the end of the string
            while excess > 0 and json_str.rstrip().endswith('}'):
                json_str = json_str.rstrip()[:-1]
                excess -= 1
        
        # Balance brackets
        open_brackets = json_str.count('[')
        close_brackets = json_str.count(']')
        if open_brackets > close_brackets:
            # Add missing closing brackets
            json_str = json_str + ']' * (open_brackets - close_brackets)
        elif close_brackets > open_brackets:
            # Remove extra closing brackets from the end
            excess = close_brackets - open_brackets
            while excess > 0 and json_str.rstrip().endswith(']'):
                json_str = json_str.rstrip()[:-1]
                excess -= 1
        
        return json_str
    
    def _fix_truncated_json(self, json_str: str) -> str:
        """Fix truncated JSON by completing unfinished strings and balancing braces."""
        # Check if string is incomplete (ends with unfinished string value)
        # Pattern: last quote not closed
        in_string = False
        escape = False
        
        for c in json_str:
            if escape:
                escape = False
                continue
            if c == '\\':
                escape = True
                continue
            if c == '"':
                in_string = not in_string
        
        # If still in string, close it
        if in_string:
            json_str = json_str.rstrip() + '"'
        
        # Balance braces
        open_braces = json_str.count('{')
        close_braces = json_str.count('}')
        open_brackets = json_str.count('[')
        close_brackets = json_str.count(']')
        
        # Add missing closing brackets/braces
        json_str = json_str + ']' * (open_brackets - close_brackets)
        json_str = json_str + '}' * (open_braces - close_braces)
        
        return json_str
    
    def _fix_escaped_quotes_in_json(self, json_str: str) -> str:
        """Fix improperly escaped quotes in JSON strings.
        
        Handles cases like: "text with \\'s" -> "text with 's"
        (LLM sometimes outputs backslash-quote for apostrophes)
        """
        # Replace \' with ' (this is not valid JSON escape)
        json_str = json_str.replace("\\'", "'")
        return json_str
    
    def _parse_llm_output(self, output: str) -> Dict[str, Any]:
        """
        Parse LLM output to extract action(s) and reasoning state updates.
        Supports multiple tool calls (each in separate ```json blocks).
        
        Returns:
            Dict with 'action' (tool_calls/submit_answer/error) and relevant fields
            - For multiple tools: action='tool_calls', tool_calls=[{tool_name, tool_args}, ...]
            - For single tool: action='tool_call', tool_name, tool_args
            - For answer: action='submit_answer', answer, explanation
        """
        # Find ALL json blocks in the output
        json_matches = re.findall(r'```json\s*(.*?)\s*```', output, re.DOTALL)
        
        if not json_matches:
            # Try to find bare JSON objects (handles multiple: {...}, {...})
            json_objects = self._extract_all_json_objects(output)
            if json_objects:
                json_matches = json_objects
            else:
                return {"action": "error", "error": "Failed to parse any JSON from output"}
        
        # Parse all JSON blocks
        parsed_actions = []
        reasoning_updates = {}
        
        for json_str in json_matches:
            json_str = json_str.strip()
            
            # Apply fixes for common LLM issues
            json_str = self._fix_escaped_quotes_in_json(json_str)
            
            # First, try to parse as-is (handles arrays and objects)
            repaired_str = self._repair_json(json_str)
            try:
                data = json.loads(repaired_str)
                # If it's a list (array of tool calls), extend parsed_actions
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            parsed_actions.append(item)
                elif isinstance(data, dict):
                    parsed_actions.append(data)
                continue  # Successfully parsed, move to next block
            except json.JSONDecodeError:
                pass
            
            # Try with truncation fix
            try:
                fixed_str = self._fix_truncated_json(repaired_str)
                data = json.loads(fixed_str)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            parsed_actions.append(item)
                elif isinstance(data, dict):
                    parsed_actions.append(data)
                continue
            except json.JSONDecodeError:
                pass
            
            # Try aggressive repair
            try:
                repaired = self._aggressive_json_repair(json_str)
                data = json.loads(repaired)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            parsed_actions.append(item)
                elif isinstance(data, dict):
                    parsed_actions.append(data)
                continue
            except json.JSONDecodeError:
                pass
            
            # Last resort: truncation fix on aggressive repair
            try:
                repaired = self._aggressive_json_repair(json_str)
                fixed_str = self._fix_truncated_json(repaired)
                data = json.loads(fixed_str)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            parsed_actions.append(item)
                elif isinstance(data, dict):
                    parsed_actions.append(data)
                continue
            except json.JSONDecodeError:
                pass
            
            # Fallback: Handle case where multiple objects are in one block: {...},{...}
            # Split by },{ pattern (without outer array brackets)
            if re.search(r'\}\s*,?\s*\{', json_str):
                parts = re.split(r'\}\s*,?\s*\{', json_str)
                for i, part in enumerate(parts):
                    if i == 0:
                        part = part.lstrip('[').strip() + '}'
                    elif i == len(parts) - 1:
                        part = '{' + part.rstrip(']').strip()
                    else:
                        part = '{' + part + '}'
                    
                    part = self._repair_json(part)
                    try:
                        data = json.loads(part)
                        if isinstance(data, dict):
                            parsed_actions.append(data)
                    except json.JSONDecodeError:
                        try:
                            repaired = self._aggressive_json_repair(part)
                            data = json.loads(repaired)
                            if isinstance(data, dict):
                                parsed_actions.append(data)
                        except json.JSONDecodeError:
                            continue
        
        if not parsed_actions:
            return {"action": "error", "error": "Failed to parse any JSON from output"}
        
        # Extract reasoning updates from first action (they apply globally)
        first_data = parsed_actions[0]
        reasoning_updates = {
            "hypothesis": first_data.get("hypothesis"),
            "ruled_out": first_data.get("ruled_out"),
            "finding": first_data.get("finding"),
            "open_question": first_data.get("open_question"),
        }
        
        # Collect all tool calls FIRST (they take priority over submit_answer)
        # This prevents LLM from "hallucinating" tool results and skipping actual execution
        tool_calls = []
        
        # Helper to extract tool call from a dict
        def extract_tool_call(tc_data, parent_data=None):
            tool_args = tc_data.get("tool_args", {})
            # Handle misplaced arguments (check both tc_data and parent if provided)
            arg_names = ["query", "frame_indices", "categories", "num_frames", 
                         "frame_index", "timestamp", "bbox", "target_object",
                         "detail_level", "max_detections"]
            for arg_name in arg_names:
                if arg_name in tc_data and arg_name not in tool_args:
                    tool_args[arg_name] = tc_data[arg_name]
                elif parent_data and arg_name in parent_data and arg_name not in tool_args:
                    tool_args[arg_name] = parent_data[arg_name]
            return {
                "tool_name": tc_data.get("tool_name"),
                "tool_args": tool_args,
                "reasoning": tc_data.get("reasoning", tc_data.get("reason", "")),
            }
        
        for data in parsed_actions:
            action = data.get("action", "")
            
            # Handle action="tool_calls" with nested tool_calls array
            if action == "tool_calls" and "tool_calls" in data:
                for tc in data.get("tool_calls", []):
                    if isinstance(tc, dict) and tc.get("tool_name"):
                        tool_calls.append(extract_tool_call(tc, data))
            
            # Handle action="tool_call" (singular)
            elif action == "tool_call":
                tool_calls.append(extract_tool_call(data))
            
            # Handle objects without explicit action but with tool_name (implicit tool_call)
            elif not action and data.get("tool_name"):
                tool_calls.append(extract_tool_call(data))
        
        # If there are tool calls, execute them (even if submit_answer also appears)
        # This prevents LLM from simulating tool results in its output
        if tool_calls:
            # Check if LLM also tried to submit_answer (hallucination pattern)
            has_submit = any(d.get("action") == "submit_answer" for d in parsed_actions)
            if has_submit:
                # Log warning: LLM tried to hallucinate tool results and answer in one turn
                import logging
                logging.getLogger(__name__).warning(
                    f"LLM output contains both tool_calls ({len(tool_calls)}) and submit_answer. "
                    "Ignoring submit_answer and executing tool_calls first."
                )
            # Fall through to return tool_calls below
        else:
            # No tool calls - check for submit_answer
            for data in parsed_actions:
                if data.get("action") == "submit_answer":
                    answer = data.get("answer")
                    if answer is not None:
                        try:
                            answer = int(answer)
                        except (ValueError, TypeError):
                            pass
                    return {
                        "action": "submit_answer",
                        "answer": answer,
                        "explanation": data.get("explanation", ""),
                        **reasoning_updates,
                    }
            
            # No tool_calls and no submit_answer
            return {"action": "error", "error": f"Unknown action in parsed data"}
        
        # Return based on number of tool calls
        if len(tool_calls) == 1:
            return {
                "action": "tool_call",
                "tool_name": tool_calls[0]["tool_name"],
                "tool_args": tool_calls[0]["tool_args"],
                "reasoning": tool_calls[0]["reasoning"],
                **reasoning_updates,
            }
        else:
            return {
                "action": "tool_calls",  # Plural for multiple
                "tool_calls": tool_calls,
                **reasoning_updates,
            }
    
    def _extract_any_answer(self, output: str, max_answer: int = 4) -> Optional[int]:
        """Try to extract any valid answer from output.
        
        Args:
            output: LLM output text to parse
            max_answer: Maximum valid answer index (inclusive)
        
        Returns:
            Extracted answer integer or None if not found
        """
        # Look for patterns like "answer: 2", "choice 3", "option 1"
        patterns = [
            r'answer["\s:]+(\d)',
            r'choice["\s:]+(\d)',
            r'option["\s:]+(\d)',
            r'"answer"\s*:\s*(\d)',
            r'select\s+(\d)',
            r'\b(\d)\b',  # Any single digit
        ]
        
        for pattern in patterns:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                try:
                    answer = int(match.group(1))
                    if 0 <= answer <= max_answer:
                        return answer
                except ValueError:
                    continue
        
        return None
    
    def process_video(
        self,
        video_path: str,
        video_id: str,
        question: str,
        choices: List[str],
        truth: int = None,
        video_output_dir: str = None,
        detailed: bool = True,
    ) -> Dict[str, Any]:
        """
        Process a single video with the agent.
        
        Args:
            video_path: Path to video file
            video_id: Video identifier
            question: Question to answer
            choices: List of multiple choice options
            truth: Ground truth answer (optional, for evaluation)
            video_output_dir: Directory for per-video outputs (logging.log, frames/)
            detailed: If True, use full logging to main logger. If False, minimal output.
        
        Returns:
            Dict with results including final_answer, explanation, tool_history, etc.
        """
        start_time = datetime.now()
        
        # Setup per-video logger if output directory provided
        video_logger = None
        simple_logger = None
        if video_output_dir:
            # video_logger writes to logging.log (detailed format, no console)
            video_logger = setup_video_logger(
                video_id=video_id,
                video_output_dir=video_output_dir,
                parent_logger=None,  # Don't propagate to main logger
            )
            # simple_logger writes to simple_log.log and optionally console
            # Console output matches simple_log format when detailed=True
            simple_logger = setup_simple_logger(
                video_id=video_id,
                video_output_dir=video_output_dir,
                console_output=detailed,  # Print to console if detailed mode
            )
        
        # Use video_logger for detailed per-video logging, main logger for summary
        active_logger = video_logger if video_logger else self.logger
        
        # Log start
        log_video_start(
            logger=self.logger,
            video_id=video_id,
            question=question,
            choices=choices,
            initial_frames=self.initial_frames,
        )
        
        # Also log to per-video logger with full details - use clean format
        if video_logger:
            lines = []
            lines.append("=" * 80)
            lines.append("VIDEO AGENT SESSION")
            lines.append("=" * 80)
            lines.append(f"Video ID: {video_id}")
            lines.append(f"Timestamp: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            lines.append("")
            lines.append("QUESTION:")
            lines.append(question)
            lines.append("")
            lines.append("CHOICES:")
            for i, choice in enumerate(choices):
                lines.append(f"  {i}. {choice}")
            lines.append("")
            lines.append(f"Initial frames to sample: {self.initial_frames}")
            lines.append(f"Max tool calls: {self.max_tool_calls}")
            lines.append(f"Model: {self.model}")
            lines.append("=" * 80)
            video_logger.info("\n".join(lines))
        
        # Log session start to simple logger
        if simple_logger:
            log_session_start_simple(
                logger=simple_logger,
                video_id=video_id,
                question=question,
                choices=choices,
                initial_frames=self.initial_frames,
                model=self.model,
                start_time=start_time,
            )
        
        # Initialize tool manager if needed
        if self.tool_manager is None:
            self.tool_manager = ToolManager(
                enabled_tools=self.enabled_tools,
                captioner=self.captioner,
                logger=self.logger,
                # Frame control parameters
                max_view_frames=self.max_view_frames,
                default_sample_frames=self.default_sample_frames,
                min_sample_frames=self.min_sample_frames,
                max_sample_frames=self.max_sample_frames,
            )
            self.tool_manager.initialize()
        
        # Load video context
        try:
            video_context = load_video_context(video_path, video_id)
        except Exception as e:
            self.logger.error(f"Failed to load video: {e}")
            return self._create_error_result(video_id, question, truth, str(e))
        
        # Sample initial frame indices
        initial_indices = sample_uniform_indices(
            video_context.total_frames,
            self.initial_frames,
        )
        
        # Try to load cached captions using ToolCache (model-aware caching)
        cache_hit = False
        initial_captions = {}
        caption_duration = 0.0
        
        # Check if tool_manager has cache (ToolClient in multiprocessing mode doesn't have _tool_cache)
        tool_cache = getattr(self.tool_manager, '_tool_cache', None)
        get_model_key = getattr(self.tool_manager, '_get_captioner_model_key', None)
        if tool_cache and get_model_key:
            model_key = get_model_key()
            cached_captions = tool_cache.get_captions_batch(
                video_path=video_path,
                frame_indices=initial_indices,
                model=model_key,
                detail_level="short",
            )
            
            # Check if all needed frames are cached
            all_cached = all(cached_captions.get(idx) is not None for idx in initial_indices)
            
            if all_cached:
                cache_hit = True
                initial_captions = {idx: cached_captions[idx] for idx in initial_indices}
                
                # Update video context with cached captions
                for idx, caption in initial_captions.items():
                    video_context.frame_captions[idx] = caption
                    if idx not in video_context.sampled_indices:
                        video_context.sampled_indices.append(idx)
                
                # Extract and save frames for output (even when captions are cached)
                extracted_frames = extract_frames(video_path, initial_indices)
                for idx, frame in extracted_frames.items():
                    video_context.frames[idx] = frame
                
                self.logger.info(f"Using cached captions for frames: {initial_indices} (model: {model_key})")
                if video_logger:
                    video_logger.info(f"Using cached captions for frames: {initial_indices} (cache hit, model: {model_key})")
        
        if not cache_hit:
            # Generate captions (no cache or incomplete cache)
            self.logger.info(f"Captioning initial frames: {initial_indices}")
            if video_logger:
                video_logger.info(f"Captioning initial frames: {initial_indices}")
            
            caption_start = datetime.now()
            
            initial_captions = self.tool_manager.caption_frames(
                video_context,
                initial_indices,
                detail_level="short",
            )
            
            caption_duration = (datetime.now() - caption_start).total_seconds() * 1000
        
        log_caption_generation(
            logger=self.logger,
            frame_indices=initial_indices,
            captions=initial_captions,
            duration_ms=caption_duration,
        )
        
        # Log full captions to per-video logger with clean format
        if video_logger:
            lines = []
            lines.append("")
            lines.append("#" * 80)
            lines.append("INITIAL FRAME CAPTIONS")
            lines.append("#" * 80)
            lines.append(f"Frames sampled: {initial_indices}")
            lines.append(f"Caption generation time: {caption_duration:.1f}ms")
            lines.append("")
            for idx in sorted(initial_captions.keys()):
                lines.append(f"--- Frame {idx} ---")
                lines.append(initial_captions[idx])
                lines.append("")
            lines.append("#" * 80)
            video_logger.info("\n".join(lines))
        
        # Generate video description using InternVideoDescription
        desc_start = datetime.now()
        try:
            video_description, desc_cache_hit = self._generate_video_description(video_path, video_context)
            video_context.video_description = video_description
            desc_duration = (datetime.now() - desc_start).total_seconds() * 1000
            
            if desc_cache_hit:
                self.logger.info(f"Video description loaded from cache")
            else:
                self.logger.info(f"Video description generated in {desc_duration:.1f}ms")
            
            if video_logger:
                lines = []
                lines.append("")
                lines.append("#" * 80)
                lines.append("VIDEO DESCRIPTION (Overall)")
                lines.append("#" * 80)
                if desc_cache_hit:
                    lines.append("Source: FROM CACHE")
                else:
                    lines.append(f"Description generation time: {desc_duration:.1f}ms")
                lines.append("")
                lines.append(video_description)
                lines.append("#" * 80)
                video_logger.info("\n".join(lines))
            
            if simple_logger:
                if desc_cache_hit:
                    simple_logger.info(f"[Video Description] FROM CACHE")
                else:
                    simple_logger.info(f"[Video Description] Generated in {desc_duration:.1f}ms")
                simple_logger.info(f"Description:\n{video_description}")
                
        except Exception as e:
            self.logger.warning(f"Failed to generate video description: {e}")
            video_context.video_description = ""
        
        # Log initial captions to simple logger
        if simple_logger:
            log_initial_captions_simple(
                logger=simple_logger,
                frame_indices=initial_indices,
                captions=initial_captions,
                fps=video_context.fps,
                duration_ms=caption_duration,
                cache_hit=cache_hit,
            )
        
        # Create initial state
        initial_state = create_initial_state(
            video_id=video_id,
            question=question,
            choices=choices,
            video_context=video_context,
            enabled_tools=self.enabled_tools,
            max_tool_calls=self.max_tool_calls,
            max_parse_errors=self.max_parse_errors,
        )
        
        # Store loggers in instance for use in nodes
        self._current_video_logger = video_logger
        self._current_simple_logger = simple_logger
        
        # Run the graph
        try:
            final_state = self.graph.invoke(initial_state)
        except Exception as e:
            self.logger.error(f"Graph execution error: {e}")
            return self._create_error_result(video_id, question, truth, str(e))
        finally:
            self._current_video_logger = None
            self._current_simple_logger = None
        
        # Extract results
        end_time = datetime.now()
        duration_s = (end_time - start_time).total_seconds()
        
        final_answer = final_state.get("final_answer")
        is_correct = final_answer == truth if truth is not None else None
        
        # Log end
        log_video_end(
            logger=self.logger,
            video_id=video_id,
            final_answer=final_answer if final_answer is not None else -1,
            truth=truth if truth is not None else -1,
            is_correct=is_correct if is_correct is not None else False,
            is_forced=final_state.get("is_forced", False),
            tool_calls=final_state.get("tool_call_count", 0),
            duration_s=duration_s,
        )
        
        # Log end to per-video logger with clean format
        if video_logger:
            status = "CORRECT" if is_correct else "WRONG"
            forced_str = " (FORCED)" if final_state.get("is_forced", False) else ""
            lines = []
            lines.append("")
            lines.append("=" * 80)
            lines.append("SESSION RESULT")
            lines.append("=" * 80)
            lines.append(f"Status: {status}{forced_str}")
            lines.append(f"Final Answer: {final_answer}")
            lines.append(f"Ground Truth: {truth}")
            lines.append(f"Explanation: {final_state.get('explanation', '')}")
            lines.append("")
            lines.append(f"Tool Calls: {final_state.get('tool_call_count', 0)}")
            lines.append(f"Frames Used: {len(video_context.frame_captions)}")
            lines.append(f"Duration: {duration_s:.1f}s")
            lines.append("=" * 80)
            video_logger.info("\n".join(lines))
        
        # Log end to simple logger
        if simple_logger:
            log_session_end_simple(
                logger=simple_logger,
                video_id=video_id,
                final_answer=final_answer if final_answer is not None else -1,
                truth=truth if truth is not None else -1,
                is_correct=is_correct if is_correct is not None else False,
                is_forced=final_state.get("is_forced", False),
                tool_calls=final_state.get("tool_call_count", 0),
                frame_count=len(video_context.frame_captions),
                duration_s=duration_s,
                start_time=start_time,
                end_time=end_time,
            )
        
        # Build result with additional data for per-video saving
        # Serialize agent_history if present
        agent_history_dict = None
        if final_state.get("agent_history"):
            agent_history_dict = final_state["agent_history"].to_dict()
        
        result = {
            "video_id": video_id,
            "question": question,
            "choices": choices,
            "final_answer": final_answer,
            "explanation": final_state.get("explanation", ""),
            "truth": truth,
            "is_correct": is_correct,
            "is_forced": final_state.get("is_forced", False),
            "is_valid": final_answer is not None,
            "tool_call_count": final_state.get("tool_call_count", 0),
            "tool_call_rounds": final_state.get("tool_call_rounds", 0),
            "tool_history": [tc.to_dict() for tc in final_state.get("tool_history", [])],
            "frame_count": len(video_context.frame_captions),
            "duration_s": duration_s,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            # Agent reasoning history
            "agent_history": agent_history_dict,
            # Additional data for per-video result.json
            "video_context_dict": video_context.to_dict(),
            "messages": final_state.get("messages", []),
            "frames": video_context.frames,  # For saving as PNG
            "fps": video_context.fps,  # For frame filename timestamps
        }
        
        return result
    
    def _create_error_result(
        self,
        video_id: str,
        question: str,
        truth: int,
        error: str,
    ) -> Dict[str, Any]:
        """Create error result with random fallback answer."""
        import random
        fallback = random.randint(0, 4)
        
        return {
            "video_id": video_id,
            "question": question,
            "final_answer": fallback,
            "explanation": f"Error: {error}",
            "truth": truth,
            "is_correct": fallback == truth if truth is not None else None,
            "is_forced": True,
            "is_valid": False,
            "error": error,
            "tool_call_count": 0,
            "tool_history": [],
            "frame_count": 0,
            # Additional fields for per-video saving
            "video_context_dict": {},
            "messages": [],
            "frames": {},
        }

