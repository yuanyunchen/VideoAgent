"""
Tool Client for Worker Processes (Multiprocessing Mode)

Lightweight client that workers use to access tools via the ToolServer.
Provides the same interface as ToolManager but routes execution through IPC.
"""

import uuid
import queue
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from multiprocessing import Queue

from .core import MessageType, ToolRequest, ToolResponse


# Per-tool timeout configuration (seconds)
# Longer timeouts for heavy-compute tools like TStar and InternVideo
TOOL_TIMEOUTS = {
    "temporal_sample_frames": 600.0,
    "temporal_spatial_sample_frames": 600.0,
    "internvideo_general_qa": 600.0,
    "internvideo_description": 600.0,
    "targeting_vqa": 450.0,
    "general_vqa": 300.0,
    "detect_objects": 300.0,
    "caption_image": 300.0,
    "detailed_captioning": 300.0,
    "default": 300.0,
}


class ToolClient:
    """
    Client for accessing tools via ToolServer.
    
    Workers use this instead of ToolManager when running in multiprocessing mode.
    Provides the same interface but routes all tool execution through the
    centralized ToolServer via IPC queues.
    """
    
    def __init__(
        self,
        worker_id: int,
        request_queue: Queue,
        response_queue: Queue,
        enabled_tools: List[str],
        logger: logging.Logger = None,
        timeout: float = 300.0,
        max_view_frames: int = 8,
        default_sample_frames: int = 16,
        min_sample_frames: int = 1,
        max_sample_frames: int = 32,
    ):
        """
        Initialize ToolClient.
        
        Args:
            worker_id: Unique ID for this worker
            request_queue: Queue to send requests to ToolServer
            response_queue: Queue to receive responses from ToolServer
            enabled_tools: List of enabled tool names
            logger: Logger instance
            timeout: Timeout for waiting for responses (seconds)
        """
        self.worker_id = worker_id
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.enabled_tools = enabled_tools
        self.logger = logger or logging.getLogger(f"ToolClient.{worker_id}")
        self.timeout = timeout
        
        self.max_view_frames = max_view_frames
        self.default_sample_frames = default_sample_frames
        self.min_sample_frames = min_sample_frames
        self.max_sample_frames = max_sample_frames
        
        self._initialized = False
    
    def initialize(self):
        """Initialize the client."""
        if self._initialized:
            return
        
        self.logger.info(f"ToolClient {self.worker_id} initialized with {len(self.enabled_tools)} enabled tools")
        self._initialized = True
    
    def get_tool_registry(self) -> Dict[str, Dict[str, Any]]:
        """Get registry of enabled tools."""
        from tools.interface import get_tool_registry
        
        base_registry = get_tool_registry(self.enabled_tools)
        
        tools_needing_frame_indices = {
            "caption_image", "detailed_captioning", "detect_objects", "detect_all_objects"
        }
        
        # Tools that support optional frame range parameters
        tools_with_frame_range = {
            "internvideo_general_qa", "temporal_sample_frames", "temporal_spatial_sample_frames"
        }
        
        for tool_name, tool_info in base_registry.items():
            if tool_name in tools_needing_frame_indices:
                tool_info["args_schema"]["frame_indices"] = {
                    "type": "List[int]",
                    "required": True,
                    "description": "Frame indices to analyze (0-based)"
                }
            
            # Update sampling tools with runtime-configured frame limits
            if tool_name in {"temporal_sample_frames", "temporal_spatial_sample_frames"}:
                if "num_frames" in tool_info["args_schema"]:
                    tool_info["args_schema"]["num_frames"]["default"] = self.default_sample_frames
                    tool_info["args_schema"]["num_frames"]["description"] = (
                        f"Number of frames to sample (default: {self.default_sample_frames}, "
                        f"range: {self.min_sample_frames}-{self.max_sample_frames})"
                    )
            
            # Ensure frame range parameters are present with proper descriptions
            if tool_name in tools_with_frame_range:
                if "start_frame" not in tool_info["args_schema"]:
                    tool_info["args_schema"]["start_frame"] = {
                        "type": "int",
                        "required": False,
                        "description": "Start frame index (0-based, optional). Default: 0 (video start)"
                    }
                if "end_frame" not in tool_info["args_schema"]:
                    tool_info["args_schema"]["end_frame"] = {
                        "type": "int",
                        "required": False,
                        "description": "End frame index (0-based, inclusive, optional). Default: last frame"
                    }
        
        return base_registry
    
    def execute(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        video_context: Any,
        max_retries: int = 1,
    ) -> Dict[str, Any]:
        """
        Execute a tool via the ToolServer with retry logic.
        
        Args:
            tool_name: Name of the tool to execute
            tool_args: Arguments for the tool
            video_context: VideoContext object with video data
            max_retries: Maximum number of retries on timeout (default: 1)
        
        Returns:
            Dict with 'success', 'result', and tool-specific data
        """
        if not self._initialized:
            self.initialize()
        
        if tool_name not in self.enabled_tools:
            return {
                "success": False,
                "result": f"Tool '{tool_name}' is not enabled. Available: {self.enabled_tools}",
            }
        
        last_result = None
        retryable_errors = {"timeout", "OOM"}  # Errors that should trigger retry
        
        for attempt in range(max_retries + 1):
            result = self._execute_once(tool_name, tool_args, video_context)
            last_result = result
            
            error_type = result.get("error", "")
            
            # If successful or not a retryable error, return immediately
            if result.get("success") or error_type not in retryable_errors:
                return result
            
            # On retryable error, log and retry
            if attempt < max_retries:
                if error_type == "OOM":
                    # OOM: wait longer to allow memory cleanup and eviction
                    import time
                    self.logger.warning(
                        f"Tool '{tool_name}' OOM, waiting 3s then retry {attempt + 1}/{max_retries}"
                    )
                    time.sleep(3.0)  # Wait for eviction and memory cleanup
                else:
                    # Timeout: retry immediately
                    self.logger.warning(
                        f"Tool '{tool_name}' timeout, retry {attempt + 1}/{max_retries}"
                    )
        
        # All retries exhausted
        return last_result
    
    def _execute_once(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        video_context: Any,
    ) -> Dict[str, Any]:
        """Execute a tool once (internal method)."""
        start_time = datetime.now()
        
        # Serialize video context
        video_context_data = {
            "video_path": video_context.video_path,
            "video_id": video_context.video_id,
            "total_frames": video_context.total_frames,
            "fps": video_context.fps,
            "duration": video_context.duration,
            "sampled_indices": list(video_context.sampled_indices),
        }
        
        request_id = str(uuid.uuid4())
        request = ToolRequest(
            msg_type=MessageType.EXECUTE_TOOL,
            request_id=request_id,
            worker_id=self.worker_id,
            tool_name=tool_name,
            tool_args=tool_args,
            video_context_data=video_context_data,
        )
        
        try:
            self.request_queue.put(request)
            self.logger.debug(f"Sent request {request_id} for tool '{tool_name}'")
        except Exception as e:
            return {
                "success": False,
                "result": f"Failed to send request: {e}",
                "error": str(e),
            }
        
        # Use per-tool timeout, fallback to default or instance timeout
        tool_timeout = TOOL_TIMEOUTS.get(tool_name, TOOL_TIMEOUTS.get("default", self.timeout))
        # Use the larger of tool-specific and instance timeout
        effective_timeout = max(tool_timeout, self.timeout)
        
        try:
            response: ToolResponse = self.response_queue.get(timeout=effective_timeout)
            
            if response.request_id != request_id:
                self.logger.warning(f"Response ID mismatch")
            
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            
            result = response.result if response.success else {
                "success": False,
                "result": response.error or "Tool execution failed",
                "error": response.error,
            }
            
            # Update video context
            if response.success:
                new_sampled = result.get("new_sampled_indices", [])
                for idx in new_sampled:
                    if idx not in video_context.sampled_indices:
                        video_context.sampled_indices.append(idx)
                
                captions = result.get("captions", {})
                for idx, caption in captions.items():
                    video_context.frame_captions[idx] = caption
            
            result["duration_ms"] = duration_ms
            return result
            
        except queue.Empty:
            return {
                "success": False,
                "result": f"Timeout waiting for response (>{effective_timeout}s)",
                "error": "timeout",
            }
        except Exception as e:
            return {
                "success": False,
                "result": f"Error receiving response: {e}",
                "error": str(e),
            }
    
    def caption_frames(
        self,
        video_context: Any,
        frame_indices: List[int],
        detail_level: str = "short",
        save_frames: bool = True,
    ) -> Dict[int, str]:
        """
        Caption multiple frames (used for initial frame captioning).
        
        Routes to view_frame tool on the ToolServer.
        """
        if not self._initialized:
            self.initialize()
        
        result = self.execute(
            tool_name="view_frame",
            tool_args={
                "frame_indices": frame_indices,
                "detail_level": detail_level,
            },
            video_context=video_context,
        )
        
        captions = result.get("captions", {})
        
        for idx, caption in captions.items():
            video_context.frame_captions[idx] = caption
            if idx not in video_context.sampled_indices:
                video_context.sampled_indices.append(idx)
        
        return captions
    
    def get_server_status(self) -> Dict[str, Any]:
        """Get status from the ToolServer."""
        request_id = str(uuid.uuid4())
        request = ToolRequest(
            msg_type=MessageType.GET_STATUS,
            request_id=request_id,
            worker_id=self.worker_id,
        )
        
        try:
            self.request_queue.put(request)
            response: ToolResponse = self.response_queue.get(timeout=10.0)
            return response.result if response.success else {"error": response.error}
        except Exception as e:
            return {"error": str(e)}
    
    def cleanup(self):
        """Cleanup (no-op for client)."""
        self.logger.info(f"ToolClient {self.worker_id} cleanup complete")




