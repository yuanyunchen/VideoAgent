"""
Tool Wrappers for VideoAgent

Provides ToolManager that adapts tools/interface/ to LangGraph agent.
Handles:
- VideoContext adaptation (frame extraction, index validation)
- Auto-captioning for frame-returning tools
- Tool instance lifecycle management
- Error handling with retry logic for GPU OOM and transient errors
- Multi-GPU resource management with intelligent tool distribution

Note: Tool descriptions and validation are defined in tools/interface/.
This module only handles VideoContext-specific adaptation.
"""

import os
import sys
import gc
import time
import traceback
import signal
import threading
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from contextlib import contextmanager
import logging

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.interface import (
    INTERFACE_MAPPING,
    FRAME_RETURNING_TOOLS,
    get_interface,
    get_tool_registry,
    validate_tool_args,
    OmniCaptionerCaptioning,
    APICaptioning,
    ViewFrame,
    YOLOWorldDetection,
    YOLOEPromptFreeDetection,
    DAMDescription,
    VideoTreeSampling,
    TStarSampling,
    VideoRAGTemporalQA,
    TStarTemporalSpatialQA,
    GeneralVQA,
    TargetingVQA,
    InternVideoGeneralQA,
    InternVideoDescription,
    Image,
    Video,
    TSTAR_FINAL_SAMPLE_FRAMES,
)

from video_agent_tools.resource_management import (
    GPUResourceManager,
    ToolPriority,
)
from video_agent_tools.resource_management.gpu_manager import get_gpu_resource_manager
from video_agent_tools.utils.tool_cache import ToolCache, get_tool_cache

# Re-export for backward compatibility
TOOL_REGISTRY = None  # Lazy loaded via get_tool_registry()


class ToolManager:
    """
    Manages tool instances and execution.
    
    Handles:
    - Lazy initialization of tool models
    - Tool execution with proper context
    - Auto-captioning for frame-returning tools
    - Error handling with retry logic for GPU OOM and transient errors
    """
    
    # Error patterns that indicate GPU memory issues (retryable after cleanup)
    GPU_OOM_PATTERNS = [
        "out of memory",
        "CUDA out of memory",
        "OutOfMemoryError",
        "CUDA error: out of memory",
        "cuda runtime error",
        "cuDNN error",
        "allocate",
        "memory allocation",
    ]
    
    # Error patterns for transient/retryable errors
    TRANSIENT_ERROR_PATTERNS = [
        "connection reset",
        "connection refused",
        "timeout",
        "rate limit",
        "service unavailable",
        "internal server error",
        "502",
        "503",
        "504",
    ]
    
    # Watchdog timeout (seconds) for heavy model initialization
    HEAVY_INIT_TIMEOUT_SEC = 180
    
    def __init__(
        self,
        enabled_tools: List[str],
        captioner: str = "gpt-4o-mini",
        logger: logging.Logger = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        retry_backoff: float = 2.0,
        # Frame control parameters
        max_view_frames: int = 8,
        default_sample_frames: int = 16,
        min_sample_frames: int = 1,
        max_sample_frames: int = 32,
        # Cache parameters
        enable_tool_cache: bool = True,
        tool_cache_dir: str = None,
    ):
        """
        Initialize ToolManager.
        
        Args:
            enabled_tools: List of tool keys to enable
            captioner: Captioner model - 'omni-captioner' for local OmniCaptioner,
                      or API model name (e.g., 'gpt-4o-mini', 'x-ai/grok-4-1-fast-reasoning')
            logger: Logger instance
            max_retries: Maximum number of retry attempts for failed tool calls
            retry_delay: Initial delay between retries (seconds)
            retry_backoff: Multiplier for delay on each retry (exponential backoff)
            max_view_frames: Maximum frames for view_frame tool
            default_sample_frames: Default number of frames for sampling tools
            min_sample_frames: Minimum number of frames for sampling tools
            max_sample_frames: Maximum number of frames for sampling tools
            enable_tool_cache: Enable caching for caption and internvideo_description tools
            tool_cache_dir: Directory for tool cache (defaults to cache/tool_cache)
        """
        self.enabled_tools = enabled_tools
        
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
        
        # Retry configuration
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.retry_backoff = retry_backoff
        self.heavy_init_timeout = self.HEAVY_INIT_TIMEOUT_SEC
        
        # Frame control parameters
        self.max_view_frames = max_view_frames
        self.default_sample_frames = default_sample_frames
        self.min_sample_frames = min_sample_frames
        self.max_sample_frames = max_sample_frames
        
        # Tool instances (lazy initialized)
        self._tool_instances: Dict[str, Any] = {}
        self._captioner: Optional[OmniCaptionerCaptioning] = None
        self._api_captioner: Optional[APICaptioning] = None
        self._initialized = False
        
        # GPU Resource Manager for multi-GPU support
        self._gpu_manager: Optional[GPUResourceManager] = None
        
        # Tool cache for caption and internvideo_description
        self._tool_cache: Optional[ToolCache] = None
        self._enable_tool_cache = enable_tool_cache
        self._tool_cache_dir = tool_cache_dir
    
    def get_tool_registry(self) -> Dict[str, Dict[str, Any]]:
        """Get registry of enabled tools with VideoContext-specific parameters.
        
        Extends the base tool registry with frame_indices parameter for tools
        that operate on specific frames.
        """
        base_registry = get_tool_registry(self.enabled_tools)
        
        # Add frame_indices parameter for tools that need it
        # These tools operate on specific frames selected by the agent
        tools_needing_frame_indices = {
            "caption_image", "detailed_captioning", "detect_objects", "detect_all_objects"
        }
        
        # Tools that support optional frame range parameters
        tools_with_frame_range = {
            "internvideo_general_qa", "temporal_sample_frames", "temporal_spatial_sample_frames"
        }
        
        for tool_name, tool_info in base_registry.items():
            # Add frame_indices for tools that need VideoContext frame selection
            # (These tools' interfaces work with images, but agent provides frame indices)
            if tool_name in tools_needing_frame_indices:
                tool_info["args_schema"]["frame_indices"] = {
                    "type": "List[int]",
                    "required": True,
                    "description": "Frame indices to analyze (0-based)"
                }
            
            # NOTE: view_frame schema is defined in its interface (ViewFrame.AGENT_INPUT_SCHEMA)
            # No special handling needed here
            
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
    
    def initialize(self):
        """Initialize core tool instances (captioner only, others are lazy)."""
        if self._initialized:
            return
        
        self.logger.info("Initializing tool instances...")
        
        # Initialize GPU Resource Manager for multi-GPU support
        self._gpu_manager = get_gpu_resource_manager(logger=self.logger)
        self._gpu_manager.log_status()
        
        # Initialize tool cache for caption and internvideo_description
        if self._enable_tool_cache:
            self._tool_cache = get_tool_cache(
                cache_dir=self._tool_cache_dir,
                enabled=True,
                logger=self.logger,
            )
            self.logger.info("Tool cache enabled")
        
        # Only initialize captioner at startup (needed for auto-caption pipeline)
        # Other tools are initialized lazily when first used
        self._init_captioner()
        
        self._initialized = True
        self.logger.info("Core tools initialized (others lazy-loaded on demand)")
    
    def _init_captioner(self):
        """Initialize the OmniCaptioner for auto-caption pipeline with error recovery."""
        if self._captioner is not None:
            return
        
        if self.use_api_captioner:
            # Use API-based captioning as primary (memory efficient)
            self.logger.info(f"Using API captioner with model: {self.api_captioner_model}")
            self._init_api_captioner()
            # Alias for primary captioner
            self._captioner = self._api_captioner
        else:
            # Use local OmniCaptioner model with GPU resource management
            target_device = None
            if self._gpu_manager:
                target_device = self._gpu_manager.select_device_for_tool("omni_captioner")
                self.logger.info(f"Selected device {target_device} for OmniCaptioner")
            
            # Retry logic with error recovery for GPU OOM
            max_retries = 3
            last_error = None
            
            for attempt in range(max_retries):
                try:
                    # Clear GPU memory before initializing
                    if attempt == 0:
                        self._clear_gpu_memory(aggressive=False)
                        self.logger.info("Cleared GPU memory before initializing OmniCaptioner")
                    else:
                        # On retry, aggressively free memory by unloading unused tools
                        self.logger.info(f"Retry {attempt + 1}/{max_retries}: Freeing memory for OmniCaptioner")
                        if self._gpu_manager:
                            # Unload unused tools to free memory
                            self._gpu_manager.free_memory_for_tool("omni_captioner", target_device)
                        self._clear_gpu_memory(aggressive=True, for_tool="omni_captioner")
                    
                    self.logger.info("Loading OmniCaptioner model...")
                    self._captioner = OmniCaptionerCaptioning(
                        model_path=self.captioner_model,
                        device=target_device
                    )
                    self._captioner.initialize()
                    
                    # Register with GPU manager for tracking
                    if self._gpu_manager and target_device:
                        self._gpu_manager.register_tool(
                            tool_name="omni_captioner",
                            instance=self._captioner,
                            device=target_device,
                            priority=ToolPriority.CRITICAL,  # Never evict captioner
                        )
                    
                    self.logger.info("OmniCaptioner loaded successfully")
                    return  # Success, exit
                    
                except Exception as e:
                    last_error = e
                    error_msg = str(e)
                    
                    # Check for CUDA OOM errors
                    if "CUDA" in error_msg or "cuda" in error_msg or "GPU" in error_msg or "memory" in error_msg.lower() or "OutOfMemory" in error_msg:
                        self.logger.warning(
                            f"OmniCaptioner GPU/memory error on attempt {attempt + 1}/{max_retries}: {e}"
                        )
                        # Clean up failed instance
                        if self._captioner is not None:
                            try:
                                del self._captioner
                            except:
                                pass
                            self._captioner = None
                        
                        # If not last attempt, wait before retry
                        if attempt < max_retries - 1:
                            time.sleep(2.0)  # Wait longer for memory to be freed
                    else:
                        # Non-memory errors: don't retry
                        self.logger.error(f"OmniCaptioner initialization failed: {e}")
                        raise
            
            # All retries failed
            self.logger.error(f"Failed to initialize OmniCaptioner after {max_retries} attempts: {last_error}")
            raise RuntimeError(f"Failed to initialize OmniCaptioner: {last_error}")
    
    def _init_api_captioner(self):
        """Initialize the API captioner."""
        if self._api_captioner is not None:
            return
        
        self.logger.info(f"Initializing API captioner with model: {self.api_captioner_model}")
        self._api_captioner = APICaptioning(
            use_aiml=True,
            default_model=self.api_captioner_model
        )
        self._api_captioner.initialize()
        self.logger.info("API captioner initialized")
    
    def _get_captioner_model_key(self) -> str:
        """Get the model key for caching based on current captioner configuration.
        
        Returns:
            Model key string for cache lookups:
            - For API captioner: "api_{model_name}"
            - For local OmniCaptioner: model path
        """
        if self.use_api_captioner:
            return f"api_{self.api_captioner_model}"
        else:
            return self.captioner_model
    
    def _is_gpu_oom_error(self, error: Exception) -> bool:
        """Check if the error is a GPU out-of-memory error.
        
        Args:
            error: The exception to check
            
        Returns:
            True if this is a GPU OOM error
        """
        error_str = str(error).lower()
        return any(pattern.lower() in error_str for pattern in self.GPU_OOM_PATTERNS)
    
    def _is_transient_error(self, error: Exception) -> bool:
        """Check if the error is a transient/retryable error.
        
        Args:
            error: The exception to check
            
        Returns:
            True if this is a transient error that may succeed on retry
        """
        error_str = str(error).lower()
        return any(pattern.lower() in error_str for pattern in self.TRANSIENT_ERROR_PATTERNS)
    
    def _clear_gpu_memory(self, aggressive: bool = False, for_tool: str = None):
        """Clear GPU memory cache to recover from OOM errors.
        
        Args:
            aggressive: If True, also evict unused tools to free more memory
            for_tool: If provided, try to free enough memory for this specific tool
        """
        try:
            import torch
            if torch.cuda.is_available():
                # Clear CUDA cache
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                # Force garbage collection
                gc.collect()
                
                # Log memory status
                for i in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(i) / 1024**3
                    reserved = torch.cuda.memory_reserved(i) / 1024**3
                    self.logger.info(
                        f"GPU {i} memory after cleanup: "
                        f"allocated={allocated:.2f}GB, reserved={reserved:.2f}GB"
                    )
                
                # Aggressive mode: use GPU resource manager to evict tools
                if aggressive and self._gpu_manager and for_tool:
                    self.logger.info(f"Attempting to free memory for tool '{for_tool}'")
                    if self._gpu_manager.free_memory_for_tool(for_tool):
                        self.logger.info(f"Successfully freed memory for '{for_tool}'")
                    else:
                        self.logger.warning(f"Could not free enough memory for '{for_tool}'")
                        
        except Exception as e:
            self.logger.warning(f"Failed to clear GPU memory: {e}")
    
    def _log_gpu_status_brief(self, reason: str):
        """Log a lightweight GPU status snapshot for telemetry around heavy loads."""
        if not self._gpu_manager:
            return
        try:
            self._gpu_manager.log_status_brief(prefix=reason)
        except Exception as e:
            self.logger.debug(f"Failed to log GPU status for '{reason}': {e}")
    
    @contextmanager
    def _init_timeout_guard(self, timeout_sec: Optional[int], description: str):
        """
        Soft timeout guard for heavy tool initialization.
        
        Uses signal.alarm in the main thread to raise TimeoutError if init hangs.
        No-op outside the main thread or when timeout_sec is falsy.
        """
        if not timeout_sec or timeout_sec <= 0:
            yield
            return
        if threading.current_thread() is not threading.main_thread():
            # signal.alarm only works in main thread; skip elsewhere
            yield
            return
        
        def _timeout_handler(signum, frame):
            raise TimeoutError(f"Timeout after {timeout_sec}s during {description}")
        
        previous_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout_sec)
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous_handler)
    
    def _execute_with_retry(
        self,
        func: Callable,
        tool_name: str,
        *args,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute a function with retry logic for GPU OOM and transient errors.
        
        Uses the GPU Resource Manager for intelligent OOM recovery:
        - First retry: clear GPU cache only
        - Second retry: aggressively evict unused tools
        - Third retry: evict more tools if needed
        
        Args:
            func: The function to execute
            tool_name: Name of the tool (for logging)
            *args: Arguments to pass to func
            **kwargs: Keyword arguments to pass to func
            
        Returns:
            Result dict from the function, or error dict if all retries failed
        """
        last_error = None
        last_traceback = None
        
        for attempt in range(self.max_retries + 1):
            try:
                result = func(*args, **kwargs)
                
                # If we had previous failures but succeeded now, log it
                if attempt > 0:
                    self.logger.info(
                        f"Tool '{tool_name}' succeeded on attempt {attempt + 1}/{self.max_retries + 1}"
                    )
                
                return result
                
            except Exception as e:
                last_error = e
                last_traceback = traceback.format_exc()
                
                # Check if this is a retryable error
                is_oom = self._is_gpu_oom_error(e)
                is_transient = self._is_transient_error(e)
                
                if is_oom:
                    self.logger.warning(
                        f"Tool '{tool_name}' GPU OOM error on attempt {attempt + 1}/{self.max_retries + 1}: {e}"
                    )
                    
                    # Progressive memory recovery strategy
                    if attempt == 0:
                        # First retry: just clear cache
                        self._clear_gpu_memory(aggressive=False)
                    else:
                        # Subsequent retries: aggressively evict tools
                        self._clear_gpu_memory(aggressive=True, for_tool=tool_name)
                        
                elif is_transient:
                    self.logger.warning(
                        f"Tool '{tool_name}' transient error on attempt {attempt + 1}/{self.max_retries + 1}: {e}"
                    )
                else:
                    # Non-retryable error, fail immediately
                    self.logger.error(
                        f"Tool '{tool_name}' non-retryable error: {e}\n{last_traceback}"
                    )
                    break
                
                # Check if we have retries left
                if attempt < self.max_retries:
                    # Calculate delay with exponential backoff
                    delay = self.retry_delay * (self.retry_backoff ** attempt)
                    self.logger.info(
                        f"Retrying '{tool_name}' in {delay:.1f}s (attempt {attempt + 2}/{self.max_retries + 1})"
                    )
                    time.sleep(delay)
                else:
                    self.logger.error(
                        f"Tool '{tool_name}' failed after {self.max_retries + 1} attempts"
                    )
        
        # All retries exhausted or non-retryable error
        error_type = "GPU OOM" if self._is_gpu_oom_error(last_error) else "execution"
        return {
            "success": False,
            "result": f"Tool '{tool_name}' {error_type} error after {self.max_retries + 1} attempts: {str(last_error)}",
            "error": str(last_error),
            "error_type": error_type,
            "attempts": self.max_retries + 1,
            "traceback": last_traceback,
        }
    
    def _validate_frame_indices(
        self,
        frame_indices: List[int],
        video_context: Any,
    ) -> tuple:
        """Validate frame indices against video context.
        
        Args:
            frame_indices: List of frame indices to validate
            video_context: VideoContext object
        
        Returns:
            Tuple of (valid_indices, error_result)
            - If valid: (list of valid indices, None)
            - If invalid: ([], error dict for return)
        """
        if not frame_indices:
            return [], {"success": False, "result": "No frame_indices provided"}
        
        valid_indices = [i for i in frame_indices if 0 <= i < video_context.total_frames]
        if not valid_indices:
            return [], {
                "success": False,
                "result": f"Invalid frame indices. Valid range: 0-{video_context.total_frames - 1}",
            }
        
        return valid_indices, None
    
    def _init_tool(self, tool_key: str, force_retry: bool = False):
        """Initialize a specific tool with improved error handling and retry logic.
        
        Uses GPU Resource Manager for device selection and intelligent memory management.
        
        Args:
            tool_key: The tool identifier to initialize
            force_retry: If True, retry even if previously failed (instance is None)
        """
        # Check if already successfully initialized
        if tool_key in self._tool_instances:
            if self._tool_instances[tool_key] is not None:
                # Update usage stats in GPU manager
                if self._gpu_manager:
                    self._gpu_manager.get_tool(tool_key)
                return  # Already initialized successfully
            elif not force_retry:
                return  # Previously failed and not forcing retry
            # force_retry=True: Remove failed entry and try again
            del self._tool_instances[tool_key]
        
        if tool_key == "caption_image":
            # Use the shared captioner
            self._tool_instances[tool_key] = self._captioner
            return
        
        # For heavy tools, proactively free memory before attempting load
        heavy_tools = {
            "detect_objects", "detect_all_objects", "temporal_qa", "temporal_spatial_qa",
            "internvideo_general_qa", "internvideo_description", "targeting_vqa"
        }
        
        if tool_key in heavy_tools and self._gpu_manager:
            # Proactively free memory for this tool before even trying to load
            self.logger.info(f"Proactively freeing memory for heavy tool '{tool_key}'")
            self._gpu_manager.free_memory_for_tool(tool_key)
            self._clear_gpu_memory(aggressive=False)
            self._log_gpu_status_brief(f"[{tool_key}] after proactive free")
        
        # Select device using GPU Resource Manager
        target_device = None
        if self._gpu_manager:
            target_device = self._gpu_manager.select_device_for_tool(tool_key)
            self.logger.info(f"Selected device {target_device} for tool '{tool_key}'")
            if tool_key in heavy_tools:
                self._log_gpu_status_brief(f"[{tool_key}] before init on {target_device}")
        
        # Retry logic for GPU-intensive tools
        heavy_tools = {
            "detect_objects", "detect_all_objects", "temporal_qa", "temporal_spatial_qa",
            "internvideo_general_qa", "internvideo_description", "targeting_vqa"
        }
        max_retries = 3 if tool_key in heavy_tools else 1
        last_error = None
        
        for attempt in range(max_retries):
            try:
                # Clear GPU memory before initializing heavy tools
                if tool_key in heavy_tools:
                    if attempt == 0:
                        self._clear_gpu_memory(aggressive=False)
                        self.logger.info(f"Cleared GPU memory before initializing {tool_key}")
                    else:
                        # On retry, aggressively free memory
                        self._clear_gpu_memory(aggressive=True, for_tool=tool_key)
                        self.logger.info(f"Retry {attempt + 1}/{max_retries}: Aggressively cleared GPU memory for {tool_key}")
                
                interface_cls = get_interface(tool_key)
                
                # Create instance with device parameter for tools that support it
                # These tools have their own GPU models that need explicit device placement
                tools_with_device_param = {
                    # InternVideo tools (~16GB)
                    "internvideo_general_qa",
                    "internvideo_description",
                    # Detection tools
                    "detect_objects",        # YOLOWorld
                    "detect_all_objects",    # YOLOE
                    # Sampling tools
                    "temporal_sample_frames",        # VideoTree (CLIP)
                    "temporal_spatial_sample_frames", # TStar
                    # Description tools
                    "describe_region",       # DAM
                }
                
                if tool_key in tools_with_device_param and target_device:
                    instance = interface_cls(device=target_device)
                else:
                    instance = interface_cls()
                
                with self._init_timeout_guard(self.heavy_init_timeout if tool_key in heavy_tools else None, f"initializing {tool_key}"):
                    instance.initialize()
                
                if tool_key in heavy_tools:
                    self._log_gpu_status_brief(f"[{tool_key}] after init on {target_device}")
                self._tool_instances[tool_key] = instance
                
                # Register with GPU Resource Manager
                if self._gpu_manager and target_device:
                    self._gpu_manager.register_tool(
                        tool_name=tool_key,
                        instance=instance,
                        device=target_device,
                    )
                
                self.logger.info(f"Successfully initialized tool: {tool_key} on {target_device or 'default device'}")
                return  # Success, exit
                
            except NotImplementedError as e:
                # Some tools may not be fully implemented - don't retry
                self.logger.warning(f"Tool {tool_key} not fully implemented: {e}")
                self._tool_instances[tool_key] = None
                return
                
            except ImportError as e:
                # Missing dependencies - don't retry
                self.logger.error(f"Tool {tool_key} missing dependencies: {e}")
                self._tool_instances[tool_key] = None
                return
                
            except Exception as e:
                last_error = e
                error_msg = str(e)
                
                # Clean up failed instance if it was created
                if 'instance' in locals() and instance is not None:
                    try:
                        # Try to clean up the failed instance
                        if hasattr(instance, 'cleanup'):
                            instance.cleanup()
                        elif hasattr(instance, 'release'):
                            instance.release()
                        del instance
                    except:
                        pass
                    instance = None
                
                # Check for specific error types and provide helpful messages
                is_timeout = isinstance(e, TimeoutError)
                is_oom_error = (
                    "CUDA" in error_msg or "cuda" in error_msg or 
                    "GPU" in error_msg or "memory" in error_msg.lower() or
                    "OutOfMemory" in error_msg
                )
                
                if is_timeout:
                    self.logger.error(
                        f"Tool {tool_key} initialization timed out after {self.heavy_init_timeout}s: {e}"
                    )
                    self._clear_gpu_memory(aggressive=True, for_tool=tool_key)
                    if self._gpu_manager:
                        self._log_gpu_status_brief(f"[{tool_key}] after timeout cleanup attempt")
                    break  # Don't keep retrying if initialization hung
                elif is_oom_error:
                    self.logger.warning(
                        f"Tool {tool_key} GPU/memory error on attempt {attempt + 1}/{max_retries}: {e}"
                    )
                    # On OOM, try to free memory using GPU manager and unload unused tools
                    if self._gpu_manager and attempt < max_retries - 1:
                        self.logger.info(f"Attempting to free memory for {tool_key} by unloading unused tools")
                        # Unload unused tools to free memory
                        self._gpu_manager.free_memory_for_tool(tool_key, target_device)
                        # Aggressively clear GPU memory
                        self._clear_gpu_memory(aggressive=True, for_tool=tool_key)
                        self._log_gpu_status_brief(f"[{tool_key}] after OOM recovery")
                    time.sleep(2.0)  # Wait longer for memory to be freed
                elif "model" in error_msg.lower() or "weight" in error_msg.lower():
                    self.logger.error(
                        f"Tool {tool_key} failed to load model/weights: {e}. "
                        "Check if model files exist and paths are correct."
                    )
                    break  # Don't retry model loading errors
                else:
                    self.logger.warning(f"Tool {tool_key} init error on attempt {attempt + 1}/{max_retries}: {e}")
                    time.sleep(0.5)  # Brief pause before retry
        
        # All retries failed
        self.logger.error(f"Failed to initialize tool {tool_key} after {max_retries} attempts: {last_error}")
        self._tool_instances[tool_key] = None
    
    def execute(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        video_context: Any,
    ) -> Dict[str, Any]:
        """
        Execute a tool with the given arguments.
        
        Includes automatic retry logic for GPU OOM and transient errors.
        
        Args:
            tool_name: Name of the tool to execute
            tool_args: Arguments for the tool
            video_context: VideoContext object with video data
        
        Returns:
            Dict with 'success', 'result' (text for agent), and optionally 'frames', 'captions'
            On failure, includes 'error', 'error_type', 'attempts', and optionally 'traceback'
        """
        if not self._initialized:
            self.initialize()
        
        if tool_name not in self.enabled_tools:
            return {
                "success": False,
                "result": f"Tool '{tool_name}' is not enabled. Available tools: {self.enabled_tools}",
            }
        
        start_time = datetime.now()
        
        # Use retry wrapper for execution
        result = self._execute_with_retry(
            self._execute_tool_internal,
            tool_name,
            tool_name,
            tool_args,
            video_context,
        )
        
        # Add timing
        duration_ms = (datetime.now() - start_time).total_seconds() * 1000
        result["duration_ms"] = duration_ms
        
        return result
    
    def _execute_tool_internal(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        video_context: Any,
    ) -> Dict[str, Any]:
        """Internal tool execution without retry logic.
        
        This method is wrapped by execute() with retry logic.
        """
        # Route to appropriate handler
        if tool_name == "caption_image":
            return self._execute_caption(tool_args, video_context)
        elif tool_name == "detailed_captioning":
            return self._execute_api_caption(tool_args, video_context)
        elif tool_name == "view_frame":
            return self._execute_view_frame(tool_args, video_context)
        elif tool_name == "temporal_sample_frames":
            return self._execute_temporal_sample(tool_args, video_context)
        elif tool_name == "temporal_spatial_sample_frames":
            return self._execute_temporal_spatial_sample(tool_args, video_context)
        elif tool_name == "detect_objects":
            return self._execute_detect_objects(tool_args, video_context)
        elif tool_name == "detect_all_objects":
            return self._execute_detect_all_objects(tool_args, video_context)
        elif tool_name == "describe_region":
            return self._execute_describe_region(tool_args, video_context)
        elif tool_name == "temporal_qa":
            return self._execute_temporal_qa(tool_args, video_context)
        elif tool_name == "temporal_spatial_qa":
            return self._execute_temporal_spatial_qa(tool_args, video_context)
        elif tool_name == "general_vqa":
            return self._execute_general_vqa(tool_args, video_context)
        elif tool_name == "targeting_vqa":
            return self._execute_targeting_vqa(tool_args, video_context)
        elif tool_name == "internvideo_general_qa":
            return self._execute_internvideo_general_qa(tool_args, video_context)
        elif tool_name == "internvideo_description":
            return self._execute_internvideo_description(tool_args, video_context)
        else:
            return {
                "success": False,
                "result": f"Unknown tool: {tool_name}",
            }
    
    def _execute_caption(
        self,
        args: Dict[str, Any],
        video_context: Any,
    ) -> Dict[str, Any]:
        """Execute caption_image tool (OmniCaptioner) with caching."""
        frame_indices = args.get("frame_indices", [])
        detail_level = args.get("detail_level", "short")
        with_objects = args.get("with_objects", False)
        
        # Validate frame indices
        valid_indices, error = self._validate_frame_indices(frame_indices, video_context)
        if error:
            return error
        
        # Model identifier for cache key
        model_key = self.captioner_model
        
        # Check cache for all frames first
        captions = {}
        indices_to_process = []
        
        if self._tool_cache and not with_objects:  # Only cache simple captions (no objects)
            cached_captions = self._tool_cache.get_captions_batch(
                video_path=video_context.video_path,
                frame_indices=valid_indices,
                model=model_key,
                detail_level=detail_level,
            )
            
            for idx in valid_indices:
                cached = cached_captions.get(idx)
                if cached is not None:
                    captions[idx] = cached
                    # Update video context with cached caption
                    video_context.frame_captions[idx] = cached
                    if idx not in video_context.sampled_indices:
                        video_context.sampled_indices.append(idx)
                else:
                    indices_to_process.append(idx)
            
            if captions:
                self.logger.info(f"Caption cache hit: {len(captions)}/{len(valid_indices)} frames")
        else:
            indices_to_process = valid_indices
        
        # If all frames were cached, skip model processing
        if not indices_to_process:
            # Format result from cache
            lines = []
            for idx in sorted(captions.keys()):
                ts = idx / video_context.fps if video_context.fps > 0 else 0
                lines.append(f"[Frame {idx} @ {ts:.1f}s]: {captions[idx]}")
            
            return {
                "success": True,
                "result": "\n".join(lines) if lines else "No captions generated",
                "captions": captions,
                "objects": None,
                "cache_hit": True,
            }
        
        # Ensure captioner is initialized
        if self._captioner is None:
            self._init_captioner()
        if self._captioner is None:
            return {"success": False, "result": "Captioner not initialized. Check model configuration."}
        
        # Extract frames for processing
        from video_agent_tools.utils.video import extract_frames
        frames = extract_frames(video_context.video_path, indices_to_process)
        
        all_objects = []
        new_captions = {}
        
        for idx, frame in frames.items():
            # Save frame data to video_context
            video_context.frames[idx] = frame
            
            # Use OmniCaptioner with detail_level and with_objects
            result = self._captioner(
                image=frame,
                detail_level=detail_level,
                with_objects=with_objects
            )
            
            caption = result.get("caption", "")
            if caption:
                captions[idx] = caption
                new_captions[idx] = caption
                # Update video context
                video_context.frame_captions[idx] = caption
                if idx not in video_context.sampled_indices:
                    video_context.sampled_indices.append(idx)
            
            # Collect objects if requested
            if with_objects and "objects" in result:
                all_objects.extend(result["objects"])
        
        # Save new captions to cache (only if not with_objects)
        if self._tool_cache and new_captions and not with_objects:
            self._tool_cache.set_captions_batch(
                video_path=video_context.video_path,
                captions=new_captions,
                model=model_key,
                detail_level=detail_level,
            )
            self.logger.info(f"Cached {len(new_captions)} new captions")
        
        # Format result
        lines = []
        for idx in sorted(captions.keys()):
            ts = idx / video_context.fps if video_context.fps > 0 else 0
            lines.append(f"[Frame {idx} @ {ts:.1f}s]: {captions[idx]}")
        
        # Add objects summary if requested
        if with_objects and all_objects:
            # Deduplicate objects across frames
            unique_objects = list(dict.fromkeys(all_objects))
            lines.append(f"\nDetected objects: {', '.join(unique_objects)}")
        
        return {
            "success": True,
            "result": "\n".join(lines) if lines else "No captions generated",
            "captions": captions,
            "objects": list(dict.fromkeys(all_objects)) if with_objects else None,
        }
    
    def _execute_api_caption(
        self,
        args: Dict[str, Any],
        video_context: Any,
    ) -> Dict[str, Any]:
        """Execute detailed_captioning tool (API-based MLLM) with caching."""
        frame_indices = args.get("frame_indices", [])
        detail_level = args.get("detail_level", "medium")
        with_objects = args.get("with_objects", False)
        
        # Validate frame indices
        valid_indices, error = self._validate_frame_indices(frame_indices, video_context)
        if error:
            return error
        
        # Model identifier for cache key (use API model name)
        model_key = f"api_{self.api_captioner_model}"
        
        # Check cache for all frames first
        captions = {}
        indices_to_process = []
        
        if self._tool_cache and not with_objects:  # Only cache simple captions (no objects)
            cached_captions = self._tool_cache.get_captions_batch(
                video_path=video_context.video_path,
                frame_indices=valid_indices,
                model=model_key,
                detail_level=detail_level,
            )
            
            for idx in valid_indices:
                cached = cached_captions.get(idx)
                if cached is not None:
                    captions[idx] = cached
                    # Update video context with cached caption
                    video_context.frame_captions[idx] = cached
                    if idx not in video_context.sampled_indices:
                        video_context.sampled_indices.append(idx)
                else:
                    indices_to_process.append(idx)
            
            if captions:
                self.logger.info(f"API caption cache hit: {len(captions)}/{len(valid_indices)} frames")
        else:
            indices_to_process = valid_indices
        
        # If all frames were cached, skip API processing
        if not indices_to_process:
            # Format result from cache
            lines = []
            for idx in sorted(captions.keys()):
                ts = idx / video_context.fps if video_context.fps > 0 else 0
                lines.append(f"[Frame {idx} @ {ts:.1f}s]: {captions[idx]}")
            
            return {
                "success": True,
                "result": "\n".join(lines) if lines else "No captions generated",
                "captions": captions,
                "objects": None,
                "cache_hit": True,
            }
        
        # Ensure API captioner is initialized
        if self._api_captioner is None:
            self._init_api_captioner()
        if self._api_captioner is None:
            return {"success": False, "result": "API captioner not initialized. Check API configuration."}
        
        # Extract frames for processing
        from video_agent_tools.utils.video import extract_frames
        frames = extract_frames(video_context.video_path, indices_to_process)
        
        all_objects = []
        new_captions = {}
        
        for idx, frame in frames.items():
            # Save frame data to video_context
            video_context.frames[idx] = frame
            
            # Use API captioner with detail_level and with_objects
            result = self._api_captioner(
                image=frame,
                detail_level=detail_level,
                with_objects=with_objects
            )
            
            if "error" in result:
                self.logger.error(f"API caption error at frame {idx}: {result['error']}")
                continue
            
            caption = result.get("caption", "")
            if caption:
                captions[idx] = caption
                new_captions[idx] = caption
                # Update video context
                video_context.frame_captions[idx] = caption
                if idx not in video_context.sampled_indices:
                    video_context.sampled_indices.append(idx)
            
            # Collect objects if requested
            if with_objects and "objects" in result:
                all_objects.extend(result["objects"])
        
        # Save new captions to cache (only if not with_objects)
        if self._tool_cache and new_captions and not with_objects:
            self._tool_cache.set_captions_batch(
                video_path=video_context.video_path,
                captions=new_captions,
                model=model_key,
                detail_level=detail_level,
            )
            self.logger.info(f"Cached {len(new_captions)} new API captions")
        
        # Format result
        lines = []
        for idx in sorted(captions.keys()):
            ts = idx / video_context.fps if video_context.fps > 0 else 0
            lines.append(f"[Frame {idx} @ {ts:.1f}s]: {captions[idx]}")
        
        # Add objects summary if requested
        if with_objects and all_objects:
            # Deduplicate objects across frames
            unique_objects = list(dict.fromkeys(all_objects))
            lines.append(f"\nDetected objects: {', '.join(unique_objects)}")
        
        return {
            "success": True,
            "result": "\n".join(lines) if lines else "No captions generated",
            "captions": captions,
            "objects": list(dict.fromkeys(all_objects)) if with_objects else None,
        }
    
    def _caption_single_frame(
        self,
        frame,
        detail_level: str = "short",
        video_path: str = None,
        frame_idx: int = None,
    ) -> Optional[str]:
        """Caption a single frame using the configured captioner (for auto-captioning).
        
        If video_path and frame_idx are provided, will check/update cache.
        """
        # Check cache first if video_path and frame_idx provided
        model_key = self._get_captioner_model_key()
        if self._tool_cache and video_path is not None and frame_idx is not None:
            cached_caption = self._tool_cache.get_caption(
                video_path=video_path,
                frame_idx=frame_idx,
                model=model_key,
                detail_level=detail_level,
            )
            if cached_caption is not None:
                return cached_caption
        
        # Call captioner
        result = self._captioner(image=frame, detail_level=detail_level)
        caption = result.get("caption")
        if not caption and "error" in result:
            self.logger.warning(f"Captioner returned error: {result.get('error')}")
        
        # Cache the result if video_path and frame_idx provided
        if caption and self._tool_cache and video_path is not None and frame_idx is not None:
            self._tool_cache.set_caption(
                video_path=video_path,
                frame_idx=frame_idx,
                model=model_key,
                detail_level=detail_level,
                caption=caption,
            )
        
        return caption
    
    def _encode_frame_to_base64(self, frame_data) -> str:
        """Encode a numpy array frame to base64 JPEG string.
        
        Args:
            frame_data: numpy array (RGB format)
        
        Returns:
            Base64 encoded JPEG string
        """
        import base64
        from io import BytesIO
        from PIL import Image as PILImage
        import numpy as np
        
        # Convert numpy array to PIL Image
        if isinstance(frame_data, np.ndarray):
            pil_img = PILImage.fromarray(frame_data.astype(np.uint8))
        else:
            pil_img = frame_data
        
        # Convert to RGB if needed
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        
        # Encode to JPEG base64
        buffer = BytesIO()
        pil_img.save(buffer, format="JPEG", quality=85)
        buffer.seek(0)
        
        return base64.b64encode(buffer.read()).decode("utf-8")
    
    def _execute_view_frame(
        self,
        args: Dict[str, Any],
        video_context: Any,
    ) -> Dict[str, Any]:
        """Execute view_frame tool - returns actual images for agent to see.
        
        This tool gives the agent direct visual access to video frames.
        The images are encoded as base64 and sent to the LLM for visual analysis.
        """
        frame_indices = args.get("frame_indices")
        frame_index = args.get("frame_index")
        timestamp = args.get("timestamp")
        
        # Determine which frames to extract
        indices_to_extract = []
        
        if frame_indices is not None and len(frame_indices) > 0:
            indices_to_extract = list(frame_indices)
        elif frame_index is not None:
            indices_to_extract = [frame_index]
        elif timestamp is not None:
            frame_idx = int(timestamp * video_context.fps)
            indices_to_extract = [frame_idx]
        else:
            return {"success": False, "result": "Provide frame_indices, frame_index, or timestamp"}
        
        # Validate frame indices
        valid_indices = [
            idx for idx in indices_to_extract 
            if 0 <= idx < video_context.total_frames
        ]
        
        if not valid_indices:
            return {
                "success": False,
                "result": f"No valid frame indices. Valid range: 0-{video_context.total_frames - 1}",
            }
        
        # Limit frames to prevent token overflow
        if len(valid_indices) > self.max_view_frames:
            self.logger.warning(f"Limiting view_frame from {len(valid_indices)} to {self.max_view_frames} frames")
            valid_indices = valid_indices[:self.max_view_frames]
        
        # Extract frames
        from video_agent_tools.utils.video import extract_frames
        frames = extract_frames(video_context.video_path, valid_indices)
        
        if not frames:
            return {"success": False, "result": "Failed to extract frames"}
        
        # Build result with image data for LLM
        image_data_list = []  # List of {base64, frame_index, timestamp}
        extracted_indices = []
        extracted_timestamps = []
        
        for idx in sorted(frames.keys()):
            frame = frames[idx]
            
            # Save to video_context for logging
            video_context.frames[idx] = frame
            if idx not in video_context.sampled_indices:
                video_context.sampled_indices.append(idx)
            
            # Calculate timestamp
            ts = idx / video_context.fps if video_context.fps > 0 else 0.0
            
            # Encode frame to base64
            base64_image = self._encode_frame_to_base64(frame)
            
            image_data_list.append({
                "base64": base64_image,
                "frame_index": idx,
                "timestamp": ts,
            })
            
            extracted_indices.append(idx)
            extracted_timestamps.append(ts)
        
        # Format text result
        if len(extracted_indices) == 1:
            idx = extracted_indices[0]
            ts = extracted_timestamps[0]
            result_text = f"Viewing Frame {idx} @ {ts:.1f}s - analyze the image to answer the question."
        else:
            lines = [f"Viewing {len(extracted_indices)} frames:"]
            for i, (idx, ts) in enumerate(zip(extracted_indices, extracted_timestamps)):
                lines.append(f"  Image {i+1}: Frame {idx} @ {ts:.1f}s")
            lines.append("Analyze the images to answer the question.")
            result_text = "\n".join(lines)
        
        return {
            "success": True,
            "result": result_text,
            "images": image_data_list,  # For LLM multimodal input
            "frame_indices": extracted_indices,
            "timestamps": extracted_timestamps,
            "has_images": True,  # Flag for graph to know this result includes images
        }
    
    def _execute_temporal_sample(
        self,
        args: Dict[str, Any],
        video_context: Any,
    ) -> Dict[str, Any]:
        """Execute temporal_sample_frames using VideoTree with auto-captioning."""
        query = args.get("query", "")
        num_frames = args.get("num_frames", self.default_sample_frames)
        # Clamp num_frames to valid range
        num_frames = max(self.min_sample_frames, min(num_frames, self.max_sample_frames))
        
        # Optional frame range parameters
        start_frame = args.get("start_frame")
        end_frame = args.get("end_frame")
        
        # Lazy initialize VideoTree sampler
        if "temporal_sample_frames" not in self._tool_instances:
            self._init_tool("temporal_sample_frames")
        
        sampler = self._tool_instances.get("temporal_sample_frames")
        
        # If sampler not available, fallback to uniform sampling
        if sampler is None:
            self.logger.warning("VideoTree sampler not available, using uniform sampling")
            return self._fallback_uniform_sample(query, num_frames, video_context, start_frame=start_frame, end_frame=end_frame)
        
        try:
            from tools.interface_base import Video
            video = Video(path=video_context.video_path)
            
            # Call the sampler with optional frame range
            result = sampler(video=video, query=query, num_frames=num_frames, start_frame=start_frame, end_frame=end_frame)
            
            if "error" in result:
                self.logger.warning(f"VideoTree error: {result['error']}, falling back to uniform")
                return self._fallback_uniform_sample(query, num_frames, video_context, start_frame=start_frame, end_frame=end_frame)
            
            # Extract frame indices and auto-caption
            frame_indices = result.get("frame_indices", [])
            
            if not frame_indices:
                return {
                    "success": True,
                    "result": "No new frames sampled.",
                }
            
            # Caption the sampled frames (with caching support)
            captions = {}
            for frame in result.get("frames", []):
                idx = frame.frame_index
                if idx is not None and frame.data is not None:
                    # Save frame data to video_context for later export
                    video_context.frames[idx] = frame.data
                    
                    caption = self._caption_single_frame(
                        frame.data, "short",
                        video_path=video_context.video_path,
                        frame_idx=idx,
                    )
                    if caption:
                        captions[idx] = caption
                        video_context.frame_captions[idx] = caption
                        if idx not in video_context.sampled_indices:
                            video_context.sampled_indices.append(idx)
            
            # Format result
            lines = [f"Sampled {len(captions)} visually diverse frames for query: '{query}'"]
            for idx in sorted(captions.keys()):
                ts = idx / video_context.fps if video_context.fps > 0 else 0
                lines.append(f"[Frame {idx} @ {ts:.1f}s]: {captions[idx]}")
            
            return {
                "success": True,
                "result": "\n".join(lines),
                "captions": captions,
            }
            
        except Exception as e:
            self.logger.error(f"VideoTree sampling failed: {e}, falling back to uniform")
            return self._fallback_uniform_sample(query, num_frames, video_context, start_frame=start_frame, end_frame=end_frame)
    
    def _execute_temporal_spatial_sample(
        self,
        args: Dict[str, Any],
        video_context: Any,
    ) -> Dict[str, Any]:
        """Execute temporal_spatial_sample_frames using TStar with auto-captioning."""
        query = args.get("query", "")
        num_frames = args.get("num_frames", self.default_sample_frames)
        # Clamp num_frames to valid range
        num_frames = max(self.min_sample_frames, min(num_frames, self.max_sample_frames))
        
        # Optional frame range parameters
        start_frame = args.get("start_frame")
        end_frame = args.get("end_frame")
        
        # Lazy initialize TStar sampler
        if "temporal_spatial_sample_frames" not in self._tool_instances:
            self._init_tool("temporal_spatial_sample_frames")
        
        sampler = self._tool_instances.get("temporal_spatial_sample_frames")
        
        # If sampler not available, fallback to uniform sampling
        if sampler is None:
            self.logger.warning("TStar sampler not available, using uniform sampling")
            return self._fallback_uniform_sample(query, num_frames, video_context, is_object_search=True, start_frame=start_frame, end_frame=end_frame)
        
        try:
            from tools.interface_base import Video
            video = Video(path=video_context.video_path)
            
            # Call the sampler with optional frame range
            result = sampler(video=video, query=query, num_frames=num_frames, start_frame=start_frame, end_frame=end_frame)
            
            if "error" in result:
                self.logger.warning(f"TStar error: {result['error']}, falling back to uniform")
                return self._fallback_uniform_sample(query, num_frames, video_context, is_object_search=True, start_frame=start_frame, end_frame=end_frame)
            
            # Get target objects and frame indices
            target_objects = result.get("target_objects", [])
            frame_indices = result.get("frame_indices", [])
            
            if not frame_indices:
                return {
                    "success": True,
                    "result": f"Searched for '{query}' but found no matching frames.",
                }
            
            # Caption the sampled frames (with caching support)
            captions = {}
            for frame in result.get("frames", []):
                idx = frame.frame_index
                if idx is not None and frame.data is not None:
                    # Save frame data to video_context for later export
                    video_context.frames[idx] = frame.data
                    
                    caption = self._caption_single_frame(
                        frame.data, "short",
                        video_path=video_context.video_path,
                        frame_idx=idx,
                    )
                    if caption:
                        captions[idx] = caption
                        video_context.frame_captions[idx] = caption
                        if idx not in video_context.sampled_indices:
                            video_context.sampled_indices.append(idx)
            
            # Format result with full information
            lines = []
            
            # Show target objects searched
            if target_objects:
                lines.append(f"Target objects searched: {', '.join(target_objects)}")
            
            # Show cue objects if available
            cue_objects = result.get("cue_objects", [])
            if cue_objects:
                lines.append(f"Cue objects: {', '.join(cue_objects)}")
            
            # Summary line
            lines.append(f"Found {len(captions)} frames matching query: '{query}'")
            lines.append("")
            
            # Show each frame with full caption
            for idx in sorted(captions.keys()):
                ts = idx / video_context.fps if video_context.fps > 0 else 0
                lines.append(f"[Frame {idx} @ {ts:.1f}s]:")
                lines.append(captions[idx])
                lines.append("")
            
            return {
                "success": True,
                "result": "\n".join(lines),
                "captions": captions,
                "target_objects": target_objects,
            }
            
        except Exception as e:
            self.logger.error(f"TStar sampling failed: {e}, falling back to uniform")
            return self._fallback_uniform_sample(query, num_frames, video_context, is_object_search=True)
    
    def _fallback_uniform_sample(
        self,
        query: str,
        num_frames: int,
        video_context: Any,
        is_object_search: bool = False,
        start_frame: int = None,
        end_frame: int = None,
    ) -> Dict[str, Any]:
        """Fallback to uniform sampling when advanced samplers are not available."""
        from video_agent_tools.utils.video import sample_indices_in_range, extract_frames
        
        # Apply optional frame range constraints
        range_start = 0 if start_frame is None else max(0, min(start_frame, video_context.total_frames - 1))
        range_end = video_context.total_frames - 1 if end_frame is None else max(range_start, min(end_frame, video_context.total_frames - 1))
        range_total = range_end - range_start + 1
        
        new_indices = sample_indices_in_range(
            total_frames=range_total,
            num_samples=num_frames,
            exclude_indices=[i - range_start for i in video_context.sampled_indices if range_start <= i <= range_end],
        )
        # Shift indices back to absolute frame positions
        new_indices = [i + range_start for i in new_indices]
        
        if not new_indices:
            return {
                "success": True,
                "result": "No new frames to sample. All frames have been viewed.",
            }
        
        frames = extract_frames(video_context.video_path, new_indices)
        
        captions = {}
        for idx, frame in frames.items():
            # Save frame data to video_context
            video_context.frames[idx] = frame
            
            caption = self._caption_single_frame(
                frame, "medium",
                video_path=video_context.video_path,
                frame_idx=idx,
            )
            if caption:
                captions[idx] = caption
                video_context.frame_captions[idx] = caption
                if idx not in video_context.sampled_indices:
                    video_context.sampled_indices.append(idx)
        
        if is_object_search:
            header = f"Searched for '{query}' and sampled {len(captions)} frames (uniform sampling):"
        else:
            header = f"Sampled {len(captions)} frames for query: '{query}' (uniform sampling):"
        
        lines = [header]
        for idx in sorted(captions.keys()):
            ts = idx / video_context.fps if video_context.fps > 0 else 0
            lines.append(f"[Frame {idx} @ {ts:.1f}s]: {captions[idx]}")
        
        return {
            "success": True,
            "result": "\n".join(lines),
            "captions": captions,
        }
    
    def _execute_detect_objects(
        self,
        args: Dict[str, Any],
        video_context: Any,
    ) -> Dict[str, Any]:
        """Execute detect_objects tool with per-frame results.
        
        Returns detection results for each frame separately with bounding box locations.
        Uses higher confidence threshold (0.3) to reduce false positives.
        """
        categories = args.get("categories", [])
        frame_indices = args.get("frame_indices", [])
        
        if not categories:
            return {"success": False, "result": "No categories provided"}
        
        # Validate frame indices
        valid_indices, error = self._validate_frame_indices(frame_indices, video_context)
        if error:
            return error
        
        # Lazy initialize the tool (with retry if previously failed)
        if "detect_objects" not in self._tool_instances or self._tool_instances.get("detect_objects") is None:
            self._init_tool("detect_objects", force_retry=True)
        
        tool = self._tool_instances.get("detect_objects")
        if tool is None:
            return {"success": False, "result": "Object detection tool not initialized. Check GPU memory or model availability."}
        
        # Extract frames
        from video_agent_tools.utils.video import extract_frames
        frames = extract_frames(video_context.video_path, valid_indices)
        
        # Convert to Image objects and keep track of frame indices
        images = []
        indices_list = []
        for idx in sorted(frames.keys()):
            images.append(Image(data=frames[idx]))
            indices_list.append(idx)
        
        # Execute detection with frame indices and higher threshold (0.3)
        result = tool(
            images=images,
            categories=categories,
            score_threshold=0.3,
            frame_indices=indices_list
        )
        
        if "error" in result:
            return {"success": False, "result": f"Detection Error: {result['error']}"}
        
        # Format result using the new per-frame format
        formatted = tool.format_output_for_agent(result)
        
        return {
            "success": True,
            "result": f"Object detection results:\n{formatted}",
        }
    
    def _execute_detect_all_objects(
        self,
        args: Dict[str, Any],
        video_context: Any,
    ) -> Dict[str, Any]:
        """Execute detect_all_objects tool with per-frame results.
        
        Returns detection results for each frame separately.
        """
        frame_indices = args.get("frame_indices", [])
        max_detections = args.get("max_detections", 50)
        
        # Validate frame indices
        valid_indices, error = self._validate_frame_indices(frame_indices, video_context)
        if error:
            return error
        
        # Lazy initialize the tool
        # Lazy initialize the tool (with retry if previously failed)
        if "detect_all_objects" not in self._tool_instances or self._tool_instances.get("detect_all_objects") is None:
            self._init_tool("detect_all_objects", force_retry=True)
        
        tool = self._tool_instances.get("detect_all_objects")
        if tool is None:
            return {"success": False, "result": "Object detection tool not initialized. Check GPU memory or model availability."}
        
        # Extract frames
        from video_agent_tools.utils.video import extract_frames
        frames = extract_frames(video_context.video_path, valid_indices)
        
        # Convert to Image objects and keep track of frame indices
        images = []
        indices_list = []
        for idx in sorted(frames.keys()):
            images.append(Image(data=frames[idx]))
            indices_list.append(idx)
        
        # Execute detection with frame indices
        result = tool(
            images=images,
            max_detections=max_detections,
            frame_indices=indices_list
        )
        
        if "error" in result:
            return {"success": False, "result": f"Detection Error: {result['error']}"}
        
        # Format result using the new per-frame format
        formatted = tool.format_output_for_agent(result)
        
        return {
            "success": True,
            "result": f"All objects detected:\n{formatted}",
        }
    
    def _execute_describe_region(
        self,
        args: Dict[str, Any],
        video_context: Any,
    ) -> Dict[str, Any]:
        """Execute describe_region tool."""
        frame_index = args.get("frame_index")
        bbox = args.get("bbox", [])
        
        if frame_index is None:
            return {"success": False, "result": "No frame_index provided"}
        if not bbox or len(bbox) != 4:
            return {"success": False, "result": "Invalid bbox. Must be [x1, y1, x2, y2]"}
        
        # Lazy initialize the tool
        if "describe_region" not in self._tool_instances:
            self._init_tool("describe_region")
        
        tool = self._tool_instances.get("describe_region")
        if tool is None:
            return {"success": False, "result": "Region description tool not initialized"}
        
        # Validate index
        if frame_index < 0 or frame_index >= video_context.total_frames:
            return {
                "success": False,
                "result": f"Frame index {frame_index} out of range. Valid: 0-{video_context.total_frames - 1}",
            }
        
        # Extract frame
        from video_agent_tools.utils.video import extract_frame
        frame = extract_frame(video_context.video_path, frame_index)
        
        if frame is None:
            return {"success": False, "result": f"Failed to extract frame {frame_index}"}
        
        # Save frame data to video_context
        video_context.frames[frame_index] = frame
        
        # Create Image object
        image = Image(data=frame)
        
        # Execute description
        result = tool(image=image, bbox=bbox)
        
        if "error" in result:
            return {"success": False, "result": f"Description Error: {result['error']}"}
        
        # Format result
        formatted = tool.format_output_for_agent(result)
        
        return {
            "success": True,
            "result": f"Region description at frame {frame_index}, bbox {bbox}:\n{formatted}",
        }
    
    def _execute_temporal_qa(
        self,
        args: Dict[str, Any],
        video_context: Any,
    ) -> Dict[str, Any]:
        """Execute temporal_qa tool (VideoRAG-based sub-question answering).
        
        This tool answers questions about events, timing, sequences, and durations
        by using VideoRAG's retrieval-augmented generation over the video.
        """
        query = args.get("query", "")
        
        if not query:
            return {"success": False, "result": "No query provided for temporal_qa"}
        
        # Lazy initialize the tool (with retry if previously failed)
        if "temporal_qa" not in self._tool_instances or self._tool_instances.get("temporal_qa") is None:
            self._init_tool("temporal_qa", force_retry=True)
        
        tool = self._tool_instances.get("temporal_qa")
        if tool is None:
            return {"success": False, "result": "Temporal QA tool not initialized. VideoRAG may not be available."}
        
        try:
            # Create Video object for the interface
            video = Video(path=video_context.video_path)
            
            # Execute the query
            result = tool(query=query, video=video)
            
            if "error" in result:
                # Clear GPU memory after error
                self._clear_gpu_memory()
                return {"success": False, "result": f"Temporal QA Error: {result['error']}"}
            
            # Format the answer
            answer = result.get("answer", "No answer returned")
            formatted = VideoRAGTemporalQA.format_output_for_agent(result)
            
            # Clear GPU memory after successful execution to free resources for other tools
            self._clear_gpu_memory()
            
            return {
                "success": True,
                "result": f"Answer to '{query}':\n{formatted}",
            }
            
        except Exception as e:
            self.logger.error(f"Temporal QA failed: {e}")
            # Clear GPU memory after exception
            self._clear_gpu_memory()
            return {"success": False, "result": f"Temporal QA failed: {str(e)}"}
    
    def _execute_temporal_spatial_qa(
        self,
        args: Dict[str, Any],
        video_context: Any,
    ) -> Dict[str, Any]:
        """Execute temporal_spatial_qa tool (TStar-based sub-question answering).
        
        This tool answers questions involving objects, locations, movements,
        and spatial-temporal relationships by using TStar's object-guided search.
        
        NOTE: Options are NOT passed to avoid leading/biasing the model's response.
        """
        query = args.get("query", "")
        # NOTE: options parameter removed to avoid leading questions
        
        if not query:
            return {"success": False, "result": "No query provided for temporal_spatial_qa"}
        
        # Lazy initialize the tool (with retry if previously failed)
        if "temporal_spatial_qa" not in self._tool_instances or self._tool_instances.get("temporal_spatial_qa") is None:
            self._init_tool("temporal_spatial_qa", force_retry=True)
        
        tool = self._tool_instances.get("temporal_spatial_qa")
        if tool is None:
            return {"success": False, "result": "Temporal-Spatial QA tool not initialized. TStar may not be available."}
        
        try:
            # Create Video object for the interface
            video = Video(path=video_context.video_path)
            
            # Execute the query - NO options passed to avoid bias
            result = tool(query=query, video=video)
            
            if "error" in result:
                # Clear GPU memory after error
                self._clear_gpu_memory()
                return {"success": False, "result": f"Temporal-Spatial QA Error: {result['error']}"}
            
            # Format the answer with full details
            lines = []
            
            # Include grounding objects information
            if "grounding_objects" in result:
                grounding = result["grounding_objects"]
                targets = grounding.get("target_objects", [])
                cues = grounding.get("cue_objects", [])
                if targets:
                    lines.append(f"Target objects searched: {', '.join(targets)}")
                if cues:
                    lines.append(f"Cue objects: {', '.join(cues)}")
            
            # Include frame timestamps
            timestamps = result.get("frame_timestamps", [])
            if timestamps:
                time_strs = [f"{t:.1f}s" for t in timestamps]
                lines.append(f"Frames analyzed: {', '.join(time_strs)}")
            
            lines.append("")
            
            # Answer
            answer = result.get("answer", "Could not determine answer.")
            lines.append("Answer:")
            lines.append(answer)
            
            # Evidence
            evidence = result.get("evidence", "")
            if evidence:
                lines.append("")
                lines.append("Evidence:")
                lines.append(evidence)
            
            # Clear GPU memory after successful execution to free resources for other tools
            self._clear_gpu_memory()
            
            return {
                "success": True,
                "result": "\n".join(lines),
            }
            
        except Exception as e:
            self.logger.error(f"Temporal-Spatial QA failed: {e}")
            # Clear GPU memory after exception
            self._clear_gpu_memory()
            return {"success": False, "result": f"Temporal-Spatial QA failed: {str(e)}"}
    
    def _execute_general_vqa(
        self,
        args: Dict[str, Any],
        video_context: Any,
    ) -> Dict[str, Any]:
        """Execute general_vqa tool (API-based visual question answering).
        
        This tool answers general visual questions using an MLLM API
        on one or more frames from the video.
        """
        query = args.get("query", "")
        frame_indices = args.get("frame_indices", [])
        
        if not query:
            return {"success": False, "result": "No query provided for general_vqa"}
        
        # If no frame indices provided, use currently sampled frames
        if not frame_indices:
            frame_indices = video_context.sampled_indices[:8] if video_context.sampled_indices else []
        
        if not frame_indices:
            return {"success": False, "result": "No frames available. Use sampling tools first."}
        
        # Lazy initialize the tool
        if "general_vqa" not in self._tool_instances:
            from tools.interface.visual_qa import GeneralVQA
            self._tool_instances["general_vqa"] = GeneralVQA(use_aiml=True)
            self._tool_instances["general_vqa"].initialize()
        
        tool = self._tool_instances.get("general_vqa")
        if tool is None:
            return {"success": False, "result": "General VQA tool not initialized."}
        
        try:
            # Extract frames
            from video_agent_tools.utils.video import extract_frames
            frames = extract_frames(video_context.video_path, frame_indices)
            
            if not frames:
                return {"success": False, "result": "Failed to extract frames for VQA."}
            
            # Save frames and create Image objects
            images = []
            for idx, frame in frames.items():
                video_context.frames[idx] = frame
                images.append(Image(data=frame))
            
            # Execute VQA
            result = tool(query=query, images=images)
            
            if "error" in result:
                return {"success": False, "result": f"General VQA Error: {result['error']}"}
            
            # Format the answer
            from tools.interface.visual_qa import GeneralVQA as GeneralVQAClass
            formatted = GeneralVQAClass.format_output_for_agent(result)
            
            return {
                "success": True,
                "result": f"Answer (based on {len(images)} frame(s)):\n{formatted}",
                "frame_indices": list(frames.keys()),
            }
            
        except Exception as e:
            self.logger.error(f"General VQA failed: {e}")
            return {"success": False, "result": f"General VQA failed: {str(e)}"}
    
    def _execute_targeting_vqa(
        self,
        args: Dict[str, Any],
        video_context: Any,
    ) -> Dict[str, Any]:
        """Execute targeting_vqa tool (VStar-based fine-grained visual question answering).
        
        This tool uses guided visual search to locate small or hard-to-see objects,
        then answers questions about them.
        """
        query = args.get("query", "")
        target_object = args.get("target_object", None)
        frame_index = args.get("frame_index", None)
        
        if not query:
            return {"success": False, "result": "No query provided for targeting_vqa"}
        
        # If no frame index provided, use the most recent sampled frame
        if frame_index is None:
            if video_context.sampled_indices:
                frame_index = video_context.sampled_indices[-1]
            else:
                return {"success": False, "result": "No frames available. Use sampling tools first."}
        
        # Validate frame index
        if not (0 <= frame_index < video_context.total_frames):
            return {
                "success": False,
                "result": f"Invalid frame index {frame_index}. Valid range: 0-{video_context.total_frames - 1}",
            }
        
        # Lazy initialize the tool
        if "targeting_vqa" not in self._tool_instances:
            from tools.interface.visual_qa import TargetingVQA
            self._tool_instances["targeting_vqa"] = TargetingVQA()
            self._tool_instances["targeting_vqa"].initialize()
        
        tool = self._tool_instances.get("targeting_vqa")
        if tool is None:
            return {"success": False, "result": "Targeting VQA tool not initialized. VStar may not be available."}
        
        try:
            # Extract frame
            from video_agent_tools.utils.video import extract_frame
            frame = extract_frame(video_context.video_path, frame_index)
            
            if frame is None:
                return {"success": False, "result": f"Failed to extract frame {frame_index}"}
            
            # Save frame data
            video_context.frames[frame_index] = frame
            
            # Execute VQA
            result = tool(
                query=query,
                images=[Image(data=frame)],
                target_object=target_object,
            )
            
            if "error" in result:
                return {"success": False, "result": f"Targeting VQA Error: {result['error']}"}
            
            # Format the answer
            from tools.interface.visual_qa import TargetingVQA as TargetingVQAClass
            formatted = TargetingVQAClass.format_output_for_agent(result)
            
            # Include search result info if available
            search_info = ""
            if "search_result" in result and result["search_result"]:
                sr = result["search_result"]
                if sr.get("success"):
                    search_info = f"\n[Visual search found '{sr.get('target_object', 'target')}' in {sr.get('path_length', 0)} steps]"
                else:
                    search_info = f"\n[Visual search did not locate target]"
            
            return {
                "success": True,
                "result": f"Answer (frame {frame_index}):\n{formatted}{search_info}",
                "frame_index": frame_index,
            }
            
        except Exception as e:
            self.logger.error(f"Targeting VQA failed: {e}")
            return {"success": False, "result": f"Targeting VQA failed: {str(e)}"}
    
    def _execute_internvideo_general_qa(
        self,
        args: Dict[str, Any],
        video_context: Any,
    ) -> Dict[str, Any]:
        """Execute internvideo_general_qa tool (InternVideo2.5-based general video Q&A).
        
        This tool uses InternVideo2.5 with 128 frames to answer general questions
        about video content. Optionally accepts start_frame and end_frame to focus
        on a specific segment.
        """
        query = args.get("query", "")
        start_frame = args.get("start_frame", None)
        end_frame = args.get("end_frame", None)
        
        if not query:
            return {"success": False, "result": "No query provided for internvideo_general_qa"}
        
        # Validate frame range if provided
        if start_frame is not None:
            if start_frame < 0 or start_frame >= video_context.total_frames:
                return {"success": False, "result": f"Invalid start_frame={start_frame}. Valid range: 0-{video_context.total_frames - 1}"}
        if end_frame is not None:
            if end_frame < 0 or end_frame >= video_context.total_frames:
                return {"success": False, "result": f"Invalid end_frame={end_frame}. Valid range: 0-{video_context.total_frames - 1}"}
            if start_frame is not None and end_frame < start_frame:
                return {"success": False, "result": f"end_frame ({end_frame}) must be >= start_frame ({start_frame})"}
        
        # Lazy initialize the tool using _init_tool for proper GPU management
        if "internvideo_general_qa" not in self._tool_instances or self._tool_instances.get("internvideo_general_qa") is None:
            self._init_tool("internvideo_general_qa", force_retry=True)
        
        tool = self._tool_instances.get("internvideo_general_qa")
        if tool is None:
            return {"success": False, "result": "InternVideo General QA tool not initialized. Check GPU memory."}
        
        try:
            # Execute with video path and optional frame range
            result = tool(
                query=query,
                video=Video(path=video_context.video_path),
                start_frame=start_frame,
                end_frame=end_frame,
            )
            
            if "error" in result:
                return {"success": False, "result": f"InternVideo QA Error: {result['error']}"}
            
            # Format the answer
            formatted = InternVideoGeneralQA.format_output_for_agent(result)
            
            return {
                "success": True,
                "result": formatted,
            }
            
        except Exception as e:
            self.logger.error(f"InternVideo General QA failed: {e}")
            return {"success": False, "result": f"InternVideo General QA failed: {str(e)}"}
    
    def _execute_internvideo_description(
        self,
        args: Dict[str, Any],
        video_context: Any,
    ) -> Dict[str, Any]:
        """Execute internvideo_description tool (InternVideo2.5-based video clip description) with caching.
        
        This tool generates a description for a specific video segment.
        REQUIRES start_frame and end_frame - full video description is already in context.
        """
        # Extract frame range arguments - these are now REQUIRED
        start_frame = args.get("start_frame", None)
        end_frame = args.get("end_frame", None)
        
        # Validate required arguments
        if start_frame is None or end_frame is None:
            return {
                "success": False, 
                "result": "Error: start_frame and end_frame are REQUIRED. "
                          "Full video description is already in the initial context. "
                          "Use this tool only to analyze a specific segment in detail. "
                          f"Valid frame range: 0-{video_context.total_frames - 1}"
            }
        
        # Validate frame range
        if start_frame < 0 or start_frame >= video_context.total_frames:
            return {"success": False, "result": f"Invalid start_frame={start_frame}. Valid range: 0-{video_context.total_frames - 1}"}
        if end_frame < start_frame or end_frame >= video_context.total_frames:
            return {"success": False, "result": f"Invalid end_frame={end_frame}. Must be >= start_frame and < {video_context.total_frames}"}
        
        # Check cache first
        if self._tool_cache:
            cached_result = self._tool_cache.get_description(
                video_path=video_context.video_path,
                start_frame=start_frame,
                end_frame=end_frame,
            )
            
            if cached_result is not None:
                self.logger.info(f"InternVideo description cache HIT: frames {start_frame}-{end_frame}")
                # Format cached result
                formatted = f"[Video Clip Description (frames {start_frame}-{end_frame})]\n{cached_result.get('description', '')}"
                return {
                    "success": True,
                    "result": formatted,
                    "cache_hit": True,
                }
        
        # Lazy initialize the tool using _init_tool for proper GPU management
        if "internvideo_description" not in self._tool_instances or self._tool_instances.get("internvideo_description") is None:
            self._init_tool("internvideo_description", force_retry=True)
        
        tool = self._tool_instances.get("internvideo_description")
        if tool is None:
            return {"success": False, "result": "InternVideo Description tool not initialized. Check GPU memory."}
        
        # Log the frame range for visibility
        frame_range_str = f"start_frame={start_frame}, end_frame={end_frame}"
        self.logger.info(f"InternVideo Clip Description: {frame_range_str}")
        
        try:
            # Execute with video path and optional frame range
            result = tool(
                video=Video(path=video_context.video_path),
                start_frame=start_frame,
                end_frame=end_frame,
            )
            
            if "error" in result:
                return {"success": False, "result": f"InternVideo Description Error: {result['error']}"}
            
            # Cache the result
            if self._tool_cache:
                self._tool_cache.set_description(
                    video_path=video_context.video_path,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    result=result,
                )
                self.logger.info(f"Cached InternVideo description for frames {start_frame}-{end_frame}")
            
            # Format the output (includes frame range info)
            formatted = InternVideoDescription.format_output_for_agent(result)
            
            return {
                "success": True,
                "result": formatted,
            }
            
        except Exception as e:
            self.logger.error(f"InternVideo Description failed: {e}")
            return {"success": False, "result": f"InternVideo Description failed: {str(e)}"}
    
    def caption_frames(
        self,
        video_context: Any,
        frame_indices: List[int],
        detail_level: str = "short",
        save_frames: bool = True,
    ) -> Dict[int, str]:
        """
        Caption multiple frames (used for initial frame captioning) with caching support.
        
        Args:
            video_context: VideoContext object
            frame_indices: Indices of frames to caption
            detail_level: Caption detail level
            save_frames: If True, save frame data to video_context.frames
        
        Returns:
            Dict mapping frame index to caption
        """
        if not self._initialized:
            self.initialize()
        
        # Model key for caching (based on current captioner configuration)
        model_key = self._get_captioner_model_key()
        
        # Check cache first for all frames
        captions = {}
        indices_to_process = []
        
        if self._tool_cache:
            cached_captions = self._tool_cache.get_captions_batch(
                video_path=video_context.video_path,
                frame_indices=frame_indices,
                model=model_key,
                detail_level=detail_level,
            )
            
            for idx in frame_indices:
                cached = cached_captions.get(idx)
                if cached is not None:
                    captions[idx] = cached
                    video_context.frame_captions[idx] = cached
                    if idx not in video_context.sampled_indices:
                        video_context.sampled_indices.append(idx)
                else:
                    indices_to_process.append(idx)
            
            if captions:
                self.logger.info(f"Caption cache hit: {len(captions)}/{len(frame_indices)} frames")
        else:
            indices_to_process = list(frame_indices)
        
        # If all frames were cached, return early
        if not indices_to_process:
            self.logger.debug(f"[DEBUG] All {len(captions)} captions from cache")
            return captions
        
        from video_agent_tools.utils.video import extract_frames
        frames = extract_frames(video_context.video_path, indices_to_process)
        
        self.logger.debug(f"[DEBUG] Extracted {len(frames)} frames from video (need to caption)")
        
        new_captions = {}
        for idx, frame in frames.items():
            # Save frame data to video_context for later use (e.g., saving to disk)
            if save_frames:
                video_context.frames[idx] = frame
            
            self.logger.debug(f"[DEBUG] Captioning frame {idx}, shape={frame.shape if hasattr(frame, 'shape') else 'unknown'}")
            caption = self._caption_single_frame(frame, detail_level)
            self.logger.debug(f"[DEBUG] Frame {idx} caption length: {len(caption) if caption else 0}")
            if caption:
                captions[idx] = caption
                new_captions[idx] = caption
                video_context.frame_captions[idx] = caption
                if idx not in video_context.sampled_indices:
                    video_context.sampled_indices.append(idx)
        
        # Cache new captions
        if self._tool_cache and new_captions:
            self._tool_cache.set_captions_batch(
                video_path=video_context.video_path,
                captions=new_captions,
                model=model_key,
                detail_level=detail_level,
            )
            self.logger.info(f"Cached {len(new_captions)} new captions")
        
        self.logger.debug(f"[DEBUG] Total captions: {len(captions)} ({len(new_captions)} new)")
        return captions
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get tool cache statistics.
        
        Returns:
            Dict with cache hit/miss counts and hit rate
        """
        if self._tool_cache:
            return self._tool_cache.get_stats()
        return {"enabled": False}
    
    def log_cache_stats(self) -> None:
        """Log tool cache statistics."""
        if self._tool_cache:
            self._tool_cache.log_stats()


# validate_tool_args is now imported from tools.interface

