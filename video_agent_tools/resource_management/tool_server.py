"""
Tool Server for Centralized GPU Tool Management (Multiprocessing Mode)

Provides a dedicated server process that manages all GPU tool instances,
allowing lightweight worker processes to access tools via IPC.

Key Features:
- Lazy loading: Tools loaded to GPU only on first request
- Dynamic scaling: Auto-replicate high-demand tools based on queue pressure
- Multi-GPU distribution: Spread tools across available GPUs
- Request queuing: FIFO queue per tool with worker-specific responses
"""

# MUST be set before any imports that might load numexpr (used by pandas, numpy, etc.)
import os
os.environ["NUMEXPR_MAX_THREADS"] = "64"
os.environ["NUMEXPR_NUM_THREADS"] = "8"

import sys
import gc
import time
import uuid
import queue
import logging
import traceback
from typing import Dict, Any, List, Optional
from datetime import datetime
from threading import Lock
from multiprocessing import Process, Queue, Event
from collections import defaultdict

from .constants import (
    TOOL_MEMORY_ESTIMATES,
    SHARED_MODEL_GROUPS,
    MEMORY_BUFFER_MB,
    QUEUE_PRESSURE_THRESHOLD,
)
from .core import GPUInfo, ToolInstance, MessageType, ToolRequest, ToolResponse

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class ToolServer:
    """
    Centralized Tool Server that manages GPU tools for all workers.
    
    Runs in a dedicated process. Workers communicate via queues.
    """
    
    def __init__(
        self,
        request_queue: Queue,
        response_queues: Dict[int, Queue],
        shutdown_event: Event,
        enabled_tools: List[str],
        captioner: str = "gpt-4o-mini",
        log_queue: Queue = None,
        output_dir: str = None,
        max_view_frames: int = 8,
        default_sample_frames: int = 16,
        min_sample_frames: int = 1,
        max_sample_frames: int = 32,
    ):
        """
        Initialize ToolServer.
        
        Args:
            request_queue: Queue for receiving tool requests from workers
            response_queues: Dict mapping worker_id to their response queue
            shutdown_event: Event to signal shutdown
            enabled_tools: List of tool names that may be requested
            captioner: Captioner model - 'omni-captioner' for local OmniCaptioner,
                      or API model name (e.g., 'gpt-4o-mini', 'x-ai/grok-4-1-fast-reasoning')
            log_queue: Optional queue for centralized logging
            output_dir: Directory for resource management log file
        """
        self.request_queue = request_queue
        self.response_queues = response_queues
        self.shutdown_event = shutdown_event
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
        
        self.log_queue = log_queue
        self.output_dir = output_dir
        
        # Frame control
        self.max_view_frames = max_view_frames
        self.default_sample_frames = default_sample_frames
        self.min_sample_frames = min_sample_frames
        self.max_sample_frames = max_sample_frames
        
        # Tool instances: tool_name -> List[ToolInstance]
        self._tool_instances: Dict[str, List[ToolInstance]] = defaultdict(list)
        
        # GPU tracking
        self._gpus: Dict[int, GPUInfo] = {}
        
        # Captioner instances
        self._captioner = None
        self._api_captioner = None
        
        # Thread safety
        self._lock = Lock()
        
        # Statistics
        self._stats = {
            "requests_processed": 0,
            "requests_waiting": 0,
            "tools_loaded": 0,
            "tools_replicated": 0,
            "tools_evicted": 0,
        }
        
        # Pending requests per tool (for replication decisions)
        self._pending_requests: Dict[str, int] = defaultdict(int)
        
        # Usage tracking for eviction decisions
        self._tool_usage_count: Dict[str, int] = defaultdict(int)  # How many times used
        self._tool_last_used: Dict[str, float] = {}  # Last usage timestamp
        
        # Initialize logging first (before anything that needs logging)
        self.logger = logging.getLogger("ToolServer")
        self._resource_log_file = None
        self._init_resource_log()
        
        # Tool cache for caption results (must be after logger initialization)
        self._tool_cache = None
        self._init_tool_cache()
    
    def _init_tool_cache(self):
        """Initialize tool cache for caption results."""
        try:
            from video_agent_tools.utils.tool_cache import get_tool_cache
            self._tool_cache = get_tool_cache(
                enabled=True,
                logger=self.logger,
            )
            self._log("info", "Tool cache initialized for ToolServer")
        except Exception as e:
            self._log("warning", f"Failed to initialize tool cache: {e}")
            self._tool_cache = None
    
    def _get_captioner_model_key(self) -> str:
        """Get the model key for caching based on current captioner configuration."""
        if self.use_api_captioner:
            return f"api_{self.api_captioner_model}"
        else:
            return self.captioner_model
    
    def _init_resource_log(self):
        """Initialize dedicated resource management log file."""
        if self.output_dir:
            log_path = os.path.join(self.output_dir, "resource_management.log")
            self._resource_log_file = open(log_path, 'a', encoding='utf-8')
            self._resource_log(f"Resource Management Log initialized")
    
    def _resource_log(self, message: str):
        """Write to dedicated resource management log only (not main log)."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted = f"{timestamp} - {message}"
        
        if self._resource_log_file:
            self._resource_log_file.write(formatted + "\n")
            self._resource_log_file.flush()
        
        # Also print to console for visibility
        print(f"[RESOURCE] {formatted}")
    
    def _log(self, level: str, message: str):
        """Log to resource management log only (not main evaluation log)."""
        # Log to resource file instead of main log
        self._resource_log(f"[{level.upper()}] {message}")
    
    def _init_gpu_info(self):
        """Initialize GPU information."""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            self._log("warning", "CUDA not available")
            return
        
        num_gpus = torch.cuda.device_count()
        self._log("info", f"Initializing ToolServer with {num_gpus} GPU(s)")
        
        for i in range(num_gpus):
            device_str = f"cuda:{i}"
            props = torch.cuda.get_device_properties(i)
            total_memory_mb = props.total_memory / (1024 ** 2)
            
            self._gpus[i] = GPUInfo(
                device_id=i,
                device_str=device_str,
                total_memory_mb=total_memory_mb,
            )
            
            self._log("info", f"GPU {i}: {props.name}, Memory: {total_memory_mb:.0f}MB")
    
    def _update_gpu_memory(self):
        """Update GPU memory statistics using actual free memory."""
        if not TORCH_AVAILABLE:
            return
        
        for gpu_id, gpu_info in self._gpus.items():
            try:
                # Use mem_get_info for actual free memory (accounts for all processes)
                free_bytes, total_bytes = torch.cuda.mem_get_info(gpu_id)
                gpu_info.allocated_memory_mb = (total_bytes - free_bytes) / (1024 ** 2)
            except Exception:
                # Fallback to memory_allocated (only this process)
                try:
                    allocated = torch.cuda.memory_allocated(gpu_id) / (1024 ** 2)
                    gpu_info.allocated_memory_mb = allocated
                except:
                    pass
    
    def _select_device_for_tool(self, tool_name: str, prefer_new_gpu: bool = False) -> str:
        """Select best GPU for loading a tool.
        
        Args:
            tool_name: Name of the tool to load
            prefer_new_gpu: If True, prefer a GPU that doesn't already have this tool
        """
        self._update_gpu_memory()
        
        memory_needed = TOOL_MEMORY_ESTIMATES.get(tool_name, 2000)
        
        # For replication - prefer GPUs without this tool already
        if prefer_new_gpu:
            existing_gpus = set()
            for inst in self._tool_instances.get(tool_name, []):
                gpu_id = int(inst.device.split(":")[1]) if ":" in inst.device else 0
                existing_gpus.add(gpu_id)
            
            # Find GPU without this tool that has enough memory
            for gpu_id, gpu_info in self._gpus.items():
                if gpu_id not in existing_gpus:
                    available = gpu_info.free_memory_mb - MEMORY_BUFFER_MB
                    if available >= memory_needed:
                        self._resource_log(f"SELECT '{tool_name}' on NEW GPU{gpu_id} for replication (need {memory_needed}MB, free {available:.0f}MB)")
                        return gpu_info.device_str
        
        # Check if tool shares model with DIFFERENT already loaded tool
        for group_name, tools in SHARED_MODEL_GROUPS.items():
            if tool_name in tools:
                for other_tool in tools:
                    if other_tool != tool_name and self._tool_instances.get(other_tool):
                        device = self._tool_instances[other_tool][0].device
                        self._resource_log(f"SELECT '{tool_name}' shares model with '{other_tool}', using {device}")
                        return device
        
        # First, try to find an empty GPU (no tools loaded)
        for gpu_id, gpu_info in self._gpus.items():
            if not gpu_info.tools_loaded:
                available = gpu_info.free_memory_mb - MEMORY_BUFFER_MB
                if available >= memory_needed:
                    self._resource_log(f"SELECT '{tool_name}' on empty GPU{gpu_id} (need {memory_needed}MB, free {available:.0f}MB)")
                    return gpu_info.device_str
        
        # Find GPU with most free memory that can fit the tool
        best_gpu = None
        best_free = -1
        
        for gpu_id, gpu_info in self._gpus.items():
            available = gpu_info.free_memory_mb - MEMORY_BUFFER_MB
            if available >= memory_needed and available > best_free:
                best_free = available
                best_gpu = gpu_id
        
        if best_gpu is not None:
            device = self._gpus[best_gpu].device_str
            gpu_info = self._gpus[best_gpu]
            self._resource_log(f"SELECT '{tool_name}' on GPU{best_gpu} (need {memory_needed}MB, free {best_free:.0f}MB, tools: {gpu_info.tools_loaded})")
            return device
        
        # No GPU has enough space - try eviction
        self._resource_log(f"NO_GPU '{tool_name}' - no GPU has {memory_needed}MB free, attempting eviction")
        
        # Try to evict least-used tool to make space
        if self._try_evict_for_tool(tool_name, memory_needed):
            # Retry finding a GPU after eviction
            self._update_gpu_memory()
            for gpu_id, gpu_info in self._gpus.items():
                available = gpu_info.free_memory_mb - MEMORY_BUFFER_MB
                if available >= memory_needed:
                    self._resource_log(f"SELECT '{tool_name}' on GPU{gpu_id} after eviction (need {memory_needed}MB, free {available:.0f}MB)")
                    return gpu_info.device_str
        
        return None
    
    def _load_tool(self, tool_name: str, prefer_new_gpu: bool = False) -> Optional[ToolInstance]:
        """Load a tool instance onto a GPU.
        
        Args:
            tool_name: Name of the tool to load
            prefer_new_gpu: If True, prefer loading on a different GPU (for replication)
        """
        from tools.interface import get_interface
        
        self._log("info", f"Loading tool: {tool_name}" + (" (prefer new GPU)" if prefer_new_gpu else ""))
        
        # Handle captioner specially
        if tool_name in ["caption_image", "view_frame", "detailed_captioning"]:
            return self._get_captioner_instance(tool_name)
        
        # Always clear GPU memory and sync before loading
        self._clear_gpu_memory()
        self._update_gpu_memory()
        
        target_device = self._select_device_for_tool(tool_name, prefer_new_gpu=prefer_new_gpu)
        
        # No GPU has enough memory - return None to signal queuing
        if target_device is None:
            self._resource_log(f"SKIP_LOAD '{tool_name}' - no GPU has enough memory, will queue")
            return None
        
        # Heavy tools need extra care
        heavy_tools = {
            "detect_objects", "detect_all_objects", "temporal_qa", "temporal_spatial_qa",
            "internvideo_general_qa", "internvideo_description", "targeting_vqa",
            "temporal_sample_frames", "temporal_spatial_sample_frames"
        }
        
        if tool_name in heavy_tools:
            # Synchronize CUDA operations before loading
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.synchronize()
                gc.collect()
                torch.cuda.empty_cache()
        
        try:
            interface_cls = get_interface(tool_name)
            
            tools_with_device_param = {
                "internvideo_general_qa", "internvideo_description",
                "detect_objects", "detect_all_objects",
                "temporal_sample_frames", "temporal_spatial_sample_frames",
                "describe_region",
            }
            
            if tool_name in tools_with_device_param:
                instance = interface_cls(device=target_device)
            else:
                instance = interface_cls()
            
            instance.initialize()
            
            memory_mb = TOOL_MEMORY_ESTIMATES.get(tool_name, 2000)
            
            tool_instance = ToolInstance(
                tool_name=tool_name,
                instance=instance,
                device=target_device,
                instance_id=str(uuid.uuid4())[:8],
                memory_mb=memory_mb,
            )
            
            self._tool_instances[tool_name].append(tool_instance)
            
            # Update GPU tracking
            gpu_id = int(target_device.split(":")[1]) if ":" in target_device else 0
            if gpu_id in self._gpus:
                self._gpus[gpu_id].tools_loaded.append(tool_name)
            
            self._stats["tools_loaded"] += 1
            
            # Detailed resource logging
            self._update_gpu_memory()
            gpu_id = int(target_device.split(":")[1]) if ":" in target_device else 0
            gpu_info = self._gpus.get(gpu_id)
            
            # Log to dedicated resource file
            instances_count = len(self._tool_instances[tool_name])
            if gpu_info:
                self._resource_log(
                    f"LOAD '{tool_name}' (instance #{instances_count}) on {target_device} | "
                    f"Est.Memory: {memory_mb}MB | "
                    f"GPU{gpu_id}: {gpu_info.allocated_memory_mb:.0f}MB / {gpu_info.total_memory_mb:.0f}MB used | "
                    f"Tools on GPU{gpu_id}: {gpu_info.tools_loaded}"
                )
            else:
                self._resource_log(f"LOAD '{tool_name}' on {target_device} | Est.Memory: {memory_mb}MB")
            
            return tool_instance
            
        except Exception as e:
            self._log("error", f"Failed to load tool '{tool_name}': {e}")
            return None
    
    def _get_captioner_instance(self, tool_name: str) -> Optional[ToolInstance]:
        """Get or create captioner instance."""
        from tools.interface import OmniCaptionerCaptioning, APICaptioning
        
        if self.use_api_captioner or tool_name == "detailed_captioning":
            if self._api_captioner is None:
                self._log("info", f"Initializing API captioner: {self.api_captioner_model}")
                self._api_captioner = APICaptioning(
                    use_aiml=True,
                    default_model=self.api_captioner_model
                )
                self._api_captioner.initialize()
                self._log("info", "API captioner initialized")
            
            return ToolInstance(
                tool_name=tool_name,
                instance=self._api_captioner,
                device="cpu",
                instance_id="api",
                memory_mb=0,
            )
        else:
            if self._captioner is None:
                target_device = self._select_device_for_tool("omni_captioner")
                self._update_gpu_memory()
                gpu_id = int(target_device.split(":")[1]) if ":" in target_device else 0
                gpu_info = self._gpus.get(gpu_id)
                
                self._resource_log(
                    f"LOAD 'omni_captioner' on {target_device} | Est.Memory: 3500MB | "
                    f"GPU{gpu_id} Before: {gpu_info.allocated_memory_mb:.0f}MB used" if gpu_info else 
                    f"LOAD 'omni_captioner' on {target_device}"
                )
                self._captioner = OmniCaptionerCaptioning(
                    model_path=self.captioner_model,
                    device=target_device
                )
                self._captioner.initialize()
                self._log("info", "OmniCaptioner loaded")
            
            return ToolInstance(
                tool_name=tool_name,
                instance=self._captioner,
                device="cuda:0",
                instance_id="omni",
                memory_mb=3500,
            )
    
    def _get_available_instance(self, tool_name: str, silent: bool = False) -> Optional[ToolInstance]:
        """Get an available tool instance, loading if necessary.
        
        Args:
            tool_name: Name of the tool to get
            silent: If True, suppress logging (used in wait loops)
        """
        with self._lock:
            # Track pending requests for this tool (only on initial request)
            if not silent:
                self._pending_requests[tool_name] += 1
            pending = self._pending_requests[tool_name]
            
            instances = self._tool_instances.get(tool_name, [])
            total = len(instances)
            busy = sum(1 for i in instances if i.in_use)
            available = total - busy
            
            if not silent:
                self._resource_log(f"GET_INSTANCE '{tool_name}' - {total} total, {busy} busy, {available} free, {pending} pending")
            
            # Find available instance
            for inst in instances:
                if not inst.in_use:
                    inst.mark_in_use()
                    if not silent:
                        self._pending_requests[tool_name] -= 1
                    self._resource_log(f"ASSIGN '{tool_name}' instance={inst.instance_id} on {inst.device} ({busy+1}/{total} busy)")
                    return inst
            
            # No instances at all - try to load first one (only on non-silent calls)
            if not instances:
                if silent:
                    # In silent/wait mode, don't try to load - just return None
                    return None
                    
                self._resource_log(f"FIRST_LOAD '{tool_name}' - no instances, {pending} pending requests")
                new_instance = self._load_tool(tool_name)
                if new_instance:
                    new_instance.mark_in_use()
                    self._pending_requests[tool_name] -= 1
                    return new_instance
                # Load failed (probably no GPU memory) - don't decrement pending, let caller handle retry
                return None
            
            # All instances busy - check if we should replicate (only on initial request)
            if not silent:
                self._resource_log(f"ALL_BUSY '{tool_name}' - {total} instances busy, {pending} pending")
                
                if self._should_replicate(tool_name):
                    self._resource_log(f"REPLICATE '{tool_name}' - creating new instance on another GPU")
                    # prefer_new_gpu=True to distribute across GPUs
                    new_instance = self._load_tool(tool_name, prefer_new_gpu=True)
                    if new_instance:
                        new_instance.mark_in_use()
                        self._stats["tools_replicated"] += 1
                        self._pending_requests[tool_name] -= 1
                        return new_instance
                
                self._pending_requests[tool_name] -= 1
            
            return None
    
    def _should_replicate(self, tool_name: str) -> bool:
        """Check if we should create another instance of a tool.
        
        Replicate when:
        1. All existing instances are busy
        2. There's enough GPU memory for a new instance
        """
        from .constants import TOOL_MAX_INSTANCES
        
        instances = self._tool_instances.get(tool_name, [])
        total_instances = len(instances)
        busy_count = sum(1 for i in instances if i.in_use)
        
        # Check if we've hit max instances (None means unlimited)
        max_instances = TOOL_MAX_INSTANCES.get(tool_name)
        if max_instances is not None and total_instances >= max_instances:
            return False
        
        # Replicate if all instances are busy
        if busy_count < total_instances:
            return False  # Some instances are free
        
        # All instances are busy - check if we have memory on any GPU
        self._update_gpu_memory()
        memory_needed = TOOL_MEMORY_ESTIMATES.get(tool_name, 2000)
        
        for gpu_id, gpu_info in self._gpus.items():
            if gpu_info.free_memory_mb - MEMORY_BUFFER_MB >= memory_needed:
                self._resource_log(
                    f"REPLICATE '{tool_name}': {busy_count}/{total_instances} busy, "
                    f"GPU{gpu_id} has {gpu_info.free_memory_mb:.0f}MB free (need {memory_needed}MB)"
                )
                return True
        
        # Don't log every time - too noisy
        return False
    
    def _release_instance(self, tool_instance: ToolInstance):
        """Release a tool instance back to the pool."""
        with self._lock:
            tool_instance.mark_available()
            # Update usage tracking
            self._tool_usage_count[tool_instance.tool_name] += 1
            self._tool_last_used[tool_instance.tool_name] = time.time()
            instances = self._tool_instances.get(tool_instance.tool_name, [])
            busy = sum(1 for i in instances if i.in_use)
            self._resource_log(f"RELEASE '{tool_instance.tool_name}' instance={tool_instance.instance_id} ({busy}/{len(instances)} busy)")
    
    def _try_evict_for_tool(self, needed_tool: str, memory_needed: float) -> bool:
        """Try to evict least-used tool(s) to make space for a new tool.
        
        Returns True if eviction was successful and space may be available.
        Strategy:
        1. First try to evict idle instances (not in use)
        2. Only evict GPU-based tools (memory > 0)
        3. Prioritize tools with low pending requests and older last-used time
        """
        with self._lock:
            # Collect individual idle instances that can be evicted
            eviction_candidates = []
            
            for tool_name, instances in self._tool_instances.items():
                if tool_name == needed_tool:
                    continue
                
                # Skip critical tools (captioner)
                if tool_name in ["omni_captioner", "api_captioner"]:
                    continue
                
                # Skip API-based tools with no GPU memory
                tool_memory = TOOL_MEMORY_ESTIMATES.get(tool_name, 0)
                if tool_memory == 0:
                    continue
                
                # Calculate base score for this tool
                pending = self._pending_requests.get(tool_name, 0)
                usage = self._tool_usage_count.get(tool_name, 0)
                last_used = self._tool_last_used.get(tool_name, 0)
                age = time.time() - last_used if last_used else float('inf')
                
                # Base score: higher pending = less likely to evict
                base_score = pending * 100 + usage - (age / 60)
                
                # Find idle instances (not in use)
                for inst in instances:
                    if not inst.in_use:
                        # Each idle instance is a candidate
                        inst_age = time.time() - inst.last_used.timestamp() if inst.last_used else float('inf')
                        inst_score = base_score - (inst_age / 60)  # Older instances have lower score
                        
                        eviction_candidates.append({
                            "tool_name": tool_name,
                            "instance": inst,
                            "score": inst_score,
                            "memory_mb": inst.memory_mb,
                            "pending": pending,
                            "usage": usage,
                            "device": inst.device,
                        })
            
            if not eviction_candidates:
                self._resource_log(f"EVICT_FAIL: no idle GPU instances to evict for '{needed_tool}'")
                return False
            
            # Sort by score (lowest first = best to evict)
            # Also prioritize instances with actual memory
            eviction_candidates.sort(key=lambda x: (x["score"], -x["memory_mb"]))
            
            # Evict instances until we have enough memory
            evicted_memory = 0
            evicted_count = 0
            for candidate in eviction_candidates:
                if evicted_memory >= memory_needed:
                    break
                
                tool_name = candidate["tool_name"]
                inst = candidate["instance"]
                
                self._resource_log(
                    f"EVICT_INSTANCE '{tool_name}' id={inst.instance_id} on {candidate['device']} "
                    f"(score={candidate['score']:.1f}, pending={candidate['pending']}, "
                    f"memory={candidate['memory_mb']}MB) to make room for '{needed_tool}'"
                )
                
                # Unload this specific instance
                self._unload_instance(inst)
                evicted_memory += candidate["memory_mb"]
                evicted_count += 1
            
            if evicted_count > 0:
                self._stats["tools_evicted"] += evicted_count
                # Clear GPU cache after eviction
                self._clear_gpu_memory()
                self._resource_log(f"EVICT_DONE: freed ~{evicted_memory}MB by evicting {evicted_count} instance(s)")
            
            return evicted_memory > 0
    
    def _unload_instance(self, tool_instance: ToolInstance):
        """Unload a tool instance from GPU and free memory."""
        try:
            tool_name = tool_instance.tool_name
            device = tool_instance.device
            memory_mb = tool_instance.memory_mb
            
            # Remove from GPU tracking
            gpu_id = int(device.split(":")[1]) if ":" in device else 0
            if gpu_id in self._gpus and tool_name in self._gpus[gpu_id].tools_loaded:
                self._gpus[gpu_id].tools_loaded.remove(tool_name)
            
            # Remove from tool instances list
            if tool_name in self._tool_instances:
                self._tool_instances[tool_name] = [
                    inst for inst in self._tool_instances[tool_name]
                    if inst.instance_id != tool_instance.instance_id
                ]
            
            # Try to properly cleanup the instance
            instance = tool_instance.instance
            
            # Call cleanup method if available
            if hasattr(instance, 'cleanup'):
                try:
                    instance.cleanup()
                except Exception as e:
                    self._resource_log(f"UNLOAD cleanup error for '{tool_name}': {e}")
            
            # Try to move model to CPU first (helps with memory release)
            if hasattr(instance, 'model') and TORCH_AVAILABLE:
                try:
                    if hasattr(instance.model, 'cpu'):
                        instance.model.cpu()
                except Exception:
                    pass
            
            # Delete all model references
            if hasattr(instance, 'model'):
                try:
                    del instance.model
                except Exception:
                    pass
            
            # Delete the instance
            del tool_instance.instance
            del instance
            
            # Force garbage collection
            gc.collect()
            
            # Clear CUDA cache for the specific device
            if TORCH_AVAILABLE and torch.cuda.is_available() and device.startswith("cuda"):
                try:
                    with torch.cuda.device(device):
                        torch.cuda.empty_cache()
                except Exception:
                    torch.cuda.empty_cache()
            
            self._resource_log(f"UNLOAD '{tool_name}' id={tool_instance.instance_id} from {device} (~{memory_mb}MB freed)")
            
        except Exception as e:
            self._resource_log(f"ERROR unloading '{tool_instance.tool_name}': {e}")
    
    def _clear_gpu_memory(self):
        """Clear GPU memory cache."""
        if TORCH_AVAILABLE and torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
    
    def _execute_tool(self, request: ToolRequest) -> ToolResponse:
        """Execute a tool request."""
        start_time = time.time()
        tool_name = request.tool_name
        video_id = request.video_context_data.get("video_id", "")[:8]
        
        # Log request arrival
        self._stats["requests_waiting"] += 1
        queue_size = self._stats["requests_waiting"]
        self._resource_log(f"REQUEST '{tool_name}' video={video_id} worker={request.worker_id} (queue={queue_size})")
        
        tool_instance = self._get_available_instance(tool_name)
        
        if tool_instance is None:
            # Wait for available instance - use longer timeout for shared resource access
            self._resource_log(f"WAITING '{tool_name}' video={video_id} - all instances busy, entering wait queue")
            max_wait = 600  # 10 minutes - longer timeout for heavy tools like internvideo
            wait_start = time.time()
            wait_count = 0
            
            while tool_instance is None and (time.time() - wait_start) < max_wait:
                time.sleep(1.0)  # Wait 1 second between checks
                wait_count += 1
                
                # Every 10 seconds, try loading or eviction
                if wait_count % 10 == 0:
                    elapsed = wait_count
                    self._resource_log(f"STILL_WAITING '{tool_name}' video={video_id} ({elapsed}s / {max_wait}s)")
                    
                    # Check if we have high queue pressure - trigger eviction
                    pending = self._pending_requests.get(tool_name, 0)
                    if pending >= 3 and not self._tool_instances.get(tool_name):
                        # Many requests waiting but no instances - try eviction
                        memory_needed = TOOL_MEMORY_ESTIMATES.get(tool_name, 2000)
                        self._resource_log(f"HIGH_PRESSURE '{tool_name}': {pending} pending, triggering eviction")
                        self._try_evict_for_tool(tool_name, memory_needed)
                    
                    # Try to get instance (may trigger loading after eviction)
                    tool_instance = self._get_available_instance(tool_name, silent=False)
                else:
                    # Use silent=True to avoid log spam
                    tool_instance = self._get_available_instance(tool_name, silent=True)
            
            if tool_instance is None:
                self._stats["requests_waiting"] -= 1
                wait_time = time.time() - wait_start
                self._resource_log(f"TIMEOUT '{tool_name}' video={video_id} after {wait_time:.1f}s")
                return ToolResponse(
                    request_id=request.request_id,
                    worker_id=request.worker_id,
                    success=False,
                    error=f"Timeout waiting for tool '{tool_name}' instance",
                    tool_name=tool_name,
                )
            else:
                wait_time = time.time() - wait_start
                self._resource_log(f"ACQUIRED '{tool_name}' video={video_id} after {wait_time:.1f}s wait")
        
        self._stats["requests_waiting"] -= 1
        self._resource_log(f"RUN '{tool_name}' on {tool_instance.device} instance={tool_instance.instance_id} video={video_id}")
        
        try:
            result = self._execute_tool_internal(
                tool_instance,
                request.tool_args,
                request.video_context_data,
            )
            
            duration_ms = (time.time() - start_time) * 1000
            
            self._resource_log(f"DONE '{tool_name}' in {duration_ms:.0f}ms success={result.get('success', False)}")
            
            return ToolResponse(
                request_id=request.request_id,
                worker_id=request.worker_id,
                success=result.get("success", False),
                result=result,
                tool_name=tool_name,
                duration_ms=duration_ms,
            )
            
        except Exception as e:
            import traceback
            error_detail = f"{type(e).__name__}: {e}"
            tb = traceback.format_exc()
            self._log("error", f"Tool execution error for '{tool_name}': {error_detail}")
            self._resource_log(f"ERROR '{tool_name}' video={video_id}: {error_detail}\n{tb}")
            return ToolResponse(
                request_id=request.request_id,
                worker_id=request.worker_id,
                success=False,
                error=error_detail,
                tool_name=tool_name,
                duration_ms=(time.time() - start_time) * 1000,
            )
        finally:
            if tool_instance:
                self._release_instance(tool_instance)
    
    def _execute_tool_internal(
        self,
        tool_instance: ToolInstance,
        tool_args: Dict[str, Any],
        video_context_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Internal tool execution with OOM error handling."""
        import gc
        
        tool_name = tool_instance.tool_name
        instance = tool_instance.instance
        
        video_path = video_context_data.get("video_path", "")
        total_frames = video_context_data.get("total_frames", 0)
        fps = video_context_data.get("fps", 30.0)
        sampled_indices = video_context_data.get("sampled_indices", [])
        
        try:
            # Route to appropriate handler
            if tool_name in ["caption_image", "detailed_captioning"]:
                return self._execute_caption(instance, tool_args, video_path, total_frames, fps, sampled_indices)
            elif tool_name == "view_frame":
                return self._execute_view_frame(instance, tool_args, video_path, total_frames, fps, sampled_indices)
            elif tool_name == "temporal_sample_frames":
                return self._execute_temporal_sample(instance, tool_args, video_path, total_frames, fps, sampled_indices)
            elif tool_name == "temporal_spatial_sample_frames":
                return self._execute_temporal_spatial_sample(instance, tool_args, video_path, total_frames, fps, sampled_indices)
            elif tool_name in ["detect_objects", "detect_all_objects"]:
                return self._execute_detection(instance, tool_name, tool_args, video_path, total_frames, fps)
            elif tool_name == "describe_region":
                return self._execute_describe_region(instance, tool_args, video_path, total_frames, fps)
            elif tool_name in ["temporal_qa", "temporal_spatial_qa"]:
                return self._execute_qa(instance, tool_name, tool_args, video_path, total_frames, fps, sampled_indices)
            elif tool_name in ["general_vqa", "targeting_vqa"]:
                return self._execute_vqa(instance, tool_name, tool_args, video_path, total_frames, fps)
            elif tool_name in ["internvideo_general_qa", "internvideo_description"]:
                return self._execute_internvideo(instance, tool_name, tool_args, video_path)
            else:
                return {"success": False, "result": f"Unknown tool: {tool_name}"}
                
        except torch.cuda.OutOfMemoryError as e:
            # Handle GPU OOM - clean up, trigger eviction, and return error
            gc.collect()
            if TORCH_AVAILABLE:
                torch.cuda.empty_cache()
            
            self._resource_log(f"OOM during '{tool_name}' execution: {e}")
            
            # Trigger eviction to free up memory for retry
            # This helps the retry attempt have a better chance of success
            memory_needed = TOOL_MEMORY_ESTIMATES.get(tool_name, 2000)
            self._resource_log(f"OOM_EVICT: triggering eviction for '{tool_name}' (need {memory_needed}MB)")
            evicted = self._try_evict_for_tool(tool_name, memory_needed)
            if evicted:
                self._resource_log(f"OOM_EVICT: eviction successful, retry may succeed")
                # Clear cache again after eviction
                gc.collect()
                if TORCH_AVAILABLE:
                    torch.cuda.empty_cache()
            else:
                self._resource_log(f"OOM_EVICT: no tools available for eviction")
            
            return {"success": False, "result": f"GPU memory insufficient during {tool_name} execution", "error": "OOM"}
        except Exception as e:
            # Catch any other errors and return gracefully
            self._resource_log(f"Error in '{tool_name}': {type(e).__name__}: {e}")
            return {"success": False, "result": f"Tool error: {e}", "error": str(e)}
    
    # Tool execution helpers (simplified)
    
    def _execute_caption(self, captioner, args, video_path, total_frames, fps, sampled_indices):
        from video_agent_tools.utils.video import extract_frames
        
        frame_indices = args.get("frame_indices", [])
        detail_level = args.get("detail_level", "short")
        
        valid_indices = [i for i in frame_indices if 0 <= i < total_frames]
        if not valid_indices:
            return {"success": False, "result": f"Invalid frame indices. Valid range: 0-{total_frames - 1}"}
        
        # Check cache first
        captions = {}
        indices_to_process = []
        model_key = self._get_captioner_model_key()
        
        if self._tool_cache:
            cached_captions = self._tool_cache.get_captions_batch(
                video_path=video_path,
                frame_indices=valid_indices,
                model=model_key,
                detail_level=detail_level,
            )
            for idx in valid_indices:
                cached = cached_captions.get(idx)
                if cached is not None:
                    captions[idx] = cached
                else:
                    indices_to_process.append(idx)
            if captions:
                self._resource_log(f"Caption cache hit: {len(captions)}/{len(valid_indices)} frames")
        else:
            indices_to_process = valid_indices
        
        # Process uncached frames
        new_sampled = []
        new_captions = {}
        
        if indices_to_process:
            frames = extract_frames(video_path, indices_to_process)
            for idx, frame in frames.items():
                result = captioner(image=frame, detail_level=detail_level)
                caption = result.get("caption", "")
                if caption:
                    captions[idx] = caption
                    new_captions[idx] = caption
                    if idx not in sampled_indices:
                        new_sampled.append(idx)
            
            # Save new captions to cache
            if self._tool_cache and new_captions:
                self._tool_cache.set_captions_batch(
                    video_path=video_path,
                    captions=new_captions,
                    model=model_key,
                    detail_level=detail_level,
                )
                self._resource_log(f"Cached {len(new_captions)} new captions")
        
        lines = [f"[Frame {idx} @ {idx/fps:.1f}s]: {captions[idx]}" for idx in sorted(captions.keys())]
        
        return {
            "success": True,
            "result": "\n".join(lines) if lines else "No captions generated",
            "captions": captions,
            "new_sampled_indices": new_sampled,
        }
    
    def _execute_view_frame(self, captioner, args, video_path, total_frames, fps, sampled_indices):
        from video_agent_tools.utils.video import extract_frames
        
        frame_spec = args.get("frame_indices", args.get("frame_index", None))
        if frame_spec is None:
            return {"success": False, "result": "No frame_indices provided"}
        
        frame_indices = [frame_spec] if isinstance(frame_spec, int) else frame_spec
        if len(frame_indices) > self.max_view_frames:
            frame_indices = frame_indices[:self.max_view_frames]
        
        valid_indices = [i for i in frame_indices if 0 <= i < total_frames]
        if not valid_indices:
            return {"success": False, "result": f"Invalid frame indices. Valid range: 0-{total_frames - 1}"}
        
        # Check cache first
        detail_level = "short"
        captions = {}
        indices_to_process = []
        model_key = self._get_captioner_model_key()
        
        if self._tool_cache:
            cached_captions = self._tool_cache.get_captions_batch(
                video_path=video_path,
                frame_indices=valid_indices,
                model=model_key,
                detail_level=detail_level,
            )
            for idx in valid_indices:
                cached = cached_captions.get(idx)
                if cached is not None:
                    captions[idx] = cached
                else:
                    indices_to_process.append(idx)
        else:
            indices_to_process = valid_indices
        
        # Process uncached frames
        new_sampled = []
        new_captions = {}
        
        if indices_to_process:
            frames = extract_frames(video_path, indices_to_process)
            for idx, frame in frames.items():
                result = captioner(image=frame, detail_level=detail_level)
                caption = result.get("caption", "")
                if caption:
                    captions[idx] = caption
                    new_captions[idx] = caption
                    if idx not in sampled_indices:
                        new_sampled.append(idx)
            
            # Save new captions to cache
            if self._tool_cache and new_captions:
                self._tool_cache.set_captions_batch(
                    video_path=video_path,
                    captions=new_captions,
                    model=model_key,
                    detail_level=detail_level,
                )
        
        lines = [f"[Frame {idx} @ {idx/fps:.1f}s]: {captions[idx]}" for idx in sorted(captions.keys())]
        
        return {
            "success": True,
            "result": "\n".join(lines) if lines else "No frames viewed",
            "captions": captions,
            "new_sampled_indices": new_sampled,
        }
    
    def _execute_temporal_sample(self, sampler, args, video_path, total_frames, fps, sampled_indices):
        from tools.interface import Video
        from video_agent_tools.utils.video import extract_frames
        
        query = args.get("query", "")
        num_frames = args.get("num_frames", self.default_sample_frames)
        num_frames = max(self.min_sample_frames, min(num_frames, self.max_sample_frames))
        
        # Optional frame range parameters
        start_frame = args.get("start_frame", None)
        end_frame = args.get("end_frame", None)
        
        if not query:
            return {"success": False, "result": "Query is required"}
        
        video = Video(path=video_path, frame_count=total_frames, fps=fps)
        result = sampler(video=video, query=query, num_frames=num_frames, start_frame=start_frame, end_frame=end_frame)
        
        # Check for errors in sampler result
        if "error" in result:
            self._resource_log(f"Temporal sampler error: {result['error'][:200]}")
            return {"success": False, "result": f"Sampler error: {result['error']}", "error": result['error']}
        
        frame_indices = result.get("frame_indices", [])
        if not frame_indices:
            return {"success": False, "result": "No frames sampled"}
        
        # Auto-caption with caching
        detail_level = "short"
        model_key = self._get_captioner_model_key()
        captions = {}
        indices_to_process = []
        
        # Check cache first
        if self._tool_cache:
            cached_captions = self._tool_cache.get_captions_batch(
                video_path=video_path,
                frame_indices=frame_indices,
                model=model_key,
                detail_level=detail_level,
            )
            for idx in frame_indices:
                cached = cached_captions.get(idx)
                if cached is not None:
                    captions[idx] = cached
                else:
                    indices_to_process.append(idx)
            if captions:
                self._resource_log(f"Temporal sample caption cache hit: {len(captions)}/{len(frame_indices)} frames")
        else:
            indices_to_process = frame_indices
        
        # Process uncached frames
        new_captions = {}
        if indices_to_process:
            frames = extract_frames(video_path, indices_to_process)
            captioner = self._captioner if self._captioner else self._api_captioner
            if captioner:
                for idx, frame in frames.items():
                    try:
                        cap_result = captioner(image=frame, detail_level=detail_level)
                        if "error" in cap_result:
                            self._resource_log(f"Caption error at frame {idx}: {cap_result.get('error', 'unknown')[:100]}")
                            captions[idx] = ""
                        else:
                            caption = cap_result.get("caption", "")
                            captions[idx] = caption
                            if caption:
                                new_captions[idx] = caption
                    except Exception as e:
                        self._resource_log(f"Caption exception at frame {idx}: {str(e)[:100]}")
                        captions[idx] = ""
            else:
                self._resource_log("No captioner available for auto-captioning")
            
            # Save new captions to cache
            if self._tool_cache and new_captions:
                self._tool_cache.set_captions_batch(
                    video_path=video_path,
                    captions=new_captions,
                    model=model_key,
                    detail_level=detail_level,
                )
        
        new_sampled = [i for i in frame_indices if i not in sampled_indices]
        
        lines = [f"Sampled {len(frame_indices)} frames for: '{query}'"]
        for idx in sorted(frame_indices):
            lines.append(f"[Frame {idx} @ {idx/fps:.1f}s]: {captions.get(idx, '')}")
        
        return {
            "success": True,
            "result": "\n".join(lines),
            "frame_indices": frame_indices,
            "captions": captions,
            "new_sampled_indices": new_sampled,
        }
    
    def _execute_temporal_spatial_sample(self, sampler, args, video_path, total_frames, fps, sampled_indices):
        from tools.interface import Video
        from video_agent_tools.utils.video import extract_frames
        
        query = args.get("query", "")
        num_frames = args.get("num_frames", self.default_sample_frames)
        # Clamp num_frames to valid range
        num_frames = max(self.min_sample_frames, min(num_frames, self.max_sample_frames))
        
        # Optional frame range parameters
        start_frame = args.get("start_frame", None)
        end_frame = args.get("end_frame", None)
        
        if not query:
            return {"success": False, "result": "Query is required"}
        
        video = Video(path=video_path, frame_count=total_frames, fps=fps)
        result = sampler(video=video, query=query, num_frames=num_frames, start_frame=start_frame, end_frame=end_frame)
        
        # Check for errors in sampler result
        if "error" in result:
            self._resource_log(f"TStar sampler error: {result['error'][:200]}")
            return {"success": False, "result": f"Sampler error: {result['error']}", "error": result['error']}
        
        frame_indices = result.get("frame_indices", [])
        regions = result.get("regions", [])
        
        if not frame_indices:
            return {"success": False, "result": "No frames sampled"}
        
        # Auto-caption with caching
        detail_level = "short"
        model_key = self._get_captioner_model_key()
        captions = {}
        indices_to_process = []
        
        # Check cache first
        if self._tool_cache:
            cached_captions = self._tool_cache.get_captions_batch(
                video_path=video_path,
                frame_indices=frame_indices,
                model=model_key,
                detail_level=detail_level,
            )
            for idx in frame_indices:
                cached = cached_captions.get(idx)
                if cached is not None:
                    captions[idx] = cached
                else:
                    indices_to_process.append(idx)
            if captions:
                self._resource_log(f"TStar sample caption cache hit: {len(captions)}/{len(frame_indices)} frames")
        else:
            indices_to_process = frame_indices
        
        # Process uncached frames
        new_captions = {}
        if indices_to_process:
            frames = extract_frames(video_path, indices_to_process)
            captioner = self._captioner if self._captioner else self._api_captioner
            if captioner:
                for idx, frame in frames.items():
                    try:
                        cap_result = captioner(image=frame, detail_level=detail_level)
                        if "error" in cap_result:
                            self._resource_log(f"Caption error at frame {idx}: {cap_result.get('error', 'unknown')[:100]}")
                            captions[idx] = ""
                        else:
                            caption = cap_result.get("caption", "")
                            captions[idx] = caption
                            if caption:
                                new_captions[idx] = caption
                    except Exception as e:
                        self._resource_log(f"Caption exception at frame {idx}: {str(e)[:100]}")
                        captions[idx] = ""
            else:
                self._resource_log("No captioner available for auto-captioning")
            
            # Save new captions to cache
            if self._tool_cache and new_captions:
                self._tool_cache.set_captions_batch(
                    video_path=video_path,
                    captions=new_captions,
                    model=model_key,
                    detail_level=detail_level,
                )
        
        new_sampled = [i for i in frame_indices if i not in sampled_indices]
        
        lines = [f"Sampled {len(frame_indices)} frames with regions for: '{query}'"]
        for i, idx in enumerate(sorted(frame_indices)):
            region_str = f", Region: {regions[i]}" if i < len(regions) else ""
            lines.append(f"[Frame {idx} @ {idx/fps:.1f}s{region_str}]: {captions.get(idx, '')}")
        
        return {
            "success": True,
            "result": "\n".join(lines),
            "frame_indices": frame_indices,
            "regions": regions,
            "captions": captions,
            "new_sampled_indices": new_sampled,
        }
    
    def _execute_detection(self, detector, tool_name, args, video_path, total_frames, fps):
        from video_agent_tools.utils.video import extract_frames
        from tools.interface import Image
        
        frame_indices = args.get("frame_indices", [])
        valid_indices = [i for i in frame_indices if 0 <= i < total_frames]
        if not valid_indices:
            return {"success": False, "result": f"Invalid frame indices"}
        
        frames = extract_frames(video_path, valid_indices)
        
        if tool_name == "detect_objects":
            # Accept both "query" (string) and "categories" (list) as input
            query = args.get("query", "")
            categories = args.get("categories", [])
            
            # Convert query string to categories list if needed
            if query and not categories:
                # Split query by comma to create categories list
                categories = [c.strip() for c in query.split(",") if c.strip()]
            elif not categories:
                return {"success": False, "result": "Query or categories required"}
            
            # Ensure categories is a list
            if isinstance(categories, str):
                categories = [categories]
            
            all_detections = []
            for idx, frame in frames.items():
                image = Image(data=frame)
                # YOLOWorldDetection expects images (list) and categories (list)
                result = detector(images=[image], categories=categories, frame_indices=[idx])
                # Check for errors in detector result
                if "error" in result:
                    self._resource_log(f"Detection error at frame {idx}: {result['error'][:200]}")
                    continue
                # Extract detections from per_frame_detections
                per_frame = result.get("per_frame_detections", {})
                for frame_idx, dets in per_frame.items():
                    for det in dets:
                        all_detections.append({"frame_index": frame_idx, "timestamp": frame_idx/fps, **det})
            
            return {"success": True, "result": f"Found {len(all_detections)} detections", "detections": all_detections}
        else:
            all_objects = {}
            for idx, frame in frames.items():
                image = Image(data=frame)
                result = detector(image=image)
                # Check for errors in detector result
                if "error" in result:
                    self._resource_log(f"Detection error at frame {idx}: {result['error'][:200]}")
                    continue
                all_objects[idx] = {"timestamp": idx/fps, "objects": result.get("objects", [])}
            
            return {"success": True, "result": "Detected objects", "all_objects": all_objects}
    
    def _execute_describe_region(self, describer, args, video_path, total_frames, fps):
        from video_agent_tools.utils.video import extract_frames
        from tools.interface import Image
        
        frame_index = args.get("frame_index", 0)
        region = args.get("region", [])
        
        if not (0 <= frame_index < total_frames):
            return {"success": False, "result": "Invalid frame index"}
        if not region or len(region) != 4:
            return {"success": False, "result": "Region must be [x1, y1, x2, y2]"}
        
        frames = extract_frames(video_path, [frame_index])
        frame = frames.get(frame_index)
        if frame is None:
            return {"success": False, "result": "Failed to extract frame"}
        
        image = Image(data=frame)
        result = describer(image=image, region=region)
        description = result.get("description", "")
        
        return {"success": True, "result": f"[Frame {frame_index}]: {description}", "description": description}
    
    def _execute_qa(self, qa_model, tool_name, args, video_path, total_frames, fps, sampled_indices):
        from tools.interface import Video
        
        question = args.get("question", "")
        if not question:
            return {"success": False, "result": "Question required"}
        
        video = Video(path=video_path, frame_count=total_frames, fps=fps)
        
        if tool_name == "temporal_spatial_qa":
            frame_indices = args.get("frame_indices", sampled_indices)
            result = qa_model(video=video, question=question, frame_indices=frame_indices)
        else:
            result = qa_model(video=video, question=question)
        
        answer = result.get("answer", "")
        return {"success": True, "result": f"Q: {question}\nA: {answer}", "answer": answer}
    
    def _execute_vqa(self, vqa_model, tool_name, args, video_path, total_frames, fps):
        from video_agent_tools.utils.video import extract_frames
        from tools.interface import Image
        
        # Accept both "question" and "query" as input
        question = args.get("question", "") or args.get("query", "")
        if not question:
            return {"success": False, "result": "Question or query required"}
        
        if tool_name == "targeting_vqa":
            frame_index = args.get("frame_index", 0)
            target_object = args.get("target_object", None) or args.get("region", None)
            
            frames = extract_frames(video_path, [frame_index])
            frame = frames.get(frame_index)
            if frame is None:
                return {"success": False, "result": "Failed to extract frame"}
            
            image = Image(data=frame)
            # TargetingVQA expects query (str), images (list), and optional target_object
            result = vqa_model(query=question, images=[image], target_object=target_object)
        else:
            # general_vqa
            frame_indices = args.get("frame_indices", [0])
            frames = extract_frames(video_path, frame_indices[:1])
            frame = list(frames.values())[0] if frames else None
            if frame is None:
                return {"success": False, "result": "Failed to extract frame"}
            
            image = Image(data=frame)
            # GeneralVQA expects query (str) and images (list)
            result = vqa_model(query=question, images=[image])
        
        answer = result.get("answer", "")
        return {"success": True, "result": f"Q: {question}\nA: {answer}", "answer": answer}
    
    def _execute_internvideo(self, model, tool_name, args, video_path):
        from tools.interface import Video
        import gc
        
        try:
            video = Video(video_path)
            
            if tool_name == "internvideo_description":
                # Get frame range from args (default to full video)
                start_frame = args.get("start_frame", 0)
                end_frame = args.get("end_frame")
                
                # Check cache first for internvideo_description
                if self._tool_cache:
                    cached_result = self._tool_cache.get_description(
                        video_path=video_path,
                        start_frame=start_frame,
                        end_frame=end_frame,
                    )
                    if cached_result is not None:
                        self._resource_log(f"InternVideo description cache HIT: frames {start_frame}-{end_frame}")
                        description = cached_result.get("description", "")
                        return {
                            "success": True,
                            "result": description,
                            "description": description,
                            "cache_hit": True,
                        }
                
                # Cache miss - run model
                result = model(video=video, start_frame=start_frame, end_frame=end_frame)
                
                # Check for errors in model response
                if "error" in result:
                    self._resource_log(f"InternVideo error: {result['error']}")
                    return {"success": False, "result": f"Model error: {result['error']}", "error": result['error']}
                
                # Cache the result
                if self._tool_cache:
                    self._tool_cache.set_description(
                        video_path=video_path,
                        start_frame=start_frame,
                        end_frame=end_frame,
                        result=result,
                    )
                    self._resource_log(f"Cached InternVideo description for frames {start_frame}-{end_frame}")
                
                return {"success": True, "result": result.get("description", ""), "description": result.get("description", "")}
            else:
                # internvideo_general_qa
                # Accept both "query" and "question" as the question parameter
                question = args.get("query") or args.get("question", "")
                if not question:
                    return {"success": False, "result": "Question/query required"}
                
                # Get optional frame range parameters
                start_frame = args.get("start_frame", None)
                end_frame = args.get("end_frame", None)
                
                # Pass query and optional frame range to InternVideoGeneralQA
                result = model(video=video, query=question, start_frame=start_frame, end_frame=end_frame)
                
                # Check for errors in model response
                if "error" in result:
                    self._resource_log(f"InternVideo error: {result['error']}")
                    return {"success": False, "result": f"Model error: {result['error']}", "error": result['error']}
                
                answer = result.get("answer", "")
                frame_range = result.get("frame_range", "")
                if not answer:
                    self._resource_log(f"InternVideo returned empty answer for: {question[:50]}...")
                
                # Format result with frame range if specified
                if frame_range and "full video" not in frame_range:
                    return {"success": True, "result": f"[Frames {frame_range}] Q: {question}\nA: {answer}", "answer": answer, "frame_range": frame_range}
                return {"success": True, "result": f"Q: {question}\nA: {answer}", "answer": answer}
        
        except torch.cuda.OutOfMemoryError as e:
            # Clean up GPU memory, trigger eviction, and return error
            gc.collect()
            torch.cuda.empty_cache()
            self._resource_log(f"OOM during {tool_name} execution: {e}")
            
            # Trigger eviction to free up memory for retry
            memory_needed = TOOL_MEMORY_ESTIMATES.get(tool_name, 2000)
            self._resource_log(f"OOM_EVICT: triggering eviction for '{tool_name}' (need {memory_needed}MB)")
            evicted = self._try_evict_for_tool(tool_name, memory_needed)
            if evicted:
                self._resource_log(f"OOM_EVICT: eviction successful, retry may succeed")
                gc.collect()
                torch.cuda.empty_cache()
            
            return {"success": False, "result": f"GPU memory insufficient for {tool_name}", "error": "OOM"}
        except Exception as e:
            self._resource_log(f"Error in {tool_name}: {type(e).__name__}: {e}")
            return {"success": False, "result": f"Tool error: {e}", "error": str(e)}
    
    def _log_gpu_status(self):
        """Log current GPU resource status."""
        self._update_gpu_memory()
        
        lines = [""]
        lines.append("=" * 60)
        lines.append("RESOURCE STATUS SUMMARY")
        lines.append("=" * 60)
        
        # GPU Memory
        lines.append("GPU Memory:")
        for gpu_id, gpu_info in self._gpus.items():
            tools_str = ", ".join(gpu_info.tools_loaded) if gpu_info.tools_loaded else "idle"
            free_mb = gpu_info.free_memory_mb
            used_pct = (gpu_info.allocated_memory_mb / gpu_info.total_memory_mb) * 100
            lines.append(
                f"  GPU{gpu_id}: {gpu_info.allocated_memory_mb:.0f}MB used ({used_pct:.0f}%) | Free: {free_mb:.0f}MB | Tools: {tools_str}"
            )
        
        # Tool instances
        lines.append("Tool Instances:")
        for tool_name, instances in self._tool_instances.items():
            in_use = sum(1 for i in instances if i.in_use)
            devices = [i.device for i in instances]
            device_str = ", ".join(devices)
            lines.append(f"  {tool_name}: {len(instances)} inst ({in_use} busy) on [{device_str}]")
        
        # Stats
        lines.append("Statistics:")
        lines.append(f"  Requests: {self._stats['requests_processed']} processed, {self._stats['requests_waiting']} waiting")
        lines.append(f"  Tools: {self._stats['tools_loaded']} loaded, {self._stats['tools_replicated']} replicated")
        lines.append("=" * 60)
        lines.append("")
        
        # Log to resource file
        for line in lines:
            self._resource_log(line)
    
    def _handle_tool_request(self, request: ToolRequest):
        """Handle a tool request in a worker thread."""
        try:
            response = self._execute_tool(request)
            worker_queue = self.response_queues.get(request.worker_id)
            if worker_queue:
                worker_queue.put(response)
        except Exception as e:
            self._resource_log(f"ERROR handling request: {e}")
            worker_queue = self.response_queues.get(request.worker_id)
            if worker_queue:
                worker_queue.put(ToolResponse(
                    request_id=request.request_id,
                    worker_id=request.worker_id,
                    success=False,
                    error=str(e),
                    tool_name=request.tool_name,
                ))
    
    def run(self):
        """Main server loop with thread pool for parallel request processing."""
        from concurrent.futures import ThreadPoolExecutor
        
        self._log("info", "ToolServer starting...")
        self._init_gpu_info()
        
        self._log("info", f"Enabled tools: {self.enabled_tools}")
        self._log("info", f"Using API captioner: {self.use_api_captioner}")
        
        # Create thread pool for parallel request handling
        # No fixed limit - let ThreadPoolExecutor manage dynamically
        # Default is min(32, os.cpu_count() + 4) which is reasonable
        self._resource_log(f"Starting with thread pool (dynamic workers, {len(self._gpus)} GPUs available)")
        
        self._log("info", "ToolServer ready")
        
        status_interval = 20  # Log status every N requests
        
        with ThreadPoolExecutor() as executor:
            while not self.shutdown_event.is_set():
                try:
                    try:
                        request = self.request_queue.get(timeout=0.5)
                    except queue.Empty:
                        continue
                    
                    if request.msg_type == MessageType.SHUTDOWN:
                        self._log("info", "Received shutdown signal")
                        self._log_gpu_status()  # Final status
                        break
                    
                    elif request.msg_type == MessageType.EXECUTE_TOOL:
                        self._stats["requests_processed"] += 1
                        
                        # Periodic status logging
                        if self._stats["requests_processed"] % status_interval == 0:
                            self._log_gpu_status()
                        
                        # Submit to thread pool for parallel execution
                        executor.submit(self._handle_tool_request, request)
                    
                    elif request.msg_type == MessageType.GET_STATUS:
                        self._update_gpu_memory()
                        status = {
                            "gpus": {k: {"device": v.device_str, "free_mb": v.free_memory_mb} for k, v in self._gpus.items()},
                            "tools_loaded": {k: len(v) for k, v in self._tool_instances.items()},
                            "stats": self._stats,
                        }
                        
                        worker_queue = self.response_queues.get(request.worker_id)
                        if worker_queue:
                            worker_queue.put(ToolResponse(
                                request_id=request.request_id,
                                worker_id=request.worker_id,
                                success=True,
                                result=status,
                            ))
                    
                except Exception as e:
                    self._log("error", f"Error in server loop: {e}\n{traceback.format_exc()}")
        
        self._log("info", f"ToolServer shutting down. Stats: {self._stats}")
        self._cleanup()
    
    def _cleanup(self):
        """Cleanup all loaded tools."""
        self._resource_log("=== CLEANUP START ===")
        self._log_gpu_status()
        
        for tool_name, instances in self._tool_instances.items():
            for inst in instances:
                try:
                    self._resource_log(f"UNLOAD '{tool_name}' from {inst.device}")
                    if hasattr(inst.instance, 'cleanup'):
                        inst.instance.cleanup()
                    del inst.instance
                except Exception as e:
                    self._resource_log(f"ERROR unloading {tool_name}: {e}")
        
        self._tool_instances.clear()
        
        for captioner in [self._captioner, self._api_captioner]:
            if captioner:
                try:
                    del captioner
                except:
                    pass
        
        self._clear_gpu_memory()
        self._resource_log("=== CLEANUP COMPLETE ===")
        
        # Close resource log file
        if self._resource_log_file:
            self._resource_log_file.close()
            self._resource_log_file = None


def run_tool_server(
    request_queue: Queue,
    response_queues: Dict[int, Queue],
    shutdown_event: Event,
    enabled_tools: List[str],
    captioner: str,
    log_queue: Queue = None,
    output_dir: str = None,
    max_view_frames: int = 8,
    default_sample_frames: int = 16,
    min_sample_frames: int = 1,
    max_sample_frames: int = 32,
):
    """Entry point for tool server process."""
    # Ensure numexpr thread limits are set in child process
    import os
    os.environ["NUMEXPR_MAX_THREADS"] = "64"
    os.environ["NUMEXPR_NUM_THREADS"] = "8"
    
    server = ToolServer(
        request_queue=request_queue,
        response_queues=response_queues,
        shutdown_event=shutdown_event,
        enabled_tools=enabled_tools,
        captioner=captioner,
        log_queue=log_queue,
        output_dir=output_dir,
        max_view_frames=max_view_frames,
        default_sample_frames=default_sample_frames,
        min_sample_frames=min_sample_frames,
        max_sample_frames=max_sample_frames,
    )
    server.run()


