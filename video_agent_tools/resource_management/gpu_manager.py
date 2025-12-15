"""
GPU Resource Manager for Single-Process Mode

Provides unified GPU memory management across tools with:
- Multi-GPU support with automatic device selection
- Memory-aware tool distribution
- OOM recovery by unloading unused tools
- Tool priority and LRU-based eviction
"""

import gc
import uuid
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from threading import Lock

from .constants import (
    TOOL_MEMORY_ESTIMATES,
    TOOL_PRIORITIES,
    SHARED_MODEL_GROUPS,
    MEMORY_BUFFER_MB,
    ToolPriority,
)
from .core import GPUInfo, ToolInfo

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class GPUResourceManager:
    """
    Manages GPU resources across multiple devices for video agent tools.
    
    For single-process mode only. For multiprocessing, use ToolServer/ToolClient.
    
    Key features:
    - Assigns tools to GPUs based on memory availability
    - Tracks tool usage for intelligent eviction
    - Provides OOM recovery through tool unloading
    - Thread-safe operations
    """
    
    def __init__(
        self,
        logger: logging.Logger = None,
        memory_buffer_mb: float = None,
    ):
        """
        Initialize the GPU Resource Manager.
        
        Args:
            logger: Logger instance
            memory_buffer_mb: Memory buffer to keep free on each GPU
        """
        self.logger = logger or logging.getLogger(__name__)
        self.memory_buffer_mb = memory_buffer_mb or MEMORY_BUFFER_MB
        
        self._lock = Lock()
        self._loaded_tools: Dict[str, ToolInfo] = {}
        self._gpus: Dict[int, GPUInfo] = {}
        
        self._init_gpu_info()
    
    def _init_gpu_info(self):
        """Initialize GPU information."""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            self.logger.warning("CUDA not available, GPU resource management disabled")
            return
        
        num_gpus = torch.cuda.device_count()
        self.logger.info(f"Initializing GPU Resource Manager with {num_gpus} GPU(s)")
        
        for i in range(num_gpus):
            device_str = f"cuda:{i}"
            props = torch.cuda.get_device_properties(i)
            total_memory_mb = props.total_memory / (1024 ** 2)
            
            self._gpus[i] = GPUInfo(
                device_id=i,
                device_str=device_str,
                total_memory_mb=total_memory_mb,
            )
            
            self.logger.info(f"GPU {i}: {props.name}, Total Memory: {total_memory_mb:.0f}MB")
        
        self._update_gpu_memory_stats()
    
    def _update_gpu_memory_stats(self):
        """Update GPU memory statistics."""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return
        
        for gpu_id, gpu_info in self._gpus.items():
            try:
                allocated = torch.cuda.memory_allocated(gpu_id) / (1024 ** 2)
                reserved = torch.cuda.memory_reserved(gpu_id) / (1024 ** 2)
                gpu_info.allocated_memory_mb = allocated
                gpu_info.reserved_memory_mb = reserved
            except Exception as e:
                self.logger.warning(f"Failed to get memory stats for GPU {gpu_id}: {e}")
    
    def get_gpu_status(self) -> Dict[int, Dict[str, Any]]:
        """Get current GPU status including memory and loaded tools."""
        self._update_gpu_memory_stats()
        
        status = {}
        for gpu_id, gpu_info in self._gpus.items():
            status[gpu_id] = {
                "device": gpu_info.device_str,
                "total_memory_mb": gpu_info.total_memory_mb,
                "allocated_memory_mb": gpu_info.allocated_memory_mb,
                "reserved_memory_mb": gpu_info.reserved_memory_mb,
                "free_memory_mb": gpu_info.free_memory_mb,
                "tools_loaded": gpu_info.tools_loaded.copy(),
            }
        return status
    
    def select_device_for_tool(self, tool_name: str) -> str:
        """
        Select the best GPU device for a tool.
        
        Strategy:
        1. If tool shares model with loaded tool, use same device
        2. Otherwise, select GPU with most free memory
        
        Args:
            tool_name: Name of the tool to load
            
        Returns:
            Device string (e.g., "cuda:0")
        """
        with self._lock:
            self._update_gpu_memory_stats()
            
            # Check if tool shares model with already loaded tool
            for group_name, tools in SHARED_MODEL_GROUPS.items():
                if tool_name in tools:
                    for other_tool in tools:
                        if other_tool in self._loaded_tools:
                            device = self._loaded_tools[other_tool].device
                            self.logger.info(
                                f"Tool '{tool_name}' shares model with '{other_tool}', "
                                f"using same device: {device}"
                            )
                            return device
            
            memory_needed = TOOL_MEMORY_ESTIMATES.get(tool_name, 2000)
            
            # Find GPU with most available memory that can fit the tool
            best_gpu = None
            best_available = -1
            
            for gpu_id, gpu_info in self._gpus.items():
                available = gpu_info.free_memory_mb - self.memory_buffer_mb
                
                if available >= memory_needed and available > best_available:
                    best_available = available
                    best_gpu = gpu_id
            
            if best_gpu is not None:
                device = self._gpus[best_gpu].device_str
                self.logger.info(
                    f"Selected {device} for '{tool_name}' "
                    f"(need {memory_needed}MB, available {best_available:.0f}MB)"
                )
                return device
            
            # No GPU has enough space - log warning and use GPU with most free memory
            self.logger.warning(
                f"No GPU has enough memory for '{tool_name}' ({memory_needed}MB needed). "
                "Will attempt to use GPU with most free memory."
            )
            
            if self._gpus:
                best_gpu_id = max(self._gpus.keys(), key=lambda g: self._gpus[g].free_memory_mb)
                return self._gpus[best_gpu_id].device_str
            
            return "cuda:0"
    
    def register_tool(
        self,
        tool_name: str,
        instance: Any,
        device: str,
        memory_mb: float = None,
        priority: ToolPriority = None,
    ):
        """
        Register a loaded tool with the resource manager.
        
        Args:
            tool_name: Name of the tool
            instance: Tool instance
            device: Device the tool is loaded on
            memory_mb: Estimated memory usage (uses default if not provided)
            priority: Tool priority for eviction
        """
        with self._lock:
            if memory_mb is None:
                memory_mb = TOOL_MEMORY_ESTIMATES.get(tool_name, 2000)
            
            if priority is None:
                priority = TOOL_PRIORITIES.get(tool_name, ToolPriority.MEDIUM)
            
            tool_info = ToolInfo(
                name=tool_name,
                instance=instance,
                device=device,
                memory_mb=memory_mb,
                priority=priority,
                last_used=datetime.now(),
                instance_id=str(uuid.uuid4())[:8],
            )
            
            self._loaded_tools[tool_name] = tool_info
            
            # Update GPU tool list
            gpu_id = self._device_to_gpu_id(device)
            if gpu_id is not None and gpu_id in self._gpus:
                if tool_name not in self._gpus[gpu_id].tools_loaded:
                    self._gpus[gpu_id].tools_loaded.append(tool_name)
            
            self.logger.info(
                f"Registered tool '{tool_name}' on {device} "
                f"(~{memory_mb:.0f}MB, priority={priority.name})"
            )
    
    def get_tool(self, tool_name: str) -> Optional[Any]:
        """
        Get a loaded tool instance and update its usage stats.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Tool instance or None if not loaded
        """
        with self._lock:
            if tool_name in self._loaded_tools:
                tool_info = self._loaded_tools[tool_name]
                tool_info.touch()
                return tool_info.instance
            return None
    
    def is_tool_loaded(self, tool_name: str) -> bool:
        """Check if a tool is currently loaded."""
        return tool_name in self._loaded_tools
    
    def unload_tool(self, tool_name: str) -> bool:
        """
        Unload a tool from GPU memory.
        
        Args:
            tool_name: Name of the tool to unload
            
        Returns:
            True if tool was unloaded, False if not found or critical
        """
        with self._lock:
            if tool_name not in self._loaded_tools:
                return False
            
            tool_info = self._loaded_tools[tool_name]
            
            if tool_info.priority == ToolPriority.CRITICAL:
                self.logger.warning(f"Cannot unload critical tool '{tool_name}'")
                return False
            
            self.logger.info(f"Unloading tool '{tool_name}' from {tool_info.device}")
            
            try:
                if hasattr(tool_info.instance, 'cleanup'):
                    tool_info.instance.cleanup()
                
                del tool_info.instance
                del self._loaded_tools[tool_name]
                
                # Update GPU tool list
                gpu_id = self._device_to_gpu_id(tool_info.device)
                if gpu_id is not None and gpu_id in self._gpus:
                    if tool_name in self._gpus[gpu_id].tools_loaded:
                        self._gpus[gpu_id].tools_loaded.remove(tool_name)
                
                self._clear_gpu_memory()
                return True
                
            except Exception as e:
                self.logger.error(f"Error unloading tool '{tool_name}': {e}")
                return False
    
    def free_memory_for_tool(self, tool_name: str, target_device: str = None) -> bool:
        """
        Try to free enough memory for a tool by evicting other tools.
        
        Uses LRU + priority-based eviction strategy.
        
        Args:
            tool_name: Name of the tool that needs memory
            target_device: Target device for the tool (optional)
            
        Returns:
            True if enough memory was freed, False otherwise
        """
        with self._lock:
            memory_needed = TOOL_MEMORY_ESTIMATES.get(tool_name, 2000)
            
            # Determine target GPU
            gpu_id = None
            if target_device:
                gpu_id = self._device_to_gpu_id(target_device)
            
            self._update_gpu_memory_stats()
            
            # Get tools that can be evicted (sorted by priority then LRU)
            eviction_candidates = []
            for name, info in self._loaded_tools.items():
                if info.priority == ToolPriority.CRITICAL:
                    continue
                if name == tool_name:
                    continue
                
                if gpu_id is not None:
                    tool_gpu = self._device_to_gpu_id(info.device)
                    if tool_gpu != gpu_id:
                        continue
                
                eviction_candidates.append(info)
            
            eviction_candidates.sort(key=lambda x: (x.priority.value, x.last_used))
            
            evicted_tools = []
            
            for candidate in eviction_candidates:
                self._update_gpu_memory_stats()
                
                if gpu_id is not None:
                    current_free = self._gpus[gpu_id].free_memory_mb
                else:
                    current_free = max(g.free_memory_mb for g in self._gpus.values()) if self._gpus else 0
                
                if current_free >= memory_needed + self.memory_buffer_mb:
                    break
                
                self.logger.info(
                    f"Evicting tool '{candidate.name}' to free memory "
                    f"(priority={candidate.priority.name})"
                )
                
                # Unload without re-acquiring lock
                if candidate.name in self._loaded_tools:
                    try:
                        if hasattr(candidate.instance, 'cleanup'):
                            candidate.instance.cleanup()
                        del candidate.instance
                        del self._loaded_tools[candidate.name]
                        
                        cgpu_id = self._device_to_gpu_id(candidate.device)
                        if cgpu_id is not None and cgpu_id in self._gpus:
                            if candidate.name in self._gpus[cgpu_id].tools_loaded:
                                self._gpus[cgpu_id].tools_loaded.remove(candidate.name)
                        
                        evicted_tools.append(candidate.name)
                    except Exception as e:
                        self.logger.error(f"Error evicting '{candidate.name}': {e}")
            
            if evicted_tools:
                self._clear_gpu_memory()
                self.logger.info(f"Evicted {len(evicted_tools)} tools: {evicted_tools}")
            
            # Final check
            self._update_gpu_memory_stats()
            
            if gpu_id is not None and gpu_id in self._gpus:
                final_free = self._gpus[gpu_id].free_memory_mb
            elif self._gpus:
                final_free = max(g.free_memory_mb for g in self._gpus.values())
            else:
                final_free = 0
            
            return final_free >= memory_needed + self.memory_buffer_mb
    
    def _clear_gpu_memory(self):
        """Clear GPU memory caches."""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return
        
        try:
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            self.logger.warning(f"Error clearing GPU memory: {e}")
    
    def _device_to_gpu_id(self, device: str) -> Optional[int]:
        """Convert device string to GPU ID."""
        if device.startswith("cuda:"):
            try:
                return int(device.split(":")[1])
            except (IndexError, ValueError):
                return 0
        elif device == "cuda":
            return 0
        return None
    
    def log_status(self):
        """Log current GPU and tool status."""
        self._update_gpu_memory_stats()
        
        self.logger.info("=" * 60)
        self.logger.info("GPU Resource Manager Status")
        self.logger.info("=" * 60)
        
        for gpu_id, gpu_info in self._gpus.items():
            self.logger.info(
                f"GPU {gpu_id}: "
                f"Allocated={gpu_info.allocated_memory_mb:.0f}MB, "
                f"Free={gpu_info.free_memory_mb:.0f}MB, "
                f"Tools={gpu_info.tools_loaded}"
            )
        
        self.logger.info(f"Loaded Tools ({len(self._loaded_tools)}):")
        for name, info in self._loaded_tools.items():
            self.logger.info(
                f"  {name}: device={info.device}, "
                f"memory={info.memory_mb:.0f}MB, "
                f"priority={info.priority.name}"
            )
        self.logger.info("=" * 60)
    
    def log_status_brief(self, prefix: str = "GPU status"):
        """Lightweight status log for telemetry."""
        self._update_gpu_memory_stats()
        for gpu_id, gpu_info in self._gpus.items():
            self.logger.info(
                f"{prefix} | GPU {gpu_id}: "
                f"alloc={gpu_info.allocated_memory_mb:.0f}MB, "
                f"free={gpu_info.free_memory_mb:.0f}MB, "
                f"tools={gpu_info.tools_loaded}"
            )


# Singleton instance for backward compatibility
_global_manager: Optional[GPUResourceManager] = None


def get_gpu_resource_manager(logger: logging.Logger = None) -> GPUResourceManager:
    """Get or create the global GPU resource manager."""
    global _global_manager
    if _global_manager is None:
        _global_manager = GPUResourceManager(logger=logger)
    return _global_manager




