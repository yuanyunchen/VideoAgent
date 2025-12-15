"""
Core Data Structures for Resource Management

Contains data classes for GPU info, tool info, and IPC messages.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional
from enum import Enum

from .constants import ToolPriority


@dataclass
class GPUInfo:
    """Information about a GPU device."""
    device_id: int
    device_str: str  # e.g., "cuda:0"
    total_memory_mb: float
    allocated_memory_mb: float = 0.0
    reserved_memory_mb: float = 0.0
    tools_loaded: List[str] = field(default_factory=list)
    
    @property
    def free_memory_mb(self) -> float:
        """Calculate approximate free memory."""
        return self.total_memory_mb - self.allocated_memory_mb
    
    @property
    def available_memory_mb(self) -> float:
        """Calculate available memory (more conservative estimate)."""
        return self.total_memory_mb - self.reserved_memory_mb


@dataclass
class ToolInfo:
    """Information about a loaded tool (for GPUResourceManager)."""
    name: str
    instance: Any
    device: str
    memory_mb: float
    priority: ToolPriority
    last_used: datetime
    use_count: int = 0
    instance_id: str = ""
    in_use: bool = False
    
    def touch(self):
        """Update last used timestamp and increment use count."""
        self.last_used = datetime.now()
        self.use_count += 1


@dataclass
class ToolInstance:
    """Wrapper for a tool instance on a specific GPU (for ToolServer)."""
    tool_name: str
    instance: Any
    device: str
    instance_id: str
    memory_mb: float
    in_use: bool = False
    last_used: datetime = field(default_factory=datetime.now)
    use_count: int = 0
    
    def mark_in_use(self):
        """Mark tool as currently in use."""
        self.in_use = True
        self.last_used = datetime.now()
        self.use_count += 1
    
    def mark_available(self):
        """Mark tool as available."""
        self.in_use = False
        self.last_used = datetime.now()


# ==============================================================================
# IPC Message Types
# ==============================================================================

class MessageType(Enum):
    """Message types for Tool Server IPC."""
    EXECUTE_TOOL = "execute_tool"
    GET_STATUS = "get_status"
    SHUTDOWN = "shutdown"
    RESPONSE = "response"


@dataclass
class ToolRequest:
    """Request message for tool execution."""
    msg_type: MessageType
    request_id: str
    worker_id: int
    tool_name: str = ""
    tool_args: Dict[str, Any] = field(default_factory=dict)
    video_context_data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass 
class ToolResponse:
    """Response message from tool server."""
    request_id: str
    worker_id: int
    success: bool
    result: Dict[str, Any] = field(default_factory=dict)
    error: str = ""
    tool_name: str = ""
    duration_ms: float = 0.0




