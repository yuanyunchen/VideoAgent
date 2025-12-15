"""
Resource Management Module for VideoAgent Tools

Provides centralized GPU resource management with:
- Single-process mode: GPUResourceManager for local tool management
- Multi-process mode: ToolServer/ToolClient for distributed tool access

Usage:
    # Single-process mode
    from resource_management import GPUResourceManager
    manager = GPUResourceManager(logger=logger)
    
    # Multi-process mode
    from resource_management import ToolServer, ToolClient, run_tool_server
"""

from .constants import (
    TOOL_MEMORY_ESTIMATES,
    TOOL_PRIORITIES,
    TOOL_MAX_INSTANCES,
    SHARED_MODEL_GROUPS,
    MEMORY_BUFFER_MB,
    QUEUE_PRESSURE_THRESHOLD,
    ToolPriority,
)

from .core import (
    GPUInfo,
    ToolInfo,
    ToolInstance,
    MessageType,
    ToolRequest,
    ToolResponse,
)

from .gpu_manager import GPUResourceManager

from .tool_server import ToolServer, run_tool_server

from .tool_client import ToolClient

__all__ = [
    # Constants
    "TOOL_MEMORY_ESTIMATES",
    "TOOL_PRIORITIES", 
    "TOOL_MAX_INSTANCES",
    "SHARED_MODEL_GROUPS",
    "MEMORY_BUFFER_MB",
    "QUEUE_PRESSURE_THRESHOLD",
    "ToolPriority",
    # Data classes
    "GPUInfo",
    "ToolInfo",
    "ToolInstance",
    "MessageType",
    "ToolRequest",
    "ToolResponse",
    # Managers
    "GPUResourceManager",
    "ToolServer",
    "ToolClient",
    "run_tool_server",
]




