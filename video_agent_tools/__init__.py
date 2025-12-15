"""
VideoAgent Tools - Video Understanding Agent Framework

Main components:
- VideoAgent: Main agent class for video Q&A
- ToolManager: Local tool management
- VideoAgentEvaluator: Batch evaluation framework

For multiprocessing:
- resource_management: Centralized GPU resource management
  - ToolServer: Manages GPU tools for all workers
  - ToolClient: Worker-side tool access
"""

from video_agent_tools.graph import VideoAgent
from video_agent_tools.tools import ToolManager
from video_agent_tools.evaluation import VideoAgentEvaluator
from video_agent_tools.state import (
    AgentState, 
    VideoContext,
    VideoMemory,
    FrameInfo,
    ToolHistoryEntry,
)

__all__ = [
    "VideoAgent",
    "ToolManager",
    "VideoAgentEvaluator",
    "AgentState",
    "VideoContext",
    "VideoMemory",
    "FrameInfo",
    "ToolHistoryEntry",
]
