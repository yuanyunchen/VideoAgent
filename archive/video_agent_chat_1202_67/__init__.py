"""
VideoAgent - Multi-agent video understanding with LLM-powered question answering.

This module implements a multi-agent system for video understanding:
- Solver Agent: Stateful agent that decides actions and provides answers
- Checker Agent: Stateless evaluator that assesses answer confidence
- Memory: Stores frame captions for context

Usage:
    from video_agent import VideoAgent
    
    config = {...}
    agent = VideoAgent(config)
    output_dir = agent.run_experiment()
"""

from video_agent.agent import VideoAgent
from video_agent.core.memory import Memory, MemoryUnit
from video_agent.agents.solver import Solver
from video_agent.agents.checker import evaluate_answer
from video_agent.agents.caption_generator import CaptionGenerator

__version__ = "2.0.0"
__all__ = [
    "VideoAgent",
    "Memory",
    "MemoryUnit",
    "Solver",
    "evaluate_answer",
    "CaptionGenerator",
]

