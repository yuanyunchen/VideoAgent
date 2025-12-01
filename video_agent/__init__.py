"""
VideoAgent - Video analysis with LLM-powered question answering.
"""

from video_agent.agent import VideoAgent
from video_agent.core.video_memory import VideoMemory
from video_agent.processors.caption_processor import CaptionProcessor
from video_agent.processors.question_processor import QuestionProcessor

__version__ = "1.0.0"
__all__ = [
    "VideoAgent",
    "VideoMemory", 
    "CaptionProcessor",
    "QuestionProcessor",
]

