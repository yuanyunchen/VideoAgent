"""
Logging Utilities for VideoAgent Tools

Structured logging for tool calls, LLM interactions, and evaluation progress.
Includes per-video logging with detailed output to individual directories.

Multiprocessing support:
- LogCollector: Process that receives logs from workers via queue
- WorkerLogger: Logger wrapper that sends logs to collector queue
"""

import os
import logging
import json
import queue
import threading
import traceback
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from enum import Enum
from multiprocessing import Process, Queue, Event

import cv2
import numpy as np


# ==============================================================================
# Multiprocessing Log Collection
# ==============================================================================

class LogLevel(Enum):
    """Log levels matching Python logging."""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


@dataclass
class LogMessage:
    """Log message for inter-process communication."""
    worker_id: int
    video_id: str
    level: LogLevel
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    # For grouping - messages with same group_id should be written together
    group_id: str = ""
    # Whether this is the last message in a group
    group_end: bool = False
    

class LogCollector:
    """
    Centralized log collector for multiprocessing.
    
    Receives log messages from workers via queue and writes them to files
    in a coordinated manner, keeping logs from the same video grouped together.
    
    Used only in multiprocessing mode (num_workers > 1).
    For single-process mode, use direct logging.
    """
    
    def __init__(
        self,
        log_queue: Queue,
        output_dir: str,
        num_workers: int,
        console_output: bool = True,
    ):
        """
        Initialize the log collector.
        
        Args:
            log_queue: Queue for receiving log messages from workers
            output_dir: Base output directory for logs
            num_workers: Number of worker processes
            console_output: Whether to also output to console
        """
        self.log_queue = log_queue
        self.output_dir = output_dir
        self.num_workers = num_workers
        self.console_output = console_output
        
        # Running state
        self._running = False
        self._shutdown_event = Event()
        
        # Buffered messages by video_id for grouping
        self._message_buffer: Dict[str, List[LogMessage]] = {}
        
        # Track which videos are complete
        self._completed_videos: set = set()
        
        # File handle for main log
        self._main_log_file = None
        self._main_log_path = os.path.join(output_dir, "logging.log")
        
        # Lock for file writing
        self._write_lock = threading.Lock()
        
        # Setup
        os.makedirs(output_dir, exist_ok=True)
    
    def _open_main_log(self):
        """Open main log file."""
        if self._main_log_file is None:
            self._main_log_file = open(self._main_log_path, 'a', encoding='utf-8')
    
    def _close_main_log(self):
        """Close main log file."""
        if self._main_log_file is not None:
            self._main_log_file.close()
            self._main_log_file = None
    
    def _write_message(self, msg: LogMessage):
        """Write a single log message."""
        # Handle timestamp
        if isinstance(msg.timestamp, datetime):
            timestamp_str = msg.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        else:
            timestamp_str = str(msg.timestamp)[:19]  # Truncate ISO format
        
        # Handle level - can be enum or string
        if isinstance(msg.level, LogLevel):
            level_str = msg.level.name
            level_enum = msg.level
        else:
            level_str = str(msg.level).upper()
            level_enum = LogLevel.INFO  # Default for color
            if level_str == "ERROR":
                level_enum = LogLevel.ERROR
            elif level_str == "WARNING":
                level_enum = LogLevel.WARNING
        
        worker_str = f"[Worker {msg.worker_id}]" if msg.worker_id >= 0 else ""
        video_str = f"[{msg.video_id}]" if msg.video_id else ""
        
        # Format: timestamp - level - [worker] [video] message
        prefix_parts = [timestamp_str, level_str]
        if worker_str:
            prefix_parts.append(worker_str)
        if video_str:
            prefix_parts.append(video_str)
        
        prefix = " - ".join(prefix_parts)
        formatted = f"{prefix} - {msg.message}"
        
        with self._write_lock:
            # Write to main log file
            self._open_main_log()
            self._main_log_file.write(formatted + "\n")
            self._main_log_file.flush()
            
            # Console output
            if self.console_output:
                # For console, use simpler format
                if level_enum == LogLevel.ERROR:
                    print(f"\033[91m{formatted}\033[0m")  # Red for errors
                elif level_enum == LogLevel.WARNING:
                    print(f"\033[93m{formatted}\033[0m")  # Yellow for warnings
                else:
                    print(formatted)
    
    def _write_buffered_messages(self, video_id: str):
        """Write all buffered messages for a video."""
        if video_id not in self._message_buffer:
            return
        
        messages = self._message_buffer.pop(video_id)
        for msg in messages:
            self._write_message(msg)
    
    def _handle_message(self, msg):
        """Handle an incoming log message.
        
        Accepts both LogMessage objects and dict messages (from ToolServer).
        """
        # Handle dict messages (from ToolServer)
        if isinstance(msg, dict):
            # Parse timestamp - could be string or datetime
            timestamp = msg.get("timestamp", datetime.now())
            if isinstance(timestamp, str):
                try:
                    timestamp = datetime.fromisoformat(timestamp)
                except:
                    timestamp = datetime.now()
            
            # Convert dict to LogMessage for consistency
            msg = LogMessage(
                timestamp=timestamp,
                level=msg.get("level", "INFO"),
                worker_id=msg.get("worker_id", -1),
                video_id=msg.get("video_id", ""),
                message=msg.get("message", str(msg)),
                group_end=msg.get("group_end", False),
            )
        
        # If no video_id, write immediately (global logs)
        if not msg.video_id:
            self._write_message(msg)
            return
        
        # Buffer by video for grouping
        if msg.video_id not in self._message_buffer:
            self._message_buffer[msg.video_id] = []
        
        self._message_buffer[msg.video_id].append(msg)
        
        # If this is end of a group (video complete), write all buffered
        if msg.group_end:
            self._write_buffered_messages(msg.video_id)
            self._completed_videos.add(msg.video_id)
    
    def run(self):
        """Run the log collector loop."""
        self._running = True
        
        while self._running and not self._shutdown_event.is_set():
            try:
                # Get message with timeout
                try:
                    msg = self.log_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                # Check for shutdown signal
                if msg is None:
                    self._running = False
                    break
                
                self._handle_message(msg)
                
            except Exception as e:
                print(f"[LogCollector] Error handling message: {e}")
                traceback.print_exc()
        
        # Flush any remaining buffered messages
        for video_id in list(self._message_buffer.keys()):
            self._write_buffered_messages(video_id)
        
        self._close_main_log()
    
    def shutdown(self):
        """Signal the collector to shutdown."""
        self._shutdown_event.set()
        self._running = False
        # Send None to unblock the queue.get()
        try:
            self.log_queue.put(None)
        except:
            pass


class WorkerLogger:
    """
    Logger wrapper for worker processes in multiprocessing mode.
    
    Sends log messages to the LogCollector via queue instead of
    writing directly to files.
    
    Provides the same interface as logging.Logger but routes messages
    through the queue.
    """
    
    def __init__(
        self,
        worker_id: int,
        log_queue: Queue,
        video_id: str = "",
        level: int = logging.DEBUG,
    ):
        """
        Initialize worker logger.
        
        Args:
            worker_id: Unique identifier for this worker
            log_queue: Queue for sending messages to collector
            video_id: Current video being processed (can be updated)
            level: Logging level
        """
        self.worker_id = worker_id
        self.log_queue = log_queue
        self.video_id = video_id
        self.level = level
        self._group_id = ""
    
    def set_video_id(self, video_id: str):
        """Update the current video ID."""
        self.video_id = video_id
    
    def start_video_group(self, video_id: str):
        """Start a new video group for message grouping."""
        self.video_id = video_id
        self._group_id = f"{video_id}_{datetime.now().timestamp()}"
    
    def end_video_group(self):
        """End the current video group."""
        # Send end marker
        msg = LogMessage(
            worker_id=self.worker_id,
            video_id=self.video_id,
            level=LogLevel.INFO,
            message="",
            group_id=self._group_id,
            group_end=True,
        )
        self._send(msg)
        self._group_id = ""
        self.video_id = ""
    
    def _send(self, msg: LogMessage):
        """Send a log message to the queue."""
        try:
            self.log_queue.put(msg)
        except Exception as e:
            # Fallback to print if queue fails
            print(f"[Worker {self.worker_id}] Log queue error: {e}")
            print(f"[Worker {self.worker_id}] {msg.message}")
    
    def _log(self, level: LogLevel, message: str):
        """Log a message at the specified level."""
        if level.value < self.level:
            return
        
        msg = LogMessage(
            worker_id=self.worker_id,
            video_id=self.video_id,
            level=level,
            message=message,
            group_id=self._group_id,
        )
        self._send(msg)
    
    def debug(self, message: str):
        """Log debug message."""
        self._log(LogLevel.DEBUG, message)
    
    def info(self, message: str):
        """Log info message."""
        self._log(LogLevel.INFO, message)
    
    def warning(self, message: str):
        """Log warning message."""
        self._log(LogLevel.WARNING, message)
    
    def error(self, message: str):
        """Log error message."""
        self._log(LogLevel.ERROR, message)
    
    def critical(self, message: str):
        """Log critical message."""
        self._log(LogLevel.CRITICAL, message)
    
    # Alias for compatibility
    warn = warning


def create_worker_logger(
    worker_id: int,
    log_queue: Queue,
    level: int = logging.DEBUG,
) -> WorkerLogger:
    """
    Create a WorkerLogger for multiprocessing.
    
    Args:
        worker_id: Unique identifier for this worker
        log_queue: Queue for sending messages to collector
        level: Logging level
        
    Returns:
        WorkerLogger instance
    """
    return WorkerLogger(
        worker_id=worker_id,
        log_queue=log_queue,
        level=level,
    )


def start_log_collector(
    log_queue: Queue,
    output_dir: str,
    num_workers: int,
    console_output: bool = True,
) -> LogCollector:
    """
    Create and start the log collector.
    
    Args:
        log_queue: Queue for receiving log messages
        output_dir: Base output directory
        num_workers: Number of worker processes
        console_output: Whether to output to console
        
    Returns:
        Started LogCollector instance (run in separate thread)
    """
    collector = LogCollector(
        log_queue=log_queue,
        output_dir=output_dir,
        num_workers=num_workers,
        console_output=console_output,
    )
    
    # Run in background thread
    thread = threading.Thread(target=collector.run, daemon=True)
    thread.start()
    
    return collector


# ==============================================================================
# Standard Logging Functions (unchanged, for single-process mode)
# ==============================================================================


def setup_logger(
    name: str,
    output_dir: str,
    level: str = "INFO",
    console_output: bool = True,
) -> logging.Logger:
    """
    Setup a logger with file and optional console output.
    
    Args:
        name: Logger name
        output_dir: Directory for log file
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        console_output: Whether to also output to console
    
    Returns:
        Configured logger
    """
    os.makedirs(output_dir, exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers = []
    
    # File handler
    log_file = os.path.join(output_dir, "logging.log")
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)  # Log everything to file
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, level.upper()))
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # File format (more detailed)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    return logger


def create_video_logger(
    video_id: str,
    output_dir: str,
    parent_logger: logging.Logger = None,
) -> logging.Logger:
    """
    Create a logger for a specific video.
    
    Args:
        video_id: Video identifier
        output_dir: Directory for video-specific logs
        parent_logger: Optional parent logger to inherit handlers
    
    Returns:
        Video-specific logger
    """
    logger_name = f"VideoAgent.Video.{video_id}"
    logger = logging.getLogger(logger_name)
    
    if parent_logger:
        # Inherit level from parent
        logger.setLevel(parent_logger.level)
        # Use parent's handlers
        logger.handlers = []
        logger.parent = parent_logger
    
    return logger


def log_tool_call(
    logger: logging.Logger,
    tool_name: str,
    tool_args: Dict[str, Any],
    result: str,
    success: bool = True,
    error: str = None,
    duration_ms: float = None,
):
    """
    Log a tool call with structured format (full output, no truncation).
    
    Args:
        logger: Logger instance
        tool_name: Name of the tool called
        tool_args: Arguments passed to the tool
        result: Tool result (full, not truncated)
        success: Whether the call succeeded
        error: Error message if failed
        duration_ms: Execution time in milliseconds
    """
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    
    # Full args (no truncation)
    args_str = json.dumps(tool_args, ensure_ascii=False, default=str)
    
    logger.info(f"[{timestamp}] [TOOL] === Tool Call ===")
    logger.info(f"[{timestamp}] [TOOL] Name: {tool_name}")
    logger.info(f"[{timestamp}] [TOOL] Args: {args_str}")
    
    if duration_ms is not None:
        logger.info(f"[{timestamp}] [TOOL] Duration: {duration_ms:.1f}ms")
    
    if success:
        logger.info(f"[{timestamp}] [TOOL] Result: {result}")
    else:
        logger.error(f"[{timestamp}] [TOOL] ERROR: {error}")
        logger.error(f"[{timestamp}] [TOOL] Partial Result: {result}")


def log_llm_interaction(
    logger: logging.Logger,
    model: str,
    input_messages: List[Dict[str, Any]],
    output: str,
    is_tool_call: bool = False,
    tool_name: str = None,
    tool_args: Dict = None,
    answer: int = None,
    duration_ms: float = None,
):
    """
    Log an LLM interaction with structured format (full output, no truncation).
    
    Args:
        logger: Logger instance
        model: Model name
        input_messages: Messages sent to LLM
        output: Raw LLM output (full, not truncated)
        is_tool_call: Whether output contains a tool call
        tool_name: Tool name if tool call
        tool_args: Tool arguments if tool call
        answer: Final answer if answer submission
        duration_ms: API call duration
    """
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    
    logger.info(f"[{timestamp}] [LLM] === LLM Interaction ===")
    logger.info(f"[{timestamp}] [LLM] Model: {model}")
    logger.info(f"[{timestamp}] [LLM] Input Messages: {len(input_messages)}")
    
    # Log last user message (full, no truncation)
    for msg in reversed(input_messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            logger.info(f"[{timestamp}] [LLM] Last User Message: {content}")
            break
    
    if duration_ms is not None:
        logger.info(f"[{timestamp}] [LLM] Duration: {duration_ms:.1f}ms")
    
    # Log full output (no truncation)
    logger.info(f"[{timestamp}] [LLM] Output: {output}")
    
    # Log parsed result
    if is_tool_call:
        logger.info(f"[{timestamp}] [LLM] Parsed: TOOL_CALL -> {tool_name}")
        if tool_args:
            args_str = json.dumps(tool_args, ensure_ascii=False, default=str)
            logger.info(f"[{timestamp}] [LLM] Tool Args: {args_str}")
    elif answer is not None:
        logger.info(f"[{timestamp}] [LLM] Parsed: ANSWER -> {answer}")


def log_video_start(
    logger: logging.Logger,
    video_id: str,
    question: str,
    choices: List[str],
    initial_frames: int,
):
    """Log the start of video processing."""
    logger.info("=" * 70)
    logger.info(f"Processing Video: {video_id}")
    logger.info("=" * 70)
    logger.info(f"Question: {question}")
    logger.info(f"Choices:")
    for i, choice in enumerate(choices):
        logger.info(f"  {i}. {choice}")
    logger.info(f"Initial frames: {initial_frames}")
    logger.info("-" * 70)


def log_video_end(
    logger: logging.Logger,
    video_id: str,
    final_answer: int,
    truth: int,
    is_correct: bool,
    is_forced: bool,
    tool_calls: int,
    duration_s: float,
):
    """Log the end of video processing."""
    status = "CORRECT" if is_correct else "WRONG"
    forced_str = " (FORCED)" if is_forced else ""
    
    logger.info("-" * 70)
    logger.info(f"[{status}] Video: {video_id}")
    logger.info(f"Final Answer: {final_answer}{forced_str}")
    logger.info(f"Ground Truth: {truth}")
    logger.info(f"Tool Calls: {tool_calls}")
    logger.info(f"Duration: {duration_s:.1f}s")
    logger.info("=" * 70)


def log_caption_generation(
    logger: logging.Logger,
    frame_indices: List[int],
    captions: Dict[int, str],
    duration_ms: float = None,
):
    """Log caption generation for frames (full captions, no truncation)."""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    
    logger.info(f"[{timestamp}] [CAPTION] === Caption Generation ===")
    logger.info(f"[{timestamp}] [CAPTION] Frames: {frame_indices}")
    
    if duration_ms:
        logger.info(f"[{timestamp}] [CAPTION] Duration: {duration_ms:.1f}ms")
    
    for idx in sorted(captions.keys()):
        caption = captions[idx]
        logger.info(f"[{timestamp}] [CAPTION] Frame {idx}: {caption}")


def setup_video_logger(
    video_id: str,
    video_output_dir: str,
    parent_logger: logging.Logger = None,
) -> logging.Logger:
    """
    Create a dedicated logger for a specific video with its own log file.
    
    Args:
        video_id: Video identifier
        video_output_dir: Directory for this video's outputs (will contain llm.log)
        parent_logger: Optional parent logger (for propagating to main log)
    
    Returns:
        Video-specific logger that writes to llm.log (LLM prompts and responses only)
    """
    os.makedirs(video_output_dir, exist_ok=True)
    
    logger_name = f"VideoAgent.Video.{video_id}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    
    # Clear existing handlers
    logger.handlers = []
    
    # File handler for per-video log - use simple format without timestamp prefix for cleaner output
    log_file = os.path.join(video_output_dir, "llm.log")
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    # Simple format - just the message, sections will have their own headers
    file_formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # If parent logger provided, also propagate to parent
    if parent_logger:
        logger.parent = parent_logger
        logger.propagate = True
    else:
        logger.propagate = False
    
    return logger


def log_llm_interaction_full(
    logger: logging.Logger,
    model: str,
    input_messages: List[Dict[str, Any]],
    output: str,
    is_tool_call: bool = False,
    tool_name: str = None,
    tool_args: Dict = None,
    answer: int = None,
    duration_ms: float = None,
    step_number: int = None,
    memory_state: str = None,
):
    """
    Log LLM interaction: INPUT prompts and OUTPUT response only.
    
    This logs ONLY:
    1. The prompts sent to LLM (all messages)
    2. The LLM's response
    
    Uses very clear visual separators for easy reading.
    """
    start_time = datetime.now()
    timestamp_str = start_time.strftime("%Y-%m-%d %H:%M:%S")
    
    # Build clear visual separator
    lines = []
    
    # ==================== INTERACTION HEADER ====================
    lines.append("")
    lines.append("")
    lines.append("█" * 100)
    lines.append("█" + " " * 98 + "█")
    step_str = f"Step {step_number}" if step_number else "LLM Call"
    header_text = f"LLM INTERACTION - {step_str}"
    padding = (98 - len(header_text)) // 2
    lines.append("█" + " " * padding + header_text + " " * (98 - padding - len(header_text)) + "█")
    lines.append("█" + " " * 98 + "█")
    lines.append("█" * 100)
    lines.append("")
    lines.append(f"┌{'─' * 98}┐")
    lines.append(f"│  Start Time: {timestamp_str:<82} │")
    lines.append(f"│  Model: {model:<88} │")
    lines.append(f"└{'─' * 98}┘")
    lines.append("")
    
    # ==================== INPUT SECTION ====================
    lines.append("╔" + "═" * 98 + "╗")
    lines.append("║" + " " * 40 + ">>> INPUT TO LLM <<<" + " " * 38 + "║")
    lines.append("╚" + "═" * 98 + "╝")
    lines.append("")
    
    # Log all messages sent to LLM
    for i, msg in enumerate(input_messages):
        role = msg.get("role", "unknown").upper()
        content = msg.get("content", "")
        
        # Role header with box
        lines.append(f"┌──────────────────────────────────────────────────────────────────────────────────────────────────┐")
        lines.append(f"│  MESSAGE {i+1}: [{role}]")
        lines.append(f"└──────────────────────────────────────────────────────────────────────────────────────────────────┘")
        
        # Handle multimodal content (with images)
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        lines.append(item.get("text", ""))
                    elif item.get("type") == "image_url":
                        lines.append("[IMAGE CONTENT - base64 encoded]")
                else:
                    lines.append(str(item))
        else:
            lines.append(str(content))
        
        lines.append("")
    
    lines.append("─" * 100)
    lines.append("")
    
    # ==================== OUTPUT SECTION ====================
    lines.append("╔" + "═" * 98 + "╗")
    lines.append("║" + " " * 40 + "<<< LLM OUTPUT >>>" + " " * 40 + "║")
    lines.append("╚" + "═" * 98 + "╝")
    lines.append("")
    
    # Full LLM output
    lines.append(output)
    lines.append("")
    
    # ==================== TIMING INFO ====================
    end_time = datetime.now()
    end_timestamp_str = end_time.strftime("%Y-%m-%d %H:%M:%S")
    
    lines.append("─" * 100)
    lines.append(f"┌{'─' * 98}┐")
    lines.append(f"│  End Time: {end_timestamp_str:<84} │")
    if duration_ms is not None:
        duration_str = f"{duration_ms:.1f}ms ({duration_ms/1000:.2f}s)"
        lines.append(f"│  LLM Response Time: {duration_str:<76} │")
    lines.append(f"└{'─' * 98}┘")
    lines.append("")
    
    # ==================== END SEPARATOR ====================
    lines.append("█" * 100)
    lines.append("")
    lines.append("")
    
    # Write all lines at once
    logger.info("\n".join(lines))


def log_tool_call_full(
    logger: logging.Logger,
    tool_name: str,
    tool_args: Dict[str, Any],
    result: str,
    success: bool = True,
    error: str = None,
    duration_ms: float = None,
    step_number: int = None,
    tool_index: int = None,
    total_tools: int = None,
):
    """
    Log a complete tool call WITHOUT truncation for per-video logs.
    Uses a clean, readable format.
    Supports multiple parallel tool calls with tool_index/total_tools.
    """
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    lines = []
    lines.append("")
    lines.append("+" * 80)
    step_str = f" [Step {step_number}]" if step_number else ""
    # Show tool index if multiple tools (e.g., "Tool 1/3")
    tool_idx_str = f" Tool {tool_index}/{total_tools}" if total_tools and total_tools > 1 else ""
    lines.append(f"TOOL EXECUTION{step_str}{tool_idx_str} @ {timestamp}")
    lines.append("+" * 80)
    status_str = "SUCCESS" if success else "FAILED"
    lines.append(f"Tool: {tool_name} ({status_str})")
    if duration_ms is not None:
        lines.append(f"Execution Time: {duration_ms:.1f}ms ({duration_ms/1000:.2f}s)")
    if error:
        lines.append(f"Error: {error}")
    lines.append("")
    
    # Log arguments
    lines.append("--- INPUT ---")
    args_str = json.dumps(tool_args, ensure_ascii=False, indent=2, default=str)
    lines.append(args_str)
    lines.append("")
    
    # Log result
    lines.append("--- OUTPUT ---")
    lines.append(result)
    lines.append("+" * 80)
    lines.append("")
    
    logger.info("\n".join(lines))


def save_video_result(
    video_output_dir: str,
    video_id: str,
    question: str,
    choices: List[str],
    final_answer: int,
    explanation: str,
    truth: int,
    is_correct: bool,
    is_forced: bool,
    is_valid: bool,
    tool_history: List[Dict[str, Any]],
    video_context_dict: Dict[str, Any],
    messages: List[Dict[str, Any]],
    duration_s: float = None,
    tool_call_rounds: int = None,
):
    """
    Save per-video result.json with memory state and conversation history.
    
    Args:
        video_output_dir: Directory for this video's outputs
        video_id: Video identifier
        question: The question asked
        choices: List of multiple choice options
        final_answer: The selected answer (0-4)
        explanation: Agent's explanation
        truth: Ground truth answer
        is_correct: Whether answer was correct
        is_forced: Whether answer was forced
        is_valid: Whether result is valid
        tool_history: List of tool call dicts
        video_context_dict: VideoContext as dict (contains frame_captions)
        messages: Full conversation message history
        duration_s: Processing duration in seconds
    """
    os.makedirs(video_output_dir, exist_ok=True)
    
    # Create simplified tool history
    tool_history_summary = []
    for tc in tool_history:
        tool_history_summary.append({
            "tool_name": tc.get("tool_name", "unknown"),
            "tool_args": tc.get("tool_args", {}),
            "success": tc.get("success", False),
            "timestamp": tc.get("timestamp", ""),
        })
    
    # Create memory state from video context
    # Sort sampled_indices chronologically for better readability
    sampled_indices = video_context_dict.get("sampled_indices", [])
    sorted_indices = sorted(sampled_indices)
    
    # Sort frame_captions by frame index
    frame_captions = video_context_dict.get("frame_captions", {})
    sorted_captions = {k: frame_captions[k] for k in sorted(frame_captions.keys(), key=lambda x: int(x) if isinstance(x, str) else x)}
    
    memory_state = {
        "video_id": video_context_dict.get("video_id", video_id),
        "total_frames": video_context_dict.get("total_frames", 0),
        "fps": video_context_dict.get("fps", 0),
        "duration": video_context_dict.get("duration", 0),
        "sampled_indices": sorted_indices,
        "frame_captions": sorted_captions,
    }
    
    # Create result structure
    result = {
        "video_id": video_id,
        "question": question,
        "choices": choices,
        "final_answer": final_answer,
        "explanation": explanation,
        "truth": truth,
        "is_correct": is_correct,
        "is_forced": is_forced,
        "is_valid": is_valid,
        "tool_call_count": len(tool_history),
        "tool_call_rounds": tool_call_rounds if tool_call_rounds is not None else len(tool_history),
        "frame_count": len(video_context_dict.get("frame_captions", {})),
        "duration_s": duration_s,
        "tool_history": tool_history,
        "tool_history_summary": tool_history_summary,
        "memory_state": memory_state,
        "conversation_history": messages,
    }
    
    result_file = os.path.join(video_output_dir, "result.json")
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)


def save_video_frames(
    video_output_dir: str,
    frames: Dict[int, Any],
    video_id: str = None,
    fps: float = 30.0,
    is_rgb: bool = True,
):
    """
    Save video frames as PNG files with frame index and timestamp in filename.
    
    Args:
        video_output_dir: Directory for this video's outputs
        frames: Dict mapping frame indices to numpy arrays
        video_id: Optional video ID for logging
        fps: Frames per second (for calculating timestamp)
        is_rgb: If True, frames are in RGB format and will be converted to BGR for saving.
                If False, frames are already in BGR format.
    """
    frames_dir = os.path.join(video_output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    
    for idx, frame in frames.items():
        if frame is None:
            continue
        
        # Calculate timestamp from frame index
        timestamp = idx / fps if fps > 0 else 0.0
        # Format: frame_{index}_{timestamp}s.png (e.g., frame_150_5.0s.png)
        frame_path = os.path.join(frames_dir, f"frame_{idx}_{timestamp:.1f}s.png")
        try:
            if isinstance(frame, np.ndarray):
                # Convert RGB to BGR for cv2.imwrite if needed
                if is_rgb and len(frame.shape) == 3 and frame.shape[2] == 3:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                else:
                    frame_bgr = frame
                cv2.imwrite(frame_path, frame_bgr)
        except Exception as e:
            # Silently skip frames that fail to save
            pass


def setup_simple_logger(
    video_id: str,
    video_output_dir: str,
    console_output: bool = False,
) -> logging.Logger:
    """
    Create a simple logger for key tool call information only.
    
    This logger outputs to simple_log.log and optionally to console.
    Contains:
    - Tool call inputs (full, not truncated)
    - Tool call outputs (full, not truncated)
    - No repeated prompts or memory state information
    
    Args:
        video_id: Video identifier
        video_output_dir: Directory for this video's outputs
        console_output: If True, also print to console (for detailed mode)
    
    Returns:
        Simple logger that writes to simple_log.log (and console if enabled)
    """
    os.makedirs(video_output_dir, exist_ok=True)
    
    logger_name = f"VideoAgent.Simple.{video_id}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    
    # Clear existing handlers
    logger.handlers = []
    
    # File handler for simple log
    log_file = os.path.join(video_output_dir, "simple_log.log")
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    # Simple format - just the message
    file_formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Also write to logging.log (unified with print output)
    main_log_file = os.path.join(video_output_dir, "logging.log")
    main_file_handler = logging.FileHandler(main_log_file, encoding='utf-8')
    main_file_handler.setLevel(logging.DEBUG)
    main_file_handler.setFormatter(file_formatter)
    logger.addHandler(main_file_handler)
    
    # Console handler (for detailed mode - prints same content as simple_log)
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        # Same simple format for console
        console_formatter = logging.Formatter('%(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # Don't propagate to parent loggers
    logger.propagate = False
    
    return logger


def log_tool_call_simple(
    logger: logging.Logger,
    tool_name: str,
    tool_args: Dict[str, Any],
    result: str,
    success: bool = True,
    error: str = None,
    duration_ms: float = None,
    step_number: int = None,
    tool_index: int = None,
    total_tools: int = None,
):
    """
    Log a tool call with FULL input and output (no truncation).
    Clean format without repeated prompts or memory information.
    Supports multiple parallel tool calls with tool_index/total_tools.
    
    Args:
        logger: Simple logger instance
        tool_name: Name of the tool called
        tool_args: Arguments passed to the tool (logged in full)
        result: Tool result (logged in full, no truncation)
        success: Whether the call succeeded
        error: Error message if failed
        duration_ms: Execution time in milliseconds
        step_number: Step number in the agent execution
        tool_index: Index of this tool in parallel execution (1-based)
        total_tools: Total number of tools being executed in parallel
    """
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    lines = []
    lines.append("")
    lines.append("=" * 60)
    step_str = f" [Step {step_number}]" if step_number else ""
    # Show tool index if multiple tools (e.g., "Tool 1/3")
    tool_idx_str = f" Tool {tool_index}/{total_tools}" if total_tools and total_tools > 1 else ""
    status_str = "SUCCESS" if success else "FAILED"
    lines.append(f"TOOL EXECUTION{step_str}{tool_idx_str} @ {timestamp}")
    lines.append(f"Tool: {tool_name} ({status_str})")
    lines.append("=" * 60)
    
    if duration_ms is not None:
        lines.append(f"Execution Time: {duration_ms:.1f}ms ({duration_ms/1000:.2f}s)")
    
    if error:
        lines.append(f"Error: {error}")
    
    lines.append("")
    lines.append("--- INPUT ---")
    # Full arguments, formatted nicely
    args_str = json.dumps(tool_args, ensure_ascii=False, indent=2, default=str)
    lines.append(args_str)
    
    lines.append("")
    lines.append("--- OUTPUT ---")
    # Full result, no truncation
    lines.append(result)
    
    lines.append("=" * 60)
    lines.append("")
    
    logger.info("\n".join(lines))


def log_agent_action_simple(
    logger: logging.Logger,
    action: str,
    llm_output: str = None,
    tool_name: str = None,
    tool_args: Dict[str, Any] = None,
    tool_calls: List[Dict[str, Any]] = None,
    answer: int = None,
    explanation: str = None,
    step_number: int = None,
    duration_ms: float = None,
    error_reason: str = None,
    is_retry: bool = False,
):
    """
    Log agent decision/action in simple format with full LLM output.
    Supports both single and multiple tool calls.
    
    Args:
        logger: Simple logger instance
        action: Action type (tool_call, tool_calls, submit_answer, error)
        llm_output: Full LLM response (including thinking, not truncated)
        tool_name: Tool name if single tool call
        tool_args: Tool arguments if single tool call
        tool_calls: List of tool calls if multiple (each with tool_name, tool_args)
        answer: Final answer if answer submission
        explanation: Explanation if answer submission
        step_number: Step number
        duration_ms: LLM call duration
        error_reason: Reason for parse error (if action is error)
        is_retry: Whether this is a retry attempt after an error
    """
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    lines = []
    lines.append("")
    lines.append("-" * 60)
    step_str = f" [Step {step_number}]" if step_number else ""
    retry_str = " (RETRY)" if is_retry else ""
    lines.append(f"AGENT DECISION{step_str}{retry_str} @ {timestamp}")
    lines.append("-" * 60)
    
    if duration_ms is not None:
        lines.append(f"LLM Response Time: {duration_ms:.1f}ms ({duration_ms/1000:.2f}s)")
    
    # Log full LLM output (complete response including thinking)
    if llm_output:
        lines.append("")
        lines.append("--- LLM FULL OUTPUT ---")
        lines.append(llm_output)
        lines.append("--- END LLM OUTPUT ---")
    
    lines.append("")
    lines.append("PARSED ACTION:")
    if action == "tool_calls" and tool_calls:
        # Multiple tool calls
        lines.append(f"  Action: CALL {len(tool_calls)} TOOLS")
        for i, tc in enumerate(tool_calls, 1):
            lines.append(f"  [{i}] {tc.get('tool_name', 'unknown')}")
            if tc.get('tool_args'):
                args_str = json.dumps(tc['tool_args'], ensure_ascii=False, indent=4, default=str)
                lines.append(f"      Args: {args_str}")
    elif action == "tool_call":
        lines.append(f"  Action: CALL TOOL")
        lines.append(f"  Tool: {tool_name}")
        if tool_args:
            args_str = json.dumps(tool_args, ensure_ascii=False, indent=2, default=str)
            lines.append(f"  Arguments:\n{args_str}")
    elif action == "submit_answer":
        lines.append(f"  Action: SUBMIT ANSWER")
        lines.append(f"  Answer: {answer}")
        if explanation:
            lines.append(f"  Explanation: {explanation}")
    elif action == "error":
        lines.append(f"  Action: PARSE ERROR")
        if error_reason:
            lines.append(f"  Reason: {error_reason}")
        lines.append("  (Will retry with error feedback)")
    else:
        lines.append(f"  Action: {action}")
    
    lines.append("-" * 60)
    
    logger.info("\n".join(lines))


def log_session_start_simple(
    logger: logging.Logger,
    video_id: str,
    question: str,
    choices: List[str],
    initial_frames: int,
    model: str,
    start_time: datetime = None,
):
    """Log session start in simple format with timestamp."""
    if start_time is None:
        start_time = datetime.now()
    
    lines = []
    lines.append("=" * 60)
    lines.append("VIDEO AGENT SESSION")
    lines.append("=" * 60)
    lines.append(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Video ID: {video_id}")
    lines.append(f"Model: {model}")
    lines.append("")
    lines.append("QUESTION:")
    lines.append(question)
    lines.append("")
    lines.append("CHOICES:")
    for i, choice in enumerate(choices):
        lines.append(f"  {i}. {choice}")
    lines.append("")
    lines.append(f"Initial frames: {initial_frames}")
    lines.append("=" * 60)
    
    logger.info("\n".join(lines))


def log_session_end_simple(
    logger: logging.Logger,
    video_id: str,
    final_answer: int,
    truth: int,
    is_correct: bool,
    is_forced: bool,
    tool_calls: int,
    frame_count: int,
    duration_s: float,
    start_time: datetime = None,
    end_time: datetime = None,
):
    """Log session end in simple format with timing summary."""
    if end_time is None:
        end_time = datetime.now()
    
    lines = []
    lines.append("")
    lines.append("=" * 60)
    lines.append("SESSION RESULT")
    lines.append("=" * 60)
    lines.append(f"End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    if start_time:
        lines.append(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Total Duration: {duration_s:.1f}s")
    lines.append("")
    status = "CORRECT" if is_correct else "WRONG"
    forced_str = " (FORCED)" if is_forced else ""
    lines.append(f"Status: {status}{forced_str}")
    lines.append(f"Final Answer: {final_answer}")
    lines.append(f"Ground Truth: {truth}")
    lines.append(f"Tool Calls: {tool_calls}")
    lines.append(f"Frames Used: {frame_count}")
    lines.append("=" * 60)
    
    logger.info("\n".join(lines))


def log_initial_captions_simple(
    logger: logging.Logger,
    frame_indices: List[int],
    captions: Dict[int, str],
    fps: float,
    duration_ms: float = None,
    cache_hit: bool = False,
):
    """Log initial frame captions in simple format (full captions, no truncation)."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    lines = []
    lines.append("")
    lines.append("-" * 60)
    lines.append(f"INITIAL FRAME CAPTIONS @ {timestamp}")
    lines.append("-" * 60)
    lines.append(f"Frames: {frame_indices}")
    if cache_hit:
        lines.append(f"Source: FROM CACHE (0.0ms)")
    elif duration_ms:
        lines.append(f"Caption Generation Time: {duration_ms:.1f}ms ({duration_ms/1000:.2f}s)")
    lines.append("")
    
    for idx in sorted(captions.keys()):
        frame_timestamp = idx / fps if fps > 0 else 0.0
        lines.append(f"[Frame {idx} @ {frame_timestamp:.1f}s]:")
        lines.append(captions[idx])
        lines.append("")
    
    lines.append("-" * 60)
    
    logger.info("\n".join(lines))


