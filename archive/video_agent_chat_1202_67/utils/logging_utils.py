"""
Logging utilities for VideoAgent.
"""

import os
import logging


def setup_logger(name: str, output_dir: str, level: str = "INFO", 
                enable_llm_logging: bool = False) -> logging.Logger:
    """
    Setup logger for experiment.
    
    Args:
        name: Logger name
        output_dir: Output directory for log files
        level: Logging level
        enable_llm_logging: Whether to enable LLM detailed logging
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    
    log_level = logging.DEBUG if enable_llm_logging else getattr(logging, level.upper())
    logger.setLevel(log_level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # File handler
    os.makedirs(output_dir, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(output_dir, "logging.log"))
    file_handler.setLevel(log_level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


def create_logger(log_file: str) -> logging.Logger:
    """
    Create logger for a specific file.
    
    Args:
        log_file: Path to log file
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(f"VideoAgent_{os.path.basename(log_file)}")
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # File handler
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    handler = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger


def create_video_logger(video_id: str, output_dir: str) -> logging.Logger:
    """
    Create logger for a specific video processing.
    
    Args:
        video_id: Video identifier
        output_dir: Output directory for log file
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(f"VideoAgent.Video.{video_id}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    os.makedirs(output_dir, exist_ok=True)
    video_log_file = os.path.join(output_dir, "logging.log")
    video_handler = logging.FileHandler(video_log_file)
    video_handler.setLevel(logging.INFO)
    video_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    video_handler.setFormatter(video_formatter)
    logger.addHandler(video_handler)
    
    return logger
