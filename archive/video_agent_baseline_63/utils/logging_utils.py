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
        enable_llm_logging: Whether to enable LLM detailed logging (uses DEBUG level)
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    
    # Set level based on llm_logging flag
    log_level = logging.DEBUG if enable_llm_logging else getattr(logging, level.upper())
    logger.setLevel(log_level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Single file handler - logging.log
    os.makedirs(output_dir, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(output_dir, "logging.log"))
    file_handler.setLevel(log_level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


def create_logger(log_file: str) -> logging.Logger:
    """
    Create logger (backward compatibility).
    
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

