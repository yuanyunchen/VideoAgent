"""
Configuration management for VideoAgent project.
"""

import os
import json
import yaml
from typing import Dict, Any

def load_config(config_name: str = "default") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_name: Name of configuration file (without .yaml extension)
        
    Returns:
        Configuration dictionary
    """
    config_file = os.path.join("configs", f"{config_name}.yaml")
    
    if not os.path.exists(config_file):
        print(f"Unknown config '{config_name}'. Using 'default'.")
        config_file = os.path.join("configs", "default.yaml")
    
    try:
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    except (FileNotFoundError, yaml.YAMLError) as e:
        print(f"Error loading config: {e}")
        raise

def list_configs() -> list:
    """List available configurations."""
    configs_dir = "configs"
    if not os.path.exists(configs_dir):
        return []
    
    configs = []
    for filename in os.listdir(configs_dir):
        if filename.endswith('.yaml'):
            configs.append(filename[:-5])  # Remove .yaml extension
    
    return sorted(configs)

def update_config(config: Dict[str, Any], **overrides) -> Dict[str, Any]:
    """Update configuration with overrides."""
    updated = config.copy()
    for key, value in overrides.items():
        if value is not None:
            updated[key] = value
    return updated

def save_config_to_output(config: Dict[str, Any], output_dir: str):
    """Save configuration to output directory as YAML."""
    os.makedirs(output_dir, exist_ok=True)
    config_file = os.path.join(output_dir, "experiment_config.yaml")
    with open(config_file, 'w') as f:
        yaml.dump(config, f, indent=2, default_flow_style=False)



# Backward compatibility constants - using default values
DEFAULT_SCHEDULER_MODEL = "gpt-4o-mini-2024-07-18"
DEFAULT_VIEWER_MODEL = "gpt-4o-mini-2024-07-18"
MAX_ROUNDS = 1
MAX_RETRIEVED_FRAMES = 5
MIN_RETRIEVED_FRAMES = 2
DEFAULT_INITIAL_FRAMES = 5
REFERENCE_LENGTH = 50
INPUT_FRAME_INTERVAL = 30
AIML_API_KEY = "fb50dec85566407bbc25ce1d28828fe7"
AIML_BASE_URL = "https://api.aimlapi.com/v1"
LLM_TEMPERATURE = 0.7
LLM_MAX_TOKENS = 10000
DATASET_DIR = "dataset"
CACHE_DIR = "cache"
OUTPUT_DIR = "output"
VIDEO_DIR = "dataset/videos"
ANNOTATION_FILE = "dataset/subset_anno.json"
TEST_VIDEO_LIST_FILE = "dataset/test_one_video.txt"
USE_CACHE = True
CACHE_LLM_FILE = "cache/cache_llm.pkl"
CACHE_CLIP_FILE = "cache/cache_clip.pkl"
LOG_LEVEL = "INFO"
DEFAULT_MAX_PROCESSES = 10
DEFAULT_MAX_TEST_VIDEOS = -1
DETAILED_CAPTION_PROMPT = f"""describe the scene in a clear, concise caption. Include key details such as:
    - Main objects or people present (tag with #C if the action is done by the camera wearer, #O if done by someone else)
    - Their spatial relationships.
    - other visual elements. 
    Focus on what is visually prominent and avoid speculation beyond what is shown. Do not provide any prediction of events happening. 
    Keep the caption length close to {REFERENCE_LENGTH} words to control detail level."""
CONFIDENCE_LEVEL_GUIDANCE = "When evaluating confidence, remain cautious and avoid overconfidence. If there's any uncertainty or need for more information, assign a confidence level of '2'. Prioritize accuracy by frequently using '2' when evidence is incomplete or unclear." 