"""
Configuration management for VideoAgent project.
"""

import os
import yaml
from typing import Dict, Any


def _load_env_file():
    """Load environment variables from .env file if it exists."""
    possible_paths = [
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), '.env'),
        os.path.join(os.getcwd(), '.env'),
    ]
    
    for env_file in possible_paths:
        if os.path.exists(env_file):
            with open(env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ.setdefault(key.strip(), value.strip())
            break


# Load .env file on module import
_load_env_file()


def load_config(config_name: str = "default") -> Dict[str, Any]:
    """
    Load configuration from YAML file with environment variable overrides.
    
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
            config = yaml.safe_load(f)
    except (FileNotFoundError, yaml.YAMLError) as e:
        print(f"Error loading config: {e}")
        raise
    
    # Override with environment variables
    config['aiml_api_key'] = os.environ.get('AIML_API_KEY', '')
    config['aiml_base_url'] = os.environ.get('AIML_BASE_URL', 'https://api.aimlapi.com/v1')
    
    return config


def list_configs() -> list:
    """List available configurations."""
    configs_dir = "configs"
    if not os.path.exists(configs_dir):
        return []
    
    configs = []
    for filename in os.listdir(configs_dir):
        if filename.endswith('.yaml'):
            configs.append(filename[:-5])
    
    return sorted(configs)


def update_config(config: Dict[str, Any], **overrides) -> Dict[str, Any]:
    """Update configuration with overrides."""
    updated = config.copy()
    for key, value in overrides.items():
        if value is not None:
            updated[key] = value
    return updated


def save_config_to_output(config: Dict[str, Any], output_dir: str):
    """Save configuration to output directory as YAML (without sensitive data)."""
    os.makedirs(output_dir, exist_ok=True)
    
    safe_config = config.copy()
    if 'aiml_api_key' in safe_config:
        safe_config['aiml_api_key'] = '***REDACTED***'
    
    config_file = os.path.join(output_dir, "experiment_config.yaml")
    with open(config_file, 'w') as f:
        yaml.dump(safe_config, f, indent=2, default_flow_style=False)


# Default values
DEFAULT_SCHEDULER_MODEL = "gpt-4o-mini-2024-07-18"
DEFAULT_VIEWER_MODEL = "gpt-4o-mini-2024-07-18"
MAX_ROUNDS = 10
MAX_RETRIEVED_FRAMES = 5
MIN_RETRIEVED_FRAMES = 2
DEFAULT_INITIAL_FRAMES = 5
REFERENCE_LENGTH = 50
INPUT_FRAME_INTERVAL = 30

# API credentials from environment
AIML_API_KEY = os.environ.get('AIML_API_KEY', '')
AIML_BASE_URL = os.environ.get('AIML_BASE_URL', 'https://api.aimlapi.com/v1')

# LLM settings
LLM_TEMPERATURE = 0.7
LLM_MAX_TOKENS = 10000

# Paths
DATASET_DIR = "data/EgoSchema_test"
CACHE_DIR = "cache"
OUTPUT_DIR = "results"
VIDEO_DIR = "data/EgoSchema_test/videos"
ANNOTATION_FILE = "data/EgoSchema_test/annotations.json"
TEST_VIDEO_LIST_FILE = "data/EgoSchema_test/video_list.txt"

# Cache settings
USE_CACHE = True
CACHE_LLM_FILE = "cache/cache_llm.pkl"
CACHE_CLIP_FILE = "cache/cache_clip.pkl"

# Logging
LOG_LEVEL = "INFO"

# Processing
DEFAULT_MAX_PROCESSES = 10
DEFAULT_MAX_TEST_VIDEOS = -1

