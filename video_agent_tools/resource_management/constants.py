"""
Constants for Resource Management

Contains tool memory estimates, priorities, and configuration constants.
"""

from enum import Enum


class ToolPriority(Enum):
    """Priority levels for tools (higher = less likely to be evicted)."""
    LOW = 1       # Rarely used, can be evicted first
    MEDIUM = 2    # Normal tools
    HIGH = 3      # Frequently used, evict last
    CRITICAL = 4  # Never evict (e.g., captioner)


# Estimated GPU memory usage per tool in MB (conservative estimates)
TOOL_MEMORY_ESTIMATES = {
    # Captioning tools
    "omni_captioner": 3500,      # OmniCaptioner ~3.5GB
    "api_captioner": 0,          # API-based, no GPU memory
    
    # InternVideo2.5 tools (~17GB actual, reserve extra for inference)
    "internvideo_general_qa": 17000,
    "internvideo_description": 17000,  # Shared with general_qa
    
    # Detection tools
    "detect_objects": 800,       # YOLO-World ~800MB
    "detect_all_objects": 600,   # YOLOE ~600MB
    
    # Sampling tools
    "temporal_sample_frames": 1500,    # VideoTree ~1.5GB
    "temporal_spatial_sample_frames": 1200,  # TStar uses MobileCLIP
    
    # Description tools
    "describe_region": 2000,     # DAM ~2GB
    
    # QA tools
    "temporal_qa": 3000,         # VideoRAG
    "temporal_spatial_qa": 1500, # TStar QA
    "general_vqa": 0,            # API-based
    "targeting_vqa": 4000,       # VStar (needs more memory for LLM)
    
    # View/caption (uses captioner)
    "view_frame": 0,             # Uses captioner
    "caption_image": 0,          # Uses captioner
    "detailed_captioning": 0,    # API-based
}

# Default tool priorities for eviction
TOOL_PRIORITIES = {
    "omni_captioner": ToolPriority.CRITICAL,
    "api_captioner": ToolPriority.CRITICAL,
    "view_frame": ToolPriority.HIGH,
    "caption_image": ToolPriority.HIGH,
    "temporal_sample_frames": ToolPriority.MEDIUM,
    "temporal_spatial_sample_frames": ToolPriority.MEDIUM,
    "internvideo_general_qa": ToolPriority.LOW,
    "internvideo_description": ToolPriority.LOW,
    "detect_objects": ToolPriority.MEDIUM,
    "detect_all_objects": ToolPriority.MEDIUM,
    "describe_region": ToolPriority.MEDIUM,
    "temporal_qa": ToolPriority.MEDIUM,
    "temporal_spatial_qa": ToolPriority.MEDIUM,
    "general_vqa": ToolPriority.MEDIUM,
    "targeting_vqa": ToolPriority.MEDIUM,
    "detailed_captioning": ToolPriority.MEDIUM,
}

# Maximum tool instances - None means no limit (dynamically determined by GPU count and memory)
# API-based tools have no GPU memory constraints
TOOL_MAX_INSTANCES = {
    # GPU-based tools: None = unlimited (auto-scale based on GPU memory)
    "omni_captioner": None,
    "internvideo_general_qa": None,
    "internvideo_description": None,
    "detect_objects": None,
    "detect_all_objects": None,
    "temporal_sample_frames": None,
    "temporal_spatial_sample_frames": None,
    "describe_region": None,
    "temporal_qa": None,
    "temporal_spatial_qa": None,
    "targeting_vqa": None,
    # API-based tools: unlimited
    "api_captioner": None,
    "general_vqa": None,
    "view_frame": None,
    "caption_image": None,
    "detailed_captioning": None,
}

# Tools that share the same underlying model (load only once)
SHARED_MODEL_GROUPS = {
    "internvideo": ["internvideo_general_qa", "internvideo_description"],
    "tstar": ["temporal_spatial_sample_frames", "temporal_spatial_qa"],
}

# Memory buffer to keep free on each GPU (MB)
MEMORY_BUFFER_MB = 3000  # Buffer for inference overhead and CUDA operations

# Queue pressure threshold for tool replication
QUEUE_PRESSURE_THRESHOLD = 3


