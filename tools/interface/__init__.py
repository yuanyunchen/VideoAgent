"""
VideoAgent Unified Interface System

Supports staged interaction with agent:
- Stage 1: get_tools_summary() -> Tool selection
- Stage 2: cls.get_input_prompt() -> Input specification
- Stage 3: cls.format_output_for_agent() -> Output presentation

Unified interface for external callers:
- get_tool_registry(enabled_tools) -> Get tool descriptions for agent prompts
- validate_tool_args(tool_name, args) -> Validate tool arguments
- execute_tool(tool_name, tool_instance, args) -> Execute tool with formatted output
"""

from typing import Dict, Any, List, Optional, Set

# Re-export base classes and types
from tools.interface_base import (
    Interface,
    InterfaceCategory,
    Video,
    Image,
    BoundingBox,
    Point,
    generate_tools_summary,
    generate_tools_summary_by_category,
)

# Import concrete interfaces
from tools.interface.videorag_qa import VideoRAGTemporalQA
from tools.interface.internvideo2_5_interface import (
    InternVideoGeneralQA,
    InternVideoDescription,
)
from tools.interface.temporal_spatial_understanding import (
    VideoTreeSampling,
    TStarSampling,
    TStarTemporalSpatialQA,
    TSTAR_SEARCH_BUDGET,
    TSTAR_FINAL_SAMPLE_FRAMES,
)
from tools.interface.object_detection import YOLOWorldDetection, YOLOEPromptFreeDetection
from tools.interface.object_description import DAMDescription
from tools.interface.view_frame import ViewFrame
from tools.interface.image_captioning import OmniCaptionerCaptioning, APICaptioning
from tools.interface.visual_qa import GeneralVQA, TargetingVQA

# Interface key -> class mapping (main tools for agent)
INTERFACE_MAPPING: Dict[str, type] = {
    # Sub-question Answering (InternVideo2.5-based)
    "internvideo_general_qa": InternVideoGeneralQA,  # General video Q&A with 128 frames
    "internvideo_description": InternVideoDescription,  # Summary + action timeline
    # Other Q&A
    "temporal_spatial_qa": TStarTemporalSpatialQA,
    "general_vqa": GeneralVQA,
    "targeting_vqa": TargetingVQA,
    # Tools - Frame Sampling
    "temporal_sample_frames": VideoTreeSampling,
    "temporal_spatial_sample_frames": TStarSampling,
    # Tools - Detection & Description
    "detect_objects": YOLOWorldDetection,
    "detect_all_objects": YOLOEPromptFreeDetection,
    "describe_region": DAMDescription,
    # Tools - Frame & Caption
    "view_frame": ViewFrame,
    "caption_image": OmniCaptionerCaptioning,
    "detailed_captioning": APICaptioning,
}

# Implementation variants (aliases, not in main tool list)
IMPLEMENTATION_VARIANTS: Dict[str, type] = {
    # Legacy alias
    "api_caption": APICaptioning,
    "api_caption_image": APICaptioning,
    # VideoRAG-based temporal QA (alternative implementation)
    "videorag_temporal_qa": VideoRAGTemporalQA,
}

# Tools that return frames and need auto-captioning
FRAME_RETURNING_TOOLS: Set[str] = {
    "view_frame",
    "temporal_sample_frames",
    "temporal_spatial_sample_frames",
}


def get_interface(key: str) -> type:
    """Get interface class by key (includes implementation variants)."""
    if key in INTERFACE_MAPPING:
        return INTERFACE_MAPPING[key]
    if key in IMPLEMENTATION_VARIANTS:
        return IMPLEMENTATION_VARIANTS[key]
    all_keys = list(INTERFACE_MAPPING.keys()) + list(IMPLEMENTATION_VARIANTS.keys())
    raise KeyError(f"Unknown interface: {key}. Available: {all_keys}")


def get_tools_summary(keys: List[str] = None) -> str:
    """Generate Stage 1 tool list for agent selection."""
    return generate_tools_summary(INTERFACE_MAPPING, keys)


def get_tools_summary_by_category(keys: List[str] = None) -> str:
    """Generate Stage 1 tool list grouped by category."""
    return generate_tools_summary_by_category(INTERFACE_MAPPING, keys)


def get_input_prompt(key: str) -> str:
    """Get Stage 2 input prompt for selected tool."""
    cls = get_interface(key)
    return cls.get_input_prompt()


def parse_tool_input(key: str, llm_output, global_context: Dict = None):
    """Parse LLM output for tool execution."""
    cls = get_interface(key)
    return cls.parse_llm_input(llm_output, global_context)


def format_tool_output(key: str, output: Dict) -> str:
    """Format Stage 3 output for agent."""
    cls = get_interface(key)
    return cls.format_output_for_agent(output)


def format_error(key: str, error_type: str, error_msg: str, suggestion: str = None) -> str:
    """Format error message for agent."""
    cls = get_interface(key)
    return cls.format_error_for_agent(error_type, error_msg, suggestion)


# =============================================================================
# Unified Tool Registry Interface (for external callers)
# =============================================================================

def get_tool_registry(enabled_tools: List[str] = None) -> Dict[str, Dict[str, Any]]:
    """Generate tool registry from Interface classes.
    
    This is the unified interface for external callers. Just pass in enabled_tools
    list and get complete tool descriptions for agent prompts.
    
    Args:
        enabled_tools: List of tool keys to include. If None, includes all.
    
    Returns:
        Dict mapping tool_name -> {name, description, args_schema, returns_frames}
    """
    if enabled_tools is None:
        enabled_tools = list(INTERFACE_MAPPING.keys())
    
    registry = {}
    for key in enabled_tools:
        if key not in INTERFACE_MAPPING:
            continue
        
        cls = INTERFACE_MAPPING[key]
        
        # Build args_schema from AGENT_INPUT_SCHEMA
        args_schema = {}
        for arg_name, arg_info in cls.AGENT_INPUT_SCHEMA.items():
            args_schema[arg_name] = {
                "type": arg_info.get("type", "any"),
                "required": arg_info.get("required", False),
                "description": arg_info.get("description", ""),
            }
            if "default" in arg_info:
                args_schema[arg_name]["default"] = arg_info["default"]
        
        registry[key] = {
            "name": cls.AGENT_NAME or key,
            "description": cls.AGENT_DESCRIPTION or cls.FUNCTIONALITY,
            "args_schema": args_schema,
            "returns_frames": key in FRAME_RETURNING_TOOLS,
        }
    
    return registry


def validate_tool_args(tool_name: str, args: Dict[str, Any]) -> Optional[str]:
    """Validate tool arguments against schema.
    
    Args:
        tool_name: Name of the tool
        args: Arguments to validate
    
    Returns:
        Error message if validation fails, None if valid
    """
    if tool_name not in INTERFACE_MAPPING:
        all_keys = list(INTERFACE_MAPPING.keys())
        return f"Unknown tool: {tool_name}. Available: {all_keys}"
    
    cls = INTERFACE_MAPPING[tool_name]
    schema = cls.AGENT_INPUT_SCHEMA
    
    # Check required arguments
    missing = []
    for arg_name, arg_info in schema.items():
        if arg_info.get("required", False) and arg_name not in args:
            missing.append(arg_name)
    
    if missing:
        return f"Missing required argument(s): {', '.join(missing)}"
    
    return None


def create_tool_instance(tool_name: str, **init_kwargs) -> Interface:
    """Create and return a tool instance.
    
    Args:
        tool_name: Name of the tool
        **init_kwargs: Arguments to pass to the constructor
    
    Returns:
        Initialized Interface instance
    """
    cls = get_interface(tool_name)
    instance = cls(**init_kwargs)
    return instance


def execute_tool(
    tool_name: str,
    tool_instance: Interface,
    args: Dict[str, Any],
) -> Dict[str, Any]:
    """Execute a tool and format output.
    
    Args:
        tool_name: Name of the tool (for formatting)
        tool_instance: Initialized tool instance
        args: Arguments to pass to the tool
    
    Returns:
        Dict with 'success', 'result' (formatted text), and raw output fields
    """
    try:
        # Execute
        raw_output = tool_instance(**args)
        
        # Check for error
        if "error" in raw_output:
            return {
                "success": False,
                "result": f"Tool error: {raw_output['error']}",
                "error": raw_output["error"],
            }
        
        # Format output
        cls = get_interface(tool_name)
        formatted = cls.format_output_for_agent(raw_output)
        
        return {
            "success": True,
            "result": formatted,
            **raw_output,
        }
        
    except Exception as e:
        return {
            "success": False,
            "result": f"Execution failed: {str(e)}",
            "error": str(e),
        }


__all__ = [
    # Base classes
    "Interface",
    "InterfaceCategory",
    "Video",
    "Image",
    "BoundingBox",
    "Point",
    # Mappings
    "INTERFACE_MAPPING",
    "IMPLEMENTATION_VARIANTS",
    "FRAME_RETURNING_TOOLS",
    "get_interface",
    # Constants
    "TSTAR_SEARCH_BUDGET",
    "TSTAR_FINAL_SAMPLE_FRAMES",
    # Stage 1: Tool Selection
    "get_tools_summary",
    "get_tools_summary_by_category",
    # Stage 2: Input Specification
    "get_input_prompt",
    "parse_tool_input",
    # Stage 3: Output Presentation
    "format_tool_output",
    "format_error",
    # Unified Interface (for external callers)
    "get_tool_registry",
    "validate_tool_args",
    "create_tool_instance",
    "execute_tool",
    # Interface Classes - QA (InternVideo2.5)
    "InternVideoGeneralQA",
    "InternVideoDescription",
    # Interface Classes - QA (Others)
    "VideoRAGTemporalQA",
    "TStarTemporalSpatialQA",
    "GeneralVQA",
    "TargetingVQA",
    # Interface Classes - Sampling
    "VideoTreeSampling",
    "TStarSampling",
    # Interface Classes - Detection & Description
    "YOLOWorldDetection",
    "YOLOEPromptFreeDetection",
    "DAMDescription",
    # Interface Classes - Frame & Caption
    "ViewFrame",
    "OmniCaptionerCaptioning",
    "APICaptioning",
]
