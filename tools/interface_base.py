"""
VideoAgent Interface Base Definitions

This module contains the base Interface class and type definitions.
Separated to avoid circular imports between interface/__init__.py and interface/*.py
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import json


# =============================================================================
# Type Definitions
# =============================================================================

@dataclass
class Video:
    """Represents a video object."""
    path: str
    fps: Optional[float] = None
    duration: Optional[float] = None
    frame_count: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None


@dataclass
class Image:
    """Represents an image/frame object."""
    path: Optional[str] = None
    data: Optional[Any] = None
    width: Optional[int] = None
    height: Optional[int] = None
    frame_index: Optional[int] = None
    timestamp: Optional[float] = None


@dataclass
class BoundingBox:
    """Represents a bounding box annotation."""
    x1: float
    y1: float
    x2: float
    y2: float
    label: Optional[str] = None
    score: Optional[float] = None


@dataclass
class Point:
    """Represents a point coordinate."""
    x: float
    y: float


class InterfaceCategory(Enum):
    """Categories of interfaces."""
    SUB_QUESTION_ANSWERING = "Sub-question Answering"
    TOOL = "Tool"


# =============================================================================
# Base Interface Class
# =============================================================================

class Interface(ABC):
    """Abstract base class for all VideoAgent interfaces.
    
    Supports staged interaction with agent:
    - Stage 1: Tool selection (AGENT_NAME + AGENT_DESCRIPTION)
    - Stage 2: Input specification (AGENT_INPUT_SCHEMA)
    - Stage 3: Output presentation (format_output_for_agent)
    """
    
    # Technical metadata (for system use)
    CATEGORY: InterfaceCategory = None
    FUNCTIONALITY: str = ""
    REFERENCE_PAPER: str = ""
    INPUT_SCHEMA: Dict[str, Any] = {}
    OUTPUT_SCHEMA: Dict[str, Any] = {}
    TOOL_SOURCES: List[str] = []
    
    # Agent-facing metadata (simplified for LLM)
    AGENT_NAME: str = ""           # Short name for the tool
    AGENT_DESCRIPTION: str = ""    # Brief description (1-2 sentences)
    AGENT_INPUT_SCHEMA: Dict[str, Any] = {}   # Simplified input params
    AGENT_OUTPUT_FORMAT: str = ""  # Brief output format description
    
    # ==========================================================================
    # Stage 1: Tool Selection
    # ==========================================================================
    
    @classmethod
    def get_tool_summary(cls) -> str:
        """Get brief summary for tool selection (Stage 1).
        
        Returns:
            One-line summary: "name: description"
        """
        return f"{cls.AGENT_NAME}: {cls.AGENT_DESCRIPTION}"
    
    # ==========================================================================
    # Stage 2: Input Specification
    # ==========================================================================
    
    @classmethod
    def get_input_prompt(cls) -> str:
        """Get input specification prompt (Stage 2).
        
        Returns:
            Formatted string describing required/optional inputs.
        """
        lines = [f"Tool: {cls.AGENT_NAME}", "", "Input Parameters:"]
        
        for param, info in cls.AGENT_INPUT_SCHEMA.items():
            required = "[required]" if info.get("required", False) else "[optional]"
            param_type = info.get("type", "any")
            desc = info.get("description", "")
            default = info.get("default")
            
            if default is not None:
                lines.append(f"  - {param} ({param_type}) {required}: {desc} (default: {default})")
            else:
                lines.append(f"  - {param} ({param_type}) {required}: {desc}")
        
        lines.extend([
            "",
            "Please provide the parameters as JSON:",
            '{"param1": value1, "param2": value2, ...}'
        ])
        
        return "\n".join(lines)
    
    # ==========================================================================
    # Stage 3: Output Presentation
    # ==========================================================================
    
    @classmethod
    def get_output_prompt(cls) -> str:
        """Get output format description (Stage 3 prefix).
        
        Returns:
            Brief description of what output to expect.
        """
        return f"Output format: {cls.AGENT_OUTPUT_FORMAT}"
    
    @classmethod
    def format_output_for_agent(cls, output: Dict[str, Any]) -> str:
        """Format execution result for agent (Stage 3).
        
        Args:
            output: Raw output dictionary from interface execution
        
        Returns:
            Formatted string message for the agent
        """
        if not output:
            return "No output returned."
        
        # Check for error
        if "error" in output:
            return f"Error: {output['error']}"
        
        # Default formatting
        lines = []
        for key, value in output.items():
            if isinstance(value, list):
                if len(value) > 5:
                    lines.append(f"{key}: [{len(value)} items]")
                else:
                    lines.append(f"{key}: {value}")
            elif isinstance(value, dict):
                lines.append(f"{key}: {json.dumps(value, ensure_ascii=False)}")
            else:
                lines.append(f"{key}: {value}")
        
        return "\n".join(lines)
    
    @classmethod
    def format_error_for_agent(cls, error_type: str, error_msg: str, suggestion: str = None) -> str:
        """Format error message for agent.
        
        Args:
            error_type: Type of error (e.g., "InvalidInput", "ExecutionError")
            error_msg: Error description
            suggestion: Optional suggestion for fixing the error
        
        Returns:
            Formatted error message
        """
        lines = [f"[{error_type}] {error_msg}"]
        if suggestion:
            lines.append(f"Suggestion: {suggestion}")
        return "\n".join(lines)
    
    # ==========================================================================
    # Input Parsing with Error Handling
    # ==========================================================================
    
    @classmethod
    def parse_llm_input(
        cls, 
        llm_output: Union[str, Dict], 
        global_context: Dict[str, Any] = None
    ) -> Tuple[Dict[str, Any], Optional[str]]:
        """Parse LLM output into interface parameters with error handling.
        
        Args:
            llm_output: LLM's output, either as JSON string or dict
            global_context: Global runtime context (video, current_frames, etc.)
        
        Returns:
            Tuple of (params_dict, error_message)
            - If successful: (params, None)
            - If error: ({}, error_message_for_agent)
        """
        # Parse JSON
        if isinstance(llm_output, str):
            try:
                params = json.loads(llm_output)
            except json.JSONDecodeError:
                # Try to extract JSON from text
                params = cls._extract_json_from_text(llm_output)
                if not params:
                    return {}, cls.format_error_for_agent(
                        "InvalidFormat",
                        "Could not parse JSON from your response.",
                        "Please provide parameters as valid JSON: {\"param\": value}"
                    )
        else:
            params = llm_output.copy() if llm_output else {}
        
        # Validate required parameters
        missing = []
        for param, info in cls.AGENT_INPUT_SCHEMA.items():
            if info.get("required", False) and param not in params:
                missing.append(param)
        
        if missing:
            return {}, cls.format_error_for_agent(
                "MissingParameter",
                f"Missing required parameter(s): {', '.join(missing)}",
                f"Please provide: {', '.join(missing)}"
            )
        
        # Inject global context
        if global_context:
            if "video" in cls.INPUT_SCHEMA and "video" not in params:
                if "video" in global_context:
                    params["video"] = global_context["video"]
            
            if "images" in cls.INPUT_SCHEMA and "images" not in params:
                if "current_frames" in global_context:
                    params["images"] = global_context["current_frames"]
            
            if "image" in cls.INPUT_SCHEMA and "image" not in params:
                if "current_frame" in global_context:
                    params["image"] = global_context["current_frame"]
        
        # Apply defaults
        for param, info in cls.INPUT_SCHEMA.items():
            if param not in params and "default" in info:
                params[param] = info["default"]
        
        return params, None
    
    @staticmethod
    def _extract_json_from_text(text: str) -> Dict[str, Any]:
        """Extract JSON object from text that may contain other content."""
        import re
        
        # Try to find JSON in code blocks
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Try to find bare JSON object
        json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
        
        return {}
    
    # ==========================================================================
    # Abstract Methods
    # ==========================================================================
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the interface (load models, etc.)."""
        pass
    
    @abstractmethod
    def __call__(self, **kwargs) -> Dict[str, Any]:
        """Execute the interface with given inputs."""
        pass


# =============================================================================
# Tool List Generation Functions
# =============================================================================

def generate_tools_summary(interface_mapping: Dict[str, type], keys: List[str] = None) -> str:
    """Generate tools summary for agent's tool selection (Stage 1).
    
    Args:
        interface_mapping: Dict mapping keys to Interface classes
        keys: Optional list of specific keys to include (None = all)
    
    Returns:
        Formatted tool list for agent
    """
    if keys is None:
        keys = list(interface_mapping.keys())
    
    lines = ["Available Tools:", ""]
    
    for i, key in enumerate(keys, 1):
        if key in interface_mapping:
            cls = interface_mapping[key]
            lines.append(f"{i}. {cls.get_tool_summary()}")
    
    lines.extend([
        "",
        "Select a tool by providing its name."
    ])
    
    return "\n".join(lines)


def generate_tools_summary_by_category(
    interface_mapping: Dict[str, type], 
    keys: List[str] = None
) -> str:
    """Generate tools summary grouped by category.
    
    Args:
        interface_mapping: Dict mapping keys to Interface classes
        keys: Optional list of specific keys to include
    
    Returns:
        Formatted tool list grouped by category
    """
    if keys is None:
        keys = list(interface_mapping.keys())
    
    # Group by category
    categories = {}
    for key in keys:
        if key in interface_mapping:
            cls = interface_mapping[key]
            cat = cls.CATEGORY.value if cls.CATEGORY else "Other"
            if cat not in categories:
                categories[cat] = []
            categories[cat].append((key, cls))
    
    lines = ["Available Tools:", ""]
    
    for cat_name, tools in categories.items():
        lines.append(f"[{cat_name}]")
        for key, cls in tools:
            lines.append(f"  - {cls.get_tool_summary()}")
        lines.append("")
    
    lines.append("Select a tool by providing its name.")
    
    return "\n".join(lines)
