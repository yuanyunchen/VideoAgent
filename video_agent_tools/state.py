"""
Agent State Definitions for LangGraph Video Agent

Defines the state schema for the ReAct agent including:
- Message history
- Video context (frames, captions)
- Tool tracking
- Result fields
"""

from typing import Dict, List, Optional, Any, TypedDict, Annotated, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import operator


# =============================================================================
# Frame and Video Memory Classes (NEW - 12.9)
# =============================================================================

@dataclass
class FrameInfo:
    """Information about a single video frame in memory."""
    frame_id: int
    timestamp: float  # in seconds
    caption: Optional[str] = None  # Basic caption from initial sampling or temporal_sample
    detailed_caption: Optional[str] = None  # From detailed_captioning tool
    viewed: bool = False  # Whether agent has seen actual image via view_frame
    source: str = "initial"  # Where this frame came from: 'initial', 'temporal_sample', 'spatial_sample', 'view_frame', 'detailed_caption'
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "caption": self.caption,
            "detailed_caption": self.detailed_caption,
            "viewed": self.viewed,
            "source": self.source,
        }


@dataclass
class VideoMemory:
    """
    Centralized video memory that persists across all agent rounds.
    
    This replaces the scattered video context information and ensures
    the agent always has access to:
    1. Video overview (metadata + description) - static after init
    2. Frame information - updated by sampling/view/caption tools
    """
    
    # Part 1: Overview (static after initialization)
    video_id: str
    video_path: str
    duration: float  # in seconds
    total_frames: int
    fps: float
    width: int
    height: int
    video_description: str = ""  # Overall video description
    
    # Part 2: Frames (updated by tools)
    # Key: frame_id, Value: FrameInfo
    frames: Dict[int, FrameInfo] = field(default_factory=dict)
    
    def add_frame(
        self,
        frame_id: int,
        timestamp: float = None,
        caption: str = None,
        detailed_caption: str = None,
        viewed: bool = False,
        source: str = "unknown",
    ) -> None:
        """Add or update a frame in memory.
        
        If frame already exists, updates only the provided fields (non-None values).
        """
        if timestamp is None:
            timestamp = frame_id / self.fps if self.fps > 0 else 0.0
        
        if frame_id in self.frames:
            # Update existing frame
            existing = self.frames[frame_id]
            if caption is not None:
                existing.caption = caption
            if detailed_caption is not None:
                existing.detailed_caption = detailed_caption
            if viewed:
                existing.viewed = True
            # Update source if more specific
            if source not in ("unknown", "initial") or existing.source == "unknown":
                existing.source = source
        else:
            # Create new frame entry
            self.frames[frame_id] = FrameInfo(
                frame_id=frame_id,
                timestamp=timestamp,
                caption=caption,
                detailed_caption=detailed_caption,
                viewed=viewed,
                source=source,
            )
    
    def add_frames_batch(
        self,
        frame_data: List[Dict[str, Any]],
        source: str = "unknown",
    ) -> None:
        """Add multiple frames at once.
        
        Args:
            frame_data: List of dicts with keys: frame_id, timestamp (optional), caption (optional)
            source: Source of these frames
        """
        for fd in frame_data:
            self.add_frame(
                frame_id=fd["frame_id"],
                timestamp=fd.get("timestamp"),
                caption=fd.get("caption"),
                detailed_caption=fd.get("detailed_caption"),
                viewed=fd.get("viewed", False),
                source=source,
            )
    
    def get_frame(self, frame_id: int) -> Optional[FrameInfo]:
        """Get frame info by ID."""
        return self.frames.get(frame_id)
    
    def get_viewed_frames(self) -> List[FrameInfo]:
        """Get all frames that have been viewed (actual images seen)."""
        return [f for f in self.frames.values() if f.viewed]
    
    def get_timeline_coverage(self) -> float:
        """Calculate what percentage of video timeline has frames in memory."""
        if not self.frames or self.total_frames == 0:
            return 0.0
        
        num_segments = 10
        segment_size = self.total_frames / num_segments
        segments_covered = set()
        
        for frame_id in self.frames.keys():
            segment = int(frame_id / segment_size)
            segment = min(segment, num_segments - 1)
            segments_covered.add(segment)
        
        return len(segments_covered) / num_segments
    
    def to_prompt(self) -> str:
        """Generate formatted prompt section for Video Memory.
        
        This is included in EVERY LLM call to ensure consistent context.
        """
        lines = []
        
        # === Part 1: Video Overview ===
        lines.append("## Video Overview")
        lines.append(f"- Duration: {self.duration:.1f} seconds")
        lines.append(f"- Total frames: {self.total_frames}")
        lines.append(f"- FPS: {self.fps:.1f}")
        lines.append(f"- Resolution: {self.width}x{self.height}")
        lines.append(f"- Valid frame indices: 0 to {self.total_frames - 1}")
        lines.append("")
        
        if self.video_description:
            lines.append("### Video Description")
            lines.append(self.video_description)
            lines.append("")
        
        # === Part 2: Frames in Memory ===
        if not self.frames:
            lines.append("## Frames in Memory")
            lines.append("No frames in memory yet.")
        else:
            # Sort frames by frame_id
            sorted_frames = sorted(self.frames.values(), key=lambda f: f.frame_id)
            
            # Separate viewed vs captioned frames
            viewed_frames = [f for f in sorted_frames if f.viewed]
            caption_frames = [f for f in sorted_frames if not f.viewed and (f.caption or f.detailed_caption)]
            
            lines.append(f"## Frames in Memory ({len(self.frames)} frames)")
            coverage = self.get_timeline_coverage()
            lines.append(f"Timeline coverage: {coverage:.0%}")
            lines.append("")
            
            # Show viewed frames first (agent has seen actual images)
            if viewed_frames:
                lines.append(f"### Viewed Frames ({len(viewed_frames)} frames - you have seen these images)")
                for f in viewed_frames:
                    frame_line = f"[Frame {f.frame_id} @ {f.timestamp:.1f}s]"
                    if f.detailed_caption:
                        lines.append(f"{frame_line}: {f.detailed_caption}")
                    elif f.caption:
                        lines.append(f"{frame_line}: {f.caption}")
                    else:
                        lines.append(f"{frame_line}: (image viewed, no caption)")
                lines.append("")
            
            # Show captioned frames (agent has text descriptions)
            if caption_frames:
                lines.append(f"### Captioned Frames ({len(caption_frames)} frames)")
                for f in caption_frames:
                    frame_line = f"[Frame {f.frame_id} @ {f.timestamp:.1f}s]"
                    if f.detailed_caption:
                        lines.append(f"{frame_line}: {f.detailed_caption}")
                    elif f.caption:
                        lines.append(f"{frame_line}: {f.caption}")
                lines.append("")
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for persistence/logging."""
        return {
            "video_id": self.video_id,
            "video_path": self.video_path,
            "duration": self.duration,
            "total_frames": self.total_frames,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "video_description": self.video_description,
            "frames": {k: v.to_dict() for k, v in self.frames.items()},
            "timeline_coverage": self.get_timeline_coverage(),
        }


@dataclass
class ToolHistoryEntry:
    """
    Single Q&A entry for tool history.
    
    Stores tool calls in a clean question-answer format,
    without the agent's reasoning process.
    """
    tool_name: str
    query: str  # The question/args in human-readable form
    answer: str  # The tool output
    frame_refs: List[int] = field(default_factory=list)  # Frame IDs for frame-content tools
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    success: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "query": self.query,
            "answer": self.answer,
            "frame_refs": self.frame_refs,
            "timestamp": self.timestamp,
            "success": self.success,
        }
    
    def to_prompt_line(self, index: int) -> str:
        """Format this entry for the tool history prompt.
        
        NOTE: No truncation - all information must be complete for LLM to see.
        """
        lines = []
        
        # Tool name and query
        if self.frame_refs:
            # Frame-content tool: show all frame references
            frame_list = ", ".join(str(f) for f in self.frame_refs)
            lines.append(f"{index}. [{self.tool_name}] {self.query}")
            lines.append(f"   → Frames: {frame_list} (see Video Memory for details)")
        else:
            # Q&A tool: show full answer (no truncation)
            lines.append(f"{index}. [{self.tool_name}] Q: {self.query}")
            lines.append(f"   → A: {self.answer}")
        
        return "\n".join(lines)


# =============================================================================
# Original Classes (keeping for compatibility)
# =============================================================================

@dataclass
class ToolCall:
    """Record of a single tool call."""
    tool_name: str
    tool_args: Dict[str, Any]
    result: str
    timestamp: str
    success: bool = True
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "tool_args": self.tool_args,
            "result": self.result,  # Full result, no truncation
            "timestamp": self.timestamp,
            "success": self.success,
            "error": self.error
        }

# 12.6 swy
@dataclass
class Hypothesis:
    """A hypothesis about which answer choice is correct."""
    choice: int
    confidence: float  # 0.0 to 1.0
    evidence: List[str]
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "choice": self.choice,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "timestamp": self.timestamp,
        }

# 12.6 swy
@dataclass
class RuledOut:
    """Record of an answer choice that has been ruled out."""
    choice: int
    reason: str
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "choice": self.choice,
            "reason": self.reason,
            "timestamp": self.timestamp,
        }

# 12.6 swy
@dataclass
class AgentHistory:
    """
    Track agent's reasoning process for self-reflection.
    
    This enables the agent to:
    - Remember what hypotheses it has formed
    - Track which choices have been ruled out
    - Avoid redundant investigations
    - Build on previous reasoning
    """
    
    # Hypotheses about answer choices (can have multiple, updated over time)
    hypotheses: List[Hypothesis] = field(default_factory=list)
    
    # Choices that have been ruled out with reasons
    ruled_out: List[RuledOut] = field(default_factory=list)
    
    # Key findings from tool calls (distilled insights)
    key_findings: List[str] = field(default_factory=list)
    
    # Questions that remain unanswered
    open_questions: List[str] = field(default_factory=list)
    
    def add_hypothesis(self, choice: int, confidence: float, evidence: List[str]) -> None:
        """Add or update a hypothesis about an answer choice."""
        # Remove existing hypothesis for same choice (update pattern)
        self.hypotheses = [h for h in self.hypotheses if h.choice != choice]
        self.hypotheses.append(Hypothesis(
            choice=choice,
            confidence=confidence,
            evidence=evidence,
            timestamp=datetime.now().isoformat(),
        ))
        # Keep sorted by confidence (highest first)
        self.hypotheses.sort(key=lambda h: h.confidence, reverse=True)
    
    def rule_out(self, choice: int, reason: str) -> None:
        """Mark a choice as ruled out."""
        # Don't duplicate
        if any(r.choice == choice for r in self.ruled_out):
            return
        self.ruled_out.append(RuledOut(
            choice=choice,
            reason=reason,
            timestamp=datetime.now().isoformat(),
        ))
    
    def add_finding(self, finding: str) -> None:
        """Add a key finding from investigation."""
        if finding not in self.key_findings:
            self.key_findings.append(finding)
    
    def add_open_question(self, question: str) -> None:
        """Add a question that needs investigation."""
        if question not in self.open_questions:
            self.open_questions.append(question)
    
    def resolve_question(self, question: str) -> None:
        """Remove a question that has been answered."""
        self.open_questions = [q for q in self.open_questions if q != question]
    
    def get_best_hypothesis(self) -> Optional[Hypothesis]:
        """Get the current best hypothesis (highest confidence)."""
        if not self.hypotheses:
            return None
        return self.hypotheses[0]
    
    def get_ruled_out_choices(self) -> List[int]:
        """Get list of choices that have been ruled out."""
        return [r.choice for r in self.ruled_out]
    
    def get_remaining_choices(self, total_choices: int) -> List[int]:
        """Get choices that haven't been ruled out."""
        ruled = set(self.get_ruled_out_choices())
        return [i for i in range(total_choices) if i not in ruled]
    
    def get_summary(self) -> str:
        """Format reasoning state for agent prompt."""
        lines = []
        
        # Current best hypothesis
        best = self.get_best_hypothesis()
        if best:
            conf_pct = f"{best.confidence:.0%}"
            lines.append(f"Current hypothesis: Choice {best.choice} ({conf_pct} confidence)")
            if best.evidence:
                lines.append(f"  Evidence: {'; '.join(best.evidence[:3])}")
        
        # Ruled out choices
        if self.ruled_out:
            ruled_strs = [f"{r.choice} ({r.reason})" for r in self.ruled_out]
            lines.append(f"Ruled out: {', '.join(ruled_strs)}")
        
        # Key findings
        if self.key_findings:
            lines.append(f"Key findings: {len(self.key_findings)}")
            for finding in self.key_findings[-3:]:  # Show last 3
                lines.append(f"  - {finding}")
        
        # Open questions
        if self.open_questions:
            lines.append(f"Open questions: {', '.join(self.open_questions[:3])}")
        
        return "\n".join(lines) if lines else ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for persistence."""
        return {
            "hypotheses": [h.to_dict() for h in self.hypotheses],
            "ruled_out": [r.to_dict() for r in self.ruled_out],
            "key_findings": self.key_findings,
            "open_questions": self.open_questions,
        }


@dataclass
class VideoContext:
    """Video-specific context maintained during agent execution."""
    video_id: str
    video_path: str
    total_frames: int
    fps: float
    duration: float
    width: int
    height: int
    
    # Frame data (numpy arrays, loaded on demand)
    frames: Dict[int, Any] = field(default_factory=dict)
    
    # Frame captions (text descriptions)
    frame_captions: Dict[int, str] = field(default_factory=dict)
    
    # Track which frames have been sampled
    sampled_indices: List[int] = field(default_factory=list)
    
    # Evidence tracking: objects and concepts observed
    observed_objects: List[str] = field(default_factory=list)
    
    # Evidence tracking: key observations made
    key_observations: List[str] = field(default_factory=list)
    
    # Video description (generated at initialization using InternVideoDescription)
    video_description: str = ""
    
    def get_timeline_coverage(self) -> float:
        """Calculate what percentage of the video timeline has been sampled.
        
        Returns:
            Float between 0.0 and 1.0 representing coverage
        """
        if not self.sampled_indices or self.total_frames == 0:
            return 0.0
        
        # Divide video into 10 segments and check how many have samples
        num_segments = 10
        segment_size = self.total_frames / num_segments
        segments_covered = set()
        
        for idx in self.sampled_indices:
            segment = int(idx / segment_size)
            segment = min(segment, num_segments - 1)  # Clamp to valid range
            segments_covered.add(segment)
        
        return len(segments_covered) / num_segments
    
    def get_evidence_summary(self) -> str:
        """Get a summary of evidence gathered so far."""
        lines = []
        
        # Timeline coverage
        coverage = self.get_timeline_coverage()
        lines.append(f"Timeline coverage: {coverage:.0%} ({len(self.sampled_indices)} frames)")
        
        # Objects observed
        if self.observed_objects:
            unique_objects = list(dict.fromkeys(self.observed_objects))[:10]  # Top 10
            lines.append(f"Objects observed: {', '.join(unique_objects)}")
        
        # Key observations
        if self.key_observations:
            lines.append(f"Key observations: {len(self.key_observations)}")
        
        return "\n".join(lines) if lines else "No evidence summary available."
    
    def get_caption_summary(self) -> str:
        """Format video info and captions for agent context."""
        lines = []
        
        # Video metadata (important for agent to know)
        lines.append("## Video Information")
        lines.append(f"- Duration: {self.duration:.1f} seconds")
        lines.append(f"- Total frames: {self.total_frames}")
        lines.append(f"- FPS: {self.fps:.1f}")
        lines.append(f"- Resolution: {self.width}x{self.height}")
        lines.append(f"- Valid frame indices: 0 to {self.total_frames - 1}")
        lines.append("")
        
        # Video description (overall summary)
        if self.video_description:
            lines.append("## Video Description (Overall)")
            lines.append(self.video_description)
            lines.append("")
        
        # Frame captions
        if not self.frame_captions:
            lines.append("## Initial Frame Captions")
            lines.append("No frame captions available yet.")
        else:
            lines.append("## Initial Frame Captions")
            lines.append(f"({len(self.frame_captions)} frames sampled)")
            lines.append("")
            
            for idx in sorted(self.frame_captions.keys()):
                caption = self.frame_captions[idx]
                timestamp = idx / self.fps if self.fps > 0 else 0
                lines.append(f"[Frame {idx} @ {timestamp:.1f}s]: {caption}")
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "video_id": self.video_id,
            "video_path": self.video_path,
            "total_frames": self.total_frames,
            "fps": self.fps,
            "duration": self.duration,
            "width": self.width,
            "height": self.height,
            "sampled_indices": self.sampled_indices,
            "frame_captions": self.frame_captions,
            "observed_objects": self.observed_objects,
            "key_observations": self.key_observations,
            "timeline_coverage": self.get_timeline_coverage(),
            "video_description": self.video_description,
        }


class AgentState(TypedDict):
    """
    State for the LangGraph Video Agent.
    
    This state is passed between nodes in the graph and tracks:
    - Conversation messages
    - Video context (metadata, captions)
    - Tool execution history
    - Evidence tracking
    - Final result
    """
    
    # Core identifiers
    video_id: str
    question: str
    choices: List[str]  # Multiple choice options (0-4)
    
    # Message history for LLM (will be reduced/managed)
    messages: Annotated[List[Dict[str, Any]], operator.add]
    
    # Video context (not passed to LLM directly, used by tools)
    # DEPRECATED: Use video_memory instead for prompt generation
    video_context: Optional[VideoContext]
    
    # NEW: Centralized video memory (12.9)
    # Contains video overview + all frame information
    # Used to generate consistent prompt content across all rounds
    video_memory: Optional[VideoMemory]
    
    # NEW: Q&A style tool history (12.9)
    # Clean question-answer format, accumulated across all rounds
    tool_history_qa: List[ToolHistoryEntry]
    
    # Tool tracking (keeping for backward compatibility)
    tool_history: List[ToolCall]
    tool_call_count: int  # Actual number of individual tool executions
    tool_call_rounds: int  # Number of LLM decision rounds (each round may request tools)
    max_tool_calls: int
    enabled_tools: List[str]
    
    # Tool type tracking (for analysis and prompting)
    sampling_tool_count: int  # temporal_sample_frames, temporal_spatial_sample_frames, view_frame
    qa_tool_count: int        # temporal_qa, temporal_spatial_qa, general_vqa, targeting_vqa
    detection_tool_count: int # detect_objects, detect_all_objects, describe_region
    
    # Agent reasoning history (hypotheses, ruled out choices, findings)
    agent_history: Optional[AgentHistory]
    
    # Pending tool calls (set by agent node, executed by tool node)
    # Supports multiple parallel tool calls (up to max_parallel_tools)
    _pending_tools: Optional[List[Dict[str, Any]]]
    
    # Visual memory: accumulated images from view_frame calls
    # Images are added to memory (up to 16 frames) and included in ALL LLM calls
    # Each image: {base64: str, frame_index: int, timestamp: float}
    _pending_images: Optional[List[Dict[str, Any]]]
    
    # One-time error feedback (not accumulated, cleared after each LLM call)
    # Used to send error/warning prompts to agent on retry
    _pending_feedback: Optional[str]
    
    # Parse error tracking
    parse_error_count: int
    max_parse_errors: int
    
    # Result
    final_answer: Optional[int]
    explanation: Optional[str]
    is_forced: bool
    is_complete: bool
    
    # Timing
    start_time: Optional[str]
    end_time: Optional[str]


def create_initial_state(
    video_id: str,
    question: str,
    choices: List[str],
    video_context: VideoContext,
    enabled_tools: List[str],
    max_tool_calls: int = 10,
    max_parse_errors: int = 3,
    video_memory: VideoMemory = None,
) -> AgentState:
    """Create initial agent state.
    
    Args:
        video_id: Unique video identifier
        question: The question to answer
        choices: List of answer choices
        video_context: VideoContext object (deprecated, for backward compatibility)
        enabled_tools: List of enabled tool names
        max_tool_calls: Maximum tool calls allowed
        max_parse_errors: Maximum parse errors before forcing answer
        video_memory: Optional VideoMemory object (if None, created from video_context)
    """
    # Create VideoMemory from VideoContext if not provided
    if video_memory is None and video_context is not None:
        video_memory = VideoMemory(
            video_id=video_context.video_id,
            video_path=video_context.video_path,
            duration=video_context.duration,
            total_frames=video_context.total_frames,
            fps=video_context.fps,
            width=video_context.width,
            height=video_context.height,
            video_description=video_context.video_description,
        )
        # Add initial frame captions to memory
        for frame_id, caption in video_context.frame_captions.items():
            timestamp = frame_id / video_context.fps if video_context.fps > 0 else 0.0
            video_memory.add_frame(
                frame_id=frame_id,
                timestamp=timestamp,
                caption=caption,
                source="initial",
            )
    
    return AgentState(
        video_id=video_id,
        question=question,
        choices=choices,
        messages=[],
        video_context=video_context,
        video_memory=video_memory,
        tool_history_qa=[],
        tool_history=[],
        tool_call_count=0,
        tool_call_rounds=0,
        max_tool_calls=max_tool_calls,
        enabled_tools=enabled_tools,
        sampling_tool_count=0,
        qa_tool_count=0,
        detection_tool_count=0,
        agent_history=AgentHistory(),  # Initialize empty reasoning history
        _pending_tools=None,  # Supports multiple parallel tool calls
        _pending_images=None,
        _pending_feedback=None,  # One-time error feedback
        parse_error_count=0,
        max_parse_errors=max_parse_errors,
        final_answer=None,
        explanation=None,
        is_forced=False,
        is_complete=False,
        start_time=datetime.now().isoformat(),
        end_time=None,
    )


# Tool type categorization
SAMPLING_TOOLS = {"temporal_sample_frames", "temporal_spatial_sample_frames", "view_frame"}
QA_TOOLS = {"temporal_qa", "temporal_spatial_qa", "general_vqa", "targeting_vqa"}
DETECTION_TOOLS = {"detect_objects", "detect_all_objects", "describe_region"}

# Frame-content tools: content goes to VideoMemory, minimal entry in tool history
# These tools produce frame captions/views that should be stored in memory
FRAME_CONTENT_TOOLS = {
    "temporal_sample_frames",
    "temporal_spatial_sample_frames", 
    "view_frame",
    "detailed_captioning",
}

# Q&A tools: full Q&A in tool history (content NOT duplicated in VideoMemory)
QA_CONTENT_TOOLS = {
    "internvideo_general_qa",
    "internvideo_description",
    "general_vqa",
    "detect_objects",
    "detect_all_objects",
    "describe_region",
}


def get_tool_type(tool_name: str) -> str:
    """Get the type category for a tool.
    
    Returns:
        One of: 'sampling', 'qa', 'detection', 'other'
    """
    if tool_name in SAMPLING_TOOLS:
        return "sampling"
    elif tool_name in QA_TOOLS:
        return "qa"
    elif tool_name in DETECTION_TOOLS:
        return "detection"
    else:
        return "other"


def is_frame_content_tool(tool_name: str) -> bool:
    """Check if tool is a frame-content tool (content goes to VideoMemory)."""
    return tool_name in FRAME_CONTENT_TOOLS


def is_qa_content_tool(tool_name: str) -> bool:
    """Check if tool is a Q&A content tool (full output in history)."""
    return tool_name in QA_CONTENT_TOOLS

