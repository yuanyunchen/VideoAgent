"""
Prompts for the LangGraph Video Agent

Contains system prompts, tool descriptions, and force answer prompts.
"""

from typing import List, Dict, Any


SYSTEM_PROMPT = """You are a video understanding expert. Your task is to answer multiple-choice questions about video content.

## How You Perceive the Video
- You receive TEXT DESCRIPTIONS (captions) of video frames, not the actual images
- Initial frame captions are provided at the start (uniformly sampled, may miss important details)
- You MUST use tools to gather sufficient information before answering
- CAUTION: Frame captions may contain inaccuracies or hallucinations - always cross-validate important observations

## Available Tools
{tool_descriptions}


## You MUST Think Before Acting


Your response MUST follow this EXACT structure:


FIRST, write your reasoning (this is REQUIRED, do not skip):

### Step 1: Analyze the Question
- What specific information does this question require?
- What are the KEY DIFFERENCES between answer choices?

### Step 2: Evaluate Evidence (BE SKEPTICAL)
- What evidence do I have? Is it from MULTIPLE independent sources?
- Can I CONFIDENTLY distinguish between choices, or is there ambiguity?
- Have I cross-validated key claims with DIFFERENT tool types (QA + sampling + detection)?
- WARNING: Single-source evidence is unreliable. Captions and tool outputs can hallucinate!

### Step 3: Decide Action
- If evidence is INSUFFICIENT or from only ONE source → Call more tools to cross-validate
- If evidence is CONTRADICTORY → Call different tool types to resolve
- ONLY submit answer when you have MULTIPLE consistent evidence sources


THEN, output exactly ONE JSON block (wrapped in ```json ... ``` markdown).


Tool call (basic):
```json
{{"action": "tool_call", "tool_name": "internvideo_general_qa", "tool_args": {{"query": "What is happening?"}}, "reason": "why this tool"}}
```

Tool call (with reasoning state - recommended):
```json
{{"action": "tool_call", "tool_name": "internvideo_general_qa", "tool_args": {{"query": "What is happening?"}}, "reason": "why this tool", "hypothesis": {{"choice": 2, "confidence": 0.6, "evidence": ["yellow mixture", "cooking in pot"]}}, "ruled_out": {{"choice": 0, "reason": "No cleaning actions observed"}}, "finding": "Person is mixing ingredients in a bowl"}}
```

Submit answer:
```json
{{"action": "submit_answer", "answer": 2, "reason": "evidence for this choice"}}
```

## Response Format
Your response structure MUST be:
1. Step 1 analysis (text)
2. Step 2 evaluation (text)
3. Step 3 decision (text)
4. ONE ```json block (tool_call or submit_answer) - NO XML TAGS
5. STOP - nothing after JSON


## Key Principles

1. **NEVER trust single-source evidence** - One tool result or caption can be wrong. Always cross-validate with DIFFERENT tool types before concluding.
2. **Be skeptical** - Frame captions hallucinate. QA tools can be wrong. Require 2-3 consistent sources.
3. **Use diverse tools** - Combine QA tools (internvideo_general_qa, temporal_spatial_qa) with sampling tools and detection to triangulate truth.
4. **Submit only when confident** - If unsure between 2+ choices, call another tool. Wrong answers from rushing are worse than using more tools.
5. **Focus on distinguishing evidence** - Find evidence that RULES OUT wrong choices, not just supports one.
6. **PRIORITIZE QA TOOLS** - Actively leverage QA tools to gather external information from specialized models.
    - **High-level analysis**: Use temporal QA and video description tools for overall action sequences, event timelines, and divide-and-conquer sub-question answering.
    - **Low-level verification**: Use visual QA and targeted QA tools to double-check critical details in important frames and verify specific visual elements.


## Reasoning State Tracking (Recommended)

Track your investigation progress by including these OPTIONAL fields in your JSON response:

| Field | Format | When to Use |
|-------|--------|-------------|
| `hypothesis` | `{{"choice": N, "confidence": 0.0-1.0, "evidence": ["...", "..."]}}` | When you form/update a guess about the answer |
| `ruled_out` | `{{"choice": N, "reason": "..."}}` | When you have evidence to eliminate a choice |
| `finding` | `"key insight string"` | When you discover important information |

Benefits: Your reasoning state will be preserved across turns, helping you avoid redundant investigations and build on previous insights. The system will remind you of your hypotheses and ruled-out choices.
"""


# Task prompt template (dynamic content - sent as user message)
# DEPRECATED: Use TASK_PROMPT_V2 instead
TASK_PROMPT = """## Current Task

Question: 
{question}

Choices:
{choices}

{initial_context}

{tool_history}

{reasoning_history}
"""

# NEW: Task prompt template v2 (12.9)
# Prompt order: Task -> Video Memory -> Tool History (Q&A format) -> Reasoning State
TASK_PROMPT_V2 = """## Current Task

Question: 
{question}

Choices:
{choices}

---

{video_memory}

---

{tool_history_qa}

{reasoning_history}
"""


FORCE_ANSWER_PROMPT = """SYSTEM NOTIFICATION: Maximum tool calls ({max_calls}) reached.

You MUST provide your final answer NOW based on all information gathered.

Think about:
1. What evidence supports each choice?
2. Which choice has the strongest support?
3. Which choices can be ruled out and why?

Then respond with:
```json
{{
    "action": "submit_answer",
    "answer": <integer 0-4>,
    "reason": "best reasoning based on available evidence"
}}
```
"""


TOOL_ERROR_PROMPT = """## CRITICAL ERROR - Action Required

**Error Type:** {error_type}
**Details:** {error_message}

### MANDATORY Before Your Next Action:

1. **ANALYZE THE ERROR**: What specifically went wrong? Was it a typo, wrong tool name, or missing argument?

2. **CHECK AVAILABLE TOOLS**: Review the tool list in the system prompt. Only use tools that are explicitly listed.

3. **VERIFY YOUR ARGUMENTS**: Each tool has specific required arguments. Double-check the schema before calling.

4. **DO NOT REPEAT THE SAME MISTAKE**: If you tried an invalid tool/argument, choose a different valid approach.

### Your Response Must Include:
- Step 1: Acknowledge what went wrong
- Step 2: State your corrected approach
- Step 3: Execute with proper format

```json
{{"action": "tool_call", "tool_name": "<valid_tool>", "tool_args": {{<proper_args>}}, "reason": "<why_this_fixes_the_issue>"}}
```
"""


PARSE_ERROR_PROMPT = """## FORMAT ERROR - Your Response Could Not Be Parsed

Your previous response did not contain valid JSON. This is a critical error that wastes a turn.

### Common Mistakes to Avoid:
- Using XML tags like `<function_call>` (WRONG)
- Missing the ```json code fence (WRONG)
- Incomplete JSON (missing closing braces)
- Text after the JSON block

### REQUIRED Response Structure:

**FIRST** - Write your analysis (mandatory):

### Step 1: Analyze the Question
[What does this question need?]

### Step 2: Evaluate Evidence
[What do I know? What's missing?]

### Step 3: Decide Action
[Tool call or submit? Why?]

**THEN** - Output exactly ONE JSON block:
```json
{{"action": "tool_call", "tool_name": "...", "tool_args": {{}}, "reason": "..."}}
```

**STOP** - Nothing after the JSON block.

Do NOT make excuses. Follow the format precisely.
"""

NO_TOOL_SUBMIT_WARNING = """## HOLD - Submitting Without Verification

You are attempting to answer based ONLY on initial frame captions, without using ANY tools.

### Why This Is Risky:
- Initial captions are sparse (only a few frames sampled from the entire video)
- Captions may contain hallucinations or miss critical details
- You have powerful tools available but haven't used them

### Before Submitting, Consider:
| Tool | Use Case |
|------|----------|
| `internvideo_general_qa` | Ask questions about the full video |
| `temporal_sample_frames` | Get frames from specific time ranges |
| `view_frame` | Examine specific frame indices in detail |

### Your Options:
1. **RECOMMENDED**: Use at least ONE tool to verify your hypothesis
2. **If truly confident**: Re-submit with detailed justification explaining why tools are unnecessary

If you choose to submit without tools, you MUST explain in your reason field:
- Why the initial captions are sufficient
- What specific evidence supports your answer
- Why additional tools would not help

Weak justifications will be considered insufficient.
"""


def format_tool_descriptions(enabled_tools: List[str], tool_registry: Dict[str, Any]) -> str:
    """Format tool descriptions from tool registry for the system prompt.
    
    Tool descriptions and arguments are fetched dynamically from the interface definitions.
    """
    lines = []
    
    for i, tool_key in enumerate(enabled_tools, 1):
        if tool_key not in tool_registry:
            continue
        
        tool_info = tool_registry[tool_key]
        name = tool_info.get("name", tool_key)
        description = tool_info.get("description", "No description")
        args_schema = tool_info.get("args_schema", {})
        
        lines.append(f"{i}. **{name}**: {description}")
        
        if args_schema:
            args_parts = []
            for arg_name, arg_info in args_schema.items():
                required = "required" if arg_info.get("required", False) else "optional"
                arg_desc = arg_info.get("description", "")
                if arg_desc:
                    args_parts.append(f"{arg_name} ({required}): {arg_desc}")
                else:
                    args_parts.append(f"{arg_name} ({required})")
            if args_parts:
                lines.append(f"   Args: {'; '.join(args_parts)}")
        
        lines.append("")
    
    return "\n".join(lines)


def format_choices(choices: List[str]) -> str:
    """Format multiple choice options."""
    lines = []
    for i, choice in enumerate(choices):
        lines.append(f"{i}. {choice}")
    return "\n".join(lines)


def format_tool_history(tool_history: List) -> str:
    """Format tool call history for agent context.
    
    This helps the agent know which tools it has ACTUALLY called,
    preventing hallucination of tool results.
    """
    if not tool_history:
        return """## Tools Called So Far
NO TOOLS CALLED YET. You only have initial frame captions and video description.
To get more information, you MUST call tools first.
WARNING: Do NOT fabricate or imagine tool results - only reference tools you have actually called!"""
    
    lines = ["## Tools Called So Far"]
    lines.append(f"({len(tool_history)} tools executed)")
    lines.append("")
    
    for i, tc in enumerate(tool_history, 1):
        # Handle both ToolCall objects and dicts
        tool_name = tc.tool_name if hasattr(tc, 'tool_name') else tc.get('tool_name', 'unknown')
        lines.append(f"{i}. {tool_name}")
    
    lines.append("")
    lines.append("You have real results from these tools in the conversation above. Only reference these actual results.")
    
    return "\n".join(lines)


# Memory context template for subsequent turns
MEMORY_CONTEXT_TEMPLATE = """## Memory Update (Turn {turn_number})

{tool_history}

{reasoning_history}
---
"""


def build_memory_context(
    tool_history: List = None,
    reasoning_history: str = "",
    turn_number: int = 1,
) -> str:
    """Build memory context to prepend to user messages in subsequent turns.
    
    This ensures the agent has up-to-date information about:
    - What tools have been called (prevents hallucination)
    - Current reasoning state (hypotheses, ruled-out choices, findings)
    
    Args:
        tool_history: List of ToolCall objects representing tools already called
        reasoning_history: Summary of agent's reasoning state from AgentHistory.get_summary()
        turn_number: Current turn number for context
        
    Returns:
        Formatted memory context string to prepend to messages
    """
    # Format tool history section
    tool_history_section = format_tool_history(tool_history or [])
    
    # Format reasoning history section if provided
    history_section = ""
    if reasoning_history:
        history_section = f"## Your Reasoning State\n{reasoning_history}"
    
    return MEMORY_CONTEXT_TEMPLATE.format(
        turn_number=turn_number,
        tool_history=tool_history_section,
        reasoning_history=history_section,
    )


def build_system_prompt(
    enabled_tools: List[str],
    tool_registry: Dict[str, Any],
    max_parallel_tools: int = 3,
) -> str:
    """Build the static system prompt (agent role + tools).
    
    This contains only static instructions that don't change per task.
    Task-specific content (question, choices, context) should use build_task_prompt().
    
    Args:
        enabled_tools: List of enabled tool names
        tool_registry: Tool definitions from ToolManager
        max_parallel_tools: Maximum tools that can be requested per turn
    """
    tool_descriptions = format_tool_descriptions(enabled_tools, tool_registry)
    
    # Add note about parallel tool calls
    if max_parallel_tools and max_parallel_tools > 1:
        parallel_note = (
            f"\nNote: You can request up to {max_parallel_tools} tool calls in ONE turn; "
            "list multiple tool_call actions together and all will be executed."
        )
        tool_descriptions = tool_descriptions + parallel_note
    
    return SYSTEM_PROMPT.format(tool_descriptions=tool_descriptions)


def build_task_prompt(
    question: str,
    choices: List[str],
    initial_context: str,
    tool_history: List = None,
    reasoning_history: str = "",
) -> str:
    """Build the dynamic task prompt (sent as user message).
    
    This contains task-specific content that changes per video/question.
    
    Args:
        question: The question to answer
        choices: List of answer choices
        initial_context: Video context (captions, metadata) from VideoContext.get_caption_summary()
        tool_history: List of ToolCall objects representing tools already called
        reasoning_history: Summary of agent's reasoning state from AgentHistory.get_summary()
    """
    formatted_choices = format_choices(choices)
    
    # Format reasoning history section if provided
    history_section = ""
    if reasoning_history:
        history_section = f"\n## Your Reasoning State\n{reasoning_history}\n"
    
    # Format tool history section
    tool_history_section = format_tool_history(tool_history or [])
    
    return TASK_PROMPT.format(
        question=question,
        choices=formatted_choices,
        initial_context=initial_context,
        tool_history=tool_history_section,
        reasoning_history=history_section,
    )


def build_force_prompt(max_calls: int) -> str:
    """Build the force answer prompt."""
    return FORCE_ANSWER_PROMPT.format(max_calls=max_calls)


def build_tool_error_prompt(error_type: str, error_message: str) -> str:
    """Build tool error prompt."""
    return TOOL_ERROR_PROMPT.format(
        error_type=error_type,
        error_message=error_message,
    )


def format_tool_history_qa(tool_history_qa: List) -> str:
    """Format tool history in Q&A format for v2 prompts.
    
    This provides a clean, accumulated view of all tool calls across all rounds,
    showing only the question and answer without the agent's reasoning process.
    
    For frame-content tools (sampling, view_frame, detailed_captioning):
        - Only shows which frames were sampled/viewed
        - Actual content is in Video Memory section
    
    For Q&A tools (internvideo_general_qa, general_vqa, etc.):
        - Shows full question and answer
    
    Args:
        tool_history_qa: List of ToolHistoryEntry objects
        
    Returns:
        Formatted string for prompt
    """
    if not tool_history_qa:
        return """## Tool Call History
No tools called yet. You have access to the Video Memory above.
To gather more information, call tools to query the video.
WARNING: Do NOT fabricate tool results - only reference tools you have actually called!"""
    
    lines = [f"## Tool Call History ({len(tool_history_qa)} calls)"]
    lines.append("")
    
    for i, entry in enumerate(tool_history_qa, 1):
        # Use the entry's to_prompt_line method if available
        if hasattr(entry, 'to_prompt_line'):
            lines.append(entry.to_prompt_line(i))
        else:
            # Fallback for dict-like entries (no truncation - all info must be complete)
            tool_name = entry.get('tool_name', 'unknown') if isinstance(entry, dict) else getattr(entry, 'tool_name', 'unknown')
            query = entry.get('query', '') if isinstance(entry, dict) else getattr(entry, 'query', '')
            answer = entry.get('answer', '') if isinstance(entry, dict) else getattr(entry, 'answer', '')
            frame_refs = entry.get('frame_refs', []) if isinstance(entry, dict) else getattr(entry, 'frame_refs', [])
            
            if frame_refs:
                # Show all frames, no truncation
                frame_list = ", ".join(str(f) for f in frame_refs)
                lines.append(f"{i}. [{tool_name}] {query}")
                lines.append(f"   → Frames: {frame_list} (see Video Memory)")
            else:
                # Show full answer, no truncation
                lines.append(f"{i}. [{tool_name}] Q: {query}")
                lines.append(f"   → A: {answer}")
        
        lines.append("")
    
    return "\n".join(lines)


def format_video_memory(video_memory) -> str:
    """Format VideoMemory object for prompt.
    
    This is a convenience wrapper that calls video_memory.to_prompt().
    If video_memory is None, returns a placeholder message.
    
    Args:
        video_memory: VideoMemory object or None
        
    Returns:
        Formatted string for prompt
    """
    if video_memory is None:
        return """## Video Memory
No video memory available. Video information was not loaded properly."""
    
    # VideoMemory has its own to_prompt method
    if hasattr(video_memory, 'to_prompt'):
        return video_memory.to_prompt()
    
    # Fallback for dict-like video_memory
    return str(video_memory)


def build_task_prompt_v2(
    question: str,
    choices: List[str],
    video_memory,
    tool_history_qa: List = None,
    reasoning_history: str = "",
) -> str:
    """Build the dynamic task prompt v2 (sent as user message).
    
    NEW: This version uses VideoMemory and Q&A-style tool history.
    
    Prompt structure:
    1. Task (question + choices)
    2. Video Memory (overview + frames)
    3. Tool History (Q&A format, accumulated across all rounds)
    4. Reasoning State (optional)
    
    Args:
        question: The question to answer
        choices: List of answer choices
        video_memory: VideoMemory object containing all video info and frames
        tool_history_qa: List of ToolHistoryEntry objects
        reasoning_history: Summary of agent's reasoning state
    """
    formatted_choices = format_choices(choices)
    
    # Format video memory section
    video_memory_section = format_video_memory(video_memory)
    
    # Format tool history in Q&A format
    tool_history_section = format_tool_history_qa(tool_history_qa or [])
    
    # Format reasoning history section if provided
    history_section = ""
    if reasoning_history:
        history_section = f"\n## Your Reasoning State\n{reasoning_history}\n"
    
    return TASK_PROMPT_V2.format(
        question=question,
        choices=formatted_choices,
        video_memory=video_memory_section,
        tool_history_qa=tool_history_section,
        reasoning_history=history_section,
    )
