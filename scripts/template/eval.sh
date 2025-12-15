#!/bin/bash

################################################################################
# VideoAgent Multi-Tools Agent Evaluation Script
#
# This script runs video question answering experiments with the LangGraph-based
# multi-tools agent architecture featuring:
#   - ReAct-style tool selection and execution
#   - Text-first architecture (agent sees captions, not images)
#   - Configurable tool set
#   - Auto-caption pipeline for frame-returning tools
#   - Multi-GPU resource management
#
# Output Structure:
#   results/<EXPERIMENT_NAME>__<MODEL>_videos_<COUNT>_<MMDD>/
#     - logging.log        : Full evaluation log with tool calls
#     - metrics.csv        : Performance metrics
#     - summary.txt        : Human-readable summary
#     - result.json        : Full results with per-video details
#     - accuracy.txt       : Quick accuracy summary
#     - experiment_config.yaml : Configuration used
#
################################################################################

################################################################################
# GPU Configuration
################################################################################
# Set visible GPUs (comma-separated, e.g., "0,1" for two GPUs)
export CUDA_VISIBLE_DEVICES=0

################################################################################
# Model Configuration
#
# Price Source: https://aimlapi.com/ai-ml-api-pricing
# Last Updated: 2025-12
#
# ============================================================================
# TEXT/REASONING MODELS - For Agent LLM
# ============================================================================
#
# API Model Name                        | Price ($/1M)    | Notes
# --------------------------------------|-----------------|------------------
# gpt-4o-mini                           | $0.15/$0.60     | Good value (default)
# x-ai/grok-4-1-fast-non-reasoning      | $0.210/$0.530   | Fast
# google/gemini-2.5-flash               | $0.32/$2.63     | Fast, 1M context
# deepseek/deepseek-chat                | $0.14/$0.28     | Budget option
#
################################################################################

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# Agent model (for decision making - which tool to call or what answer to give)
AGENT_MODEL="x-ai/grok-4-1-fast-reasoning"

# ============================================================================
# TOOLS CONFIGURATION
# ============================================================================
# Available tools (from tools/interface/):
#   - caption_image              : Generate caption for frame(s) [OmniCaptioner]
#   - view_frame                 : View specific frame (returns caption)
#   - temporal_sample_frames     : Sample diverse frames (returns captions)
#   - temporal_spatial_sample_frames : Find frames with objects (returns captions)
#   - detect_objects             : Detect specific objects [YOLO-World]
#   - detect_all_objects         : Detect all objects [YOLOE]
#   - describe_region            : Describe specific region [DAM]
#
# NOTE: Tools that return frames (view_frame, temporal_*_sample_frames) 
#       automatically caption the frames before returning to agent.
# ============================================================================

# Tools to enable (comma-separated list)
# ============================================================================
# AVAILABLE TOOLS (from tools/interface/__init__.py INTERFACE_MAPPING):
# ============================================================================
#
# QA Tools - InternVideo2.5-based:
#   - internvideo_general_qa          : General video Q&A with 128 frames [InternVideo2.5]
#   - internvideo_description         : Video summary + action timeline [InternVideo2.5]
#
# QA Tools - Others:
#   - temporal_spatial_qa             : Temporal-spatial sub-question QA [TStar]
#   - general_vqa                     : General visual QA [API-based]
#   - targeting_vqa                   : Fine-grained visual QA [VStar]
#
# Frame Sampling:
#   - temporal_sample_frames          : Sample diverse frames temporally [VideoTree]
#   - temporal_spatial_sample_frames  : Find frames with specific objects [TStar]
#
# Detection & Description:
#   - detect_objects                  : Detect specific object categories [YOLO-World]
#   - detect_all_objects              : Detect all objects without prompts [YOLOE]
#   - describe_region                 : Describe specific region [DAM]
#
# Frame & Caption:
#   - view_frame                      : View specific frame (returns caption)
#   - caption_image                   : Generate caption for frames [OmniCaptioner]
#   - detailed_captioning             : Generate detailed caption via API [MLLM]
#
# ============================================================================
# Notes:
#   - caption_image is redundant since sampling tools auto-caption frames
#   - Frame-returning tools (view_frame, temporal_sample_frames, 
#     temporal_spatial_sample_frames) automatically caption frames before 
#     returning to agent
# ============================================================================
# detect_all_objects,describe_region,temporal_spatial_qa,caption_image,targeting_vqa
TOOLS="internvideo_general_qa,internvideo_description,general_vqa,temporal_sample_frames,temporal_spatial_sample_frames,detect_objects,view_frame,detailed_captioning"

# ============================================================================
# EXPERIMENT SETTINGS
# ============================================================================

# Test round name (used for organizing results)
EXPERIMENT_NAME="eval"

# Number of test videos to run (0 or -1 = full dataset)
COUNT=0

# Maximum tool calls per video before forced answer
MAX_TOOL_CALLS=20

# Maximum parallel tools per agent interaction
# This controls how many tools the agent can request in a single turn
# All requested tools will be executed, but this helps manage GPU memory
MAX_PARALLEL_TOOLS=3

# Number of frames to caption at initialization
INITIAL_FRAMES=5

# Detailed console output (true/false)
# When true: print detailed info during processing
# When false: only show progress bar
# Note: Output files always save full details regardless of this setting
DETAILED="true"

# ============================================================================
# MULTIPROCESSING SETTINGS
# ============================================================================
# Number of worker processes for parallel video processing.
#   - 1 = Single process mode (default, same as before)
#   - >1 = Parallel processing with multiple workers
#
# Notes:
#   - Each worker loads its own tools (requires more GPU memory)
#   - Use with multiple GPUs for best performance
#   - Recommended: Set to number of GPUs available
NUM_WORKERS=1

# ============================================================================
# FRAME CONTROL SETTINGS
# ============================================================================
# These parameters control frame extraction and sampling limits.
#
# MAX_VIEW_FRAMES: Maximum frames for view_frame tool (prevents token overflow)
#   - Limits how many frames agent can request to view at once
#   - Higher values = more visual context but more tokens/cost
#   - Recommended: 4-16
MAX_VIEW_FRAMES=8

# DEFAULT_SAMPLE_FRAMES: Default number of frames for sampling tools
#   - Used by temporal_sample_frames and temporal_spatial_sample_frames
#   - Agent can override this within MIN/MAX limits
DEFAULT_SAMPLE_FRAMES=5

# MIN_SAMPLE_FRAMES: Minimum frames for sampling tools
MIN_SAMPLE_FRAMES=2

# MAX_SAMPLE_FRAMES: Maximum frames for sampling tools
#   - Caps agent's frame request to prevent excessive processing
MAX_SAMPLE_FRAMES=8

# ============================================================================
# CAPTIONER CONFIGURATION
# ============================================================================
# Captioner for generating frame descriptions.
#
# Options:
#   - "omni-captioner"  : Use local OmniCaptioner model (requires GPU memory)
#   - "<model_name>"    : Use API-based MLLM captioning (e.g., "gpt-4o-mini", "gpt-4o")
#
# API captioning uses less memory but costs API credits.
# ============================================================================

CAPTIONER="omni-captioner"

################################################################################
# Advanced Settings (usually don't need to change)

# Dataset paths
VIDEO_LIST="data/EgoSchema_test/video_list.txt"
ANNOTATION_FILE="data/EgoSchema_test/annotations.json"
VIDEO_DIR="data/EgoSchema_test/videos"

# Output directory
OUTPUT_DIR="results"

################################################################################
# Environment Setup

# Get script directory and find project root by looking for .env file
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

# Traverse up to find project root (where .env file exists)
while [ "$PROJECT_ROOT" != "/" ] && [ ! -f "$PROJECT_ROOT/.env" ]; do
    PROJECT_ROOT="$(cd "$PROJECT_ROOT/.." && pwd)"
done

# Change to project root to ensure relative paths work
cd "$PROJECT_ROOT" || exit 1

# Load environment variables from .env if it exists
ENV_FILE="$PROJECT_ROOT/.env"
if [ -f "$ENV_FILE" ]; then
    set -a
    . "$ENV_FILE"
    set +a
fi

# Check if API key is set
if [ -z "$AIML_API_KEY" ] && [ -z "$OPENAI_API_KEY" ]; then
    echo "[ERROR] No API key found."
    echo "Please set AIML_API_KEY or OPENAI_API_KEY in your .env file:"
    echo "  export AIML_API_KEY=your_api_key_here"
    exit 1
fi

################################################################################
# Derived Settings (auto-computed)

# Convert count for CLI (0 means use all, -1 means all)
if [ "$COUNT" -eq 0 ] || [ "$COUNT" -eq -1 ]; then
    MAX_VIDEOS="-1"
else
    MAX_VIDEOS="$COUNT"
fi

################################################################################
# Print Configuration Summary

echo "============================================================"
echo "VideoAgent Multi-Tools Agent Evaluation"
echo "============================================================"
echo ""
echo "Agent Model: $AGENT_MODEL"
echo ""
echo "Enabled Tools:"
# POSIX-compatible tool list parsing
OLD_IFS="$IFS"
IFS=','
for tool in $TOOLS; do
    echo "  - $tool"
done
IFS="$OLD_IFS"
echo ""
echo "Settings:"
echo "  Experiment Name:   $EXPERIMENT_NAME"
echo "  Count:             $COUNT (MAX_VIDEOS=$MAX_VIDEOS)"
echo "  Max Tool Calls:    $MAX_TOOL_CALLS"
echo "  Max Parallel Tools: $MAX_PARALLEL_TOOLS"
echo "  Initial Frames:    $INITIAL_FRAMES"
echo "  Num Workers:       $NUM_WORKERS"
echo "  Detailed Output:   $DETAILED"
echo ""
echo "Frame Control:"
echo "  Max View Frames:   $MAX_VIEW_FRAMES"
echo "  Sample Frames:     $MIN_SAMPLE_FRAMES - $MAX_SAMPLE_FRAMES (default: $DEFAULT_SAMPLE_FRAMES)"
echo ""
echo "Captioner:"
if [ "$CAPTIONER" = "omni-captioner" ]; then
    echo "  Type:  Local OmniCaptioner"
    echo "  Model: U4R/OmniCaptioner"
else
    echo "  Type:  API-based MLLM"
    echo "  Model: $CAPTIONER"
fi
echo ""
echo "============================================================"
echo ""

################################################################################
# Run Evaluation

echo "[INFO] Starting evaluation at $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

python -m video_agent_tools.cli \
    --model "$AGENT_MODEL" \
    --tools "$TOOLS" \
    --max-tool-calls "$MAX_TOOL_CALLS" \
    --max-parallel-tools "$MAX_PARALLEL_TOOLS" \
    --initial-frames "$INITIAL_FRAMES" \
    --max-view-frames "$MAX_VIEW_FRAMES" \
    --default-sample-frames "$DEFAULT_SAMPLE_FRAMES" \
    --min-sample-frames "$MIN_SAMPLE_FRAMES" \
    --max-sample-frames "$MAX_SAMPLE_FRAMES" \
    --num-workers "$NUM_WORKERS" \
    --captioner "$CAPTIONER" \
    --annotation-file "$ANNOTATION_FILE" \
    --video-dir "$VIDEO_DIR" \
    --video-list "$VIDEO_LIST" \
    --max-videos "$MAX_VIDEOS" \
    --output-dir "$OUTPUT_DIR" \
    --experiment-name "$EXPERIMENT_NAME" \
    --detailed "$DETAILED"

EXIT_CODE=$?

################################################################################
# Post-Processing

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "[SUCCESS] Evaluation completed at $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    
    # Find and display the output directory
    OUTPUT_PATH=$(ls -td ${OUTPUT_DIR}/${EXPERIMENT_NAME}__* 2>/dev/null | head -1)
    if [ -n "$OUTPUT_PATH" ]; then
        echo "Output directory: $OUTPUT_PATH"
        echo ""
        
        # Display summary if exists
        if [ -f "$OUTPUT_PATH/summary.txt" ]; then
            echo "--- Summary ---"
            cat "$OUTPUT_PATH/summary.txt"
            echo ""
        fi
    fi
else
    echo "[ERROR] Evaluation failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi

echo "============================================================"

