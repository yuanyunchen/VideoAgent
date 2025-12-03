#!/bin/bash

################################################################################
# VideoAgent Multi-Agent Evaluation Script Template
#
# This script runs video question answering experiments with the new multi-agent
# architecture featuring:
#   - Solver Agent (stateful): Decision making and answer generation
#   - Viewer Agent: Frame caption generation (requires vision model)
#   - Checker Agent (stateless): Confidence evaluation and feedback
#
# Output Structure:
#   results/<ROUND_NAME>__<SCHEDULER>_viewer_<VIEWER>_videos_<COUNT>_<MMDD>/
#     - logging.log     : Full evaluation log
#     - metrics.csv     : Performance metrics
#     - summary.txt     : Human-readable summary
#     - result.json     : Full results with per-video details
#     - accuracy.txt    : Quick accuracy summary
#     - videos/         : Per-video outputs with frames and logs
#
################################################################################

################################################################################
# Model Configuration
#
# Price Source: https://aimlapi.com/ai-ml-api-pricing
# Full model list: configs/models.yaml
# Last Updated: 2025-12
#
# ============================================================================
# VISION MODELS (support image input) - Required for VIEWER
# ============================================================================
#
# API Model Name                        | Price ($/1M)    | Notes
# --------------------------------------|-----------------|------------------
# gpt-4o                                | $2.50/$10.00    | Excellent quality
# gpt-4o-mini                           | $0.15/$0.60     | Best value (default)
# x-ai/grok-4-1-fast-non-reasoning      | $0.210/$0.530   | Vision + fast
# x-ai/grok-4-1-fast-reasoning          | $0.210/$0.530   | Vision + thinking
# google/gemini-2.5-flash               | $0.32/$2.63     | Fast, 1M context
# alibaba/qwen-vl-max-latest            | $0.40/$1.20     | Strong visual
#
# ============================================================================
# TEXT/REASONING MODELS - For SCHEDULER and CHECKER
# ============================================================================
#
# API Model Name                        | Price ($/1M)    | Notes
# --------------------------------------|-----------------|------------------
# gpt-4o-mini                           | $0.15/$0.60     | Good value
# x-ai/grok-4-1-fast-non-reasoning      | $0.210/$0.530   | Fast
# x-ai/grok-4-1-fast-reasoning          | $0.210/$0.530   | With thinking
# deepseek/deepseek-chat                | $0.14/$0.28     | Budget option
# alibaba/qwen3-235b-a22b-thinking-2507 | $0.242/$2.415   | Best reasoning
#
################################################################################

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
# Scheduler: Solver agent - makes decisions (retrieve frames or answer)
# Viewer:    Caption agent - generates frame captions (MUST support vision!)
# Checker:   Evaluation agent - assesses answer confidence (1-10 scale)
# ============================================================================

# Scheduler model (for Solver agent - decision making)
SCHEDULER_MODEL="gpt-4o-mini"

# Viewer model (for caption generation - MUST support vision/images)
VIEWER_MODEL="gpt-4o-mini"

# Checker model (for confidence evaluation)
# If not set, defaults to SCHEDULER_MODEL
CHECKER_MODEL="gpt-4o-mini"

# ============================================================================
# EXPERIMENT SETTINGS
# ============================================================================

# Test round name (used for organizing results)
ROUND_NAME="evaluation"

# Number of test cases to run (0 or -1 = full dataset)
COUNT=0

# Maximum rounds of interaction (Solver submits answer, Checker evaluates)
# Each round: Solver can retrieve frames OR submit answer
MAX_ROUNDS=5

# Confidence threshold (1-10) to accept an answer
# Higher = stricter, requires more confident answers
# Recommended: 7-8 for balanced, 9+ for strict
CONFIDENCE_THRESHOLD=8

# ============================================================================
# PROCESSING SETTINGS
# ============================================================================

# Multiprocessing (1 = sequential, higher = parallel)
MAX_PROCESSES=1

# Detailed logging (true/false)
DETAILED="true"

# Cache LLM responses (true/false)
USE_CACHE="true"

# Initial frames to sample per video
INITIAL_FRAMES=5

################################################################################
# Advanced Settings (usually don't need to change)

# Configuration file to use as base
CONFIG="default"

# Dataset paths
VIDEO_LIST="data/EgoSchema_test/video_list.txt"
ANNOTATION_FILE="data/EgoSchema_test/annotations.json"
VIDEO_DIR="data/EgoSchema_test/videos"

################################################################################
# Environment Setup

# Load environment variables from .env if it exists
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

# Check if API key is set
if [ -z "$AIML_API_KEY" ]; then
    echo "[ERROR] AIML_API_KEY environment variable is not set."
    echo "Please set it in your .env file or export it directly:"
    echo "  export AIML_API_KEY=your_api_key_here"
    exit 1
fi

################################################################################
# Derived Settings (auto-computed)

# Use SCHEDULER_MODEL for checker if not specified
if [ -z "$CHECKER_MODEL" ]; then
    CHECKER_MODEL="$SCHEDULER_MODEL"
fi

# Convert count for CLI (0 means use all, -1 means all)
if [ "$COUNT" -eq 0 ] || [ "$COUNT" -eq -1 ]; then
    MAX_VIDEOS="-1"
else
    MAX_VIDEOS="$COUNT"
fi

# Set logging based on detailed mode
if [ "$DETAILED" = "true" ]; then
    LLM_LOGGING="--llm-logging"
else
    LLM_LOGGING=""
fi

# Set cache flag
if [ "$USE_CACHE" = "true" ]; then
    CACHE_FLAG=""
else
    CACHE_FLAG="--no-cache"
fi

################################################################################
# Print Configuration Summary

echo "============================================================"
echo "VideoAgent Multi-Agent Evaluation"
echo "============================================================"
echo ""
echo "Models:"
echo "  Scheduler (Solver):  $SCHEDULER_MODEL"
echo "  Viewer (Caption):    $VIEWER_MODEL"
echo "  Checker (Evaluate):  $CHECKER_MODEL"
echo ""
echo "Settings:"
echo "  Round Name:          $ROUND_NAME"
echo "  Count:               $COUNT (MAX_VIDEOS=$MAX_VIDEOS)"
echo "  Max Rounds:          $MAX_ROUNDS"
echo "  Confidence:          $CONFIDENCE_THRESHOLD/10"
echo "  Initial Frames:      $INITIAL_FRAMES"
echo "  Max Processes:       $MAX_PROCESSES"
echo "  Detailed:            $DETAILED"
echo "  Use Cache:           $USE_CACHE"
echo ""
echo "============================================================"
echo ""

################################################################################
# Run Evaluation

echo "[INFO] Starting evaluation at $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

python -m video_agent.cli \
    --config "$CONFIG" \
    --experiment-name "$ROUND_NAME" \
    --scheduler-model "$SCHEDULER_MODEL" \
    --viewer-model "$VIEWER_MODEL" \
    --checker-model "$CHECKER_MODEL" \
    --video-list "$VIDEO_LIST" \
    --annotation-file "$ANNOTATION_FILE" \
    --video-dir "$VIDEO_DIR" \
    --max-videos "$MAX_VIDEOS" \
    --max-rounds "$MAX_ROUNDS" \
    --confidence-threshold "$CONFIDENCE_THRESHOLD" \
    --initial-frames "$INITIAL_FRAMES" \
    --max-processes "$MAX_PROCESSES" \
    $LLM_LOGGING \
    $CACHE_FLAG

EXIT_CODE=$?

################################################################################
# Post-Processing

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "[SUCCESS] Evaluation completed at $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    
    # Find and display the output directory
    OUTPUT_DIR=$(ls -td results/${ROUND_NAME}__* 2>/dev/null | head -1)
    if [ -n "$OUTPUT_DIR" ]; then
        echo "Output directory: $OUTPUT_DIR"
        echo ""
        
        # Display summary if exists
        if [ -f "$OUTPUT_DIR/summary.txt" ]; then
            echo "--- Summary ---"
            cat "$OUTPUT_DIR/summary.txt"
            echo ""
        fi
    fi
else
    echo "[ERROR] Evaluation failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi

echo "============================================================"
