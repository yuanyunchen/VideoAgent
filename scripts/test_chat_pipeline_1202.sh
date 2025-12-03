#!/bin/bash

################################################################################
# Test Script: Multi-Agent Chat Pipeline
# Date: 2025-12-02
#
# Tests the new multi-agent video understanding system with:
#   - All 3 models using x-ai/grok-4-1-fast-non-reasoning
#   - Confidence threshold: 8/10 (default)
#   - Max 3 rounds of interaction (answer submissions)
#   - Tests on 5 videos for quick validation
#
################################################################################

# ============================================================================
# MODEL CONFIGURATION - All using Grok 4.1 Fast (Non-Reasoning)
# ============================================================================

SCHEDULER_MODEL="x-ai/grok-4-1-fast-non-reasoning"
VIEWER_MODEL="x-ai/grok-4-1-fast-non-reasoning"
CHECKER_MODEL="x-ai/grok-4-1-fast-non-reasoning"

# ============================================================================
# EXPERIMENT SETTINGS
# ============================================================================

ROUND_NAME="chat_pipeline_no_checker_1202"
COUNT=100                    # Test on 5 videos
MAX_ROUNDS=3               # 3 rounds of interaction (answer submissions)
CONFIDENCE_THRESHOLD=8     # Default confidence threshold (1-10)
INITIAL_FRAMES=5           # Start with 5 frames

# ============================================================================
# PROCESSING SETTINGS
# ============================================================================

MAX_PROCESSES=25            # Parallel workers
DETAILED="true"            # Enable detailed logging
USE_CACHE="false"          # Disable cache for fresh test

################################################################################
# Dataset paths
VIDEO_LIST="data/EgoSchema_test/video_list.txt"
ANNOTATION_FILE="data/EgoSchema_test/annotations.json"
VIDEO_DIR="data/EgoSchema_test/videos"
CONFIG="default"

################################################################################
# Environment Setup

if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

if [ -z "$AIML_API_KEY" ]; then
    echo "[ERROR] AIML_API_KEY environment variable is not set."
    exit 1
fi

################################################################################
# Derived Settings

MAX_VIDEOS="$COUNT"

if [ "$DETAILED" = "true" ]; then
    LLM_LOGGING="--llm-logging"
else
    LLM_LOGGING=""
fi

if [ "$USE_CACHE" = "true" ]; then
    CACHE_FLAG=""
else
    CACHE_FLAG="--no-cache"
fi

################################################################################
# Print Configuration

echo "============================================================"
echo "Test: Multi-Agent Chat Pipeline"
echo "============================================================"
echo ""
echo "Models (all using Grok 4.1 Fast Non-Reasoning):"
echo "  Scheduler (Solver):  $SCHEDULER_MODEL"
echo "  Viewer (Caption):    $VIEWER_MODEL"
echo "  Checker (Evaluate):  $CHECKER_MODEL"
echo ""
echo "Settings:"
echo "  Round Name:          $ROUND_NAME"
echo "  Videos:              $COUNT"
echo "  Max Rounds:          $MAX_ROUNDS"
echo "  Confidence:          $CONFIDENCE_THRESHOLD/10"
echo "  Initial Frames:      $INITIAL_FRAMES"
echo ""
echo "============================================================"
echo ""

################################################################################
# Run Test

echo "[INFO] Starting test at $(date '+%Y-%m-%d %H:%M:%S')"
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
    echo "[SUCCESS] Test completed at $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    
    OUTPUT_DIR=$(ls -td results/${ROUND_NAME}__* 2>/dev/null | head -1)
    if [ -n "$OUTPUT_DIR" ]; then
        echo "Output directory: $OUTPUT_DIR"
        echo ""
        
        if [ -f "$OUTPUT_DIR/summary.txt" ]; then
            echo "--- Summary ---"
            cat "$OUTPUT_DIR/summary.txt"
        fi
    fi
else
    echo "[ERROR] Test failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi

echo "============================================================"

