#!/bin/bash
################################################################################
# Test Script - 8 Videos with Grok Model
# Created: 2025-12-01
# Purpose: Test agent workflow, model performance, and evaluation pipeline
################################################################################

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
SCHEDULER_MODEL="x-ai/grok-4-1-fast-non-reasoning"
VIEWER_MODEL="x-ai/grok-4-1-fast-non-reasoning"

# ============================================================================
# TEST CONFIGURATION
# ============================================================================
ROUND_NAME="test_grok_1video"
DATASET="EgoSchema_test"
COUNT=1
MAX_ROUNDS=5
DETAILED="true"
MAX_PROCESSES=4  # Parallel processing
USE_CACHE="false"  # No cache for fresh test

# ============================================================================
# Advanced Settings
# ============================================================================
CONFIG="default"
CAPTION_METHOD="multi_level"

################################################################################
# Environment Setup
################################################################################

cd "$(dirname "$0")/.." || exit 1

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
################################################################################

# Dataset paths - use EgoSchema_test
VIDEO_LIST="data/EgoSchema_test/video_list.txt"
ANNOTATION_FILE="data/EgoSchema_test/annotations.json"
VIDEO_DIR="data/EgoSchema_test/videos"

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
################################################################################

echo "============================================================"
echo "VideoAgent Test Run"
echo "============================================================"
echo ""
echo "Configuration:"
echo "  Round Name:       $ROUND_NAME"
echo "  Scheduler Model:  $SCHEDULER_MODEL"
echo "  Viewer Model:     $VIEWER_MODEL"
echo "  Dataset:          $DATASET"
echo "  Video Count:      $COUNT"
echo "  Max Rounds:       $MAX_ROUNDS"
echo "  Max Processes:    $MAX_PROCESSES"
echo "  Use Cache:        $USE_CACHE"
echo ""
echo "============================================================"
echo ""

################################################################################
# Run Test
################################################################################

echo "[INFO] Starting test at $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

python -m video_agent.cli \
    --config "$CONFIG" \
    --experiment-name "$ROUND_NAME" \
    --scheduler-model "$SCHEDULER_MODEL" \
    --viewer-model "$VIEWER_MODEL" \
    --video-list "$VIDEO_LIST" \
    --annotation-file "$ANNOTATION_FILE" \
    --video-dir "$VIDEO_DIR" \
    --max-videos "$MAX_VIDEOS" \
    --max-rounds "$MAX_ROUNDS" \
    --max-processes "$MAX_PROCESSES" \
    --caption-method "$CAPTION_METHOD" \
    $LLM_LOGGING \
    $CACHE_FLAG

EXIT_CODE=$?

################################################################################
# Post-Processing
################################################################################

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "[SUCCESS] Test completed at $(date '+%Y-%m-%d %H:%M:%S')"
    
    OUTPUT_DIR=$(ls -td results/${ROUND_NAME}__* 2>/dev/null | head -1)
    if [ -n "$OUTPUT_DIR" ]; then
        echo ""
        echo "Output directory: $OUTPUT_DIR"
        echo ""
        
        # Show summary
        if [ -f "$OUTPUT_DIR/summary.txt" ]; then
            echo "============================================================"
            echo "SUMMARY"
            echo "============================================================"
            cat "$OUTPUT_DIR/summary.txt"
        fi
        
        # Show metrics
        if [ -f "$OUTPUT_DIR/metrics.csv" ]; then
            echo ""
            echo "============================================================"
            echo "METRICS"
            echo "============================================================"
            cat "$OUTPUT_DIR/metrics.csv"
        fi
        
        # Show answer details
        if [ -f "$OUTPUT_DIR/answer.json" ]; then
            echo ""
            echo "============================================================"
            echo "ANSWER DETAILS (first 50 lines)"
            echo "============================================================"
            head -50 "$OUTPUT_DIR/answer.json"
        fi
    fi
else
    echo "[ERROR] Test failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi

echo ""
echo "============================================================"
echo "Test Complete"
echo "============================================================"
