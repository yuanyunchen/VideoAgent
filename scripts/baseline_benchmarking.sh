#!/bin/bash

################################################################################
# Baseline Benchmarking Script
# Model: Grok 4.1 Fast Non-Reasoning (Vision + Text)
# Videos: 100
# Processes: 16
################################################################################

# Model Configuration
SCHEDULER_MODEL="x-ai/grok-4-1-fast-reasoning"
VIEWER_MODEL="x-ai/grok-4-1-fast-non-reasoning"

# Test Configuration
ROUND_NAME="reasoning_baseline_benchmarking"
COUNT=100
MAX_ROUNDS=3
MAX_PROCESSES=16

# Dataset - using EgoSchema_test
ANNOTATION_FILE="data/EgoSchema_test/annotations.json"
VIDEO_LIST="data/EgoSchema_test/video_list.txt"
VIDEO_DIR="data/EgoSchema_test/videos"

# Other settings
USE_CACHE="true"
DETAILED="true"
CONFIG="default"

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
    exit 1
fi

################################################################################
# Derived Settings

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
echo "Baseline Benchmarking"
echo "============================================================"
echo "Scheduler Model:  $SCHEDULER_MODEL"
echo "Viewer Model:     $VIEWER_MODEL"
echo "Max Processes:    $MAX_PROCESSES"
echo "Video Count:      $COUNT"
echo "Max Rounds:       $MAX_ROUNDS"
echo "Video List:       $VIDEO_LIST"
echo "============================================================"
echo ""

################################################################################
# Run Evaluation

python -m video_agent.cli \
    --config "$CONFIG" \
    --scheduler-model "$SCHEDULER_MODEL" \
    --viewer-model "$VIEWER_MODEL" \
    --experiment-name "$ROUND_NAME" \
    --video-list "$VIDEO_LIST" \
    --annotation-file "$ANNOTATION_FILE" \
    --video-dir "$VIDEO_DIR" \
    --max-videos "$COUNT" \
    --max-rounds "$MAX_ROUNDS" \
    --max-processes "$MAX_PROCESSES" \
    $LLM_LOGGING \
    $CACHE_FLAG

EXIT_CODE=$?

################################################################################
# Show Results

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "[SUCCESS] Benchmarking completed"
    
    OUTPUT_DIR=$(ls -td results/${ROUND_NAME}__* 2>/dev/null | head -1)
    if [ -n "$OUTPUT_DIR" ]; then
        echo "Output: $OUTPUT_DIR"
        echo ""
        
        if [ -f "$OUTPUT_DIR/summary.txt" ]; then
            cat "$OUTPUT_DIR/summary.txt"
        fi
    fi
else
    echo "[ERROR] Benchmarking failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi

