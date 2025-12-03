#!/bin/bash

################################################################################
# Direct Frame Evaluation Script
# Evaluates model accuracy by showing only the first N frames (no captions)
################################################################################

# Model Configuration
MODEL="x-ai/grok-4-1-fast-non-reasoning"

# Frame Configuration
# Number of initial frames to show the model (without captions)
NUM_FRAMES=5

# Test Configuration
ROUND_NAME="polished_5_frame_eval"
COUNT=100

# Parallel Processing
# Number of worker processes for parallel evaluation
MAX_PROCESSES=32

# Dataset - using EgoSchema_test
ANNOTATION_FILE="data/EgoSchema_test/annotations.json"
VIDEO_LIST="data/EgoSchema_test/video_list.txt"
VIDEO_DIR="data/EgoSchema_test/videos"

# Output
OUTPUT_DIR="results"

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
# Print Configuration

echo "============================================================"
echo "Direct Frame Evaluation"
echo "============================================================"
echo "Model:          $MODEL"
echo "Num Frames:     $NUM_FRAMES"
echo "Video Count:    $COUNT"
echo "Max Processes:  $MAX_PROCESSES"
echo "Video List:     $VIDEO_LIST"
echo "============================================================"
echo ""

################################################################################
# Run Evaluation

python -m evaluation.direct_frame_eval \
    --model "$MODEL" \
    --num-frames "$NUM_FRAMES" \
    --max-videos "$COUNT" \
    --max-processes "$MAX_PROCESSES" \
    --experiment-name "$ROUND_NAME" \
    --video-dir "$VIDEO_DIR" \
    --annotation-file "$ANNOTATION_FILE" \
    --video-list "$VIDEO_LIST" \
    --output-dir "$OUTPUT_DIR"

EXIT_CODE=$?

################################################################################
# Show Results

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "[SUCCESS] Evaluation completed"
    
    OUTPUT_PATH=$(ls -td results/${ROUND_NAME}__* 2>/dev/null | head -1)
    if [ -n "$OUTPUT_PATH" ]; then
        echo "Output: $OUTPUT_PATH"
    fi
else
    echo "[ERROR] Evaluation failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi
