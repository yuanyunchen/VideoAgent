#!/bin/bash

################################################################################
# VideoAgent Evaluation Script Template
#
# This script runs video question answering experiments with configurable
# models, datasets, and parameters. Results are saved with standardized
# output structure for easy comparison and analysis.
#
# Output Structure:
#   results/<ROUND_NAME>_<MODEL>_<COUNT>_<MMDD>[_HHMM]/
#     - logging.log     : Full evaluation log
#     - metrics.csv     : Performance metrics (accuracy, improvement rate, etc.)
#     - summary.txt     : Human-readable summary with case analysis
#     - stats.json      : Full statistics in JSON format
#     - answer.json     : Detailed predictions with Q&A
#     - result.json     : Raw results (backward compatible)
#     - videos/         : Per-video outputs with frames and logs
#
# Key Metrics:
#   - Accuracy:           Final answer accuracy
#   - First Round Acc:    Accuracy after first round only
#   - Improvement Rate:   % of initially wrong answers that became correct
#   - Case Types:         MAINTAINED, IMPROVED, DEGRADED, FAILED
#
################################################################################

################################################################################
# Evaluation Settings
#
# Price Source: https://aimlapi.com/ai-ml-api-pricing
# Full model list: configs/models.yaml
# Last Updated: 2025-01
#
# ============================================================================
# MULTIMODAL MODELS (for viewer_model - caption generation)
# ============================================================================
# These models support VISION (image input) - use for frame captioning
#
# API Model Name                        | Price ($/1M)    | Rating | Notes
# --------------------------------------|-----------------|--------|------------------
# openai/gpt-4.1                        | $2.00/$8.00     | *****  | Latest GPT with vision
# openai/gpt-4.1-mini                   | $0.40/$1.60     | ****   | Fast, good value
# gpt-4o                                | $2.50/$10.00    | *****  | Excellent quality
# gpt-4o-mini                           | $0.15/$0.60     | ****   | Best value (default)
# anthropic/claude-4-sonnet             | $3.00/$15.00    | *****  | Latest Claude vision
# anthropic/claude-4.5-sonnet           | $3.00/$15.00    | *****  | Excellent reasoning
# claude-3-haiku                        | $0.25/$1.25     | ***    | Budget option
# google/gemini-2.5-pro                 | $1.25/$10.00    | *****  | Best long context
# google/gemini-2.5-flash               | $0.15/$0.60     | ****   | Fast, 1M context
# alibaba/qwen-vl-max-latest            | $0.40/$1.20     | ****   | Strong visual
# alibaba/qwen-vl-plus                  | $0.15/$0.45     | ***    | Budget Qwen vision
#
# ============================================================================
# TEXT/REASONING MODELS (for scheduler_model - Q&A, evaluation)
# ============================================================================
# These models can be used for reasoning/answering tasks
#
# API Model Name                                | Price ($/1M)    | Think  | Vision | Rating
# ----------------------------------------------|-----------------|--------|--------|--------
# --- xAI Grok Models (Multimodal!) ---
# x-ai/grok-4-1-fast-reasoning                  | $0.210/$0.530   | ~1.8k  | YES    | ****+
# x-ai/grok-4-1-fast-non-reasoning              | $0.210/$0.530   | -      | YES    | ****+
# --- Thinking/Reasoning Models (Text-only) ---
# alibaba/qwen3-next-80b-a3b-thinking           | $0.158/$1.600   | ~1.7k  | NO     | ****
# alibaba/qwen3-235b-a22b-thinking-2507         | $0.242/$2.415   | ~2.2k  | NO     | *****
# deepseek/deepseek-reasoner-v3.1               | $0.294/$0.441   | ~5.6k  | NO     | *****
# deepseek/deepseek-reasoner                    | $0.294/$0.441   | ~500   | NO     | ****
# minimax/m2                                    | $0.315/$1.260   | ~1.9k  | NO     | ****+
# --- Standard Text Models ---
# openai/gpt-oss-120b                           | $0.20/$0.80     | -      | NO     | ****
# openai/gpt-oss-20b                            | $0.05/$0.20     | -      | NO     | ***
# deepseek/deepseek-v3                          | $0.14/$0.28     | -      | NO     | ****
# deepseek/deepseek-chat                        | $0.14/$0.28     | -      | NO     | ****
# alibaba/qwen-turbo-latest                     | $0.05/$0.15     | -      | NO     | ***
#
# NOTE: x-ai/grok-4-1-fast-* models support BOTH vision and text!
#       They can be used as both SCHEDULER and VIEWER model.
# ============================================================================

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
# Scheduler: Handles Q&A, confidence evaluation, segment planning (text-only OK)
# Viewer:    Generates frame captions (MUST support vision/images)
# ============================================================================

# Scheduler model (for Q&A and evaluation - can be text-only)
SCHEDULER_MODEL="gpt-4o-mini"

# Viewer model (for frame captioning - MUST support vision/images)
# WARNING: If using text-only model here, caption generation will fail!
VIEWER_MODEL="gpt-4o-mini"

# Test round name (used for organizing results)
# Results will be saved to: results/<ROUND_NAME>_<MODEL>_<COUNT>_<MMDD>/
ROUND_NAME="evaluation"

# Dataset configuration
# Options: subset_valid (default), subset, test_one_video
DATASET="subset_valid"

# Number of test cases to run (0 = full dataset, -1 = all available)
COUNT=0

# Maximum rounds for iterative refinement (1-5 recommended)
MAX_ROUNDS=3

# Detailed output mode (true/false)
# When true: enables LLM logging and verbose output
DETAILED="true"

# Multiprocessing settings
# MAX_PROCESSES: number of parallel workers (1 = sequential, 0 = auto)
MAX_PROCESSES=4

# Cache settings (true/false)
# When true: reuses LLM responses from cache
USE_CACHE="true"

################################################################################
# Advanced Settings (usually don't need to change)

# Configuration file to use as base
CONFIG="default"

# Caption method: multi_level, detailed, group
CAPTION_METHOD="multi_level"

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

# Use SCHEDULER_MODEL for viewer if not specified
if [ -z "$VIEWER_MODEL" ]; then
    VIEWER_MODEL="$SCHEDULER_MODEL"
fi

# Map dataset name to file
case "$DATASET" in
    "subset_valid")
        VIDEO_LIST="data/video_lists/subset_valid.txt"
        ;;
    "subset")
        VIDEO_LIST="data/video_lists/subset.txt"
        ;;
    "test_one_video")
        VIDEO_LIST="data/video_lists/test_one_video.txt"
        ;;
    *)
        VIDEO_LIST="data/video_lists/${DATASET}.txt"
        ;;
esac

# Convert count for CLI (0 means use all, -1 means all)
if [ "$COUNT" -eq 0 ]; then
    MAX_VIDEOS="-1"
else
    MAX_VIDEOS="$COUNT"
fi

# Set logging based on detailed mode
if [ "$DETAILED" = "true" ]; then
    LLM_LOGGING="--llm-logging"
    echo "[INFO] Detailed mode enabled - LLM interactions will be logged"
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
echo "VideoAgent Evaluation"
echo "============================================================"
echo ""
echo "Configuration:"
echo "  Round Name:       $ROUND_NAME"
echo "  Scheduler Model:  $SCHEDULER_MODEL"
echo "  Viewer Model:     $VIEWER_MODEL"
echo "  Dataset:        $DATASET"
echo "  Video List:     $VIDEO_LIST"
echo "  Count:          $COUNT (MAX_VIDEOS=$MAX_VIDEOS)"
echo "  Max Rounds:     $MAX_ROUNDS"
echo "  Max Processes:  $MAX_PROCESSES"
echo "  Detailed:       $DETAILED"
echo "  Use Cache:      $USE_CACHE"
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
    --max-videos "$MAX_VIDEOS" \
    --max-rounds "$MAX_ROUNDS" \
    --max-processes "$MAX_PROCESSES" \
    --caption-method "$CAPTION_METHOD" \
    $LLM_LOGGING \
    $CACHE_FLAG

EXIT_CODE=$?

################################################################################
# Post-Processing

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "[SUCCESS] Evaluation completed at $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    echo "Results saved to: results/${ROUND_NAME}__*"
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
        
        # Display accuracy if exists
        if [ -f "$OUTPUT_DIR/accuracy.txt" ]; then
            echo "--- Accuracy ---"
            cat "$OUTPUT_DIR/accuracy.txt"
            echo ""
        fi
    fi
else
    echo "[ERROR] Evaluation failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi

echo "============================================================"

