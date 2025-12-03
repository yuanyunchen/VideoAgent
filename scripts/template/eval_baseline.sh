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
# Last Updated: 2025-12
#
# ============================================================================
# VISION MODELS (support image input)
# ============================================================================
# Use for: viewer_model (frame captioning), or as scheduler if vision needed
#
# API Model Name                        | Price ($/1M)    | Think  | Rating | Notes
# --------------------------------------|-----------------|--------|--------|------------------
# --- OpenAI ---
# openai/gpt-5.1                        | $1.31/$10.50    | -      | *****  | Latest GPT-5.1
# openai/gpt-5.1-chat-latest            | $1.31/$10.50    | -      | *****  | GPT-5.1 chat variant
# openai/gpt-4.1-mini                   | $0.42/$1.68     | -      | ****   | Fast, good value
# openai/gpt-4.1-nano                   | $0.11/$0.42     | -      | ***    | Budget option
# gpt-4o                                | $2.50/$10.00    | -      | *****  | Excellent quality
# gpt-4o-mini                           | $0.15/$0.60     | -      | ****   | Best value (default)
# --- Anthropic ---
# anthropic/claude-4.5-sonnet           | $3.00/$15.00    | -      | *****  | Excellent reasoning
# anthropic/claude-haiku-4.5            | $1.05/$5.25     | -      | ***    | Budget option
# --- Google ---
# google/gemini-3-pro-preview           | $4.20/$18.90    | -      | *****  | Latest Gemini 3
# google/gemini-2.5-pro                 | $1.31/$10.50    | -      | *****  | Best long context
# google/gemini-2.5-flash               | $0.32/$2.63     | -      | ****   | Fast, 1M context
# --- Alibaba ---
# alibaba/qwen-vl-max-latest            | $0.40/$1.20     | -      | ****   | Strong visual
# alibaba/qwen-vl-plus                  | $0.15/$0.45     | -      | ***    | Budget Qwen vision
# --- xAI (Vision + Reasoning!) ---
# x-ai/grok-4-1-fast-reasoning          | $0.210/$0.530   | ~1.8k  | ****+  | Vision + thinking
# x-ai/grok-4-1-fast-non-reasoning      | $0.210/$0.530   | -      | ****+  | Vision, no thinking
#
# ============================================================================
# TEXT-ONLY MODELS (no vision support)
# ============================================================================
# Use for: scheduler_model (Q&A, evaluation, reasoning)
#
# API Model Name                                | Price ($/1M)    | Think  | Rating | Notes
# ----------------------------------------------|-----------------|--------|--------|------------------
# --- Thinking/Reasoning Models ---
# alibaba/qwen3-next-80b-a3b-thinking           | $0.158/$1.600   | ~1.7k  | ****   | MoE thinking
# alibaba/qwen3-235b-a22b-thinking-2507         | $0.242/$2.415   | ~2.2k  | *****  | Best Qwen reasoning
# deepseek/deepseek-reasoner-v3.1               | $0.294/$0.441   | ~5.6k  | *****  | Deep reasoning
# deepseek/deepseek-reasoner                    | $0.294/$0.441   | ~500   | ****   | Fast reasoning
# minimax/m2                                    | $0.315/$1.260   | ~1.9k  | ****+  | Balanced
# --- Standard Text Models ---
# openai/gpt-oss-120b                           | $0.20/$0.80     | -      | ****   | Large OSS model
# openai/gpt-oss-20b                            | $0.05/$0.20     | -      | ***    | Small OSS model
# deepseek/deepseek-chat                        | $0.14/$0.28     | -      | ****   | Good value
# alibaba/qwen-turbo-latest                     | $0.05/$0.15     | -      | ***    | Budget option
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
# Default: EgoSchema_test (500 videos)
DATASET="EgoSchema_test"

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

# Map dataset name to paths
VIDEO_LIST="data/EgoSchema_test/video_list.txt"
ANNOTATION_FILE="data/EgoSchema_test/annotations.json"
VIDEO_DIR="data/EgoSchema_test/videos"

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

