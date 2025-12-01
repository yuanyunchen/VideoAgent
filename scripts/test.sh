#!/bin/bash

# VideoAgent - Default Configuration Run
# ====================================
# Basic experiment with default settings

# python main.py \
#     --config default \
#     --experiment-name "first_test_after_reconstruction" \
#     --max-videos 100 \
#     --max-round 5 \
#     --aiml-api-key "fb50dec85566407bbc25ce1d28828fe7"

python main.py \
    --config default \
    --aiml-api-key "fb50dec85566407bbc25ce1d28828fe7" \
    --experiment-name "new_base_model" \
    --max-videos 10 \
    --scheduler-model Qwen/Qwen3-235B-A22B-fp8-tput \
    --viewer-model Qwen/Qwen3-235B-A22B-fp8-tput

