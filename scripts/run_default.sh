#!/bin/bash

# VideoAgent - Default Configuration Run
# ====================================
# Basic experiment with default settings


python main.py \
    --config default \
    --experiment-name "default_experiment" \
    --max-videos 1 \
    --aiml-api-key "fb50dec85566407bbc25ce1d28828fe7"

echo "✅ Default experiment completed!" 