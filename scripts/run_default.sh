#!/bin/bash

# VideoAgent - Default Configuration Run
# ====================================
# Basic experiment with default settings

# Load environment variables from .env if it exists
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Check if API key is set
if [ -z "$AIML_API_KEY" ]; then
    echo "Error: AIML_API_KEY environment variable is not set."
    echo "Please set it in your .env file or export it directly."
    exit 1
fi

python -m video_agent.cli \
    --config default \
    --experiment-name "default_experiment" \
    --max-videos 1

echo "Default experiment completed!" 