#!/usr/bin/env python3
"""
VideoAgent CLI interface with YAML configuration management.
"""

import argparse
import sys
from video_agent import VideoAgent
from utils.config import load_config, list_configs, update_config

# ============================================================================
# CLI INTERFACE
# ============================================================================

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="VideoAgent: Video analysis with question answering",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Configuration selection
    parser.add_argument(
        "--config", 
        default="default",
        help="Configuration to use (default: %(default)s)"
    )
    parser.add_argument(
        "--list-configs", 
        action="store_true",
        help="List available configurations and exit"
    )
    
    # Model overrides
    parser.add_argument("--scheduler-model", help="Override scheduler model")
    parser.add_argument("--viewer-model", help="Override viewer model")
    
    # API configuration
    parser.add_argument("--aiml-api-key", dest="aiml_api_key", help="Override AIML API key")
    
    # Processing overrides
    parser.add_argument("--max-rounds", type=int, help="Override max rounds")
    parser.add_argument("--max-videos", type=int, dest="max_test_videos", help="Override max videos")
    parser.add_argument("--caption-method", help="Override caption method")
    parser.add_argument("--video-processing-method", dest="video_processing_method", help="Override video processing method")
    
    # Execution overrides
    parser.add_argument("--experiment-name", dest="experiment_name", help="Override experiment name")
    parser.add_argument("--no-multiprocess", action="store_false", dest="multi_process", help="Disable multiprocessing")
    parser.add_argument("--max-processes", type=int, dest="max_processes", help="Override max processes (default: 1)")
    parser.add_argument("--llm-logging", action="store_true", dest="enable_llm_logging", help="Enable LLM logging")
    
    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Handle list configs
    if args.list_configs:
        configs = list_configs()
        print("Available configurations:")
        for config_name in configs:
            print(f"  - {config_name}")
        return
    
    # Load and update configuration
    config = load_config(args.config)
    
    # Apply overrides
    overrides = {k: v for k, v in vars(args).items() 
                if v is not None and k not in ['config', 'list_configs']}
    
    if overrides:
        config = update_config(config, **overrides)
    
    # Initialize and run VideoAgent
    agent = VideoAgent(config=config)
    output_dir = agent.run_experiment()
    
    print(f"Experiment completed. Results saved to: {output_dir}")

if __name__ == "__main__":
    main()
