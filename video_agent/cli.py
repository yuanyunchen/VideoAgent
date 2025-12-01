#!/usr/bin/env python3
"""
VideoAgent CLI interface with YAML configuration management.
"""

import argparse
import os
import sys
from datetime import datetime

from video_agent.agent import VideoAgent
from video_agent.utils.config import load_config, list_configs, update_config


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="VideoAgent: Video analysis with question answering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m video_agent.cli --config default --max-videos 10
  python -m video_agent.cli --experiment-name test --scheduler-model gpt-4o
  python -m video_agent.cli --list-configs
        """
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
    parser.add_argument("--scheduler-model", help="Model for Q&A and evaluation")
    parser.add_argument("--viewer-model", help="Model for caption generation")
    
    # API configuration
    parser.add_argument("--aiml-api-key", dest="aiml_api_key", help="Override AIML API key")
    
    # Processing overrides
    parser.add_argument("--max-rounds", type=int, dest="max_rounds", help="Maximum analysis rounds (default: 5)")
    parser.add_argument("--max-videos", type=int, dest="max_test_videos", help="Number of videos to process (-1 for all)")
    parser.add_argument("--caption-method", dest="caption_method", 
                       choices=["multi_level", "detailed", "group"],
                       help="Caption generation method")
    parser.add_argument("--video-processing-method", dest="video_processing_method", 
                       help="Video processing method")
    
    # Execution overrides
    parser.add_argument("--experiment-name", dest="experiment_name", 
                       help="Experiment name for output directory")
    parser.add_argument("--no-multiprocess", action="store_false", dest="multi_process", 
                       help="Disable multiprocessing")
    parser.add_argument("--max-processes", type=int, dest="max_processes", 
                       help="Number of parallel processes (default: auto)")
    parser.add_argument("--llm-logging", action="store_true", dest="enable_llm_logging", 
                       help="Enable detailed LLM logging")
    parser.add_argument("--no-cache", action="store_false", dest="use_cache",
                       help="Disable LLM response caching")
    
    # Output options
    parser.add_argument("--quiet", "-q", action="store_true",
                       help="Suppress progress output")
    
    return parser.parse_args()


def print_header(config):
    """Print experiment header."""
    print("=" * 60)
    print("VideoAgent Evaluation")
    print("=" * 60)
    print(f"Start Time:      {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Experiment:      {config.get('experiment_name', 'default')}")
    print(f"Scheduler Model: {config.get('scheduler_model', 'N/A')}")
    print(f"Viewer Model:    {config.get('viewer_model', 'N/A')}")
    print(f"Max Videos:      {config.get('max_test_videos', -1)}")
    print(f"Max Rounds:      {config.get('max_rounds', 1)}")
    print(f"Max Processes:   {config.get('max_processes', 1)}")
    print("=" * 60)
    print()


def print_summary(output_dir):
    """Print summary from results."""
    summary_file = os.path.join(output_dir, "summary.txt")
    if os.path.exists(summary_file):
        print()
        with open(summary_file, 'r') as f:
            print(f.read())
    else:
        accuracy_file = os.path.join(output_dir, "accuracy.txt")
        if os.path.exists(accuracy_file):
            print()
            print("-" * 40)
            print("Results")
            print("-" * 40)
            with open(accuracy_file, 'r') as f:
                print(f.read())


def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Handle list configs
    if args.list_configs:
        configs = list_configs()
        print("Available configurations:")
        for config_name in configs:
            print(f"  - {config_name}")
        return 0
    
    # Load and update configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Error loading configuration: {e}", file=sys.stderr)
        return 1
    
    # Apply overrides
    overrides = {k: v for k, v in vars(args).items() 
                if v is not None and k not in ['config', 'list_configs', 'quiet']}
    
    if overrides:
        config = update_config(config, **overrides)
    
    # Print header
    if not args.quiet:
        print_header(config)
    
    # Initialize and run VideoAgent
    try:
        agent = VideoAgent(config=config)
        output_dir = agent.run_experiment()
    except KeyboardInterrupt:
        print("\nExperiment interrupted by user.")
        return 130
    except Exception as e:
        print(f"Error during experiment: {e}", file=sys.stderr)
        return 1
    
    # Print summary
    if not args.quiet:
        print_summary(output_dir)
        print(f"\nResults saved to: {output_dir}")
    else:
        print(output_dir)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

