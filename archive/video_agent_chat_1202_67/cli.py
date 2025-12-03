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
        description="VideoAgent: Multi-agent video understanding with question answering",
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
    parser.add_argument("--scheduler-model", help="Model for Solver agent (decision making)")
    parser.add_argument("--viewer-model", help="Model for caption generation (must support vision)")
    parser.add_argument("--checker-model", help="Model for Checker agent (confidence evaluation)")
    
    # API configuration
    parser.add_argument("--aiml-api-key", dest="aiml_api_key", help="Override AIML API key")
    
    # Processing overrides
    parser.add_argument("--max-rounds", type=int, dest="max_rounds", 
                       help="Maximum steps in main loop (default: 10)")
    parser.add_argument("--max-videos", type=int, dest="max_test_videos", 
                       help="Number of videos to process (-1 for all)")
    parser.add_argument("--initial-frames", type=int, dest="default_initial_frames",
                       help="Number of initial frames to sample (default: 5)")
    parser.add_argument("--confidence-threshold", type=int, dest="confidence_threshold",
                       help="Confidence threshold to accept answer (1-10, default: 8)")
    
    # Data path overrides
    parser.add_argument("--video-list", dest="test_video_list_file",
                       help="Path to video list file")
    parser.add_argument("--annotation-file", dest="annotation_file",
                       help="Path to annotation JSON file")
    parser.add_argument("--video-dir", dest="video_dir",
                       help="Path to video directory")
    
    # Execution overrides
    parser.add_argument("--experiment-name", dest="experiment_name", 
                       help="Experiment name for output directory")
    parser.add_argument("--max-processes", type=int, dest="max_processes", 
                       help="Number of parallel processes (default: 1)")
    parser.add_argument("--llm-logging", action="store_true", dest="enable_llm_logging", 
                       help="Enable detailed LLM logging")
    parser.add_argument("--no-cache", action="store_false", dest="use_cache",
                       help="Disable LLM response caching")
    
    # Output options
    parser.add_argument("--quiet", "-q", action="store_true",
                       help="Suppress progress output")
    
    # Backward compatibility with old CLI
    parser.add_argument("--caption-method", dest="caption_method",
                       help="(Deprecated) Caption method - ignored in new system")
    
    return parser.parse_args()


def print_header(config):
    """Print experiment header."""
    print("=" * 60)
    print("VideoAgent Multi-Agent Evaluation")
    print("=" * 60)
    print(f"Start Time:      {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Experiment:      {config.get('experiment_name', 'default')}")
    print(f"Scheduler Model: {config.get('scheduler_model', 'N/A')}")
    print(f"Viewer Model:    {config.get('viewer_model', 'N/A')}")
    print(f"Checker Model:   {config.get('checker_model', config.get('scheduler_model', 'N/A'))}")
    print(f"Max Videos:      {config.get('max_test_videos', -1)}")
    print(f"Max Steps:       {config.get('max_rounds', 10)}")
    print(f"Confidence:      {config.get('confidence_threshold', 8)}/10")
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
    
    # Apply overrides from command line
    overrides = {k: v for k, v in vars(args).items() 
                if v is not None and k not in ['config', 'list_configs', 'quiet', 'caption_method']}
    
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
        import traceback
        traceback.print_exc()
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

