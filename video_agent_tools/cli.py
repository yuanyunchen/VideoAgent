"""
CLI Entry Point for VideoAgent Tools

Command-line interface for running video agent evaluation.
"""

import os
import sys
import argparse

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from video_agent_tools.evaluation import VideoAgentEvaluator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="VideoAgent Tools - Multi-Tools Video Understanding Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings
  python -m video_agent_tools.cli --annotation-file data/annotations.json --video-dir data/videos
  
  # Run with specific tools
  python -m video_agent_tools.cli --tools caption_image,view_frame,detect_objects
  
  # Run on limited videos
  python -m video_agent_tools.cli --max-videos 10
        """,
    )
    
    # Required arguments
    parser.add_argument(
        "--annotation-file",
        type=str,
        default="data/EgoSchema_test/annotations.json",
        help="Path to annotations JSON file",
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        default="data/EgoSchema_test/videos",
        help="Directory containing video files",
    )
    
    # Optional arguments
    parser.add_argument(
        "--video-list",
        type=str,
        default=None,
        help="Path to file with video IDs to include (one per line)",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=-1,
        help="Maximum videos to process (-1 for all)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="LLM model name for agent",
    )
    parser.add_argument(
        "--tools",
        type=str,
        default="view_frame,temporal_sample_frames,temporal_spatial_sample_frames,detect_objects,detect_all_objects,describe_region",
        help="Comma-separated list of tools to enable",
    )
    parser.add_argument(
        "--max-tool-calls",
        type=int,
        default=10,
        help="Maximum tool calls per video before forced answer",
    )
    parser.add_argument(
        "--max-parallel-tools",
        type=int,
        default=3,
        help="Maximum tools agent can request in a single turn (all will be executed)",
    )
    parser.add_argument(
        "--initial-frames",
        type=int,
        default=5,
        help="Number of frames to caption at initialization",
    )
    parser.add_argument(
        "--captioner",
        type=str,
        default="gpt-4o-mini",
        help="Captioner model: 'omni-captioner' for local OmniCaptioner, or API model name (e.g., 'gpt-4o-mini', 'x-ai/grok-4-1-fast-reasoning')",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Base output directory for results",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="tools_agent",
        help="Name for this experiment (used in output directory)",
    )
    parser.add_argument(
        "--detailed",
        type=str,
        default="true",
        choices=["true", "false"],
        help="Detailed console output (true/false). Files always save full details.",
    )
    
    # Multiprocessing
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of worker processes (1 = single process, >1 = parallel processing)",
    )
    
    # Frame control parameters
    parser.add_argument(
        "--max-view-frames",
        type=int,
        default=8,
        help="Maximum frames for view_frame tool (prevents token overflow)",
    )
    parser.add_argument(
        "--default-sample-frames",
        type=int,
        default=16,
        help="Default number of frames for sampling tools",
    )
    parser.add_argument(
        "--min-sample-frames",
        type=int,
        default=1,
        help="Minimum number of frames for sampling tools",
    )
    parser.add_argument(
        "--max-sample-frames",
        type=int,
        default=32,
        help="Maximum number of frames for sampling tools",
    )
    parser.add_argument(
        "--restore-path",
        type=str,
        default=None,
        help="Path to previous result directory to restore from. Will load completed videos and continue with unfinished ones.",
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Parse tools list
    enabled_tools = [t.strip() for t in args.tools.split(",") if t.strip()]
    
    # Parse detailed flag
    detailed = args.detailed.lower() == "true"
    
    print("=" * 70)
    print("VideoAgent Tools Evaluation")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Tools: {enabled_tools}")
    print(f"Max Tool Calls: {args.max_tool_calls}")
    print(f"Max Parallel Tools: {args.max_parallel_tools}")
    print(f"Initial Frames: {args.initial_frames}")
    print(f"Max Videos: {args.max_videos if args.max_videos > 0 else 'all'}")
    print(f"Num Workers: {args.num_workers}")
    print(f"Detailed Output: {detailed}")
    print(f"Frame Limits: view={args.max_view_frames}, sample={args.min_sample_frames}-{args.max_sample_frames} (default={args.default_sample_frames})")
    print(f"Captioner: {args.captioner}")
    print("=" * 70)
    print()
    
    # Create evaluator
    evaluator = VideoAgentEvaluator(
        model=args.model,
        enabled_tools=enabled_tools,
        max_tool_calls=args.max_tool_calls,
        max_parallel_tools=args.max_parallel_tools,
        initial_frames=args.initial_frames,
        captioner=args.captioner,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        detailed=detailed,
        num_workers=args.num_workers,
        # Frame control parameters
        max_view_frames=args.max_view_frames,
        default_sample_frames=args.default_sample_frames,
        min_sample_frames=args.min_sample_frames,
        max_sample_frames=args.max_sample_frames,
    )
    
    # Run evaluation
    output_dir = evaluator.run(
        annotation_file=args.annotation_file,
        video_dir=args.video_dir,
        video_list_file=args.video_list,
        max_videos=args.max_videos,
        restore_path=args.restore_path,
    )
    
    print()
    print("=" * 70)
    print(f"Results saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()

