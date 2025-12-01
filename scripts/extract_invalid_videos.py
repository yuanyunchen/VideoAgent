#!/usr/bin/env python3
"""
Extract invalid video directories from experiment results.
"""

import os
import json
import shutil
import argparse
from pathlib import Path

def extract_invalid_videos(input_dir, output_dir):
    """
    Extract invalid video directories from experiment results.
    
    Args:
        input_dir: Path to the experiment results directory
        output_dir: Path to the output directory where invalid videos will be copied
    
    Returns:
        List of invalid video IDs
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Check if input directory exists
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    
    # Check for result.json file
    result_file = input_path / "result.json"
    if not result_file.exists():
        raise FileNotFoundError(f"result.json not found in {input_dir}")
    
    # Read the result.json file
    with open(result_file, 'r') as f:
        results = json.load(f)
    
    # Find invalid videos (final_answer = -1)
    invalid_videos = []
    for video_id, video_result in results.items():
        if video_result.get("final_answer") == -1:
            invalid_videos.append(video_id)
    
    print(f"Found {len(invalid_videos)} invalid videos:")
    for video_id in invalid_videos:
        print(f"  - {video_id}")
    
    if not invalid_videos:
        print("No invalid videos found.")
        return invalid_videos
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Copy invalid video directories
    videos_dir = input_path / "videos"
    if not videos_dir.exists():
        raise FileNotFoundError(f"Videos directory not found: {videos_dir}")
    
    # Copy experiment metadata
    metadata_files = ["accuracy.txt", "experiment_config.yaml", "result.json"]
    for metadata_file in metadata_files:
        src_file = input_path / metadata_file
        if src_file.exists():
            dst_file = output_path / metadata_file
            shutil.copy2(src_file, dst_file)
            print(f"Copied metadata: {metadata_file}")
    
    # Create a filtered result.json with only invalid videos
    filtered_results = {vid: results[vid] for vid in invalid_videos}
    with open(output_path / "filtered_result.json", 'w') as f:
        json.dump(filtered_results, f, indent=2)
    
    # Copy invalid video directories
    invalid_videos_dir = output_path / "videos"
    invalid_videos_dir.mkdir(exist_ok=True)
    
    copied_count = 0
    for video_id in invalid_videos:
        src_video_dir = videos_dir / video_id
        dst_video_dir = invalid_videos_dir / video_id
        
        if src_video_dir.exists():
            if dst_video_dir.exists():
                shutil.rmtree(dst_video_dir)
            shutil.copytree(src_video_dir, dst_video_dir)
            copied_count += 1
            print(f"Copied video directory: {video_id}")
        else:
            print(f"Warning: Video directory not found: {video_id}")
    
    print(f"\nSummary:")
    print(f"  Total invalid videos: {len(invalid_videos)}")
    print(f"  Successfully copied: {copied_count}")
    print(f"  Output directory: {output_path}")
    
    # Create a summary file
    summary = {
        "input_directory": str(input_path),
        "output_directory": str(output_path),
        "total_invalid_videos": len(invalid_videos),
        "successfully_copied": copied_count,
        "invalid_video_ids": invalid_videos,
        "extraction_timestamp": str(Path().absolute())
    }
    
    with open(output_path / "extraction_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    return invalid_videos

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Extract invalid video directories from experiment results"
    )
    parser.add_argument(
        "input_dir",
        help="Path to the experiment results directory"
    )
    parser.add_argument(
        "output_dir",
        help="Path to the output directory where invalid videos will be copied"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only show what would be extracted without copying"
    )
    
    args = parser.parse_args()
    
    try:
        if args.dry_run:
            print("DRY RUN MODE - No files will be copied")
            
        invalid_videos = extract_invalid_videos(args.input_dir, args.output_dir)
        
        if invalid_videos:
            print(f"\nExtraction completed successfully!")
        else:
            print(f"\nNo invalid videos found to extract.")
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 
    
"""
python extract_invalid_videos.py "results/first_test_after_reconstruction__gpt-4o-mini-2024-07-18_viewer_gpt-4o-mini-2024-07-18_numbers_100" "results/valid_video_cases/first_test_after_reconstruction"
"""