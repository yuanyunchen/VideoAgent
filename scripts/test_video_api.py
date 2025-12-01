#!/usr/bin/env python3
"""
Test script for video input to various LLM APIs.
Tests if models support direct video input vs frame-based input.

Usage:
    python scripts/test_video_api.py [--model MODEL] [--video VIDEO_PATH]

Examples:
    python scripts/test_video_api.py
    python scripts/test_video_api.py --model gpt-4o
    python scripts/test_video_api.py --model gemini --video path/to/video.mp4
"""

import os
import sys
import argparse
import base64
import cv2
import tempfile
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()


def extract_video_clip(video_path: str, start_sec: float = 0, duration: float = 3.0) -> str:
    """
    Extract a clip from video using OpenCV.
    Returns path to the extracted clip.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    start_frame = int(start_sec * fps)
    end_frame = int((start_sec + duration) * fps)
    
    # Create temporary output file
    output_path = tempfile.mktemp(suffix='.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    for _ in range(end_frame - start_frame):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
    
    cap.release()
    out.release()
    
    print(f"Extracted {duration}s clip to: {output_path}")
    print(f"  FPS: {fps}, Resolution: {width}x{height}")
    print(f"  Frames: {end_frame - start_frame}")
    
    return output_path


def extract_frames(video_path: str, num_frames: int = 5) -> list:
    """
    Extract evenly spaced frames from video.
    Returns list of numpy arrays (BGR images).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = [int(i * total_frames / (num_frames + 1)) for i in range(1, num_frames + 1)]
    
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    
    cap.release()
    print(f"Extracted {len(frames)} frames from video")
    return frames


def image_to_base64(image) -> str:
    """Convert numpy image array to base64 string."""
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer.tobytes()).decode('utf-8')


def video_to_base64(video_path: str) -> str:
    """Convert video file to base64 string."""
    with open(video_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def test_openai_video_direct(client: OpenAI, video_path: str, model: str = "gpt-4o-mini"):
    """
    Test 1: Try direct video input (likely will fail for most models).
    """
    print(f"\n{'='*60}")
    print(f"TEST 1: Direct Video Input - {model}")
    print(f"{'='*60}")
    
    video_base64 = video_to_base64(video_path)
    file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
    print(f"Video size: {file_size_mb:.2f} MB")
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe what happens in this video clip in detail."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:video/mp4;base64,{video_base64}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=500
        )
        print(f"SUCCESS! Model response:")
        print(response.choices[0].message.content)
        return True
    except Exception as e:
        print(f"FAILED: {type(e).__name__}")
        print(f"Error: {str(e)[:500]}")
        return False


def test_openai_frames(client: OpenAI, frames: list, model: str = "gpt-4o-mini"):
    """
    Test 2: Send multiple frames as images.
    """
    print(f"\n{'='*60}")
    print(f"TEST 2: Frame-based Input ({len(frames)} frames) - {model}")
    print(f"{'='*60}")
    
    content = [{"type": "text", "text": "These are frames from a 3-second video clip. Describe what happens in this video in detail."}]
    
    for i, frame in enumerate(frames):
        base64_img = image_to_base64(frame)
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_img}"
            }
        })
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            max_tokens=500
        )
        print(f"SUCCESS! Model response:")
        print(response.choices[0].message.content)
        return True
    except Exception as e:
        print(f"FAILED: {type(e).__name__}")
        print(f"Error: {str(e)[:500]}")
        return False


def test_gemini_video(api_key: str, video_path: str):
    """
    Test 3: Try Gemini API with video input.
    Gemini has native video understanding capability.
    """
    print(f"\n{'='*60}")
    print(f"TEST 3: Gemini Video Input")
    print(f"{'='*60}")
    
    try:
        import google.generativeai as genai
        
        genai.configure(api_key=api_key)
        
        # Upload video file
        print("Uploading video to Gemini...")
        video_file = genai.upload_file(video_path)
        print(f"Uploaded: {video_file.uri}")
        
        # Wait for processing
        import time
        while video_file.state.name == "PROCESSING":
            print("Processing video...")
            time.sleep(2)
            video_file = genai.get_file(video_file.name)
        
        if video_file.state.name == "FAILED":
            raise ValueError(f"Video processing failed: {video_file.state.name}")
        
        # Generate content
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content([
            "Describe what happens in this video clip in detail.",
            video_file
        ])
        
        print(f"SUCCESS! Model response:")
        print(response.text)
        
        # Clean up
        genai.delete_file(video_file.name)
        return True
        
    except ImportError:
        print("SKIPPED: google-generativeai not installed")
        print("Install with: pip install google-generativeai")
        return None
    except Exception as e:
        print(f"FAILED: {type(e).__name__}")
        print(f"Error: {str(e)[:500]}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test video input for various LLM APIs")
    parser.add_argument("--model", type=str, default="all",
                       help="Model to test: gpt-4o-mini, gpt-4o, gemini, or all")
    parser.add_argument("--video", type=str, default=None,
                       help="Path to video file (optional)")
    parser.add_argument("--frames", type=int, default=5,
                       help="Number of frames to extract (default: 5)")
    parser.add_argument("--duration", type=float, default=3.0,
                       help="Clip duration in seconds (default: 3)")
    args = parser.parse_args()
    
    print("=" * 60)
    print("VIDEO API INPUT TEST")
    print("=" * 60)
    
    # Get API keys
    openai_api_key = os.environ.get('OPENAI_API_KEY') or os.environ.get('AIML_API_KEY')
    openai_base_url = os.environ.get('OPENAI_BASE_URL') or os.environ.get('AIML_BASE_URL')
    gemini_api_key = os.environ.get('GOOGLE_API_KEY') or os.environ.get('GEMINI_API_KEY')
    
    if not openai_api_key and args.model in ['all', 'gpt-4o-mini', 'gpt-4o']:
        print("ERROR: No API key found. Set OPENAI_API_KEY or AIML_API_KEY")
        sys.exit(1)
    
    if openai_api_key:
        print(f"OpenAI API Key: {'*' * 20}...{openai_api_key[-4:]}")
    if openai_base_url:
        print(f"OpenAI Base URL: {openai_base_url}")
    if gemini_api_key:
        print(f"Gemini API Key: {'*' * 20}...{gemini_api_key[-4:]}")
    
    # Find a test video
    if args.video:
        source_video = args.video
    else:
        video_dir = project_root / "data" / "videos"
        test_videos = list(video_dir.glob("*.mp4"))[:1]
        
        if not test_videos:
            video_dir = project_root / "dataset" / "videos"
            test_videos = list(video_dir.glob("*.mp4"))[:1]
        
        if not test_videos:
            print("ERROR: No test videos found in data/videos or dataset/videos")
            sys.exit(1)
        
        source_video = str(test_videos[0])
    
    print(f"\nSource video: {source_video}")
    
    # Extract clip
    clip_path = extract_video_clip(source_video, start_sec=5, duration=args.duration)
    
    # Extract frames for frame-based test
    frames = extract_frames(clip_path, num_frames=args.frames)
    
    results = {}
    
    # Initialize OpenAI client if needed
    client = None
    if openai_api_key and args.model in ['all', 'gpt-4o-mini', 'gpt-4o']:
        client_kwargs = {"api_key": openai_api_key}
        if openai_base_url:
            client_kwargs["base_url"] = openai_base_url
        client = OpenAI(**client_kwargs)
    
    # Run tests based on model selection
    if args.model in ['all', 'gpt-4o-mini']:
        # Test 1: Direct video input with GPT-4o-mini
        results['gpt-4o-mini_video'] = test_openai_video_direct(client, clip_path, "gpt-4o-mini")
        
        # Test 2: Frame-based input with GPT-4o-mini
        results['gpt-4o-mini_frames'] = test_openai_frames(client, frames, "gpt-4o-mini")
    
    if args.model in ['all', 'gpt-4o']:
        # Test 3: GPT-4o
        print(f"\n{'='*60}")
        print("TEST: GPT-4o Frame-based Input")
        print(f"{'='*60}")
        try:
            results['gpt-4o_frames'] = test_openai_frames(client, frames, "gpt-4o")
        except Exception as e:
            print(f"GPT-4o not available: {e}")
            results['gpt-4o_frames'] = None
    
    if args.model in ['all', 'gemini']:
        # Test 4: Gemini
        if gemini_api_key:
            results['gemini_video'] = test_gemini_video(gemini_api_key, clip_path)
        else:
            print(f"\n{'='*60}")
            print("TEST: Gemini - SKIPPED (no GOOGLE_API_KEY)")
            print("Set GOOGLE_API_KEY env var to test Gemini native video input")
            print(f"{'='*60}")
            results['gemini_video'] = None
    
    # Cleanup
    try:
        os.remove(clip_path)
        print(f"\nCleaned up temp file: {clip_path}")
    except:
        pass
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for test_name, result in results.items():
        status = "PASS" if result else ("SKIP" if result is None else "FAIL")
        print(f"  {test_name}: {status}")
    
    print("\nConclusion:")
    if results.get('gpt-4o-mini_video'):
        print("  - GPT-4o-mini supports direct video input!")
    elif 'gpt-4o-mini_video' in results:
        print("  - GPT-4o-mini does NOT support direct video input")
        if results.get('gpt-4o-mini_frames'):
            print("  - Use frame-based approach instead (works!)")
    
    if results.get('gemini_video'):
        print("  - Gemini supports native video input!")
    
    print("\nNote: For video understanding, use one of these approaches:")
    print("  1. Frame extraction: Extract N frames, send as images (works with all vision models)")
    print("  2. Gemini native: Use google-generativeai with video file upload")


if __name__ == "__main__":
    main()

