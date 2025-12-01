#!/usr/bin/env python3
"""
Test script to verify if a model supports vision/multimodal input.
Usage: python scripts/test_model_vision.py [model_name]
"""

import os
import sys
import base64
from pathlib import Path

# Load environment variables from .env
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ[key] = value.strip('"').strip("'")

from openai import OpenAI

def create_test_image():
    """Create a simple 100x100 red PNG for testing."""
    import io
    try:
        from PIL import Image
        # Create a 100x100 red image
        img = Image.new('RGB', (100, 100), color='red')
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode('utf-8')
    except ImportError:
        # Fallback: use a pre-generated 10x10 red PNG
        # This is a valid 10x10 red PNG image
        red_10x10_png = (
            "iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAIAAAACUFjqAAAADklEQVQY"
            "02P4z4APMFAXAHkpB/zy2g8GAAAAAElFTkSuQmCC"
        )
        return red_10x10_png

def test_model_text_only(client, model_name):
    """Test if model works with text-only input."""
    print(f"\n[TEST 1] Text-only request to {model_name}...")
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": "Say 'Hello' and nothing else."}
            ],
            max_tokens=50
        )
        result = response.choices[0].message.content
        print(f"  Response: {result[:100]}")
        print("  [PASS] Text-only works")
        return True
    except Exception as e:
        print(f"  [FAIL] Error: {e}")
        return False

def test_model_vision(client, model_name):
    """Test if model works with vision/image input."""
    print(f"\n[TEST 2] Vision/Image request to {model_name}...")
    
    test_image_b64 = create_test_image()
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What color is this image? Reply with just the color name."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{test_image_b64}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=50
        )
        result = response.choices[0].message.content
        print(f"  Response: {result[:100]}")
        print("  [PASS] Vision/multimodal works!")
        return True
    except Exception as e:
        error_str = str(e)
        print(f"  [FAIL] Error: {error_str[:200]}")
        if "not support" in error_str.lower() or "multimodal" in error_str.lower():
            print("  -> Model does NOT support vision/images")
        return False

def main():
    # Default model to test
    model_name = "x-ai/grok-4-1-fast-non-reasoning"
    
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    
    api_key = os.environ.get("AIML_API_KEY")
    base_url = os.environ.get("AIML_BASE_URL", "https://api.aimlapi.com/v1")
    
    if not api_key:
        print("[ERROR] AIML_API_KEY not set")
        sys.exit(1)
    
    print("=" * 60)
    print(f"Testing Model: {model_name}")
    print("=" * 60)
    print(f"API Base URL: {base_url}")
    print(f"API Key: {api_key[:8]}...{api_key[-4:]}")
    
    client = OpenAI(api_key=api_key, base_url=base_url)
    
    # Run tests
    text_ok = test_model_text_only(client, model_name)
    vision_ok = test_model_vision(client, model_name)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Text-only:  {'SUPPORTED' if text_ok else 'FAILED'}")
    print(f"Vision:     {'SUPPORTED (Multimodal)' if vision_ok else 'NOT SUPPORTED (Text-only)'}")
    
    if vision_ok:
        print(f"\n-> {model_name} can be used as VIEWER_MODEL")
    else:
        print(f"\n-> {model_name} can ONLY be used as SCHEDULER_MODEL")
    
    print("=" * 60)

if __name__ == "__main__":
    main()

