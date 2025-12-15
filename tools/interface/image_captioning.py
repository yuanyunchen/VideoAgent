"""
Image Captioning Interfaces

OmniCaptionerCaptioning: Local model-based captioning with configurable detail levels
APICaptioning: API-based captioning using OpenAI vision API
"""

import os
import sys
import base64
import tempfile
import argparse
import hashlib
import json
import logging
import time
from io import BytesIO
from pathlib import Path
from typing import Dict, Any, Optional, Union, List

import numpy as np
from PIL import Image as PILImage

from tools.interface_base import Interface, InterfaceCategory, Image

logger = logging.getLogger(__name__)


class OmniCaptionerCaptioning(Interface):
    """OmniCaptioner for image captioning."""
    
    CATEGORY = InterfaceCategory.TOOL
    FUNCTIONALITY = "Generate captions with configurable detail levels."
    REFERENCE_PAPER = "OmniCaptioner (2024)"
    TOOL_SOURCES = ["OmniCaptioner"]
    
    INPUT_SCHEMA = {
        "image": {"type": "Image", "required": True},
        "detail_level": {"type": "string", "required": False, "default": "short"},
        "with_objects": {"type": "bool", "required": False, "default": False}
    }
    
    OUTPUT_SCHEMA = {
        "caption": {"type": "string"},
        "objects": {"type": "List[str]", "required": False}
    }
    
    # Agent-facing
    AGENT_NAME = "caption_image"
    AGENT_DESCRIPTION = "Generate a description of the current frame. Can optionally detect all visible objects."
    
    AGENT_INPUT_SCHEMA = {
        "detail_level": {
            "type": "string",
            "required": False,
            "default": "short",
            "description": "'short' (1-2 sentences), 'medium' (paragraph), or 'tag' (comma-separated tags)"
        },
        "with_objects": {
            "type": "bool",
            "required": False,
            "default": False,
            "description": "If true, also detect and list all visible objects"
        }
    }
    
    AGENT_OUTPUT_FORMAT = "Text description of the image (+ object list if with_objects=true)"
    
    # Detail level mapping to OmniCaptioner prompt types
    # Using "Natural" prompts for video frames (more appropriate than AIGC prompts)
    DETAIL_LEVEL_MAP = {
        "short": "Short_Natural",
        "medium": "Medium_Natural",
        "tag": "Tag_Natural",  # Structured tags for natural images
    }
    
    VALID_DETAIL_LEVELS = {"short", "medium", "tag"}
    
    # System prompts for natural image captioning (from OmniCaptioner)
    SYSTEM_PROMPTS = {
        "Detailed_Natural": "You are a helpful natural image captioner. Provide a comprehensive description of the natural image, including the main subject, background elements, lighting conditions, color distribution, textures, spatial arrangement, and any potential dynamic context.",
        "Medium_Natural": "You are a helpful natural image captioner. Describe the main content, background in the medium-length text.",
        "Short_Natural": "You are a helpful natural image captioner. Describe the main content, background in the short-length text.",
        # Tag prompts for structured keyword/tag output
        "Tag_Natural": "You are a helpful image tagger. Generate descriptive tags for natural images, capturing key subjects, actions, objects, scene elements, and visual attributes.",
    }
    
    QUESTIONS = {
        "Detailed_Natural": "Describe this image in detail.",
        "Medium_Natural": "Can you describe this image with a medium-length caption?",
        "Short_Natural": "Can you provide a brief caption for this image?",
        "Tag_Natural": "Generate natural English tags describing this image, including subjects, actions, objects, and scene elements. Output as comma-separated tags.",
    }
    
    def __init__(self, model_path: str = "U4R/OmniCaptioner", device: str = None):
        """Initialize OmniCaptionerCaptioning.
        
        Args:
            model_path: Path to OmniCaptioner model (HuggingFace ID or local path)
            device: Target device (e.g., "cuda:0"). If None, uses device_map="auto"
        """
        self.model_path = model_path
        self.device = device  # None means use device_map="auto"
        self.model = None
        self.processor = None
        self._initialized = False
        self._object_detector = None
    
    @classmethod
    def format_output_for_agent(cls, output: Dict[str, Any]) -> str:
        if "error" in output:
            return f"Error: {output['error']}"
        
        result_parts = []
        if "caption" in output:
            result_parts.append(output["caption"])
        
        if "objects" in output and output["objects"]:
            objects = output["objects"]
            result_parts.append(f"\nDetected objects: {', '.join(objects)}")
        
        return "\n".join(result_parts) if result_parts else "Could not generate caption."
    
    def initialize(self) -> None:
        """Load the OmniCaptioner model and processor."""
        if self._initialized:
            return
        
        import torch
        from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
        
        print(f"Loading OmniCaptioner model from {self.model_path}...")
        
        # Determine device_map based on whether explicit device is provided
        # When device is provided, we use explicit device placement for centralized control
        # Otherwise, use "auto" for automatic device mapping
        if self.device is not None:
            device_map = {"": self.device}  # Map all modules to the specified device
            print(f"Using explicit device: {self.device}")
        else:
            device_map = "auto"
            print("Using device_map='auto'")
        
        # Check if we should use local files only (for offline mode)
        import os
        local_only = os.environ.get("HF_HUB_OFFLINE", "0") == "1" or os.environ.get("TRANSFORMERS_OFFLINE", "0") == "1"
        
        # Try flash_attention_2 first, fall back to sdpa if not available
        try:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_path,
                attn_implementation="flash_attention_2",
                low_cpu_mem_usage=True,
                torch_dtype=torch.bfloat16,
                device_map=device_map,
                local_files_only=local_only,
            )
        except ImportError:
            print("Flash Attention 2 not available, using SDPA instead...")
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_path,
                attn_implementation="sdpa",
                low_cpu_mem_usage=True,
                torch_dtype=torch.bfloat16,
                device_map=device_map,
                local_files_only=local_only,
            )
        
        # Use processor from Qwen official repo
        # Reduce max_pixels to limit memory usage (1280*28*28 ~ 1M pixels)
        self.processor = Qwen2VLProcessor.from_pretrained(
            "Qwen/Qwen2-VL-7B-Instruct",
            min_pixels=2*28*28,
            max_pixels=1280*28*28,
            local_files_only=local_only,
        )
        
        self._initialized = True
        print("OmniCaptioner model loaded successfully.")
    
    def _get_prompt_for_level(self, detail_level: str) -> tuple:
        """Get system prompt and question for the given detail level.
        
        Args:
            detail_level: 'short' or 'medium'
        
        Returns:
            Tuple of (system_prompt, question)
        
        Raises:
            ValueError: If detail_level is not valid
        """
        level = detail_level.lower()
        if level not in self.VALID_DETAIL_LEVELS:
            raise ValueError(
                f"Invalid detail_level: '{detail_level}'. "
                f"Must be one of: {', '.join(sorted(self.VALID_DETAIL_LEVELS))}"
            )
        prompt_type = self.DETAIL_LEVEL_MAP[level]
        return self.SYSTEM_PROMPTS[prompt_type], self.QUESTIONS[prompt_type]
    
    def _detect_objects(self, image: Union[Image, np.ndarray, PILImage.Image]) -> List[str]:
        """Detect all objects in the image using YOLOE.
        
        Args:
            image: Image to analyze
            
        Returns:
            List of unique object labels
        """
        if self._object_detector is None:
            from tools.interface.object_detection import YOLOEPromptFreeDetection
            self._object_detector = YOLOEPromptFreeDetection()
            self._object_detector.initialize()
        
        # Prepare image
        if isinstance(image, Image):
            img_obj = image
        elif isinstance(image, np.ndarray):
            img_obj = Image(data=image)
        elif isinstance(image, PILImage.Image):
            img_obj = Image(data=np.array(image))
        else:
            return []
        
        # Detect objects with reasonable threshold
        result = self._object_detector(
            images=[img_obj],
            score_threshold=0.15,  # Higher threshold for cleaner results
            max_detections=50
        )
        
        if "error" in result or "detections" not in result:
            return []
        
        # Get unique labels sorted by frequency
        label_counts = {}
        for det in result["detections"]:
            label = det.get("label", "unknown")
            label_counts[label] = label_counts.get(label, 0) + 1
        
        # Sort by count (descending) and return labels
        sorted_labels = sorted(label_counts.keys(), key=lambda x: -label_counts[x])
        return sorted_labels
    
    def _prepare_image(self, image: Union[Image, np.ndarray, PILImage.Image, str]) -> PILImage.Image:
        """Convert input to PIL Image.
        
        Args:
            image: Image object, numpy array, PIL Image, or file path
        
        Returns:
            PIL Image in RGB format
        """
        if isinstance(image, Image):
            if image.data is not None:
                if isinstance(image.data, np.ndarray):
                    return PILImage.fromarray(image.data.astype(np.uint8))
                elif isinstance(image.data, PILImage.Image):
                    return image.data.convert("RGB")
            elif image.path is not None:
                return PILImage.open(image.path).convert("RGB")
            else:
                raise ValueError("Image object has no data or path")
        elif isinstance(image, np.ndarray):
            return PILImage.fromarray(image.astype(np.uint8))
        elif isinstance(image, PILImage.Image):
            return image.convert("RGB")
        elif isinstance(image, str):
            return PILImage.open(image).convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
    
    def __call__(
        self,
        image: Union[Image, np.ndarray, PILImage.Image, str],
        detail_level: str = "short",
        with_objects: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate a caption for the image.
        
        Args:
            image: Image to caption (Image object, numpy array, PIL Image, or path)
            detail_level: Level of detail ('short' or 'medium')
            with_objects: If True, also detect and list all visible objects
        
        Returns:
            Dict with 'caption' key and optionally 'objects' list
        """
        # Validate detail_level first
        level = detail_level.lower()
        if level not in self.VALID_DETAIL_LEVELS:
            return {
                "error": f"Invalid detail_level: '{detail_level}'. "
                         f"Must be one of: {', '.join(sorted(self.VALID_DETAIL_LEVELS))}"
            }
        
        if not self._initialized:
            self.initialize()
        
        import torch
        import gc
        
        result = {}
        
        try:
            # Prepare image
            pil_image = self._prepare_image(image)
            
            # Get prompts for detail level
            system_prompt, question = self._get_prompt_for_level(detail_level)
            
            # Build messages
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": question},
                    ],
                },
            ]
            
            # Process input
            text = self.processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            inputs = self.processor(
                text=[text], 
                images=[pil_image], 
                padding=True, 
                return_tensors="pt"
            ).to('cuda')
            
            # Generate
            output_ids = self.model.generate(
                **inputs,
                top_p=0.8,
                temperature=0.2,
                do_sample=True,
                max_new_tokens=2048,
                top_k=10,
                repetition_penalty=1.2
            )
            
            # Decode
            generated_ids = [output_ids[len(input_ids):] 
                           for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
            output_text = self.processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=True
            )
            
            result["caption"] = output_text[0] if output_text else ""
            
            # Cleanup
            gc.collect()
            torch.cuda.empty_cache()
            
            # Optionally detect objects
            if with_objects:
                objects = self._detect_objects(image)
                result["objects"] = objects
            
            return result
            
        except Exception as e:
            return {"error": f"Failed to generate caption: {str(e)}"}


class APICaptioning(Interface):
    """API-based image captioning using OpenAI-compatible vision API.
    
    Supports OpenAI, AIML API, and other OpenAI-compatible endpoints.
    Will automatically use AIML API as fallback if OpenAI is not accessible.
    """
    
    CATEGORY = InterfaceCategory.TOOL
    FUNCTIONALITY = "Generate captions using vision-language API (MLLM)."
    REFERENCE_PAPER = "N/A (API)"
    TOOL_SOURCES = ["Vision API"]
    
    INPUT_SCHEMA = {
        "image": {"type": "Image", "required": True},
        "model": {"type": "string", "required": False, "default": "x-ai/grok-4-1-fast-non-reasoning"},
        "detail_level": {"type": "string", "required": False, "default": "medium"},
        "with_objects": {"type": "bool", "required": False, "default": False}
    }
    
    OUTPUT_SCHEMA = {
        "caption": {"type": "string"},
        "objects": {"type": "List[str]", "required": False}
    }
    
    # Agent-facing
    AGENT_NAME = "detailed_captioning"
    AGENT_DESCRIPTION = "Generate detailed image description using API-based MLLM. Can optionally detect all visible objects."
    
    AGENT_INPUT_SCHEMA = {
        "detail_level": {
            "type": "string",
            "required": False,
            "default": "medium",
            "description": "'short' (1-2 sentences) or 'medium' (paragraph)"
        },
        "with_objects": {
            "type": "bool",
            "required": False,
            "default": False,
            "description": "If true, also detect and list all visible objects"
        }
    }
    
    AGENT_OUTPUT_FORMAT = "Text description of the image (+ object list if with_objects=true)"
    
    # Detail level to prompt mapping
    DETAIL_PROMPTS = {
        "short": "Describe this image briefly in 1-2 sentences, focusing on the main subject and action.",
        "medium": "Describe this image in detail, including the main subjects, their actions, and the setting.",
    }
    
    VALID_DETAIL_LEVELS = {"short", "medium"}
    
    def __init__(
        self, 
        api_key: Optional[str] = None, 
        base_url: Optional[str] = None,
        use_aiml: bool = False,
        default_model: str = "x-ai/grok-4-1-fast-non-reasoning"
    ):
        """Initialize APICaptioning.
        
        Args:
            api_key: API key. If None, reads from env vars.
            base_url: Optional base URL for API (for custom endpoints)
            use_aiml: If True, use AIML API instead of OpenAI
            default_model: Default model to use
        """
        self.api_key = api_key
        self.base_url = base_url
        self.use_aiml = use_aiml
        self.default_model = default_model
        self.client = None
        self._initialized = False
        self._object_detector = None
    
    @classmethod
    def format_output_for_agent(cls, output: Dict[str, Any]) -> str:
        if "error" in output:
            return f"Error: {output['error']}"
        
        result_parts = []
        if "caption" in output:
            result_parts.append(output["caption"])
        
        if "objects" in output and output["objects"]:
            objects = output["objects"]
            result_parts.append(f"\nDetected objects: {', '.join(objects)}")
        
        return "\n".join(result_parts) if result_parts else "Could not generate caption."
    
    def initialize(self) -> None:
        """Initialize the OpenAI-compatible client."""
        if self._initialized:
            return
        
        from openai import OpenAI
        from dotenv import load_dotenv
        
        load_dotenv()
        
        # Determine which API to use
        if self.use_aiml or self.base_url:
            # Use AIML API or custom endpoint
            api_key = self.api_key or os.environ.get("AIML_API_KEY")
            base_url = self.base_url or os.environ.get("AIML_BASE_URL", "https://api.aimlapi.com/v1")
            
            if not api_key:
                raise ValueError(
                    "AIML API key not found. Set AIML_API_KEY environment variable "
                    "or pass api_key to constructor."
                )
            
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            # Try OpenAI first
            api_key = self.api_key or os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OpenAI API key not found. Set OPENAI_API_KEY environment variable "
                    "or pass api_key to constructor."
                )
            
            self.client = OpenAI(api_key=api_key)
        
        self._initialized = True
    
    def _encode_image(self, image: Union[Image, np.ndarray, PILImage.Image, str]) -> str:
        """Encode image to base64 string.
        
        Args:
            image: Image to encode
        
        Returns:
            Base64 encoded image string
        """
        if isinstance(image, Image):
            if image.data is not None:
                if isinstance(image.data, np.ndarray):
                    pil_img = PILImage.fromarray(image.data.astype(np.uint8))
                elif isinstance(image.data, PILImage.Image):
                    pil_img = image.data
                else:
                    raise ValueError(f"Unsupported image data type: {type(image.data)}")
            elif image.path is not None:
                pil_img = PILImage.open(image.path)
            else:
                raise ValueError("Image object has no data or path")
        elif isinstance(image, np.ndarray):
            pil_img = PILImage.fromarray(image.astype(np.uint8))
        elif isinstance(image, PILImage.Image):
            pil_img = image
        elif isinstance(image, str):
            pil_img = PILImage.open(image)
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        # Convert to RGB if needed
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        
        # Encode to base64
        buffer = BytesIO()
        pil_img.save(buffer, format="JPEG", quality=95)
        buffer.seek(0)
        
        return base64.b64encode(buffer.read()).decode("utf-8")
    
    def _detect_objects(self, image: Union[Image, np.ndarray, PILImage.Image]) -> List[str]:
        """Detect all objects in the image using YOLOE.
        
        Args:
            image: Image to analyze
            
        Returns:
            List of unique object labels
        """
        if self._object_detector is None:
            from tools.interface.object_detection import YOLOEPromptFreeDetection
            self._object_detector = YOLOEPromptFreeDetection()
            self._object_detector.initialize()
        
        # Prepare image
        if isinstance(image, Image):
            img_obj = image
        elif isinstance(image, np.ndarray):
            img_obj = Image(data=image)
        elif isinstance(image, PILImage.Image):
            img_obj = Image(data=np.array(image))
        else:
            return []
        
        # Detect objects with reasonable threshold
        result = self._object_detector(
            images=[img_obj],
            score_threshold=0.15,  # Higher threshold for cleaner results
            max_detections=50
        )
        
        if "error" in result or "detections" not in result:
            return []
        
        # Get unique labels sorted by frequency
        label_counts = {}
        for det in result["detections"]:
            label = det.get("label", "unknown")
            label_counts[label] = label_counts.get(label, 0) + 1
        
        # Sort by count (descending) and return labels
        sorted_labels = sorted(label_counts.keys(), key=lambda x: -label_counts[x])
        return sorted_labels
    
    def __call__(
        self,
        image: Union[Image, np.ndarray, PILImage.Image, str],
        model: str = None,
        detail_level: str = "medium",
        with_objects: bool = False,
        prompt: str = None,
        max_tokens: int = 1024,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate a caption for the image using OpenAI-compatible API.
        
        Args:
            image: Image to caption
            model: Model to use (default: self.default_model)
            detail_level: 'short' or 'medium'
            with_objects: If True, also detect and list all visible objects
            prompt: Custom prompt (overrides detail_level)
            max_tokens: Maximum tokens in response
        
        Returns:
            Dict with 'caption' key and optionally 'objects' list
        """
        if not self._initialized:
            self.initialize()
        
        result = {}
        
        try:
            # Encode image
            base64_image = self._encode_image(image)
            
            # Use custom prompt or get from detail level
            if prompt is None:
                level = detail_level.lower()
                if level not in self.VALID_DETAIL_LEVELS:
                    return {
                        "error": f"Invalid detail_level: '{detail_level}'. "
                                 f"Must be one of: {', '.join(sorted(self.VALID_DETAIL_LEVELS))}"
                    }
                prompt = self.DETAIL_PROMPTS[level]
            
            # Use provided model or default
            model = model or self.default_model
            
            # Build request
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]
            
            # Call API
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens
            )
            
            result["caption"] = response.choices[0].message.content
            
            # Optionally detect objects
            if with_objects:
                objects = self._detect_objects(image)
                result["objects"] = objects
            
            return result
            
        except Exception as e:
            return {"error": f"API call failed: {str(e)}"}


# ========= OmniCaptioner batch pre-processing (migrated from preprocess_omnicaptioner.py) =========

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OMNICAPTIONER_CACHE_DIR = PROJECT_ROOT / "tools" / "cache" / "omnicaptioner"


def _omni_cache_dir_for_level(base_cache_dir: Union[str, Path], detail_level: str) -> str:
    """Return cache directory for the given detail level."""
    base_cache_dir = Path(base_cache_dir)
    if detail_level == "medium":
        return str(base_cache_dir)
    return str(base_cache_dir / detail_level)


def _omni_video_cache_path(video_path: str, cache_dir: str) -> str:
    """Cache file path for a video."""
    video_hash = hashlib.md5(os.path.abspath(video_path).encode()).hexdigest()[:8]
    video_name = os.path.basename(video_path).split(".")[0]
    return os.path.join(cache_dir, f"{video_name}_{video_hash}.json")


def omnicaptioner_is_cached(video_path: str, cache_dir: str, num_frames: int = 10) -> bool:
    """Check whether a video already has cached captions."""
    cache_path = _omni_video_cache_path(video_path, cache_dir)
    if not os.path.exists(cache_path):
        return False

    try:
        with open(cache_path, "r") as f:
            cache_data = json.load(f)
        captions = cache_data.get("captions", {})
        return len(captions) >= num_frames
    except Exception:
        return False


def omnicaptioner_load_cache(video_path: str, cache_dir: str) -> Optional[Dict[str, Any]]:
    """Load cached captions for a video."""
    cache_path = _omni_video_cache_path(video_path, cache_dir)
    if not os.path.exists(cache_path):
        return None
    try:
        with open(cache_path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _save_captions_to_cache(
    video_path: str,
    cache_dir: str,
    captions: Dict[int, str],
    frame_indices: List[int],
    total_frames: int,
    fps: float,
    detail_level: str = "short",
) -> str:
    """Persist captions to a JSON cache file."""
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = _omni_video_cache_path(video_path, cache_dir)
    cache_data = {
        "video_path": os.path.abspath(video_path),
        "video_name": os.path.basename(video_path),
        "total_frames": total_frames,
        "fps": fps,
        "num_cached_frames": len(captions),
        "frame_indices": frame_indices,
        "detail_level": detail_level,
        "captions": {str(k): v for k, v in captions.items()},
        "cached_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(cache_path, "w") as f:
        json.dump(cache_data, f, indent=2, ensure_ascii=False)
    return cache_path


def _update_master_cache(master_cache_path: str, video_name: str, frame_idx: int, caption: str) -> None:
    """Update aggregated cache JSON in real time."""
    if os.path.exists(master_cache_path):
        with open(master_cache_path, "r") as f:
            master_data = json.load(f)
    else:
        master_data = {}

    if video_name not in master_data:
        master_data[video_name] = {}
    master_data[video_name][str(frame_idx)] = caption

    with open(master_cache_path, "w") as f:
        json.dump(master_data, f, indent=2, ensure_ascii=False)


def _sample_uniform_indices(total_frames: int, num_samples: int) -> List[int]:
    """Sample frame indices uniformly from a video."""
    if num_samples >= total_frames:
        return list(range(total_frames))
    step = (total_frames - 1) / (num_samples - 1) if num_samples > 1 else 0
    indices = []
    for i in range(num_samples):
        idx = int(round(i * step))
        idx = min(idx, total_frames - 1)
        indices.append(idx)
    return sorted(set(indices))


def _extract_frames(video_path: str, frame_indices: List[int]) -> Dict[int, Any]:
    """Extract specific frames from a video as RGB numpy arrays."""
    import cv2  # Local import to avoid hard dependency on import

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    frames = {}
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames[idx] = frame_rgb
    cap.release()
    return frames


def _get_video_info(video_path: str) -> Dict[str, Any]:
    """Retrieve video metadata."""
    import cv2  # Local import to avoid hard dependency on import

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    info = {
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return info


def preprocess_videos_with_omnicaptioner(
    video_dir: str,
    cache_dir: Union[str, Path] = DEFAULT_OMNICAPTIONER_CACHE_DIR,
    num_frames: int = 10,
    detail_level: str = "short",
    limit: int = 0,
    skip_existing: bool = True,
    master_cache_name: str = "all_captions.json",
    model_path: str = "U4R/OmniCaptioner",
    captioner: Optional[OmniCaptionerCaptioning] = None,
) -> Dict[str, Any]:
    """
    Batch pre-process videos with OmniCaptioner to cache frame captions.

    Args:
        video_dir: Directory containing videos
        cache_dir: Base directory for cache files
        num_frames: Number of frames to sample per video
        detail_level: Caption detail level ('short', 'medium', or 'tag')
        limit: Max number of videos to process (0 = all)
        skip_existing: Skip already cached videos if present
        master_cache_name: Name of the aggregated cache JSON
        model_path: Model path for OmniCaptioner
        captioner: Optional pre-initialized OmniCaptionerCaptioning

    Returns:
        Dict with processing statistics
    """
    detail_level = detail_level.lower()
    if detail_level not in OmniCaptionerCaptioning.VALID_DETAIL_LEVELS:
        raise ValueError(f"Invalid detail_level: {detail_level}")

    actual_cache_dir = _omni_cache_dir_for_level(cache_dir, detail_level)

    video_files = sorted(
        [
            os.path.join(video_dir, f)
            for f in os.listdir(video_dir)
            if f.endswith((".mp4", ".avi", ".mov", ".mkv"))
        ]
    )
    if limit > 0:
        video_files = video_files[:limit]

    stats = {
        "total": len(video_files),
        "processed": 0,
        "skipped": 0,
        "failed": 0,
        "failed_videos": [],
    }

    cap_logger = logger.getChild("omnicaptioner_preprocess")
    if not cap_logger.handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    cap_logger.info("OmniCaptioner pre-processing started")
    cap_logger.info("Cache dir: %s", actual_cache_dir)
    cap_logger.info("Frames per video: %s", num_frames)
    cap_logger.info("Detail level: %s", detail_level)

    captioner = captioner or OmniCaptionerCaptioning(model_path=model_path)
    captioner.initialize()

    os.makedirs(actual_cache_dir, exist_ok=True)
    master_cache_path = os.path.join(actual_cache_dir, master_cache_name)

    for i, video_path in enumerate(video_files):
        video_name = os.path.basename(video_path)
        video_id = video_name.split(".")[0]

        if skip_existing and omnicaptioner_is_cached(video_path, actual_cache_dir, num_frames):
            cap_logger.info("[%d/%d] Skipping (cached): %s", i + 1, len(video_files), video_name)
            stats["skipped"] += 1
            continue

        cap_logger.info("[%d/%d] Processing: %s", i + 1, len(video_files), video_name)

        try:
            start_time = time.time()
            video_info = _get_video_info(video_path)
            total_frames = video_info["total_frames"]
            fps = video_info["fps"]
            frame_indices = _sample_uniform_indices(total_frames, num_frames)
            cap_logger.info("  Sampling %d frames: %s", len(frame_indices), frame_indices)

            frames = _extract_frames(video_path, frame_indices)
            cap_logger.info("  Extracted %d frames", len(frames))

            captions: Dict[int, str] = {}
            for j, (idx, frame_data) in enumerate(frames.items()):
                image = Image(data=frame_data)
                result = captioner(image=image, detail_level=detail_level, with_objects=False)
                if "error" in result:
                    cap_logger.warning("  Frame %s caption error: %s", idx, result["error"])
                    continue
                caption_text = result.get("caption", "")
                captions[idx] = caption_text
                _update_master_cache(master_cache_path, video_id, idx, caption_text)
                cap_logger.info("  [%d/%d] Frame %s captioned and saved", j + 1, len(frames), idx)

            _save_captions_to_cache(
                video_path=video_path,
                cache_dir=actual_cache_dir,
                captions=captions,
                frame_indices=frame_indices,
                total_frames=total_frames,
                fps=fps,
                detail_level=detail_level,
            )

            elapsed = time.time() - start_time
            cap_logger.info("  Completed in %.1fs, cached %d captions", elapsed, len(captions))
            stats["processed"] += 1

            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                import gc
                gc.collect()
            except Exception:
                pass

        except Exception as e:
            import traceback
            cap_logger.error("  Failed: %s", e)
            traceback.print_exc()
            stats["failed"] += 1
            stats["failed_videos"].append(video_name)

    return stats


def test_omnicaptioner_cache(video_dir: str, cache_dir: Union[str, Path], num_tests: int = 3) -> bool:
    """Quickly verify that OmniCaptioner caches load correctly."""
    cap_logger = logger.getChild("omnicaptioner_test")
    if not cap_logger.handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    cap_logger.info("=" * 60)
    cap_logger.info("Testing OmniCaptioner Cache Loading")
    cap_logger.info("=" * 60)

    video_files = sorted(
        [
            os.path.join(video_dir, f)
            for f in os.listdir(video_dir)
            if f.endswith((".mp4", ".avi", ".mov", ".mkv"))
        ]
    )
    cached_videos = [v for v in video_files if omnicaptioner_is_cached(v, cache_dir)]
    if not cached_videos:
        cap_logger.error("No cached videos found!")
        return False

    cap_logger.info("Found %d cached videos", len(cached_videos))
    test_videos = cached_videos[:num_tests]
    all_passed = True

    for video_path in test_videos:
        video_name = os.path.basename(video_path)
        cap_logger.info("Testing: %s", video_name)
        try:
            start_time = time.time()
            cache_data = omnicaptioner_load_cache(video_path, cache_dir)
            load_time = time.time() - start_time
            if cache_data is None:
                cap_logger.error("  [FAIL] Could not load cache")
                all_passed = False
                continue
            captions = cache_data.get("captions", {})
            frame_indices = cache_data.get("frame_indices", [])
            cap_logger.info("  [PASS] Loaded from cache in %.4fs", load_time)
            cap_logger.info("  Cached frames: %d", len(captions))
            cap_logger.info("  Frame indices: %s...", frame_indices[:5])
            if captions:
                first_key = list(captions.keys())[0]
                sample = str(captions[first_key])[:100]
                cap_logger.info("  Sample caption: %s...", sample)
        except Exception as e:
            cap_logger.error("  [FAIL] Error: %s", e)
            all_passed = False

    return all_passed


def _omnicaptioner_cli() -> int:
    """CLI entry point for OmniCaptioner pre-processing."""
    parser = argparse.ArgumentParser(description="Pre-process videos with OmniCaptioner")
    parser.add_argument("--video_dir", type=str, required=True, help="Directory containing videos")
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=str(DEFAULT_OMNICAPTIONER_CACHE_DIR),
        help="Directory for cache files",
    )
    parser.add_argument("--num_frames", type=int, default=5, help="Number of frames to sample per video")
    parser.add_argument(
        "--detail_level",
        type=str,
        default="short",
        choices=["short", "medium", "tag"],
        help="Caption detail level ('short', 'medium', or 'tag' for natural English tags)",
    )
    parser.add_argument("--limit", type=int, default=0, help="Max videos to process (0 = all)")
    parser.add_argument("--skip_existing", action="store_true", default=True, help="Skip already cached videos")
    parser.add_argument("--test_only", action="store_true", help="Only test cache loading")
    parser.add_argument("--num_tests", type=int, default=3, help="Number of videos to test")
    parser.add_argument(
        "--model_path",
        type=str,
        default="U4R/OmniCaptioner",
        help="Model path or HF repo id for OmniCaptioner",
    )
    args = parser.parse_args()

    if not os.path.exists(args.video_dir):
        logger.error("Video directory not found: %s", args.video_dir)
        return 1

    actual_cache_dir = _omni_cache_dir_for_level(args.cache_dir, args.detail_level)

    if args.test_only:
        success = test_omnicaptioner_cache(args.video_dir, actual_cache_dir, args.num_tests)
        return 0 if success else 1

    stats = preprocess_videos_with_omnicaptioner(
        video_dir=args.video_dir,
        cache_dir=args.cache_dir,
        num_frames=args.num_frames,
        detail_level=args.detail_level,
        limit=args.limit,
        skip_existing=args.skip_existing,
        master_cache_name="all_captions.json",
        model_path=args.model_path,
    )

    logger.info("=" * 60)
    logger.info("Pre-processing Summary")
    logger.info("=" * 60)
    logger.info("Total videos: %d", stats["total"])
    logger.info("Processed: %d", stats["processed"])
    logger.info("Skipped (cached): %d", stats["skipped"])
    logger.info("Failed: %d", stats["failed"])
    if stats["failed_videos"]:
        logger.info("Failed videos:")
        for v in stats["failed_videos"]:
            logger.info("  - %s", v)

    if stats["processed"] > 0 or stats["skipped"] > 0:
        success = test_omnicaptioner_cache(args.video_dir, actual_cache_dir, args.num_tests)
        if success:
            logger.info("[SUCCESS] All cache loading tests passed!")
        else:
            logger.error("[FAILED] Some cache loading tests failed!")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_omnicaptioner_cli())
