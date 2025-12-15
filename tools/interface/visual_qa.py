"""
Visual Q&A Interfaces

Two approaches:
- GeneralVQA: Use MLLM API for general visual questions (same model as scheduler)
- TargetingVQA: Use VStar for fine-grained visual problems (small objects, precise localization)
"""

import os
import sys
import base64
from io import BytesIO
from typing import Dict, Any, Optional, Union, List

import numpy as np
from PIL import Image as PILImage

from tools.interface_base import Interface, InterfaceCategory, Image


class GeneralVQA(Interface):
    """API-based Visual Q&A for general visual questions.
    
    Uses OpenAI-compatible vision API to answer questions about one or more frames.
    """
    
    CATEGORY = InterfaceCategory.SUB_QUESTION_ANSWERING
    FUNCTIONALITY = "Answer general visual questions using MLLM API."
    REFERENCE_PAPER = "N/A (API)"
    TOOL_SOURCES = ["Vision API"]
    
    INPUT_SCHEMA = {
        "query": {"type": "string", "required": True},
        "images": {"type": "List[Image]", "required": True},
        "model": {"type": "string", "required": False, "default": None}  # None = use scheduler model
    }
    
    OUTPUT_SCHEMA = {
        "answer": {"type": "string"}
    }
    
    # Agent-facing
    AGENT_NAME = "general_vqa"
    AGENT_DESCRIPTION = "Answer general visual questions about scenes, actions, attributes, and relationships."
    
    AGENT_INPUT_SCHEMA = {
        "query": {
            "type": "string",
            "required": True,
            "description": "Visual question about the current frame(s)"
        }
    }
    
    AGENT_OUTPUT_FORMAT = "Text answer based on visual analysis"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        use_aiml: bool = False,
        default_model: str = "gpt-4o-mini"
    ):
        """Initialize GeneralVQA.
        
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
    
    @classmethod
    def format_output_for_agent(cls, output: Dict[str, Any]) -> str:
        if "error" in output:
            return f"Error: {output['error']}"
        if "answer" in output:
            return output["answer"]
        return "Could not get answer."
    
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
                # Fallback to AIML API
                api_key = os.environ.get("AIML_API_KEY")
                base_url = os.environ.get("AIML_BASE_URL", "https://api.aimlapi.com/v1")
                if api_key:
                    self.client = OpenAI(api_key=api_key, base_url=base_url)
                else:
                    raise ValueError(
                        "No API key found. Set OPENAI_API_KEY or AIML_API_KEY environment variable."
                    )
            else:
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
    
    def __call__(
        self,
        query: str,
        images: List[Union[Image, np.ndarray, PILImage.Image, str]],
        model: str = None,
        max_tokens: int = 1024,
        **kwargs
    ) -> Dict[str, Any]:
        """Answer a visual question about one or more images.
        
        Args:
            query: The question to answer
            images: List of images to analyze
            model: Model to use (default: self.default_model)
            max_tokens: Maximum tokens in response
        
        Returns:
            Dict with 'answer' key
        """
        if not self._initialized:
            self.initialize()
        
        if not query:
            return {"error": "Query is required"}
        
        if not images or len(images) == 0:
            return {"error": "At least one image is required"}
        
        try:
            # Build content with images
            content = []
            
            for i, image in enumerate(images):
                base64_image = self._encode_image(image)
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                })
            
            # Add the question
            if len(images) > 1:
                question_text = f"I'm showing you {len(images)} frames from a video. {query}"
            else:
                question_text = query
            
            content.append({
                "type": "text",
                "text": question_text
            })
            
            # Use provided model or default
            model = model or self.default_model
            
            # Build request
            messages = [
                {
                    "role": "user",
                    "content": content
                }
            ]
            
            # Call API
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens
            )
            
            return {"answer": response.choices[0].message.content}
            
        except Exception as e:
            return {"error": f"API call failed: {str(e)}"}


class TargetingVQA(Interface):
    """VStar-based Visual Q&A for fine-grained visual problems.
    
    Uses guided visual search to locate small or hard-to-see objects,
    then answers questions about them.
    """
    
    CATEGORY = InterfaceCategory.SUB_QUESTION_ANSWERING
    FUNCTIONALITY = "Answer fine-grained visual questions using guided visual search."
    REFERENCE_PAPER = "V* (2024)"
    TOOL_SOURCES = ["VStar"]
    
    INPUT_SCHEMA = {
        "query": {"type": "string", "required": True},
        "images": {"type": "List[Image]", "required": True},
        "target_object": {"type": "string", "required": False, "default": None}
    }
    
    OUTPUT_SCHEMA = {
        "answer": {"type": "string"},
        "search_result": {"type": "object", "required": False}
    }
    
    # Agent-facing
    AGENT_NAME = "targeting_vqa"
    AGENT_DESCRIPTION = "Answer fine-grained visual questions - small objects, precise details, hard-to-see elements."
    
    AGENT_INPUT_SCHEMA = {
        "query": {
            "type": "string",
            "required": True,
            "description": "Question about specific/small objects or fine details"
        },
        "target_object": {
            "type": "string",
            "required": False,
            "description": "Optional: specific object to search for and focus on"
        }
    }
    
    AGENT_OUTPUT_FORMAT = "Text answer with visual search result"
    
    def __init__(
        self,
        vsm_model_path: str = "craigwu/seal_vsm_7b",
        vqa_model_path: str = "craigwu/seal_vqa_7b",
        confidence_high: float = 0.5,
        confidence_low: float = 0.3,
        minimum_size: int = 224,
        minimum_size_scale: float = 4.0
    ):
        """Initialize TargetingVQA.
        
        Args:
            vsm_model_path: Path to VSM model
            vqa_model_path: Path to VQA model
            confidence_high: High confidence threshold for detection
            confidence_low: Low confidence threshold for detection
            minimum_size: Minimum patch size for visual search
            minimum_size_scale: Scale factor for minimum size calculation
        """
        self.vsm_model_path = vsm_model_path
        self.vqa_model_path = vqa_model_path
        self.confidence_high = confidence_high
        self.confidence_low = confidence_low
        self.minimum_size = minimum_size
        self.minimum_size_scale = minimum_size_scale
        
        self._vsm = None
        self._vqa_llm = None
        self._initialized = False
    
    @classmethod
    def format_output_for_agent(cls, output: Dict[str, Any]) -> str:
        if "error" in output:
            return f"Error: {output['error']}"
        
        result_parts = []
        if "answer" in output:
            result_parts.append(output["answer"])
        
        if "search_result" in output and output["search_result"]:
            sr = output["search_result"]
            if sr.get("success"):
                result_parts.append(f"\n[Visual search: found target in {sr.get('path_length', 0)} steps]")
        
        return "\n".join(result_parts) if result_parts else "Could not locate or answer."
    
    def initialize(self) -> None:
        """Initialize VStar models (VSM and VQA_LLM)."""
        if self._initialized:
            return
        
        # Add vstar to path
        vstar_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "models", "vstar"
        )
        if vstar_path not in sys.path:
            sys.path.insert(0, vstar_path)
        
        # Import VStar components
        from visual_search import parse_args, VSM
        from vstar_bench_eval import VQA_LLM, expand2square
        
        # Store expand2square for later use
        self._expand2square = expand2square
        
        # Initialize VSM
        print(f"Loading VSM model from {self.vsm_model_path}...")
        vsm_args = parse_args([])
        vsm_args.version = self.vsm_model_path
        self._vsm = VSM(vsm_args)
        print("VSM model loaded successfully.")
        
        # Initialize VQA_LLM
        print(f"Loading VQA model from {self.vqa_model_path}...")
        
        class VQAArgs:
            vqa_model_path = self.vqa_model_path
            vqa_model_base = None
            conv_type = "v1"
        
        self._vqa_llm = VQA_LLM(VQAArgs())
        print("VQA model loaded successfully.")
        
        self._initialized = True
    
    def _prepare_image(self, image: Union[Image, np.ndarray, PILImage.Image, str]) -> PILImage.Image:
        """Convert input to PIL Image.
        
        Args:
            image: Image to prepare
        
        Returns:
            PIL Image in RGB format
        """
        if isinstance(image, Image):
            if image.data is not None:
                if isinstance(image.data, np.ndarray):
                    return PILImage.fromarray(image.data.astype(np.uint8)).convert("RGB")
                elif isinstance(image.data, PILImage.Image):
                    return image.data.convert("RGB")
            elif image.path is not None:
                return PILImage.open(image.path).convert("RGB")
            else:
                raise ValueError("Image object has no data or path")
        elif isinstance(image, np.ndarray):
            return PILImage.fromarray(image.astype(np.uint8)).convert("RGB")
        elif isinstance(image, PILImage.Image):
            return image.convert("RGB")
        elif isinstance(image, str):
            return PILImage.open(image).convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
    
    def _extract_target_object(self, query: str) -> Optional[str]:
        """Extract target object from query using simple heuristics.
        
        Args:
            query: The question
        
        Returns:
            Extracted target object or None
        """
        import re
        
        # Common patterns for object references
        patterns = [
            r"what (?:is|are) the (.+?)(?:\?|$)",
            r"where (?:is|are) the (.+?)(?:\?|$)",
            r"find (?:the )?(.+?)(?:\?|$)",
            r"locate (?:the )?(.+?)(?:\?|$)",
            r"look for (?:the )?(.+?)(?:\?|$)",
            r"(?:the|a|an) (.+?) (?:is|are|in|on|at)",
        ]
        
        query_lower = query.lower()
        for pattern in patterns:
            match = re.search(pattern, query_lower)
            if match:
                target = match.group(1).strip()
                # Clean up common suffixes
                target = re.sub(r"\s+(doing|holding|wearing|in the image|in this image).*", "", target)
                if target and len(target) < 50:  # Sanity check
                    return target
        
        return None
    
    def __call__(
        self,
        query: str,
        images: List[Union[Image, np.ndarray, PILImage.Image, str]],
        target_object: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Answer a fine-grained visual question using visual search.
        
        Args:
            query: The question to answer
            images: List of images to analyze (uses first image for search)
            target_object: Optional specific object to search for
        
        Returns:
            Dict with 'answer' and optionally 'search_result' keys
        """
        if not self._initialized:
            self.initialize()
        
        if not query:
            return {"error": "Query is required"}
        
        if not images or len(images) == 0:
            return {"error": "At least one image is required"}
        
        # Import visual_search function
        vstar_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "models", "vstar"
        )
        if vstar_path not in sys.path:
            sys.path.insert(0, vstar_path)
        from visual_search import visual_search
        
        try:
            # Prepare the first image
            pil_image = self._prepare_image(images[0])
            
            # Determine target object
            if target_object is None:
                target_object = self._extract_target_object(query)
            
            search_result = None
            object_crops = None
            
            # Perform visual search if we have a target object
            if target_object:
                smallest_size = max(
                    int(np.ceil(min(pil_image.width, pil_image.height) / self.minimum_size_scale)),
                    self.minimum_size
                )
                
                final_step, path_length, search_successful, all_valid_boxes = visual_search(
                    self._vsm,
                    pil_image,
                    target_object,
                    target_bbox=None,
                    smallest_size=smallest_size,
                    confidence_high=self.confidence_high,
                    confidence_low=self.confidence_low
                )
                
                search_result = {
                    "success": search_successful,
                    "path_length": path_length,
                    "target_object": target_object
                }
                
                # If search successful, extract object crop for VQA
                if search_successful and final_step is not None:
                    bbox = final_step.get('detection_result')
                    if bbox is not None:
                        search_patch = final_step.get('bbox', [0, 0, 0, 0])
                        # Convert to absolute coordinates
                        abs_bbox = [
                            bbox[0].item() + search_patch[0] if hasattr(bbox[0], 'item') else bbox[0] + search_patch[0],
                            bbox[1].item() + search_patch[1] if hasattr(bbox[1], 'item') else bbox[1] + search_patch[1],
                            bbox[2].item() if hasattr(bbox[2], 'item') else bbox[2],
                            bbox[3].item() if hasattr(bbox[3], 'item') else bbox[3]
                        ]
                        search_result["bbox"] = abs_bbox
                        
                        # Get object crop
                        object_crops = self._vqa_llm.get_object_crop(pil_image, abs_bbox, patch_scale=1.5)
                        object_crops = object_crops.unsqueeze(0)  # Add batch dimension
            
            # Expand image to square for VQA
            background_color = tuple(int(x*255) for x in self._vqa_llm.image_processor.image_mean)
            image_square, _, _ = self._expand2square(pil_image, background_color)
            
            # Generate answer
            answer = self._vqa_llm.free_form_inference(
                image_square,
                query,
                max_new_tokens=256,
                object_crops=object_crops
            )
            
            result = {"answer": answer}
            if search_result:
                result["search_result"] = search_result
            
            return result
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"error": f"VQA failed: {str(e)}"}
