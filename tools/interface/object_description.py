"""
Object Description Interfaces
"""

import os
import sys
from typing import Dict, Any, Union
import numpy as np
from PIL import Image as PILImage

from tools.interface_base import Interface, InterfaceCategory, Image, BoundingBox, Point


class DAMDescription(Interface):
    """DAM (Describe Anything Model) for region description."""
    
    CATEGORY = InterfaceCategory.TOOL
    FUNCTIONALITY = "Generate detailed descriptions for specific image regions."
    REFERENCE_PAPER = "DAM (2024)"
    TOOL_SOURCES = ["DAM"]
    
    INPUT_SCHEMA = {
        "image": {"type": "Image", "required": True},
        "location": {"type": "Union[Point, BoundingBox]", "required": True},
        "query": {"type": "string", "required": False, "default": "Describe this object."}
    }
    
    OUTPUT_SCHEMA = {
        "description": {"type": "string"}
    }
    
    # Agent-facing
    AGENT_NAME = "describe_region"
    AGENT_DESCRIPTION = "Get detailed description of a specific object or region by pointing to it."
    
    AGENT_INPUT_SCHEMA = {
        "location": {
            "type": "object",
            "required": True,
            "description": "Point {x, y} or box {x1, y1, x2, y2} to describe"
        },
        "query": {
            "type": "string",
            "required": False,
            "default": "Describe this object.",
            "description": "Specific question about the region (optional)"
        }
    }
    
    AGENT_OUTPUT_FORMAT = "Text description of the specified region"
    
    # Model paths
    DAM_DIR = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models", "describe-anything"
    )
    DEFAULT_MODEL_PATH = "nvidia/DAM-3B"
    
    def __init__(self, model_path: str = None, device: str = "cuda:0", load_8bit: bool = False):
        self.model_path = model_path or self.DEFAULT_MODEL_PATH
        self.device = device
        self.load_8bit = load_8bit
        self.model = None
    
    @classmethod
    def format_output_for_agent(cls, output: Dict[str, Any]) -> str:
        if "error" in output:
            return f"Error: {output['error']}"
        if "description" in output:
            return output["description"]
        return "Could not generate description."
    
    def initialize(self) -> None:
        """Initialize the DAM model."""
        if self.model is not None:
            return
        
        # Add DAM to path
        if self.DAM_DIR not in sys.path:
            sys.path.insert(0, self.DAM_DIR)
        
        from dam import DescribeAnythingModel, disable_torch_init
        
        disable_torch_init()
        
        # For newer transformers compatibility with siglip, we need to pass
        # device_map as specific device instead of "auto" or "cuda"
        self.model = DescribeAnythingModel(
            model_path=self.model_path,
            conv_mode="v1",
            prompt_mode="full+focal_crop",
            device=self.device,
            device_map=self.device,  # Override device_map to use specific device
            load_8bit=self.load_8bit
        )
    
    def _image_to_pil(self, image: Image) -> PILImage.Image:
        """Convert Image object to PIL Image."""
        if image.data is not None:
            data = image.data
            # Already PIL Image
            if isinstance(data, PILImage.Image):
                return data.convert('RGB')
            # Numpy array (BGR format from OpenCV)
            if isinstance(data, np.ndarray):
                # Convert BGR to RGB
                if len(data.shape) == 3 and data.shape[2] == 3:
                    rgb = data[:, :, ::-1]
                else:
                    rgb = data
                return PILImage.fromarray(rgb).convert('RGB')
        elif image.path is not None:
            return PILImage.open(image.path).convert('RGB')
        
        raise ValueError("Image has no data or path")
    
    def _create_mask_from_location(
        self,
        location: Union[Point, BoundingBox, Dict],
        image_size: tuple
    ) -> PILImage.Image:
        """Create a mask image from location specification.
        
        Args:
            location: Point, BoundingBox, or dict with coordinates
            image_size: (width, height) of the image
            
        Returns:
            PIL Image mask (white = region of interest)
        """
        width, height = image_size
        mask_np = np.zeros((height, width), dtype=np.uint8)
        
        # Parse location
        if isinstance(location, Point):
            # For point, create a small circular region
            cx, cy = int(location.x), int(location.y)
            radius = max(10, min(width, height) // 20)
            y_coords, x_coords = np.ogrid[:height, :width]
            dist = np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2)
            mask_np[dist <= radius] = 255
            
        elif isinstance(location, BoundingBox):
            x1, y1 = int(location.x1), int(location.y1)
            x2, y2 = int(location.x2), int(location.y2)
            x1, x2 = max(0, x1), min(width, x2)
            y1, y2 = max(0, y1), min(height, y2)
            mask_np[y1:y2, x1:x2] = 255
            
        elif isinstance(location, dict):
            if 'x' in location and 'y' in location:
                # Point format
                cx, cy = int(location['x']), int(location['y'])
                radius = max(10, min(width, height) // 20)
                y_coords, x_coords = np.ogrid[:height, :width]
                dist = np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2)
                mask_np[dist <= radius] = 255
            elif 'x1' in location and 'y1' in location:
                # BoundingBox format
                x1, y1 = int(location['x1']), int(location['y1'])
                x2, y2 = int(location['x2']), int(location['y2'])
                x1, x2 = max(0, x1), min(width, x2)
                y1, y2 = max(0, y1), min(height, y2)
                mask_np[y1:y2, x1:x2] = 255
            else:
                raise ValueError(f"Unknown location dict format: {location}")
        else:
            raise ValueError(f"Unknown location type: {type(location)}")
        
        return PILImage.fromarray(mask_np)
    
    def __call__(
        self,
        image: Image,
        location: Union[Point, BoundingBox, Dict],
        query: str = "Describe this object.",
        **kwargs
    ) -> Dict[str, Any]:
        """Generate description for a specific region in the image.
        
        Args:
            image: Image object containing the image
            location: Point or BoundingBox specifying the region
            query: Question/prompt for the description
            
        Returns:
            Dict with 'description' string
        """
        if self.model is None:
            self.initialize()
        
        try:
            # Convert image to PIL
            pil_image = self._image_to_pil(image)
            
            # Create mask from location
            mask = self._create_mask_from_location(location, pil_image.size)
            
            # Format query with image token
            if "<image>" not in query:
                query = f"<image>\n{query}"
            
            # Get description
            description = self.model.get_description(
                image_pil=pil_image,
                mask_pil=mask,
                query=query,
                streaming=False,
                temperature=0.2,
                top_p=0.5,
                num_beams=1,
                max_new_tokens=512
            )
            
            return {"description": description}
            
        except Exception as e:
            import traceback
            return {"error": f"{str(e)}\n{traceback.format_exc()}"}
