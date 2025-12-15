"""
Object Detection Interfaces
"""

import os
import sys
from typing import Dict, Any, List, Optional
import numpy as np
import cv2

from tools.interface_base import Interface, InterfaceCategory, Image, BoundingBox


class YOLOWorldDetection(Interface):
    """YOLO-World open-vocabulary object detection."""
    
    CATEGORY = InterfaceCategory.TOOL
    FUNCTIONALITY = "Open-vocabulary object detection."
    REFERENCE_PAPER = "YOLO-World (2024)"
    TOOL_SOURCES = ["YOLO-World"]
    
    INPUT_SCHEMA = {
        "images": {"type": "List[Image]", "required": True},
        "categories": {"type": "List[str]", "required": True},
        "score_threshold": {"type": "float", "required": False, "default": 0.3},
        "frame_indices": {"type": "List[int]", "required": False}
    }
    
    OUTPUT_SCHEMA = {
        "per_frame_detections": {"type": "Dict[int, List[object]]"}
    }
    
    # Agent-facing
    AGENT_NAME = "detect_objects"
    AGENT_DESCRIPTION = "Detect and locate specific objects you specify in the current frame(s)."
    
    AGENT_INPUT_SCHEMA = {
        "categories": {
            "type": "List[str]",
            "required": True,
            "description": "Objects to detect (e.g., ['person', 'car', 'dog'])"
        }
    }
    
    AGENT_OUTPUT_FORMAT = "Per-frame detections with object locations (bounding boxes)"
    
    # Model paths
    YOLOWORLD_DIR = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models", "YOLO-World"
    )
    
    def __init__(self, device: str = "cuda:0"):
        self.device = device
        self.detector = None
    
    @classmethod
    def format_output_for_agent(cls, output: Dict[str, Any]) -> str:
        """Format per-frame detection results with bounding boxes for the agent."""
        if "error" in output:
            return f"Error: {output['error']}"
        
        per_frame = output.get("per_frame_detections", {})
        if not per_frame:
            return "No objects detected in any frame."
        
        lines = []
        for frame_idx in sorted(per_frame.keys()):
            detections = per_frame[frame_idx]
            if detections:
                det_strs = []
                for det in detections:
                    label = det.get("label", "unknown")
                    score = det.get("score", 0)
                    box = det.get("box_str", "")
                    det_strs.append(f"{label}({score:.2f}) at {box}")
                lines.append(f"Frame {frame_idx}: {', '.join(det_strs)}")
            else:
                lines.append(f"Frame {frame_idx}: No objects detected")
        
        return "\n".join(lines)
    
    def initialize(self) -> None:
        """Initialize the YOLO-World detector."""
        if self.detector is not None:
            return
        
        # Add YOLO-World to path
        if self.YOLOWORLD_DIR not in sys.path:
            sys.path.insert(0, self.YOLOWORLD_DIR)
        
        from yoloworld_inference import YOLOWorldDetector
        self.detector = YOLOWorldDetector(device=self.device)
    
    def _image_to_numpy(self, image: Image) -> np.ndarray:
        """Convert Image object to numpy array (BGR format)."""
        if image.data is not None:
            data = image.data
            # Handle PIL Image
            if hasattr(data, 'convert'):
                data = np.array(data.convert('RGB'))
                # RGB to BGR
                return data[:, :, ::-1].copy()
            # Handle numpy array
            if isinstance(data, np.ndarray):
                # Check if RGB (3 channels) and convert to BGR
                if len(data.shape) == 3 and data.shape[2] == 3:
                    return data
                return data
        elif image.path is not None:
            return cv2.imread(image.path)
        
        raise ValueError("Image has no data or path")
    
    def __call__(
        self,
        images: List[Image],
        categories: List[str],
        score_threshold: float = 0.3,
        frame_indices: Optional[List[int]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Run object detection on images.
        
        Args:
            images: List of Image objects to process
            categories: List of category names to detect
            score_threshold: Confidence threshold for detections (default 0.3)
            frame_indices: Optional list of frame indices corresponding to images
            
        Returns:
            Dict with 'per_frame_detections' containing per-frame results with bboxes
        """
        if self.detector is None:
            self.initialize()
        
        # Generate frame indices if not provided
        if frame_indices is None:
            frame_indices = list(range(len(images)))
        
        try:
            per_frame_detections = {}
            
            for i, image in enumerate(images):
                frame_idx = frame_indices[i] if i < len(frame_indices) else i
                img_np = self._image_to_numpy(image)
                
                results = self.detector.detect(
                    image=img_np,
                    categories=categories,
                    score_thr=score_threshold
                )
                
                frame_detections = []
                for box, label_text, score in zip(
                    results['boxes'],
                    results['label_texts'],
                    results['scores']
                ):
                    # Format bbox as [x1, y1, x2, y2]
                    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                    box_str = f"[{x1},{y1},{x2},{y2}]"
                    
                    bbox = BoundingBox(
                        x1=float(box[0]),
                        y1=float(box[1]),
                        x2=float(box[2]),
                        y2=float(box[3]),
                        label=label_text,
                        score=float(score)
                    )
                    frame_detections.append({
                        "box": bbox,
                        "box_str": box_str,
                        "label": label_text,
                        "score": float(score)
                    })
                
                per_frame_detections[frame_idx] = frame_detections
            
            return {"per_frame_detections": per_frame_detections}
            
        except Exception as e:
            return {"error": str(e)}


class YOLOEPromptFreeDetection(Interface):
    """YOLOE prompt-free detection (all objects)."""
    
    CATEGORY = InterfaceCategory.TOOL
    FUNCTIONALITY = "Detect all objects without specifying categories."
    REFERENCE_PAPER = "YOLOE (2025)"
    TOOL_SOURCES = ["YOLOE"]
    
    INPUT_SCHEMA = {
        "images": {"type": "List[Image]", "required": True},
        "score_threshold": {"type": "float", "required": False, "default": 0.001},
        "max_detections": {"type": "int", "required": False, "default": 100},
        "frame_indices": {"type": "List[int]", "required": False}
    }
    
    OUTPUT_SCHEMA = {
        "per_frame_detections": {"type": "Dict[int, List[object]]"}
    }
    
    # Agent-facing
    AGENT_NAME = "detect_all_objects"
    AGENT_DESCRIPTION = "Detect ALL visible objects without specifying what to look for."
    
    AGENT_INPUT_SCHEMA = {
        "max_detections": {
            "type": "int",
            "required": False,
            "default": 100,
            "description": "Maximum objects to detect"
        }
    }
    
    AGENT_OUTPUT_FORMAT = "Per-frame list of all detected objects with labels"
    
    # Model paths
    YOLOE_DIR = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models", "yoloe"
    )
    
    def __init__(self, device: str = "cuda:0", model_size: str = "v8l"):
        """Initialize YOLOE detector.
        
        Args:
            device: CUDA device (e.g., 'cuda:0')
            model_size: Model size variant ('v8s', 'v8m', 'v8l', '11s', '11m', '11l')
                        Default is 'v8l' as it's pre-downloaded.
        """
        self.device = device
        self.model_size = model_size
        self.model = None
        self.vocab_names = None
    
    @classmethod
    def format_output_for_agent(cls, output: Dict[str, Any]) -> str:
        """Format per-frame detection results for the agent."""
        if "error" in output:
            return f"Error: {output['error']}"
        
        per_frame = output.get("per_frame_detections", {})
        if not per_frame:
            return "No objects detected in any frame."
        
        lines = []
        for frame_idx in sorted(per_frame.keys()):
            detections = per_frame[frame_idx]
            if detections:
                # Group by label for readability
                label_counts = {}
                for det in detections:
                    label = det.get("label", "unknown")
                    if label not in label_counts:
                        label_counts[label] = 0
                    label_counts[label] += 1
                
                sorted_labels = sorted(label_counts.items(), key=lambda x: -x[1])[:10]
                label_strs = [f"{label}({count})" for label, count in sorted_labels]
                lines.append(f"Frame {frame_idx}: {', '.join(label_strs)}")
            else:
                lines.append(f"Frame {frame_idx}: No objects detected")
        
        return "\n".join(lines)
    
    def initialize(self) -> None:
        """Initialize the YOLOE prompt-free detector."""
        if self.model is not None:
            return
        
        import gc
        import torch
        import torch.nn as nn
        
        # Add YOLOE to path
        if self.YOLOE_DIR not in sys.path:
            sys.path.insert(0, self.YOLOE_DIR)
        
        # Change working directory temporarily for relative path resolution
        original_cwd = os.getcwd()
        os.chdir(self.YOLOE_DIR)
        
        try:
            from ultralytics import YOLOE
            from huggingface_hub import hf_hub_download
            
            # Determine model filenames based on model_size
            model_variant = self.model_size  # e.g., 'v8s', 'v8m', 'v8l', '11s'
            vocab_cache_path = os.path.join(self.YOLOE_DIR, "pretrain", f"yoloe-{model_variant}-vocab.pt")
            pf_filename = f"yoloe-{model_variant}-seg-pf.pt"
            
            # Check if we have cached vocabulary
            if os.path.exists(vocab_cache_path):
                # Load cached vocabulary (much faster and less memory)
                print(f"Loading cached vocabulary from: {vocab_cache_path}")
                cache = torch.load(vocab_cache_path, map_location='cpu')
                self.vocab_names = cache['names']
                vocab_info = cache['vocab_info']
                
                # Load prompt-free model first
                pf_model_path = hf_hub_download(
                    repo_id="jameslahm/yoloe",
                    filename=pf_filename
                )
                
                self.model = YOLOE(pf_model_path)
                self.model.to(self.device)
                
                # Reconstruct vocab from cache as Conv2d layers
                vocab = nn.ModuleList()
                for i, info in enumerate(vocab_info):
                    conv = nn.Conv2d(
                        in_channels=info['in_channels'],
                        out_channels=info['out_channels'],
                        kernel_size=info['kernel_size']
                    )
                    vocab.append(conv)
                
                # Load state dict
                vocab.load_state_dict(cache['vocab_state_dict'])
                vocab.to(self.device)
                
                self.model.set_vocab(vocab, names=self.vocab_names)
                self.model.model.model[-1].is_fused = True
                self.model.model.model[-1].conf = 0.001
                self.model.model.model[-1].max_det = 1000
                
            else:
                # No cache - compute vocabulary (memory intensive)
                yaml_name = f"yoloe-{model_variant}.yaml"
                seg_filename = f"yoloe-{model_variant}-seg.pt"
                
                # Load vocabulary names
                vocab_path = os.path.join(self.YOLOE_DIR, "tools", "ram_tag_list.txt")
                with open(vocab_path, 'r') as f:
                    self.vocab_names = [x.strip() for x in f.readlines()]
                
                # Load unfused model to get vocab
                unfused_model_path = os.path.join(self.YOLOE_DIR, "pretrain", seg_filename)
                if not os.path.exists(unfused_model_path):
                    unfused_model_path = hf_hub_download(
                        repo_id="jameslahm/yoloe",
                        filename=seg_filename
                    )
                
                unfused_model = YOLOE(yaml_name)
                unfused_model.load(unfused_model_path)
                unfused_model.eval()
                unfused_model.to(self.device)
                
                vocab = unfused_model.get_vocab(self.vocab_names)
                
                # Clean up unfused model
                del unfused_model
                gc.collect()
                torch.cuda.empty_cache()
                
                # Load prompt-free model
                pf_model_path = hf_hub_download(
                    repo_id="jameslahm/yoloe",
                    filename=pf_filename
                )
                
                self.model = YOLOE(pf_model_path)
                self.model.to(self.device)
                self.model.set_vocab(vocab, names=self.vocab_names)
                self.model.model.model[-1].is_fused = True
                self.model.model.model[-1].conf = 0.001
                self.model.model.model[-1].max_det = 1000
            
        finally:
            os.chdir(original_cwd)
    
    def _image_to_numpy(self, image: Image) -> np.ndarray:
        """Convert Image object to numpy array (BGR format)."""
        if image.data is not None:
            data = image.data
            # Handle PIL Image
            if hasattr(data, 'convert'):
                data = np.array(data.convert('RGB'))
                # RGB to BGR
                return data[:, :, ::-1].copy()
            # Handle numpy array
            if isinstance(data, np.ndarray):
                return data
        elif image.path is not None:
            return cv2.imread(image.path)
        
        raise ValueError("Image has no data or path")
    
    def __call__(
        self,
        images: List[Image],
        score_threshold: float = 0.001,
        max_detections: int = 100,
        frame_indices: Optional[List[int]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Run prompt-free object detection on images.
        
        Args:
            images: List of Image objects to process
            score_threshold: Confidence threshold for detections
            max_detections: Maximum number of detections to return per frame
            frame_indices: Optional list of frame indices corresponding to images
            
        Returns:
            Dict with 'per_frame_detections' containing per-frame results
        """
        if self.model is None:
            self.initialize()
        
        # Generate frame indices if not provided
        if frame_indices is None:
            frame_indices = list(range(len(images)))
        
        try:
            per_frame_detections = {}
            
            for i, image in enumerate(images):
                frame_idx = frame_indices[i] if i < len(frame_indices) else i
                img_np = self._image_to_numpy(image)
                
                # Run prediction
                results = self.model.predict(
                    source=img_np,
                    conf=score_threshold,
                    verbose=False
                )
                
                frame_detections = []
                
                if len(results) > 0:
                    result = results[0]
                    boxes = result.boxes
                    
                    if boxes is not None and len(boxes) > 0:
                        # Get detections
                        det_boxes = boxes.xyxy.cpu().numpy()
                        det_scores = boxes.conf.cpu().numpy()
                        det_classes = boxes.cls.cpu().numpy().astype(int)
                        
                        # Convert to BoundingBox format
                        for j, (box, score, cls_id) in enumerate(zip(det_boxes, det_scores, det_classes)):
                            if j >= max_detections:
                                break
                            
                            label = self.vocab_names[cls_id] if cls_id < len(self.vocab_names) else f"class_{cls_id}"
                            
                            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                            box_str = f"[{x1},{y1},{x2},{y2}]"
                            
                            bbox = BoundingBox(
                                x1=float(box[0]),
                                y1=float(box[1]),
                                x2=float(box[2]),
                                y2=float(box[3]),
                                label=label,
                                score=float(score)
                            )
                            frame_detections.append({
                                "box": bbox,
                                "box_str": box_str,
                                "label": label,
                                "score": float(score)
                            })
                
                # Sort by score and limit
                frame_detections.sort(key=lambda x: -x["score"])
                frame_detections = frame_detections[:max_detections]
                
                per_frame_detections[frame_idx] = frame_detections
            
            return {"per_frame_detections": per_frame_detections}
            
        except Exception as e:
            import traceback
            return {"error": f"{str(e)}\n{traceback.format_exc()}"}
