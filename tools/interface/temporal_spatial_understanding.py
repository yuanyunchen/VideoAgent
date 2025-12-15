"""
Temporal-Spatial Understanding Interfaces (TStar + VideoTree)

- TStarTemporalSpatialQA: Temporal-spatial QA using object-guided search
- TStarSampling: Object-guided frame sampling (TStar searcher)
- VideoTreeSampling: Visual-diversity frame sampling (VideoTree)
"""

import os
import sys
from typing import Dict, Any, Optional, List

import numpy as np

from tools.interface_base import Interface, InterfaceCategory, Video, Image

# Global constants
TSTAR_SEARCH_BUDGET = 128  # max frames to search during detection
TSTAR_FINAL_SAMPLE_FRAMES = 16  # default real sampling count


class VideoTreeSampling(Interface):
    """VideoTree-based temporal frame sampling using visual clustering."""

    CATEGORY = InterfaceCategory.TOOL
    FUNCTIONALITY = "Sample visually diverse frames to capture events and temporal changes using CLIP-based clustering."
    REFERENCE_PAPER = "VideoTree: Adaptive Tree-based Video Representation for LLM Reasoning (2024)"
    TOOL_SOURCES = ["VideoTree", "CLIP"]

    INPUT_SCHEMA = {
        "video": {"type": "Video", "required": True},
        "query": {"type": "string", "required": True},
        "num_frames": {"type": "int", "required": False, "default": TSTAR_FINAL_SAMPLE_FRAMES},
        "start_frame": {"type": "int", "required": False, "default": None},
        "end_frame": {"type": "int", "required": False, "default": None},
    }

    OUTPUT_SCHEMA = {
        "frames": {"type": "List[Image]"},
        "timestamps": {"type": "List[float]"},
        "frame_indices": {"type": "List[int]"},
    }

    AGENT_NAME = "temporal_sample_frames"
    AGENT_DESCRIPTION = (
        "Sample visually diverse frames to understand events, actions, and temporal changes. "
        "Uses visual clustering to capture different scenes/moments. Best for understanding "
        "WHAT happened, sequences, and changes over time. "
        "Query should be a QUESTION about events/actions you want to explore. "
        "Optionally specify a frame range to focus on a specific segment."
    )

    AGENT_INPUT_SCHEMA = {
        "query": {
            "type": "string",
            "required": True,
            "description": "A QUESTION about events or actions (e.g., 'What activities does the person perform?', 'How does the scene change over time?', 'What is the sequence of events?')",
        },
        "num_frames": {
            "type": "int",
            "required": False,
            "default": TSTAR_FINAL_SAMPLE_FRAMES,
            "description": "Number of frames to sample (default: 16)",
        },
        "start_frame": {
            "type": "int",
            "required": False,
            "description": "Start frame index (0-based, optional). Default: 0 (video start)",
        },
        "end_frame": {
            "type": "int",
            "required": False,
            "description": "End frame index (0-based, inclusive, optional). Default: last frame",
        },
    }

    AGENT_OUTPUT_FORMAT = "Captions of sampled frames with timestamps"

    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        sample_fps: float = 1.0,
        device: str = "cuda:0",
    ):
        self.model_name = model_name
        self.sample_fps = sample_fps
        self.device = device

        self._model = None
        self._processor = None
        self._initialized = False

    def initialize(self) -> None:
        if self._initialized:
            return

        import torch
        from transformers import CLIPModel, CLIPProcessor

        self._processor = CLIPProcessor.from_pretrained(self.model_name)
        self._model = CLIPModel.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
        ).to(self.device).eval()

        self._initialized = True

    @classmethod
    def format_output_for_agent(cls, output: Dict[str, Any]) -> str:
        if "error" in output:
            return f"Error: {output['error']}"

        captions = output.get("captions", {})
        timestamps = output.get("timestamps", [])
        frame_indices = output.get("frame_indices", [])

        if captions:
            lines = [f"Sampled {len(captions)} visually diverse frames:"]
            for idx in sorted(captions.keys()):
                ts = timestamps[list(frame_indices).index(idx)] if idx in frame_indices else 0
                lines.append(f"[Frame {idx} @ {ts:.1f}s]: {captions[idx]}")
            return "\n".join(lines)
        if timestamps:
            return f"Sampled {len(timestamps)} frames at: {[f'{t:.1f}s' for t in timestamps]}"
        return "No frames sampled."

    def __call__(
        self,
        video: Video,
        query: str,
        num_frames: int = TSTAR_FINAL_SAMPLE_FRAMES,
        start_frame: int = None,
        end_frame: int = None,
        **kwargs,
    ) -> Dict[str, Any]:
        num_frames = max(1, num_frames)

        if not self._initialized:
            self.initialize()

        import torch
        import cv2
        from PIL import Image as PILImage
        from scipy.cluster.hierarchy import linkage, fcluster

        video_path = video.path if isinstance(video, Video) else video

        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Apply frame range constraints
            range_start = 0 if start_frame is None else max(0, min(start_frame, total_frames - 1))
            range_end = total_frames - 1 if end_frame is None else max(range_start, min(end_frame, total_frames - 1))

            frame_interval = max(1, int(fps / self.sample_fps))
            # Sample only within the specified frame range
            sample_indices = list(range(range_start, range_end + 1, frame_interval))
            if len(sample_indices) < num_frames:
                sample_indices = list(range(range_start, range_end + 1))

            frames_data = []
            features_list = []

            for idx in sample_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    continue
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = PILImage.fromarray(frame_rgb)
                frames_data.append((idx, frame_rgb, pil_image))

                # Some videos contain occasional frames with slightly different sizes
                # (e.g., due to encoding glitches). Enabling padding and batching the
                # single image prevents tensor creation errors in HF processors.
                try:
                    inputs = self._processor(
                        images=[pil_image],
                        padding=True,
                        return_tensors="pt",
                    ).to(self.device)
                except ValueError:
                    # Standardize the frame to the processor's expected crop size and retry.
                    crop_size = getattr(self._processor, "image_processor", None)
                    if crop_size and getattr(crop_size, "crop_size", None):
                        target_w = crop_size.crop_size.get("width", 224)
                        target_h = crop_size.crop_size.get("height", 224)
                    else:
                        target_w = target_h = 224
                    safe_image = pil_image.convert("RGB").resize((target_w, target_h))
                    inputs = self._processor(
                        images=[safe_image],
                        padding=True,
                        return_tensors="pt",
                    ).to(self.device)
                with torch.no_grad(), torch.cuda.amp.autocast():
                    image_features = self._model.get_image_features(**inputs)
                    features_list.append(image_features.cpu().numpy().flatten())

            cap.release()

            if len(features_list) < num_frames:
                selected_indices = [f[0] for f in frames_data]
            else:
                features = np.array(features_list)
                features = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
                linked = linkage(features, method="ward")
                cluster_labels = fcluster(linked, num_frames, criterion="maxclust")

                selected_indices = []
                for cluster_id in range(1, num_frames + 1):
                    cluster_mask = cluster_labels == cluster_id
                    if not np.any(cluster_mask):
                        continue

                    cluster_features = features[cluster_mask]
                    cluster_frames = [frames_data[i] for i, m in enumerate(cluster_mask) if m]

                    centroid = cluster_features.mean(axis=0)
                    distances = np.linalg.norm(cluster_features - centroid, axis=1)
                    best_idx = np.argmin(distances)
                    selected_indices.append(cluster_frames[best_idx][0])

            selected_indices = sorted(selected_indices)

            result_frames = []
            result_timestamps = []
            for idx in selected_indices:
                for frame_idx, frame_rgb, _ in frames_data:
                    if frame_idx == idx:
                        result_frames.append(
                            Image(
                                data=frame_rgb,
                                frame_index=idx,
                                timestamp=idx / fps if fps > 0 else 0,
                            )
                        )
                        result_timestamps.append(idx / fps if fps > 0 else 0)
                        break

            return {
                "frames": result_frames,
                "timestamps": result_timestamps,
                "frame_indices": selected_indices,
            }
        except Exception as e:
            import traceback

            return {"error": f"{str(e)}\n{traceback.format_exc()}"}


class TStarSampling(Interface):
    """TStar-based object-guided spatial frame sampling."""

    CATEGORY = InterfaceCategory.TOOL
    FUNCTIONALITY = "Search video for frames containing specific objects mentioned in query using object detection."
    REFERENCE_PAPER = "T*: Grounded Object-Centric Tree Search for Long-Video Understanding (2024)"
    TOOL_SOURCES = ["TStar", "YOLO-World"]

    INPUT_SCHEMA = {
        "video": {"type": "Video", "required": True},
        "query": {"type": "string", "required": True},
        "num_frames": {"type": "int", "required": False, "default": TSTAR_FINAL_SAMPLE_FRAMES},
        "start_frame": {"type": "int", "required": False, "default": None},
        "end_frame": {"type": "int", "required": False, "default": None},
    }

    OUTPUT_SCHEMA = {
        "frames": {"type": "List[Image]"},
        "timestamps": {"type": "List[float]"},
        "frame_indices": {"type": "List[int]"},
        "target_objects": {"type": "List[str]"},
        "cue_objects": {"type": "List[str]"},
    }

    AGENT_NAME = "temporal_spatial_sample_frames"
    AGENT_DESCRIPTION = (
        "Search video for frames based on objects, locations, or spatial relationships. "
        "Uses object detection to find relevant frames. Best for questions about "
        "WHERE things are, WHAT objects appear, spatial arrangements, and object interactions. "
        "Query should be a QUESTION mentioning the objects/locations you want to find. "
        "Optionally specify a frame range to focus on a specific segment."
    )

    AGENT_INPUT_SCHEMA = {
        "query": {
            "type": "string",
            "required": True,
            "description": "A QUESTION about objects or spatial details (e.g., 'Where is the cup located?', 'What tools does the person use?', 'Are there any vehicles in the scene?', 'What objects are on the table?')",
        },
        "num_frames": {
            "type": "int",
            "required": False,
            "default": TSTAR_FINAL_SAMPLE_FRAMES,
            "description": f"Number of frames to retrieve (default: {TSTAR_FINAL_SAMPLE_FRAMES})",
        },
        "start_frame": {
            "type": "int",
            "required": False,
            "description": "Start frame index (0-based, optional). Default: 0 (video start)",
        },
        "end_frame": {
            "type": "int",
            "required": False,
            "description": "End frame index (0-based, inclusive, optional). Default: last frame",
        },
    }

    AGENT_OUTPUT_FORMAT = "Captions of frames containing the target objects"

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        heuristic: str = "owl-vit",
        search_budget: int = TSTAR_SEARCH_BUDGET,
        confidence_threshold: float = 0.3,
        device: str = "cuda:0",
        output_dir: str = "./output/tstar",
        grid_rows: int = 4,
        grid_cols: int = 4,
    ):
        self.model = model
        self.heuristic_type = heuristic
        self.search_budget = search_budget
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.output_dir = output_dir
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols

        self._grounder = None
        self._heuristic = None
        self._initialized = False

    def initialize(self) -> None:
        if self._initialized:
            return

        from dotenv import load_dotenv

        load_dotenv()

        api_key = os.environ.get("AIML_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "TStar requires an API key for VLM grounding. "
                "Set AIML_API_KEY or OPENAI_API_KEY environment variable."
            )

        tstar_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "models",
            "TStar",
        )
        if tstar_path not in sys.path:
            sys.path.insert(0, tstar_path)

        from TStar.interface_grounding import TStarUniversalGrounder
        from TStar.TStarFramework import initialize_heuristic

        self._grounder = TStarUniversalGrounder(
            model_name=self.model,
            num_frames=8,
        )
        # Pass device to heuristic to enable multi-GPU support
        self._heuristic = initialize_heuristic(self.heuristic_type, device=self.device)

        self._initialized = True

    @classmethod
    def format_output_for_agent(cls, output: Dict[str, Any]) -> str:
        if "error" in output:
            return f"Error: {output['error']}"

        target_objects = output.get("target_objects", [])
        captions = output.get("captions", {})
        timestamps = output.get("timestamps", [])
        frame_indices = output.get("frame_indices", [])

        lines = []
        if target_objects:
            lines.append(f"Searched for: {', '.join(target_objects)}")

        if captions:
            lines.append(f"Found {len(captions)} frames:")
            for idx in sorted(captions.keys()):
                ts = timestamps[list(frame_indices).index(idx)] if idx in frame_indices else 0
                lines.append(f"[Frame {idx} @ {ts:.1f}s]: {captions[idx]}")
        elif timestamps:
            lines.append(f"Found {len(timestamps)} frames at: {[f'{t:.1f}s' for t in timestamps]}")
        else:
            lines.append("No frames with target objects found.")

        return "\n".join(lines)

    def __call__(
        self,
        video: Video,
        query: str,
        num_frames: int = TSTAR_FINAL_SAMPLE_FRAMES,
        start_frame: int = None,
        end_frame: int = None,
        **kwargs,
    ) -> Dict[str, Any]:
        num_frames = max(1, num_frames)

        if not self._initialized:
            self.initialize()

        tstar_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "models",
            "TStar",
        )
        if tstar_path not in sys.path:
            sys.path.insert(0, tstar_path)

        from TStar.interface_searcher import TStarSearcher
        import cv2

        video_path = video.path if isinstance(video, Video) else video

        try:
            # Get video info for frame range validation
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            # Apply frame range constraints
            range_start = 0 if start_frame is None else max(0, min(start_frame, total_frames - 1))
            range_end = total_frames - 1 if end_frame is None else max(range_start, min(end_frame, total_frames - 1))
            has_frame_range = (range_start > 0 or range_end < total_frames - 1)
            
            target_objects, cue_objects = self._grounder.inference_query_grounding(
                video_path=video_path,
                question=query,
                options="",
            )

            if not target_objects:
                target_objects = ["person", "object"]
            if not cue_objects:
                cue_objects = ["background"]

            # Request more frames if we have a range constraint (will filter later)
            search_frames = num_frames * 2 if has_frame_range else num_frames
            
            searcher = TStarSearcher(
                video_path=video_path,
                heuristic=self._heuristic,
                target_objects=target_objects,
                cue_objects=cue_objects,
                search_nframes=search_frames,
                image_grid_shape=(self.grid_rows, self.grid_cols),
                search_budget=self.search_budget,
                output_dir=self.output_dir,
                confidence_threshold=self.confidence_threshold,
            )

            frames_np, timestamps = searcher.search(final_samples=search_frames)

            frame_indices = [int(ts * fps) for ts in timestamps]

            # Filter frames within the specified range
            result_frames = []
            result_timestamps = []
            result_indices = []
            for i, (frame_np, ts) in enumerate(zip(frames_np, timestamps)):
                idx = frame_indices[i]
                if range_start <= idx <= range_end:
                    if isinstance(frame_np, np.ndarray):
                        result_frames.append(
                            Image(
                                data=frame_np,
                                frame_index=idx,
                                timestamp=ts,
                            )
                        )
                        result_timestamps.append(ts)
                        result_indices.append(idx)
                    # Stop once we have enough frames
                    if len(result_frames) >= num_frames:
                        break

            return {
                "frames": result_frames,
                "timestamps": result_timestamps,
                "frame_indices": result_indices,
                "target_objects": target_objects,
                "cue_objects": cue_objects,
            }
        except Exception as e:
            import traceback

            return {"error": f"{str(e)}\n{traceback.format_exc()}"}


class TStarTemporalSpatialQA(Interface):
    """TStar-based Temporal-Spatial Q&A Interface."""

    CATEGORY = InterfaceCategory.SUB_QUESTION_ANSWERING
    FUNCTIONALITY = "Answer questions involving temporal and spatial relationships."
    REFERENCE_PAPER = "T* (2024)"
    TOOL_SOURCES = ["TStar"]

    INPUT_SCHEMA = {
        "query": {"type": "string", "required": True},
        "video": {"type": "Video", "required": True},
        "model": {"type": "string", "required": False, "default": "gpt-4o-mini"},
    }

    OUTPUT_SCHEMA = {
        "answer": {"type": "string"},
        "evidence": {"type": "string"},
    }

    AGENT_NAME = "temporal_spatial_qa"
    AGENT_DESCRIPTION = "Answer questions involving objects in space and time - movements, locations, trajectories."

    AGENT_INPUT_SCHEMA = {
        "query": {
            "type": "string",
            "required": True,
            "description": "Question about spatial-temporal relationships",
        }
    }

    AGENT_OUTPUT_FORMAT = "Text answer with evidence about spatial-temporal information"

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        heuristic: str = "owl-vit",
        search_nframes: int = TSTAR_FINAL_SAMPLE_FRAMES,
        grid_rows: int = 4,
        grid_cols: int = 4,
        confidence_threshold: float = 0.6,
        search_budget: int = TSTAR_SEARCH_BUDGET,
        output_dir: str = "./output/tstar",
        device: str = "cuda:0",
    ):
        self.model = model
        self.heuristic_type = heuristic
        self.search_nframes = search_nframes
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.confidence_threshold = confidence_threshold
        self.search_budget = search_budget
        self.output_dir = output_dir
        self.device = device

        self._grounder = None
        self._heuristic = None
        self._initialized = False

    def initialize(self) -> None:
        if self._initialized:
            return

        from dotenv import load_dotenv

        load_dotenv()

        api_key = os.environ.get("AIML_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "TStar requires an API key for VLM grounding. "
                "Set AIML_API_KEY or OPENAI_API_KEY environment variable."
            )

        tstar_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "models",
            "TStar",
        )
        if tstar_path not in sys.path:
            sys.path.insert(0, tstar_path)

        from TStar.interface_grounding import TStarUniversalGrounder
        from TStar.TStarFramework import initialize_heuristic

        self._grounder = TStarUniversalGrounder(
            model_name=self.model,
            num_frames=8,
        )
        # Pass device to heuristic to enable multi-GPU support
        self._heuristic = initialize_heuristic(self.heuristic_type, device=self.device)

        self._initialized = True

    @classmethod
    def format_output_for_agent(cls, output: Dict[str, Any]) -> str:
        if "error" in output:
            return f"Error: {output['error']}"

        answer = output.get("answer", "Could not determine answer.")
        evidence = output.get("evidence", "")

        if evidence:
            return f"Answer: {answer}\nEvidence: {evidence}"
        return f"Answer: {answer}"

    def __call__(
        self,
        query: str,
        video: Video,
        model: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        if not self._initialized:
            self.initialize()

        tstar_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "models",
            "TStar",
        )
        if tstar_path not in sys.path:
            sys.path.insert(0, tstar_path)

        from TStar.interface_searcher import TStarSearcher
        from PIL import Image as PILImage

        video_path = video.path if isinstance(video, Video) else video

        if model and model != self.model:
            from TStar.interface_grounding import TStarUniversalGrounder

            grounder = TStarUniversalGrounder(model_name=model, num_frames=8)
        else:
            grounder = self._grounder

        try:
            target_objects, cue_objects = grounder.inference_query_grounding(
                video_path=video_path,
                question=query,
                options="",
            )

            if not target_objects:
                target_objects = ["person", "object"]
            if not cue_objects:
                cue_objects = ["background"]

            searcher = TStarSearcher(
                video_path=video_path,
                heuristic=self._heuristic,
                target_objects=target_objects,
                cue_objects=cue_objects,
                search_nframes=self.search_nframes,
                image_grid_shape=(self.grid_rows, self.grid_cols),
                search_budget=self.search_budget,
                output_dir=self.output_dir,
                confidence_threshold=self.confidence_threshold,
            )

            frames_np, timestamps = searcher.search(final_samples=self.search_nframes)

            frames_pil: List[PILImage.Image] = []
            for frame in frames_np:
                if isinstance(frame, np.ndarray):
                    frames_pil.append(PILImage.fromarray(frame))
                else:
                    frames_pil.append(frame)

            answer = grounder.inference_openend_qa(
                frames=frames_pil,
                question=query,
            )

            evidence_parts = []
            if target_objects:
                evidence_parts.append(f"Searched for objects: {', '.join(target_objects)}")
            if timestamps:
                time_str = ", ".join([f"{t:.1f}s" for t in timestamps[:5]])
                if len(timestamps) > 5:
                    time_str += f" (+{len(timestamps) - 5} more)"
                evidence_parts.append(f"Analyzed frames at: {time_str}")

            evidence = ". ".join(evidence_parts) if evidence_parts else ""

            return {
                "answer": answer,
                "evidence": evidence,
                "grounding_objects": {
                    "target_objects": target_objects,
                    "cue_objects": cue_objects,
                },
                "frame_timestamps": timestamps,
            }
        except Exception as e:
            import traceback

            traceback.print_exc()
            return {"error": str(e)}


__all__ = [
    "VideoTreeSampling",
    "TStarSampling",
    "TStarTemporalSpatialQA",
    "TSTAR_SEARCH_BUDGET",
    "TSTAR_FINAL_SAMPLE_FRAMES",
]

