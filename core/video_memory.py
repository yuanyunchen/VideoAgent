"""
Video Memory module for VideoAgent.
"""

import os
import cv2
import json
import numpy as np
from typing import List, Dict, Any, Optional, Union

class VideoMemory:
    """Manages video frames and associated analysis data."""
    
    def __init__(self, video_id: str, question: str, answer: str = "", video_path: Optional[str] = None, 
                 choices: Optional[List[str]] = None, truth: Optional[str] = None, 
                 video_frame_tups: Optional[List] = None):
        """
        Initialize VideoMemory.
        
        Args:
            video_id: Unique identifier for the video
            question: Question to be answered
            answer: Ground truth answer (optional)
            video_path: Path to video file
            choices: Multiple choice options (optional)
            truth: Ground truth label for multiple choice (optional)
            video_frame_tups: Pre-loaded video frame tuples (backward compatibility)
        """
        self.video_id = video_id
        self.question = question
        self.answer = answer
        self.video_path = video_path
        self.choices = choices or []
        self.truth = truth
        
        # Load video frames
        self.video_frames = []
        self.original_idx = []
        
        if video_frame_tups:
            # Backward compatibility: use provided frame tuples
            self.video_frames = [tup[1] for tup in video_frame_tups]
            self.original_idx = [tup[0] for tup in video_frame_tups]
        elif video_path and os.path.exists(video_path):
            self._load_video_frames()
        
        self.n_frames = len(self.video_frames)
        
        # Analysis state - using lists for backward compatibility
        self.sampled_idx = []
        self.sampled_frames = []
        self.captions = {}  # Maps frame_idx to caption
        
        # For backward compatibility with original main.py
        self.detailed_captions = self.captions  # Alias
        self.overview = ""
        self.events = {}  # Maps frame_idx to event description
        
        # New format event descriptions
        self.event_descriptions = ""
        
        # Results
        self.predicted_answer = ""
        self.confidence = 1
        
        # Multi-round tracking
        self.rounds_history = []
        self.current_round = 0
        
        # Analysis tracking
        self.answer_analysis = ""
        self.confidence_analysis = ""
        
        # Initialize with basic sampling
        if self.n_frames > 0:
            self._initial_sampling()
    
    def _load_video_frames(self, interval: int = 30) -> None:
        """Load frames from video file."""
        cap = cv2.VideoCapture(self.video_path)
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx % interval == 0:
                self.video_frames.append(frame)
                self.original_idx.append(frame_idx)
            
            frame_idx += 1
        
        cap.release()
    
    def _initial_sampling(self, num_frames: int = 5) -> None:
        """Initial frame sampling."""
        if self.n_frames <= num_frames:
            indices = list(range(self.n_frames))
        else:
            indices = np.linspace(0, self.n_frames - 1, num=num_frames, dtype=int).tolist()
        
        self.add_sampled_frames(indices)
    
    def add_sampled_frames(self, indices: List[int]) -> None:
        """Add new sampled frame indices."""
        for idx in indices:
            if idx not in self.sampled_idx and 0 <= idx < len(self.video_frames):
                self.sampled_idx.append(idx)
                self.sampled_frames.append(self.video_frames[idx])
        
        # Keep indices sorted
        combined = list(zip(self.sampled_idx, self.sampled_frames))
        combined.sort(key=lambda x: x[0])
        self.sampled_idx, self.sampled_frames = zip(*combined) if combined else ([], [])
        self.sampled_idx = list(self.sampled_idx)
        self.sampled_frames = list(self.sampled_frames)
    
    def update_caption(self, frame_idx: int, caption: str) -> None:
        """Update caption for a specific frame."""
        self.captions[frame_idx] = caption
        self.detailed_captions[frame_idx] = caption  # Backward compatibility
    
    def start_new_round(self) -> None:
        """Start a new processing round."""
        self.current_round += 1
    
    def save_round_result(self, answer: str, confidence: int, frame_count: int, 
                         captions: Optional[dict] = None, event_descriptions: str = "") -> None:
        """Save result for current round."""
        round_result = {
            "round": self.current_round,
            "answer": answer,
            "confidence": confidence,
            "frame_count": frame_count
        }
        self.rounds_history.append(round_result)
    
    def save_to_directory(self, output_dir: str) -> None:
        """Save memory state to directory."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save question with multiple choice options
        question_file = os.path.join(output_dir, "question.txt")
        with open(question_file, "w") as f:
            f.write(f"Here is the question: \n{self.question}\n\n")
            if self.choices:
                f.write("Here are the choices: \n")
                for i, choice in enumerate(self.choices):
                    f.write(f"{i}. {choice}\n")
        
        # Save memory summary
        memory_file = os.path.join(output_dir, "memory.txt")
        with open(memory_file, "w") as f:
            f.write(str(self))
        
        # Save results
            result = {
                "video_id": self.video_id,
                "question": self.question,
                "predicted_answer": self.predicted_answer,
                "actual_answer": self.answer,
                "confidence": self.confidence,
                "rounds_history": self.rounds_history,
                "total_rounds": self.current_round,
                "answer_analysis": getattr(self, 'answer_analysis', ''),
                "confidence_analysis": getattr(self, 'confidence_analysis', '')
            }
        
        result_file = os.path.join(output_dir, "result.json")
        with open(result_file, "w") as f:
            json.dump(result, f, indent=2)
        
        # Save frames
        frames_dir = os.path.join(output_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        
        for i, frame_idx in enumerate(self.sampled_idx):
            frame_path = os.path.join(frames_dir, f"{frame_idx}.png")
            cv2.imwrite(frame_path, self.sampled_frames[i])
    
    def __str__(self) -> str:
        """
        String representation of memory content - restored from original main.py format.
        """
        content = []
        
        # Write Overview
        content.append("=== Overview: overview of the video. ===\n")
        content.append(f"{self.overview.strip() if self.overview else 'No overview provided.'}\n\n")

        # Write Events
        content.append("=== Events: abstract understanding ===\n")
        if self.events:
            for idx in self.sampled_idx:
                if idx in self.events:
                    event = self.events[idx]
                    content.append(f"Frame {idx}: {event}\n")
        else:
            content.append("No events recorded.\n")
        content.append("\n")
        
        # Write Captions
        content.append("=== Detailed Captions : visual elements ===\n")
        if self.sampled_idx and self.captions:
            for idx in self.sampled_idx:
                if idx in self.captions:
                    content.append(f"Frame {idx}: {self.captions[idx]}\n")
            else:
                content.append("No captions available.\n")
        
        return "".join(content)

    def frame_formatted(self, frame_indices: List[int]) -> Dict[str, str]:
        """Format frame captions for display - backward compatibility."""
        return {f"Frame {idx}": self.captions[idx] for idx in frame_indices if idx in self.captions}
    
    def update_event(self, frame_idx: int, event: str) -> None:
        """Update event description - backward compatibility."""
        if not hasattr(self, 'events'):
            self.events = {}
        self.events[frame_idx] = event
    
    def update_overview(self, overview: str) -> None:
        """Update overview - backward compatibility."""
        self.overview = overview
    
    def output(self, output_dir: str) -> None:
        """Backward compatibility method."""
        self.save_to_directory(output_dir)
    
    @property
    def detailed_captions(self) -> Dict[int, str]:
        """Backward compatibility property."""
        return self.captions
    
    @detailed_captions.setter
    def detailed_captions(self, value: Dict[int, str]) -> None:
        """Backward compatibility setter."""
        self.captions = value 