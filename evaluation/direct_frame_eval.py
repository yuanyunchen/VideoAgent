"""
Direct Frame Evaluation Pipeline.
Evaluates model accuracy when given only the first N frames (images) without captions.
"""

import os
import json
import csv
import random
import argparse
import logging
import multiprocessing as mp
from datetime import datetime
from typing import Dict, Any, List
from tqdm import tqdm

from video_agent.utils.api import get_llm_response
from video_agent.utils.video import get_video_frames
from video_agent.utils.parsing import parse_video_annotation, parse_text_find_number
from video_agent.utils.logging_utils import setup_logger


# System prompt for direct frame question answering
SYSTEM_PROMPT_DIRECT_FRAME = """You are an expert video analyst tasked with answering questions based on visual information from video frames. 
"""


def _build_direct_frame_prompt(question: str, choices: List[str], num_frames: int, total_video_frames: int) -> str:
    """
    Build a well-structured prompt for direct frame question answering.
    
    Args:
        question: The question to answer
        choices: List of multiple choice options
        num_frames: Number of frames provided
        total_video_frames: Total frames in the original video
        
    Returns:
        Formatted prompt string
    """
    answer_format = {"final_answer": "<number 0-4>"}
    
    prompt = f"""You are provided with {num_frames} frames from the beginning of a video.
The video contains approximately {total_video_frames} frames in total (sampled at 1 frame per second).
These frames represent the initial portion of the video content.

------------------------------------------------------------------------
Task: Analyze the provided frames and answer the following multiple choice question.

Question: {question}

Choices:
"""
    
    for i, choice in enumerate(choices):
        prompt += f"  {i}. {choice}\n"
    
    prompt += f"""
------------------------------------------------------------------------
Instructions:
1. Carefully observe each frame for visual details, objects, actions, and context.
2. Identify the main activity or scenario depicted across the frames.
3. Consider what the camera wearer (#C) appears to be doing.
4. Evaluate each choice against your observations.
5. Select the most appropriate answer based on available visual evidence.

Note: If the frames provide insufficient information to answer with certainty, 
make your best judgment based on the most prominent visual cues.

------------------------------------------------------------------------
Format your response as follows:
- First, write your step-by-step analysis as plain text, describing:
  * Key observations from each frame
  * The overall activity or context
  * How you evaluated each choice
- Then, on a new line, include the JSON output enclosed in triple backticks:

```json
{json.dumps(answer_format)}
```

Replace <number 0-4> with your chosen answer (0, 1, 2, 3, or 4).
"""
    
    return prompt


def _process_video_worker(args) -> Dict:
    """Worker function for multiprocessing."""
    video_info, config, output_dir = args
    
    video_id = video_info["video_id"]
    video_dir = config["video_dir"]
    model = config["model"]
    num_frames = config["num_frames"]
    enable_logging = config.get("enable_logging", True)
    
    video_path = os.path.join(video_dir, f"{video_id}.mp4")
    
    # Setup per-video logging if enabled
    video_logger = None
    if enable_logging and output_dir:
        video_output_dir = os.path.join(output_dir, "videos", video_id)
        os.makedirs(video_output_dir, exist_ok=True)
        
        video_logger = logging.getLogger(f"DirectFrameEval.Video.{video_id}")
        video_logger.setLevel(logging.INFO)
        video_logger.handlers.clear()
        
        video_log_file = os.path.join(video_output_dir, "logging.log")
        video_handler = logging.FileHandler(video_log_file)
        video_handler.setLevel(logging.INFO)
        video_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        video_handler.setFormatter(video_formatter)
        video_logger.addHandler(video_handler)
        
        video_logger.info(f"=== Starting video processing: {video_id} ===")
    
    # Load frames
    try:
        frames = get_video_frames(video_path, interval=30)
        if frames is None or len(frames) == 0:
            if video_logger:
                video_logger.error(f"Cannot load frames from {video_path}")
            return {"video_id": video_id, "error": "Cannot load frames", "success": False}
        
        total_video_frames = len(frames)
        if video_logger:
            video_logger.info(f"Video loaded: {total_video_frames} total frames (sampled at interval=30)")
            
    except Exception as e:
        if video_logger:
            video_logger.error(f"Error loading frames: {e}")
        return {"video_id": video_id, "error": str(e), "success": False}
    
    # Get first N frames
    selected_frames = frames[:num_frames]
    selected_indices = list(range(num_frames))
    
    if video_logger:
        video_logger.info(f"Selected frames: {selected_indices} (first {num_frames} frames)")
    
    # Build question prompt
    question = video_info.get("question", "")
    choices = []
    for i in range(5):
        option_key = f"option {i}"
        if option_key in video_info:
            choices.append(video_info[option_key])
    
    truth = video_info.get("truth")
    
    if video_logger:
        video_logger.info(f"Question: {question}")
        video_logger.info(f"Choices: {choices}")
        video_logger.info(f"Ground truth: {truth}")
    
    # Format prompt
    prompt = _build_direct_frame_prompt(question, choices, num_frames, total_video_frames)
    
    if video_logger:
        video_logger.info("=== LLM Request ===")
        video_logger.info(f"Model: {model}")
        video_logger.info(f"System prompt: {SYSTEM_PROMPT_DIRECT_FRAME}")
        video_logger.info(f"User prompt:\n{prompt}")
    
    # Get model response
    try:
        response = get_llm_response(
            model=model,
            query=prompt,
            images=selected_frames,
            system_prompt=SYSTEM_PROMPT_DIRECT_FRAME,
            logger=video_logger
        )
        
        if video_logger:
            video_logger.info("=== LLM Response ===")
            video_logger.info(f"Response:\n{response}")
        
        # Parse answer
        answer = parse_text_find_number(response, "final_answer")
        is_valid = True
        
        if video_logger:
            video_logger.info(f"Parsed answer: {answer}")
        
        # Fallback if parsing failed
        if answer == -1:
            is_valid = False
            answer = random.randint(0, 4)
            if video_logger:
                video_logger.warning(f"Parsing failed, using fallback random answer: {answer}")
        
        # Check correctness
        try:
            truth_int = int(truth) if truth is not None else -1
        except (ValueError, TypeError):
            truth_int = -1
        
        is_correct = answer == truth_int
        
        if video_logger:
            status = "CORRECT" if is_correct else "WRONG"
            video_logger.info(f"=== Result: {status} ===")
            video_logger.info(f"Predicted: {answer}, Truth: {truth_int}")
        
        # Save per-video result
        if enable_logging and output_dir:
            video_result = {
                "video_id": video_id,
                "question": question,
                "choices": choices,
                "predicted_answer": answer,
                "truth": truth_int,
                "is_correct": is_correct,
                "is_valid": is_valid,
                "num_frames_used": len(selected_frames),
                "total_video_frames": total_video_frames
            }
            result_file = os.path.join(video_output_dir, "result.json")
            with open(result_file, "w") as f:
                json.dump(video_result, f, indent=2)
        
        return {
            "video_id": video_id,
            "question": question,
            "choices": choices,
            "predicted_answer": answer,
            "truth": truth_int,
            "is_correct": is_correct,
            "is_valid": is_valid,
            "num_frames_used": len(selected_frames),
            "total_video_frames": total_video_frames,
            "response": response,
            "success": True
        }
        
    except Exception as e:
        if video_logger:
            video_logger.error(f"Error getting LLM response: {e}")
        
        # Return fallback result
        fallback_answer = random.randint(0, 4)
        try:
            truth_int = int(truth) if truth is not None else -1
        except (ValueError, TypeError):
            truth_int = -1
        
        if video_logger:
            video_logger.warning(f"Using fallback random answer: {fallback_answer}")
            video_logger.info(f"=== Result: {'CORRECT' if fallback_answer == truth_int else 'WRONG'} (fallback) ===")
        
        return {
            "video_id": video_id,
            "question": question,
            "choices": choices,
            "predicted_answer": fallback_answer,
            "truth": truth_int,
            "is_correct": fallback_answer == truth_int,
            "is_valid": False,
            "num_frames_used": len(selected_frames) if 'selected_frames' in dir() else num_frames,
            "error": str(e),
            "success": True
        }


class DirectFrameEvaluator:
    """Evaluate model by showing first N frames directly without captions."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = config.get("model", "x-ai/grok-4-1-fast-non-reasoning")
        self.num_frames = config.get("num_frames", 5)
        self.video_dir = config.get("video_dir", "data/EgoSchema_test/videos")
        self.annotation_file = config.get("annotation_file", "data/EgoSchema_test/annotations.json")
        self.video_list_file = config.get("video_list_file", "data/EgoSchema_test/video_list.txt")
        self.max_videos = config.get("max_videos", -1)
        self.output_dir = config.get("output_dir", "results")
        self.experiment_name = config.get("experiment_name", "direct_frame_eval")
        self.max_processes = config.get("max_processes", 1)
        self.enable_logging = config.get("enable_logging", True)
    
    def run(self) -> str:
        """Run evaluation and return output directory path."""
        # Create output directory
        timestamp = datetime.now().strftime("%m%d")
        output_name = f"{self.experiment_name}__{self.model.replace('/', '_')}_frames_{self.num_frames}_videos_{self.max_videos}_{timestamp}"
        output_dir = os.path.join(self.output_dir, output_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # Setup logging
        logger = setup_logger(
            name="DirectFrameEval",
            output_dir=output_dir,
            level="INFO"
        )
        
        logger.info(f"Starting Direct Frame Evaluation")
        logger.info(f"Model: {self.model}")
        logger.info(f"Num Frames: {self.num_frames}")
        logger.info(f"Max Videos: {self.max_videos}")
        logger.info(f"Max Processes: {self.max_processes}")
        logger.info(f"Enable Logging: {self.enable_logging}")
        
        # Load annotations
        videos_info = parse_video_annotation(
            annotation_file=self.annotation_file,
            video_list_file=self.video_list_file,
            max_videos=self.max_videos
        )
        
        logger.info(f"Loaded {len(videos_info)} videos")
        
        # Process videos
        if self.max_processes > 1:
            logger.info(f"Using multiprocessing with {self.max_processes} workers")
            results, stats = self._run_multiprocess(videos_info, output_dir, logger)
        else:
            logger.info("Using single process mode")
            results, stats = self._run_sequential(videos_info, output_dir, logger)
        
        # Calculate final statistics
        valid_count = stats["total"]
        accuracy = stats["correct"] / valid_count if valid_count > 0 else 0
        
        stats["accuracy"] = accuracy
        stats["valid"] = valid_count
        
        logger.info(f"Evaluation completed")
        logger.info(f"Accuracy: {accuracy:.2%} ({stats['correct']}/{valid_count})")
        
        # Save results
        self._save_results(results, stats, output_dir)
        
        return output_dir
    
    def _run_sequential(self, videos_info: List[Dict], output_dir: str, logger: logging.Logger):
        """Run evaluation sequentially."""
        results = []
        stats = {
            "total": 0,
            "correct": 0,
            "invalid": 0,
            "fallback_used": 0
        }
        
        # Prepare worker config
        worker_config = {
            "model": self.model,
            "num_frames": self.num_frames,
            "video_dir": self.video_dir,
            "enable_logging": self.enable_logging
        }
        
        pbar = tqdm(videos_info, desc="Evaluating", unit="video")
        
        for video_info in pbar:
            video_id = video_info["video_id"]
            
            # Process video using the worker function for consistency
            result = _process_video_worker((video_info, worker_config, output_dir))
            
            if result.get("success", False):
                results.append(result)
                stats["total"] += 1
                
                if result.get("is_valid", True):
                    if result["is_correct"]:
                        stats["correct"] += 1
                else:
                    stats["fallback_used"] += 1
                    if result["is_correct"]:
                        stats["correct"] += 1
                
                # Log result
                status = "CORRECT" if result["is_correct"] else "WRONG"
                logger.info(f"[{status}] {video_id}: pred={result['predicted_answer']}, truth={result['truth']}, valid={result['is_valid']}")
            else:
                stats["invalid"] += 1
                logger.warning(f"Failed to process video: {video_id} - {result.get('error', 'Unknown error')}")
            
            # Update progress bar
            acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
            pbar.set_postfix({"Acc": f"{acc:.1%}", "Correct": stats["correct"]})
        
        pbar.close()
        return results, stats
    
    def _run_multiprocess(self, videos_info: List[Dict], output_dir: str, logger: logging.Logger):
        """Run evaluation with multiprocessing."""
        results = []
        stats = {
            "total": 0,
            "correct": 0,
            "invalid": 0,
            "fallback_used": 0
        }
        
        # Prepare worker config
        worker_config = {
            "model": self.model,
            "num_frames": self.num_frames,
            "video_dir": self.video_dir,
            "enable_logging": self.enable_logging
        }
        
        # Prepare arguments
        worker_args = [(video_info, worker_config, output_dir) for video_info in videos_info]
        
        # Use multiprocessing pool
        with mp.Pool(processes=self.max_processes) as pool:
            pbar = tqdm(total=len(videos_info), desc="Evaluating", unit="video")
            
            for result in pool.imap_unordered(_process_video_worker, worker_args):
                if result.get("success", False):
                    results.append(result)
                    stats["total"] += 1
                    
                    if result.get("is_valid", True):
                        if result["is_correct"]:
                            stats["correct"] += 1
                    else:
                        stats["fallback_used"] += 1
                        if result["is_correct"]:
                            stats["correct"] += 1
                    
                    # Log result
                    status = "CORRECT" if result["is_correct"] else "WRONG"
                    logger.info(f"[{status}] {result['video_id']}: pred={result['predicted_answer']}, truth={result['truth']}, valid={result['is_valid']}")
                else:
                    stats["invalid"] += 1
                    logger.warning(f"Failed to process video: {result.get('video_id', 'unknown')} - {result.get('error', 'Unknown error')}")
                
                # Update progress bar
                acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
                pbar.update(1)
                pbar.set_postfix({"Acc": f"{acc:.1%}", "Correct": stats["correct"]})
            
            pbar.close()
        
        return results, stats
    
    def _save_results(self, results: List[Dict], stats: Dict, output_dir: str):
        """Save all result files."""
        
        # 1. Save result.json
        result_output = {
            "_summary": stats,
            "results": {r["video_id"]: {
                "question": r.get("question", ""),
                "choices": r.get("choices", []),
                "predicted_answer": r.get("predicted_answer"),
                "truth": r.get("truth"),
                "is_correct": r.get("is_correct"),
                "is_valid": r.get("is_valid"),
                "num_frames_used": r.get("num_frames_used"),
                "total_video_frames": r.get("total_video_frames", 0)
            } for r in results}
        }
        with open(os.path.join(output_dir, "result.json"), "w") as f:
            json.dump(result_output, f, indent=2)
        
        # 2. Save metrics.csv
        with open(os.path.join(output_dir, "metrics.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            writer.writerow(["total_videos", stats["total"]])
            writer.writerow(["valid_videos", stats["valid"]])
            writer.writerow(["invalid_videos", stats["invalid"]])
            writer.writerow(["fallback_used", stats["fallback_used"]])
            writer.writerow(["correct_answers", stats["correct"]])
            writer.writerow(["accuracy", f"{stats['accuracy']:.4f}"])
            writer.writerow(["num_frames", self.num_frames])
            writer.writerow(["max_processes", self.max_processes])
            writer.writerow(["model", self.model])
        
        # 3. Save summary.txt
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(os.path.join(output_dir, "summary.txt"), "w") as f:
            f.write("=" * 70 + "\n")
            f.write("Direct Frame Evaluation Summary\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Output Directory: {output_dir}\n\n")
            f.write(f"Model: {self.model}\n")
            f.write(f"Frames Used: {self.num_frames} (first frames only)\n")
            f.write(f"Max Processes: {self.max_processes}\n\n")
            
            f.write("-" * 70 + "\n")
            f.write("Results\n")
            f.write("-" * 70 + "\n")
            f.write(f"Total Videos:       {stats['total']}\n")
            f.write(f"Valid Videos:       {stats['valid']}\n")
            f.write(f"Invalid Videos:     {stats['invalid']}\n")
            f.write(f"Fallback Used:      {stats['fallback_used']}\n")
            f.write(f"Correct Answers:    {stats['correct']}\n")
            f.write(f"Accuracy:           {stats['accuracy']:.2%} ({stats['correct']}/{stats['valid']})\n\n")
            
            f.write("-" * 70 + "\n")
            f.write("Method Description\n")
            f.write("-" * 70 + "\n")
            f.write(f"This evaluation uses only the first {self.num_frames} frames from each video.\n")
            f.write("No captions are generated - the model sees raw images directly.\n")
            f.write("This tests the model's ability to understand video content from\n")
            f.write("limited visual information without any text-based intermediaries.\n\n")
            f.write("The model is provided with:\n")
            f.write("- First N frames from the video\n")
            f.write("- The multiple choice question\n")
            f.write("- Instructions using #C/#O notation for camera wearer actions\n\n")
            
            f.write("=" * 70 + "\n")
        
        # 4. Save accuracy.txt (backward compatible)
        with open(os.path.join(output_dir, "accuracy.txt"), "w") as f:
            f.write(f"number_videos: {stats['total']}\n")
            f.write(f"mean_accuracy: {stats['accuracy']}\n")
            f.write(f"num_frames: {self.num_frames}\n")
            f.write(f"max_processes: {self.max_processes}\n")
            f.write(f"invalid_videos: {stats['invalid']}\n")
            f.write(f"fallback_videos: {stats['fallback_used']}\n")
            f.write(f"model: {self.model}\n")
        
        # 5. Save config
        config_output = {
            "model": self.model,
            "num_frames": self.num_frames,
            "max_processes": self.max_processes,
            "video_dir": self.video_dir,
            "annotation_file": self.annotation_file,
            "video_list_file": self.video_list_file,
            "max_videos": self.max_videos,
            "experiment_name": self.experiment_name,
            "enable_logging": self.enable_logging
        }
        with open(os.path.join(output_dir, "experiment_config.yaml"), "w") as f:
            for key, value in config_output.items():
                f.write(f"{key}: {value}\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Direct Frame Evaluation")
    
    parser.add_argument("--model", default="x-ai/grok-4-1-fast-non-reasoning",
                       help="Model to use for evaluation")
    parser.add_argument("--num-frames", type=int, default=5,
                       help="Number of initial frames to show (default: 5)")
    parser.add_argument("--max-videos", type=int, default=100,
                       help="Maximum number of videos to evaluate (default: 100)")
    parser.add_argument("--max-processes", type=int, default=1,
                       help="Number of parallel processes (default: 1)")
    parser.add_argument("--experiment-name", default="direct_frame_eval",
                       help="Experiment name for output directory")
    parser.add_argument("--video-dir", default="data/EgoSchema_test/videos",
                       help="Path to video directory")
    parser.add_argument("--annotation-file", default="data/EgoSchema_test/annotations.json",
                       help="Path to annotation file")
    parser.add_argument("--video-list", default="data/EgoSchema_test/video_list.txt",
                       help="Path to video list file")
    parser.add_argument("--output-dir", default="results",
                       help="Output directory for results")
    parser.add_argument("--no-logging", action="store_true",
                       help="Disable per-video logging")
    
    args = parser.parse_args()
    
    config = {
        "model": args.model,
        "num_frames": args.num_frames,
        "max_videos": args.max_videos,
        "max_processes": args.max_processes,
        "experiment_name": args.experiment_name,
        "video_dir": args.video_dir,
        "annotation_file": args.annotation_file,
        "video_list_file": args.video_list,
        "output_dir": args.output_dir,
        "enable_logging": not args.no_logging
    }
    
    # Print header
    print("=" * 60)
    print("Direct Frame Evaluation")
    print("=" * 60)
    print(f"Model:         {config['model']}")
    print(f"Num Frames:    {config['num_frames']}")
    print(f"Max Videos:    {config['max_videos']}")
    print(f"Max Processes: {config['max_processes']}")
    print(f"Logging:       {'Enabled' if config['enable_logging'] else 'Disabled'}")
    print("=" * 60)
    print()
    
    evaluator = DirectFrameEvaluator(config)
    output_dir = evaluator.run()
    
    # Print results
    print()
    print(f"Results saved to: {output_dir}")
    
    summary_file = os.path.join(output_dir, "summary.txt")
    if os.path.exists(summary_file):
        print()
        with open(summary_file, "r") as f:
            print(f.read())
    
    return 0


if __name__ == "__main__":
    exit(main())
