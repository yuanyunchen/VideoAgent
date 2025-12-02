"""
VideoAgent - Main orchestrator for video analysis experiments.
"""

import os
import json
import csv
import logging
import random
import multiprocessing as mp
from datetime import datetime
from typing import Dict, Any, Optional, List
from collections import Counter
from tqdm import tqdm

from video_agent.core.video_memory import VideoMemory
from video_agent.processors.caption_processor import CaptionProcessor
from video_agent.processors.question_processor import QuestionProcessor
from video_agent.utils.cache import CacheManager
from video_agent.utils.logging_utils import setup_logger
from video_agent.utils.parsing import parse_video_annotation, parse_text_find_number
from video_agent.utils.video import get_video_frames
from video_agent.utils.config import save_config_to_output


class EvalStatistics:
    """Track evaluation statistics during experiment."""
    
    def __init__(self):
        self.total = 0
        self.correct = 0
        self.first_round_correct = 0
        self.improved_cases = 0  # Wrong -> Correct after more rounds
        self.degraded_cases = 0  # Correct -> Wrong after more rounds (shouldn't happen often)
        self.failed_cases = 0    # Wrong -> Wrong
        self.maintained_cases = 0  # Correct -> Correct
        self.total_rounds = 0
        self.total_frames = 0
        self.invalid_videos = 0  # Now tracks videos that used fallback
        self.fallback_videos = 0  # Videos that used fallback mechanism
        self.confidence_distribution = Counter()
        self.rounds_distribution = Counter()
        self.case_types = Counter()
    
    def update(self, result: Dict):
        """Update statistics with a new result."""
        self.total += 1
        
        predicted = result.get("predicted_answer", "-1")
        truth = result.get("truth")
        rounds = result.get("rounds", 1)
        frame_count = result.get("frame_count", 0)
        answers_list = result.get("answers", [])
        confidence_list = result.get("confidence", [])
        is_valid_flag = result.get("is_valid", True)  # New validity flag
        
        # Handle parsing
        try:
            predicted_int = int(predicted) if str(predicted).lstrip('-').isdigit() else -1
            label = int(truth) if truth is not None and str(truth).isdigit() else -1
        except (ValueError, AttributeError):
            predicted_int = -1
            label = -1
        
        # Track fallback usage
        if not is_valid_flag:
            self.fallback_videos += 1
        
        # With fallback mechanism, predicted should never be -1
        # But keep backward compatibility check
        has_valid_prediction = predicted_int != -1
        is_correct = predicted_int == label and has_valid_prediction
        
        if not has_valid_prediction:
            self.invalid_videos += 1
            self.case_types["INVALID"] += 1
            return
        
        # Track correct answers
        if is_correct:
            self.correct += 1
        
        # Track rounds and frames
        self.total_rounds += rounds
        self.total_frames += frame_count
        self.rounds_distribution[rounds] += 1
        
        # Track confidence
        if confidence_list:
            final_confidence = confidence_list[-1] if isinstance(confidence_list, list) else confidence_list
            self.confidence_distribution[final_confidence] += 1
        
        # Determine case type based on first vs final answer
        first_correct = False
        if answers_list and len(answers_list) > 0:
            first_answer = answers_list[0]
            try:
                first_answer_int = int(first_answer) if str(first_answer).lstrip('-').isdigit() else -1
                first_correct = first_answer_int == label and first_answer_int != -1
            except (ValueError, AttributeError):
                first_correct = False
        
        if first_correct and is_correct:
            self.first_round_correct += 1
            self.maintained_cases += 1
            self.case_types["MAINTAINED"] += 1
        elif first_correct and not is_correct:
            self.first_round_correct += 1  # First round was correct, even if final is wrong
            self.degraded_cases += 1
            self.case_types["DEGRADED"] += 1
        elif not first_correct and is_correct:
            self.improved_cases += 1
            self.case_types["IMPROVED"] += 1
        else:
            self.failed_cases += 1
            self.case_types["FAILED"] += 1
    
    def get_accuracy(self) -> float:
        """Get current accuracy."""
        valid = self.total - self.invalid_videos
        return self.correct / valid if valid > 0 else 0.0
    
    def get_first_round_accuracy(self) -> float:
        """Get first round accuracy."""
        valid = self.total - self.invalid_videos
        return self.first_round_correct / valid if valid > 0 else 0.0
    
    def get_improvement_rate(self) -> float:
        """Get improvement rate (improved / initially wrong)."""
        initially_wrong = self.improved_cases + self.failed_cases
        return self.improved_cases / initially_wrong if initially_wrong > 0 else 0.0
    
    def get_avg_rounds(self) -> float:
        """Get average rounds per video."""
        valid = self.total - self.invalid_videos
        return self.total_rounds / valid if valid > 0 else 0.0
    
    def get_avg_frames(self) -> float:
        """Get average frames per video."""
        valid = self.total - self.invalid_videos
        return self.total_frames / valid if valid > 0 else 0.0
    
    def get_progress_dict(self) -> Dict[str, str]:
        """Get dictionary for progress bar postfix."""
        acc = self.get_accuracy()
        first_acc = self.get_first_round_accuracy()
        valid_rate = (self.total - self.invalid_videos) / self.total if self.total > 0 else 1.0
        
        return {
            'Acc': f'{acc:.1%}',
            'First': f'{first_acc:.1%}',
            'Imp': f'{self.improved_cases}',
            'Deg': f'{self.degraded_cases}',
            'Valid': f'{valid_rate:.1%}'
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        valid = self.total - self.invalid_videos
        return {
            "total": self.total,
            "valid": valid,
            "invalid": self.invalid_videos,
            "fallback_used": self.fallback_videos,
            "correct": self.correct,
            "accuracy": self.get_accuracy(),
            "first_round_correct": self.first_round_correct,
            "first_round_accuracy": self.get_first_round_accuracy(),
            "improved_cases": self.improved_cases,
            "degraded_cases": self.degraded_cases,
            "failed_cases": self.failed_cases,
            "maintained_cases": self.maintained_cases,
            "improvement_rate": self.get_improvement_rate(),
            "avg_rounds": self.get_avg_rounds(),
            "avg_frames": self.get_avg_frames(),
            "total_rounds": self.total_rounds,
            "total_frames": self.total_frames,
            "confidence_distribution": dict(self.confidence_distribution),
            "rounds_distribution": dict(self.rounds_distribution),
            "case_types": dict(self.case_types)
        }


def _process_video_worker(video_info: Dict, config: Dict[str, Any], output_dir: str, video_index: int) -> Dict:
    """
    Worker function for multiprocessing video processing.
    
    Args:
        video_info: Video information dictionary
        config: Configuration dictionary
        output_dir: Output directory path
        video_index: Index of the video being processed
        
    Returns:
        Result dictionary
    """
    try:
        # Create a local VideoAgent instance for this process
        agent = VideoAgent(config)
        
        # Create a minimal logger for this process
        logger = logging.getLogger(f"VideoAgent.Worker.{video_index}")
        logger.setLevel(logging.INFO)
        
        # Create logs dictionary for this process
        logs = {}
        
        # Process the video
        result = agent._process_video(video_info, output_dir, logger, logs)
        
        return {
            "success": True,
            "result": result,
            "video_index": video_index,
            "video_id": video_info['video_id']
        }
        
    except Exception as e:
        # Return error result with fallback random answer
        fallback_answer = random.randint(0, 4)
        return {
            "success": False,
            "error": str(e),
            "video_index": video_index,
            "video_id": video_info['video_id'],
            "result": {
                "video_id": video_info['video_id'],
                "question": video_info.get("question", ""),
                "predicted_answer": str(fallback_answer),
                "actual_answer": video_info.get("answer", ""),
                "confidence": [random.randint(1, 3)],
                "answers": [fallback_answer],
                "rounds": 0,
                "frame_count": 0,
                "truth": video_info.get("truth"),
                "rounds_history": [],
                "is_valid": False,
            }
        }


def _process_video_worker_unpack(args):
    """Unpack arguments for _process_video_worker to work with pool.imap."""
    return _process_video_worker(*args)


class VideoAgent:
    """Main VideoAgent class for coordinating video analysis experiments."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize VideoAgent with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Initialize processors
        self.caption_processor = CaptionProcessor(config)
        self.question_processor = QuestionProcessor(config)
        
        # Initialize cache manager
        self.cache_manager = CacheManager(
            llm_cache_file=config.get("cache_llm_file", "cache/cache_llm.pkl"),
            clip_cache_file=config.get("cache_clip_file", "cache/cache_clip.pkl"),
            use_cache=config.get("use_cache", True)
        )
        
        # Set max_processes (default 1 for single-process mode)
        self.max_processes = config.get("max_processes", 1)
        
        # Handle backward compatibility with multi_process parameter
        if config.get("multi_process", False) and self.max_processes == 1:
            # If multi_process is True but max_processes is 1, use CPU count
            self.max_processes = min(mp.cpu_count(), 4)  # Cap at 4 to avoid overwhelming the system
        
        # Ensure max_processes is valid
        self.max_processes = max(1, min(self.max_processes, mp.cpu_count()))
    
    def run_experiment(self, video_list_file: Optional[str] = None) -> str:
        """
        Run video analysis experiment.
        
        Args:
            video_list_file: Optional path to video list file
            
        Returns:
            Output directory path
        """
        # Setup experiment
        experiment_name = self.config.get("experiment_name", "default_experiment")
        max_test_videos = self.config.get("max_test_videos", -1)
        
        # Generate output directory with expected format
        scheduler_model = self.config.get("scheduler_model", "gpt-4o-mini-2024-07-18")
        viewer_model = self.config.get("viewer_model", "gpt-4o-mini-2024-07-18")
        model_info = f"{os.path.basename(scheduler_model)}_viewer_{os.path.basename(viewer_model)}_numbers_{max_test_videos}"
        output_dir = os.path.join(
            self.config.get("output_dir", "results"),
            f"{experiment_name}__{model_info}"
        )
        os.makedirs(output_dir, exist_ok=True)
        
        # Save configuration to output directory
        save_config_to_output(self.config, output_dir)
        
        # Setup logging
        logger = setup_logger(
            name="VideoAgent",
            output_dir=output_dir,
            level=self.config.get("log_level", "INFO"),
            enable_llm_logging=self.config.get("enable_llm_logging", False)
        )
        
        # Load video annotations
        annotation_file = self.config.get("annotation_file", "data/EgoSchema_test/annotations.json")
        video_list = video_list_file or self.config.get("test_video_list_file")
        
        videos_info = parse_video_annotation(
            annotation_file=annotation_file,
            video_list_file=video_list,
            max_videos=max_test_videos
        )
        
        logger.info(f"Starting experiment: {experiment_name}")
        logger.info(f"Processing {len(videos_info)} videos")
        logger.info(f"Configuration saved to: {os.path.join(output_dir, 'experiment_config.yaml')}")
        
        # Process videos with progress bar
        results = []
        logs = {}
        
        if self.max_processes > 1:
            # Use multiprocessing
            logger.info(f"Using multiprocessing with {self.max_processes} processes")
            logger.info("Note: Progress updates may be less frequent in multiprocessing mode")
            results = self._process_videos_multiprocessing(videos_info, output_dir, logger)
        else:
            # Use single process (original behavior)
            logger.info("Using single process mode")
            results = self._process_videos_sequential(videos_info, output_dir, logger, logs)
        
        # Filter out None results
        results = [r for r in results if r is not None]
        
        # Generate global result files
        self._save_global_results(results, output_dir)
        
        logger.info(f"Experiment completed. Results in: {output_dir}")
        return output_dir
    
    def _process_videos_sequential(self, videos_info: List[Dict], output_dir: str, logger: logging.Logger, logs: Dict) -> List[Dict]:
        """Process videos sequentially with detailed statistics tracking."""
        results = []
        stats = EvalStatistics()
        
        # Create progress bar with detailed format
        pbar = tqdm(
            videos_info, 
            desc="Processing", 
            unit="video",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}"
        )
        
        for i, video_info in enumerate(pbar):
            video_id = video_info['video_id']
            
            logger.info(f"Processing video {i+1}/{len(videos_info)}: {video_id}")
            result = self._process_video(video_info, output_dir, logger, logs)
            
            if result:
                results.append(result)
                stats.update(result)
                
                # Log detailed result
                predicted = result.get('predicted_answer', 'N/A')
                truth = result.get('truth', 'N/A')
                rounds = result.get('rounds', 1)
                status = "OK" if str(predicted) == str(truth) else "WRONG"
                logger.info(f"Completed [{status}] video {i+1}: {video_id} | Pred={predicted} Truth={truth} Rounds={rounds}")
            else:
                logger.warning(f"No result for video {i+1}: {video_id}")
            
            # Update progress bar with comprehensive stats
            pbar.set_postfix(stats.get_progress_dict())
        
        pbar.close()
        
        # Store statistics for later use
        self._eval_stats = stats
        
        return results
    
    def _process_videos_multiprocessing(self, videos_info: List[Dict], output_dir: str, logger: logging.Logger) -> List[Dict]:
        """Process videos using multiprocessing with statistics tracking."""
        results = []
        stats = EvalStatistics()
        
        logger.info(f"Starting multiprocessing with {self.max_processes} processes")
        
        # Prepare arguments for each video
        video_args = [(video_info, self.config, output_dir, i) for i, video_info in enumerate(videos_info)]
        
        # Use multiprocessing with progress bar
        with mp.Pool(processes=self.max_processes) as pool:
            # Create progress bar
            pbar = tqdm(
                total=len(videos_info), 
                desc="Processing", 
                unit="video",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}"
            )
            
            # Process videos in parallel using imap_unordered for better throughput
            try:
                # Use imap_unordered to get results as soon as they complete (not in order)
                worker_results_iter = pool.imap_unordered(_process_video_worker_unpack, video_args)
                
                # Process results as they complete
                for i, worker_result in enumerate(worker_results_iter):
                    if worker_result["success"]:
                        result = worker_result["result"]
                        results.append(result)
                        stats.update(result)
                        logger.info(f"Completed video {i+1}/{len(videos_info)}: {worker_result['video_id']}")
                    else:
                        logger.error(f"Error processing video {worker_result['video_id']}: {worker_result['error']}")
                        result = worker_result["result"]
                        results.append(result)
                        stats.update(result)  # Also update stats for failed videos (with fallback)
                    
                    # Update progress bar with comprehensive stats
                    pbar.update(1)
                    pbar.set_postfix(stats.get_progress_dict())
                    
            except Exception as e:
                logger.error(f"Error in multiprocessing: {e}")
                pbar.close()
                raise
            
            pbar.close()
        
        # Store statistics for later use
        self._eval_stats = stats
        
        return results
    
    def _process_video(self, video_info: Dict, output_dir: str, logger: logging.Logger, logs: Dict) -> Dict:
        """Process a single video following original main.py workflow."""
        video_id = video_info["video_id"]
        video_output_dir = os.path.join(output_dir, "videos", video_id)
        os.makedirs(video_output_dir, exist_ok=True)
        
        # Create individual video logger if enabled
        video_logger = None
        if self.config.get("logging_process", True):
            video_logger = logging.getLogger(f"VideoAgent.Video.{video_id}")
            video_logger.setLevel(logging.INFO)
            video_logger.handlers.clear()
            
            video_log_file = os.path.join(video_output_dir, "logging.log")
            video_handler = logging.FileHandler(video_log_file)
            video_handler.setLevel(logging.INFO)
            video_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            video_handler.setFormatter(video_formatter)
            video_logger.addHandler(video_handler)
            
            video_logger.info(f"=== Starting video processing: {video_id} ===")
            video_logger.info(f"Question: {video_info['question']}")
        
        # Extract multiple choice options if available
        choices = []
        for i in range(5):
            option_key = f"option {i}"
            if option_key in video_info:
                choices.append(video_info[option_key])
        
        # Load video frames
        video_dir = self.config.get("video_dir", "data/EgoSchema_test/videos")
        video_path = os.path.join(video_dir, f"{video_id}.mp4")
        
        # Create video frame tuples for backward compatibility
        try:
            original_video_frames = get_video_frames(video_path, interval=30)
            if original_video_frames is None:
                logger.error(f"Cannot open {video_path}")
                return {}
            video_frame_tups = [(i * 30, original_video_frames[i]) for i in range(len(original_video_frames))]
        except Exception as e:
            logger.error(f"Error loading video frames: {e}")
            return {}
        
        # Initialize video memory
        video_memory = VideoMemory(
            video_id=video_id,
            question=video_info["question"],
            answer=video_info.get("answer", ""),
            choices=choices if choices else [],
            truth=video_info.get("truth"),
            video_frame_tups=video_frame_tups
        )
        
        if video_logger:
            video_logger.info(f"Video loaded: {video_memory.n_frames} total frames")
            video_logger.info(f"Initial sampled frames: {video_memory.sampled_idx}")
        
        # Format question for processing
        formatted_question = video_memory.question
        if choices:
            formatted_question = (
                f"Here is the question: \n{video_memory.question}\n\n"
                + "Here are the choices: \n"
                + "\n".join([f"{i}. {choice}" for i, choice in enumerate(choices)])
            )
        
        # Multi-round processing following original main.py workflow
        max_rounds = self.config.get("max_rounds", 1)
        confidence = 0
        round_num = 0
        answer = -1
        
        # Initialize lists to collect answers and confidence for each round
        answers_list = []
        confidence_list = []
        
        # Track validity: True if result is from normal parsing, False if from fallback
        is_valid_answer = True
        is_valid_confidence = True
        
        # Initial frames are already sampled in VideoMemory.__init__
        new_sampled_index = list(video_memory.sampled_idx)
        
        while round_num < max_rounds and confidence < 3:
            if video_logger:
                video_logger.info(f"=== Starting Round {round_num + 1} ===")
                video_logger.info(f"Processing frames: {new_sampled_index}")
            
            # Step 1: Generate captions (only for new frames)
            caption_response = self.caption_processor._write_multi_level_captions(
                video_memory, new_sampled_index, video_logger
            )
            
            if video_logger:
                video_logger.info(f"=== Captioning ===")
                video_logger.info(f"LLM response:\n {caption_response}")
                video_logger.info(f"Video Memory:\n{video_memory}")
            
            # Step 2: Answer question
            answer_str = self.question_processor.answer_question(video_memory, video_logger)
            answer = parse_text_find_number(answer_str, "final_answer")
            
            if video_logger:
                video_logger.info(f"=== Answer === LLM response:")
                video_logger.info(f"'{answer_str}'")
                video_logger.info(f"Answer: {answer}")
            
            # If answer is -1 (error), apply fallback mechanism
            if answer == -1:
                is_valid_answer = False
                # Try to get previous valid answer
                valid_answers = [a for a in answers_list if a != -1]
                if valid_answers:
                    answer = valid_answers[-1]
                    if video_logger:
                        video_logger.warning(f"Answer parsing failed, falling back to previous answer: {answer}")
                else:
                    # No previous valid answer, use random
                    answer = random.randint(0, 4)
                    if video_logger:
                        video_logger.warning(f"Answer parsing failed, no previous valid answer, using random: {answer}")
            
            # Collect answer for this round
            answers_list.append(answer)
            
            # Step 3: Evaluate confidence
            confidence_str = self.question_processor.evaluate_confidence(
                video_memory, video_memory.question, answer_str, video_logger
            )
            confidence = parse_text_find_number(confidence_str, "confidence")
            
            if video_logger:
                video_logger.info(f"=== Self Evaluation ===")
                video_logger.info(f"LLM response:\n '{confidence_str}'")
                video_logger.info(f"Confidence: {confidence}")
            
            # If confidence is -1 (error), apply fallback mechanism
            if confidence == -1:
                is_valid_confidence = False
                # Try to get previous valid confidence
                valid_confidences = [c for c in confidence_list if c != -1]
                if valid_confidences:
                    confidence = valid_confidences[-1]
                    if video_logger:
                        video_logger.warning(f"Confidence parsing failed, falling back to previous confidence: {confidence}")
                else:
                    # No previous valid confidence, use random (1-3)
                    confidence = random.randint(1, 3)
                    if video_logger:
                        video_logger.warning(f"Confidence parsing failed, no previous valid confidence, using random: {confidence}")
            
            # Collect confidence for this round
            confidence_list.append(confidence)
            
            # Step 4: If confidence < 3, generate new segments for next round
            if confidence < 3 and round_num < max_rounds - 1:
                new_sampled_index, segment_response = self.question_processor.generate_segment_steps(
                    video_memory, formatted_question, video_logger
                )
                
                if video_logger:
                    video_logger.info(f"=== Generate Segment Steps ===")
                    video_logger.info(f"LLM response:\n '{segment_response}'")
                    video_logger.info(f"new_sampled_idx: {new_sampled_index}")
            
            round_num += 1
        
        # Determine overall validity
        is_valid = is_valid_answer and is_valid_confidence
        
        # Final results
        video_memory.predicted_answer = str(answer)
        video_memory.confidence = confidence
        
        if video_logger:
            video_logger.info(f"=== Video processing completed after {round_num} rounds ===")
            video_logger.info(f"Final answer: {video_memory.predicted_answer}")
            video_logger.info(f"Final confidence: {video_memory.confidence}")
            video_logger.info(f"Is valid (no fallback used): {is_valid}")
        
        # Backward compatibility for logs
        label = int(video_info.get("truth", 0)) if video_info.get("truth") else 0
        # Mark as correct if answer matches the truth
        corr = int(label == answer)
        count_frame = len(video_memory.sampled_idx)
        
        logs[video_id] = {
            "final_answer": answer,
            "answers": answers_list,
            "confidence": confidence_list,
            "rounds": round_num,
            "label": label,
            "corr": corr,
            "count_frame": count_frame,
            "is_valid": is_valid,
        }
        
        # Save video results
        video_memory.save_to_directory(video_output_dir)
        
        # Save formatted question (already handled in video_memory.save_to_directory)
        # But keep for backward compatibility
        question_file = os.path.join(video_output_dir, "question.txt")
        with open(question_file, "w") as f:
            f.write(formatted_question)
        
        # Save individual result
        result_file = os.path.join(video_output_dir, "result.json")
        with open(result_file, "w") as f:
            json.dump(logs[video_id], f)
        
        logger.info(f"Finished video: {video_id}/{answer}/{video_info.get('truth', 'N/A')} (valid={is_valid})")
        
        return {
            "video_id": video_id,
            "question": video_memory.question,
            "predicted_answer": video_memory.predicted_answer,
            "actual_answer": video_memory.answer,
            "confidence": confidence_list,
            "answers": answers_list,
            "rounds": round_num,
            "frame_count": len(video_memory.sampled_idx),
            "truth": video_memory.truth,
            "rounds_history": video_memory.rounds_history,
            "is_valid": is_valid,
        }
    
    def _save_global_results(self, results: List[Dict], output_dir: str):
        """Save global result files with comprehensive output formats."""
        if not results:
            return
        
        # Use stored statistics or compute new ones
        if hasattr(self, '_eval_stats'):
            stats = self._eval_stats
        else:
            stats = EvalStatistics()
            for result in results:
                stats.update(result)
        
        # Prepare data for global results
        video_results = {}
        
        for result in results:
            video_id = result.get("video_id", "unknown")
            predicted = result.get("predicted_answer", "")
            truth = result.get("truth")
            frame_count = result.get("frame_count", 0)
            confidence = result.get("confidence", 1)
            rounds = result.get("rounds", 1)
            question = result.get("question", "")
            answers_list = result.get("answers", [])
            
            # Convert predicted answer
            try:
                predicted_int = int(predicted) if str(predicted).lstrip('-').isdigit() else -1
                label = int(truth) if truth is not None and str(truth).isdigit() else -1
            except (ValueError, AttributeError):
                predicted_int = -1
                label = -1
            
            is_correct = 1 if predicted_int == label and predicted_int != -1 else 0
            
            # Determine case type
            first_correct = False
            if answers_list and len(answers_list) > 0:
                try:
                    first_answer_int = int(answers_list[0]) if str(answers_list[0]).lstrip('-').isdigit() else -1
                    first_correct = first_answer_int == label and first_answer_int != -1
                except (ValueError, AttributeError):
                    first_correct = False
            
            if predicted_int == -1:
                case_type = "INVALID"
            elif first_correct and is_correct:
                case_type = "MAINTAINED"
            elif first_correct and not is_correct:
                case_type = "DEGRADED"
            elif not first_correct and is_correct:
                case_type = "IMPROVED"
            else:
                case_type = "FAILED"
            
            # Get validity flag (defaults to True for backward compatibility)
            is_valid_flag = result.get("is_valid", True)
            
            # Format for result.json (merged format with question)
            video_results[video_id] = {
                "question": question,
                "final_answer": predicted_int,
                "answers": answers_list,
                "confidence": confidence if isinstance(confidence, list) else [confidence],
                "rounds": rounds,
                "label": label,
                "corr": is_correct,
                "count_frame": frame_count,
                "case_type": case_type,
                "is_valid": is_valid_flag,
            }
        
        # Get stats dict for summary
        stats_dict = stats.to_dict()
        
        # Add summary stats to result.json
        result_output = {
            "_summary": {
                "total": stats_dict["total"],
                "valid": stats_dict["valid"],
                "invalid": stats_dict["invalid"],
                "fallback_used": stats_dict["fallback_used"],
                "correct": stats_dict["correct"],
                "accuracy": stats_dict["accuracy"],
                "first_round_correct": stats_dict["first_round_correct"],
                "first_round_accuracy": stats_dict["first_round_accuracy"],
                "improved_cases": stats_dict["improved_cases"],
                "degraded_cases": stats_dict["degraded_cases"],
                "failed_cases": stats_dict["failed_cases"],
                "maintained_cases": stats_dict["maintained_cases"],
                "improvement_rate": stats_dict["improvement_rate"],
                "avg_rounds": stats_dict["avg_rounds"],
                "avg_frames": stats_dict["avg_frames"],
                "total_rounds": stats_dict["total_rounds"],
                "total_frames": stats_dict["total_frames"],
            },
            "results": video_results
        }
        
        # 1. Save result.json (merged format with summary)
        result_file = os.path.join(output_dir, "result.json")
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result_output, f, indent=2)
        
        # 2. Save metrics.csv (comprehensive)
        metrics_file = os.path.join(output_dir, "metrics.csv")
        with open(metrics_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            writer.writerow(["total_videos", stats_dict["total"]])
            writer.writerow(["valid_videos", stats_dict["valid"]])
            writer.writerow(["invalid_videos", stats_dict["invalid"]])
            writer.writerow(["fallback_used", stats_dict["fallback_used"]])
            writer.writerow(["correct_answers", stats_dict["correct"]])
            writer.writerow(["accuracy", f"{stats_dict['accuracy']:.4f}"])
            writer.writerow(["first_round_correct", stats_dict["first_round_correct"]])
            writer.writerow(["first_round_accuracy", f"{stats_dict['first_round_accuracy']:.4f}"])
            writer.writerow(["improved_cases", stats_dict["improved_cases"]])
            writer.writerow(["degraded_cases", stats_dict["degraded_cases"]])
            writer.writerow(["failed_cases", stats_dict["failed_cases"]])
            writer.writerow(["maintained_cases", stats_dict["maintained_cases"]])
            writer.writerow(["improvement_rate", f"{stats_dict['improvement_rate']:.4f}"])
            writer.writerow(["avg_rounds", f"{stats_dict['avg_rounds']:.2f}"])
            writer.writerow(["avg_frames", f"{stats_dict['avg_frames']:.2f}"])
        
        # 3. Save summary.txt (human-readable with analysis)
        summary_file = os.path.join(output_dir, "summary.txt")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("VideoAgent Evaluation Summary\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Output Directory: {output_dir}\n\n")
            
            # Overall Results
            f.write("-" * 70 + "\n")
            f.write("Overall Results\n")
            f.write("-" * 70 + "\n")
            f.write(f"Total Videos:       {stats_dict['total']}\n")
            f.write(f"Valid Videos:       {stats_dict['valid']}\n")
            f.write(f"Invalid Videos:     {stats_dict['invalid']}\n")
            f.write(f"Fallback Used:      {stats_dict['fallback_used']}\n")
            f.write(f"Correct Answers:    {stats_dict['correct']}\n\n")
            
            # Accuracy Metrics
            f.write("-" * 70 + "\n")
            f.write("Accuracy Metrics\n")
            f.write("-" * 70 + "\n")
            f.write(f"Final Accuracy:         {stats_dict['accuracy']:.2%} ({stats_dict['correct']}/{stats_dict['valid']})\n")
            f.write(f"First Round Accuracy:   {stats_dict['first_round_accuracy']:.2%} ({stats_dict['first_round_correct']}/{stats_dict['valid']})\n")
            
            # Calculate improvement delta
            acc_delta = stats_dict['accuracy'] - stats_dict['first_round_accuracy']
            delta_sign = "+" if acc_delta >= 0 else ""
            f.write(f"Accuracy Improvement:   {delta_sign}{acc_delta:.2%}\n\n")
            
            # Case Analysis
            f.write("-" * 70 + "\n")
            f.write("Case Type Analysis\n")
            f.write("-" * 70 + "\n")
            f.write(f"MAINTAINED (Correct -> Correct):  {stats_dict['maintained_cases']}\n")
            f.write(f"IMPROVED   (Wrong -> Correct):    {stats_dict['improved_cases']}\n")
            f.write(f"DEGRADED   (Correct -> Wrong):    {stats_dict['degraded_cases']}\n")
            f.write(f"FAILED     (Wrong -> Wrong):      {stats_dict['failed_cases']}\n\n")
            
            # Improvement Analysis
            f.write("-" * 70 + "\n")
            f.write("Improvement Analysis\n")
            f.write("-" * 70 + "\n")
            initially_wrong = stats_dict['improved_cases'] + stats_dict['failed_cases']
            f.write(f"Initially Wrong:        {initially_wrong}\n")
            f.write(f"Successfully Improved:  {stats_dict['improved_cases']}\n")
            f.write(f"Improvement Rate:       {stats_dict['improvement_rate']:.2%}\n\n")
            
            # Resource Usage
            f.write("-" * 70 + "\n")
            f.write("Resource Usage\n")
            f.write("-" * 70 + "\n")
            f.write(f"Total Rounds:       {stats_dict['total_rounds']}\n")
            f.write(f"Total Frames:       {stats_dict['total_frames']}\n")
            f.write(f"Avg Rounds/Video:   {stats_dict['avg_rounds']:.2f}\n")
            f.write(f"Avg Frames/Video:   {stats_dict['avg_frames']:.1f}\n\n")
            
            # Rounds Distribution
            if stats_dict['rounds_distribution']:
                f.write("-" * 70 + "\n")
                f.write("Rounds Distribution\n")
                f.write("-" * 70 + "\n")
                for rounds, count in sorted(stats_dict['rounds_distribution'].items()):
                    pct = count / stats_dict['valid'] * 100 if stats_dict['valid'] > 0 else 0
                    f.write(f"  {rounds} round(s): {count} videos ({pct:.1f}%)\n")
                f.write("\n")
            
            # Confidence Distribution
            if stats_dict['confidence_distribution']:
                f.write("-" * 70 + "\n")
                f.write("Confidence Distribution\n")
                f.write("-" * 70 + "\n")
                for conf, count in sorted(stats_dict['confidence_distribution'].items()):
                    pct = count / stats_dict['valid'] * 100 if stats_dict['valid'] > 0 else 0
                    f.write(f"  Confidence {conf}: {count} videos ({pct:.1f}%)\n")
                f.write("\n")
            
            f.write("=" * 70 + "\n")
        
        # 4. Save accuracy.txt (backward compatible)
        accuracy_file = os.path.join(output_dir, "accuracy.txt")
        with open(accuracy_file, 'w', encoding='utf-8') as f:
            f.write(f"number_videos: {stats_dict['total']}\n")
            f.write(f"mean_accuracy: {stats_dict['accuracy']}\n")
            f.write(f"mean_frame: {stats_dict['avg_frames']}\n")
            f.write(f"mean_rounds: {stats_dict['avg_rounds']}\n")
            f.write(f"invalid_videos: {stats_dict['invalid']}\n")
            f.write(f"fallback_videos: {stats_dict['fallback_used']}\n")
            f.write(f"valid_only_accuracy: {stats_dict['accuracy']}\n")
            f.write(f"valid_only_mean_frames: {stats_dict['avg_frames']}\n")
            f.write(f"first_round_accuracy: {stats_dict['first_round_accuracy']}\n")
            f.write(f"improved_cases: {stats_dict['improved_cases']}\n")
            f.write(f"degraded_cases: {stats_dict['degraded_cases']}\n")
            f.write(f"improvement_rate: {stats_dict['improvement_rate']}\n")


# Backward compatibility functions
def run(viewer_model, scheduler_model, round_name, max_test_video_numbers=-1, max_round=1,
        viewer_function=None, llm_logging=False, multi_process=True, use_cache=True, max_processes=1):
    """Backward compatibility function to match original main.py interface."""
    
    config = {
        "viewer_model": viewer_model,
        "scheduler_model": scheduler_model,
        "experiment_name": round_name,
        "max_test_videos": max_test_video_numbers,
        "max_rounds": max_round,
        "enable_llm_logging": llm_logging,
        "use_cache": use_cache,
        "multi_process": multi_process,
        "logging_process": True,
        "max_processes": max_processes,
        # Default paths
        "annotation_file": "data/EgoSchema_test/annotations.json",
        "test_video_list_file": "data/EgoSchema_test/video_list.txt",
        "video_dir": "data/EgoSchema_test/videos",
        "output_dir": "results"
    }
    
    agent = VideoAgent(config)
    return agent.run_experiment()

