"""
VideoAgent - Main orchestrator for video analysis experiments.
"""

import os
import json
import logging
import multiprocessing as mp
from functools import partial
from typing import Dict, Any, Optional, List
from tqdm import tqdm
from core.video_memory import VideoMemory
from processors.caption_processor import CaptionProcessor
from processors.question_processor import QuestionProcessor
from utils.general import CacheManager, setup_logger, parse_video_annotation, parse_text_find_number
from utils.config import save_config_to_output


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
        # Return error result
        return {
            "success": False,
            "error": str(e),
            "video_index": video_index,
            "video_id": video_info['video_id'],
            "result": {
                "video_id": video_info['video_id'],
                "question": video_info.get("question", ""),
                "predicted_answer": "-1",
                "actual_answer": video_info.get("answer", ""),
                "confidence": [-1],
                "answers": [-1],
                "rounds": 0,
                "frame_count": 0,
                "truth": video_info.get("truth"),
                "rounds_history": []
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
            self.config.get("output_dir", "output"),
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
        annotation_file = self.config.get("annotation_file", "dataset/subset_anno.json")
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
        """Process videos sequentially (original single-process behavior)."""
        results = []
        
        # Create progress bar
        pbar = tqdm(videos_info, desc="Processing videos", unit="video", 
                   bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {desc}")
        
        for i, video_info in enumerate(pbar):
            video_id = video_info['video_id']
            
            logger.info(f"Processing video {i+1}/{len(videos_info)}: {video_id}")
            result = self._process_video(video_info, output_dir, logger, logs)
            results.append(result)
            logger.info(f"Completed video {i+1}/{len(videos_info)}: {video_id}")
            
            # Calculate accuracy and valid rate
            valid_results = [r for r in results if str(r.get('predicted_answer', '')) != '-1' and str(r.get('confidence', '')) != '-1']
            correct_count = len([r for r in valid_results if str(r.get('predicted_answer', '')) == str(r.get('truth', ''))])
            valid_processed = len(valid_results)
            accuracy = correct_count / valid_processed if valid_processed > 0 else 0.0
            valid_rate = valid_processed / (i + 1) if (i + 1) > 0 else 0.0
            
            pbar.set_postfix({"Accuracy": f"{accuracy:.2%}", "Valid": f"{valid_rate:.2%}"})
        
        pbar.close()
        return results
    
    def _process_videos_multiprocessing(self, videos_info: List[Dict], output_dir: str, logger: logging.Logger) -> List[Dict]:
        """Process videos using multiprocessing."""
        results = []
        
        logger.info(f"Starting multiprocessing with {self.max_processes} processes")
        
        # Prepare arguments for each video
        video_args = [(video_info, self.config, output_dir, i) for i, video_info in enumerate(videos_info)]
        
        # Use multiprocessing with progress bar
        with mp.Pool(processes=self.max_processes) as pool:
            # Create progress bar
            pbar = tqdm(total=len(videos_info), desc="Processing videos", unit="video",
                       bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {desc}")
            
            # Process videos in parallel using imap for better progress tracking
            try:
                # Use imap to get results as they complete
                worker_results_iter = pool.imap(_process_video_worker_unpack, video_args)
                
                # Process results as they complete
                for i, worker_result in enumerate(worker_results_iter):
                    if worker_result["success"]:
                        results.append(worker_result["result"])
                        logger.info(f"Completed video {i+1}/{len(videos_info)}: {worker_result['video_id']}")
                    else:
                        logger.error(f"Error processing video {worker_result['video_id']}: {worker_result['error']}")
                        results.append(worker_result["result"])
                    
                    # Update progress bar
                    pbar.update(1)
                    
                    # Calculate and display stats
                    valid_results = [r for r in results if str(r.get('predicted_answer', '')) != '-1' and str(r.get('confidence', '')) != '-1']
                    correct_count = len([r for r in valid_results if str(r.get('predicted_answer', '')) == str(r.get('truth', ''))])
                    valid_processed = len(valid_results)
                    accuracy = correct_count / valid_processed if valid_processed > 0 else 0.0
                    valid_rate = valid_processed / (i + 1) if (i + 1) > 0 else 0.0
                    
                    pbar.set_postfix({"Accuracy": f"{accuracy:.2%}", "Valid": f"{valid_rate:.2%}"})
                    
            except Exception as e:
                logger.error(f"Error in multiprocessing: {e}")
                pbar.close()
                raise
            
            pbar.close()
        
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
        video_path = os.path.join(self.config.get("video_dir", "dataset/videos"), f"{video_id}.mp4")
        
        # Create video frame tuples for backward compatibility
        try:
            from utils.general import get_video_frames
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
            
            # Collect answer for this round
            answers_list.append(answer)
            
            if video_logger:
                video_logger.info(f"=== Answer === LLM response:")
                video_logger.info(f"'{answer_str}'")
                video_logger.info(f"Answer: {answer}")
            
            # If answer is -1 (error), set confidence to -1 and break the loop
            if answer == -1:
                confidence = -1
                confidence_list.append(confidence)
                if video_logger:
                    video_logger.error("Error in question processing, terminating video processing")
                break
            
            # Step 3: Evaluate confidence
            confidence_str = self.question_processor.evaluate_confidence(
                video_memory, video_memory.question, answer_str, video_logger
            )
            confidence = parse_text_find_number(confidence_str, "confidence")
            
            # Collect confidence for this round
            confidence_list.append(confidence)
            
            if video_logger:
                video_logger.info(f"=== Self Evaluation ===")
                video_logger.info(f"LLM response:\n '{confidence_str}'")
                video_logger.info(f"Confidence: {confidence}")
            
            # If confidence is -1 (error), break the loop
            if confidence == -1:
                if video_logger:
                    video_logger.error("Error in confidence evaluation, terminating video processing")
                break
            
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
        
        # Final results
        video_memory.predicted_answer = str(answer)
        video_memory.confidence = confidence
        
        if video_logger:
            video_logger.info(f"=== Video processing completed after {round_num} rounds ===")
            video_logger.info(f"Final answer: {video_memory.predicted_answer}")
            video_logger.info(f"Final confidence: {video_memory.confidence}")
        
        # Backward compatibility for logs
        label = int(video_info.get("truth", 0)) if video_info.get("truth") else 0
        # Only mark as correct if answer is valid (not -1) and matches the truth
        corr = int(label == answer and answer != -1)
        count_frame = len(video_memory.sampled_idx)
        
        logs[video_id] = {
            "final_answer": answer,
            "answers": answers_list,
            "confidence": confidence_list,
            "rounds": round_num,
            "label": label,
            "corr": corr,
            "count_frame": count_frame,
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
        
        logger.info(f"Finished video: {video_id}/{answer}/{video_info.get('truth', 'N/A')}")
        
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
            "rounds_history": video_memory.rounds_history
        }
    
    def _save_global_results(self, results: List[Dict], output_dir: str):
        """Save global result files (accuracy.txt and result.json)."""
        if not results:
            return
        
        # Prepare data for global results - matching original main.py format
        video_results = {}
        total_videos = len(results)
        correct_answers = 0
        total_frames = 0
        invalid_videos = 0  # Track videos with errors (final_answer = -1)
        
        for result in results:
            video_id = result["video_id"]
            predicted = result.get("predicted_answer", "")
            truth = result.get("truth")
            frame_count = result.get("frame_count", 0)
            confidence = result.get("confidence", 1)
            rounds = result.get("rounds", 1)
            
            # Convert predicted answer to integer for multiple choice
            try:
                predicted_int = int(predicted) if str(predicted).lstrip('-').isdigit() else 0
            except (ValueError, AttributeError):
                predicted_int = 0
            
            # Check if this is an invalid video (error occurred, final_answer = -1)
            is_invalid = (predicted_int == -1) and (confidence != -1)
            if is_invalid:
                invalid_videos += 1
            
            # Use truth as label
            try:
                label = int(truth) if truth is not None and str(truth).isdigit() else 0
            except (ValueError, AttributeError):
                label = 0
            
            # Check if prediction is correct (only for valid videos)
            is_correct = 1 if predicted_int == label and not is_invalid else 0
            if not is_invalid:  # Only count correct answers for valid videos
                correct_answers += is_correct
                total_frames += frame_count
            
            # Format for result.json - matching original main.py format
            video_results[video_id] = {
                "final_answer": predicted_int,
                "answers": result.get("answers", []),
                "confidence": result.get("confidence", []),
                "rounds": rounds,
                "label": label,
                "corr": is_correct,
                "count_frame": frame_count,
            }
        
        # Save result.json in original format
        result_file = os.path.join(output_dir, "result.json")
        with open(result_file, 'w') as f:
            json.dump(video_results, f, indent=2)
        
        # Calculate and save accuracy.txt - matching original format
        valid_videos = total_videos - invalid_videos
        mean_accuracy = correct_answers / total_videos if total_videos > 0 else 0.0
        mean_frames = total_frames / total_videos if total_videos > 0 else 0.0
        valid_accuracy = correct_answers / valid_videos if valid_videos > 0 else 0.0
        valid_mean_frames = total_frames / valid_videos if valid_videos > 0 else 0.0
        
        accuracy_file = os.path.join(output_dir, "accuracy.txt")
        with open(accuracy_file, 'w') as f:
            f.write(f"number_videos: {total_videos}\n")
            f.write(f"mean_accuracy: {mean_accuracy}\n")
            f.write(f"mean_frame: {mean_frames}\n")
            f.write(f"invalid_videos: {invalid_videos}\n")
            f.write(f"valid_only_accuracy: {valid_accuracy}\n")
            f.write(f"valid_only_mean_frames: {valid_mean_frames}\n")
        

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
        # Default paths - updated to use existing files
        "annotation_file": "dataset/subset_anno.json",
        "test_video_list_file": "dataset/subset.txt",  # Use subset.txt instead of test_one_video.txt
        "video_dir": "dataset/videos",
        "output_dir": "results"
    }
    
    agent = VideoAgent(config)
    return agent.run_experiment()

# Additional backward compatibility - removed dynamic property assignment due to linter issues 