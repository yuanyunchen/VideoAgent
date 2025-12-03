"""
VideoAgent - Main orchestrator for multi-agent video understanding.
Implements the algorithm flow with Solver and Checker agents.
"""

import os
import json
import csv
import random
import logging
import multiprocessing as mp
from datetime import datetime
from typing import Dict, Any, Optional, List
from collections import Counter
from tqdm import tqdm

from video_agent.core.memory import Memory
from video_agent.agents.solver import Solver
from video_agent.agents.checker import evaluate_answer, format_feedback_message
from video_agent.agents.caption_generator import CaptionGenerator
from video_agent.utils.video import get_video_frames, sample_frame_indices, sample_frame_indices_in_range
from video_agent.utils.parsing import parse_video_annotation
from video_agent.utils.logging_utils import setup_logger, create_video_logger
from video_agent.utils.config import save_config_to_output


class EvalStatistics:
    """Track evaluation statistics during experiment."""
    
    def __init__(self):
        self.total = 0
        self.correct = 0
        self.first_round_correct = 0
        self.improved_cases = 0      # Wrong -> Correct after more rounds
        self.degraded_cases = 0      # Correct -> Wrong after more rounds
        self.failed_cases = 0        # Wrong -> Wrong
        self.maintained_cases = 0    # Correct -> Correct
        self.invalid_videos = 0
        self.total_steps = 0
        self.total_frames = 0
        self.confidence_distribution = Counter()
        self.steps_distribution = Counter()
        self.case_types = Counter()
    
    def update(self, result: Dict):
        """Update statistics with a new result."""
        self.total += 1
        
        predicted = result.get("final_answer", -1)
        truth = result.get("truth")
        steps = result.get("total_steps", 1)
        frame_count = result.get("frame_count", 0)
        is_valid = result.get("is_valid", True)
        answers_list = result.get("answers_history", [])
        
        try:
            predicted_int = int(predicted) if str(predicted).lstrip('-').isdigit() else -1
            label = int(truth) if truth is not None and str(truth).isdigit() else -1
        except (ValueError, AttributeError):
            predicted_int = -1
            label = -1
        
        # Handle invalid cases (errors, fallbacks)
        if not is_valid:
            self.invalid_videos += 1
            self.case_types["INVALID"] += 1
            # Still track if the fallback happened to be correct
            if predicted_int != -1 and predicted_int == label:
                # Note: we don't count this in "correct" as it was a fallback
                pass
            return
        
        if predicted_int == -1:
            self.invalid_videos += 1
            self.case_types["INVALID"] += 1
            return
        
        is_correct = predicted_int == label
        
        if is_correct:
            self.correct += 1
        
        self.total_steps += steps
        self.total_frames += frame_count
        self.steps_distribution[steps] += 1
        
        confidence = result.get("confidence_score", 0)
        self.confidence_distribution[confidence] += 1
        
        # Determine first answer correctness
        first_correct = False
        if answers_list and len(answers_list) > 0:
            first_answer = answers_list[0]
            try:
                first_answer_int = int(first_answer) if str(first_answer).lstrip('-').isdigit() else -1
                first_correct = first_answer_int == label and first_answer_int != -1
            except (ValueError, AttributeError):
                first_correct = False
        else:
            # If no answers_list, treat final answer as first answer
            first_correct = is_correct
        
        # Classify case type
        if first_correct and is_correct:
            self.first_round_correct += 1
            self.maintained_cases += 1
            self.case_types["MAINTAINED"] += 1
        elif first_correct and not is_correct:
            self.first_round_correct += 1
            self.degraded_cases += 1
            self.case_types["DEGRADED"] += 1
        elif not first_correct and is_correct:
            self.improved_cases += 1
            self.case_types["IMPROVED"] += 1
        else:
            self.failed_cases += 1
            self.case_types["FAILED"] += 1
    
    def get_accuracy(self) -> float:
        valid = self.total - self.invalid_videos
        return self.correct / valid if valid > 0 else 0.0
    
    def get_first_round_accuracy(self) -> float:
        valid = self.total - self.invalid_videos
        return self.first_round_correct / valid if valid > 0 else 0.0
    
    def get_improvement_rate(self) -> float:
        initially_wrong = self.improved_cases + self.failed_cases
        return self.improved_cases / initially_wrong if initially_wrong > 0 else 0.0
    
    def get_avg_steps(self) -> float:
        valid = self.total - self.invalid_videos
        return self.total_steps / valid if valid > 0 else 0.0
    
    def get_avg_frames(self) -> float:
        valid = self.total - self.invalid_videos
        return self.total_frames / valid if valid > 0 else 0.0
    
    def get_progress_dict(self) -> Dict[str, str]:
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
        valid = self.total - self.invalid_videos
        return {
            "total": self.total,
            "valid": valid,
            "invalid": self.invalid_videos,
            "correct": self.correct,
            "accuracy": self.get_accuracy(),
            "first_round_correct": self.first_round_correct,
            "first_round_accuracy": self.get_first_round_accuracy(),
            "improved_cases": self.improved_cases,
            "degraded_cases": self.degraded_cases,
            "failed_cases": self.failed_cases,
            "maintained_cases": self.maintained_cases,
            "improvement_rate": self.get_improvement_rate(),
            "avg_steps": self.get_avg_steps(),
            "avg_frames": self.get_avg_frames(),
            "total_steps": self.total_steps,
            "total_frames": self.total_frames,
            "confidence_distribution": dict(self.confidence_distribution),
            "steps_distribution": dict(self.steps_distribution),
            "case_types": dict(self.case_types)
        }


def _process_video_worker(video_info: Dict, config: Dict[str, Any], output_dir: str, video_index: int) -> Dict:
    """Worker function for multiprocessing video processing."""
    try:
        agent = VideoAgent(config)
        result = agent._process_single_video(video_info, output_dir)
        return {
            "success": True,
            "result": result,
            "video_index": video_index,
            "video_id": video_info['video_id']
        }
    except Exception as e:
        fallback_answer = random.randint(0, 4)
        return {
            "success": False,
            "error": str(e),
            "video_index": video_index,
            "video_id": video_info['video_id'],
            "result": {
                "video_id": video_info['video_id'],
                "question": video_info.get("question", ""),
                "final_answer": fallback_answer,
                "explanation": f"Error: {str(e)}",
                "confidence_score": 0,
                "total_steps": 0,
                "frame_count": 0,
                "truth": video_info.get("truth"),
                "is_valid": False,
                "error": str(e),
                "answers_history": [],
            }
        }


def _process_video_worker_unpack(args):
    """Unpack arguments for multiprocessing."""
    return _process_video_worker(*args)


class VideoAgent:
    """
    Main VideoAgent orchestrator implementing the multi-agent video understanding algorithm.
    
    Algorithm Flow:
    1. Initialize: Load video frames, sample initial frames, generate captions
    2. Create Solver with initial memory
    3. Main Loop (max_steps iterations):
       a. Get Solver decision
       b. If retrieve_more_frames: fetch frames, add to memory, notify Solver
       c. If answer_question: call Checker
          - If confidence >= 4: SUCCESS, return result
          - If confidence < 4: send feedback to Solver, continue
       d. Handle JSON parse errors with retry message
    4. Terminate: Return best answer or fallback
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize VideoAgent with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.scheduler_model = config.get("scheduler_model", "gpt-4o-mini")
        self.viewer_model = config.get("viewer_model", "gpt-4o-mini")
        self.checker_model = config.get("checker_model", config.get("scheduler_model", "gpt-4o-mini"))
        self.max_steps = config.get("max_rounds", 10)
        self.initial_frames = config.get("default_initial_frames", 5)
        self.max_processes = config.get("max_processes", 1)
        self.confidence_threshold = config.get("confidence_threshold", 8)
    
    def run_experiment(self, video_list_file: Optional[str] = None) -> str:
        """
        Run video analysis experiment on multiple videos.
        
        Args:
            video_list_file: Optional path to video list file
            
        Returns:
            Output directory path
        """
        experiment_name = self.config.get("experiment_name", "default_experiment")
        max_test_videos = self.config.get("max_test_videos", -1)
        
        # Generate output directory name
        scheduler_model = os.path.basename(self.scheduler_model)
        viewer_model = os.path.basename(self.viewer_model)
        timestamp = datetime.now().strftime("%m%d")
        model_info = f"{scheduler_model}_viewer_{viewer_model}_videos_{max_test_videos}_{timestamp}"
        output_dir = os.path.join(
            self.config.get("output_dir", "results"),
            f"{experiment_name}__{model_info}"
        )
        os.makedirs(output_dir, exist_ok=True)
        
        # Save configuration
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
        logger.info(f"Scheduler model: {self.scheduler_model}")
        logger.info(f"Viewer model: {self.viewer_model}")
        
        # Process videos
        if self.max_processes > 1:
            results = self._process_videos_multiprocessing(videos_info, output_dir, logger)
        else:
            results = self._process_videos_sequential(videos_info, output_dir, logger)
        
        results = [r for r in results if r is not None]
        
        # Save global results
        self._save_global_results(results, output_dir)
        
        logger.info(f"Experiment completed. Results in: {output_dir}")
        return output_dir
    
    def _process_videos_sequential(self, videos_info: List[Dict], output_dir: str, 
                                   logger: logging.Logger) -> List[Dict]:
        """Process videos sequentially."""
        results = []
        stats = EvalStatistics()
        
        pbar = tqdm(
            videos_info, 
            desc="Processing", 
            unit="video",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}"
        )
        
        for i, video_info in enumerate(pbar):
            video_id = video_info['video_id']
            logger.info(f"Processing video {i+1}/{len(videos_info)}: {video_id}")
            
            result = self._process_single_video(video_info, output_dir)
            
            if result:
                results.append(result)
                stats.update(result)
                
                predicted = result.get('final_answer', 'N/A')
                truth = result.get('truth', 'N/A')
                status = "OK" if str(predicted) == str(truth) else "WRONG"
                logger.info(f"[{status}] {video_id}: Pred={predicted} Truth={truth}")
            
            pbar.set_postfix(stats.get_progress_dict())
        
        self._eval_stats = stats
        return results
    
    def _process_videos_multiprocessing(self, videos_info: List[Dict], output_dir: str, 
                                        logger: logging.Logger) -> List[Dict]:
        """Process videos using multiprocessing."""
        results = []
        stats = EvalStatistics()
        
        video_args = [(video_info, self.config, output_dir, i) for i, video_info in enumerate(videos_info)]
        
        with mp.Pool(processes=self.max_processes) as pool:
            pbar = tqdm(
                total=len(videos_info), 
                desc="Processing", 
                unit="video",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}"
            )
            
            try:
                worker_results_iter = pool.imap_unordered(_process_video_worker_unpack, video_args)
                
                for worker_result in worker_results_iter:
                    result = worker_result["result"]
                    results.append(result)
                    stats.update(result)
                    
                    if worker_result["success"]:
                        logger.info(f"Completed: {worker_result['video_id']}")
                    else:
                        logger.error(f"Error {worker_result['video_id']}: {worker_result['error']}")
                    
                    pbar.update(1)
                    pbar.set_postfix(stats.get_progress_dict())
                    
            except Exception as e:
                logger.error(f"Multiprocessing error: {e}")
                pbar.close()
                raise
            
            pbar.close()
        
        self._eval_stats = stats
        return results
    
    def _process_single_video(self, video_info: Dict, output_dir: str) -> Dict:
        """
        Process a single video using the multi-agent algorithm.
        
        This implements the main algorithm flow:
        1. Initialize Memory with initial frame captions
        2. Create Solver and send initial message
        3. Main loop: Get Solver decision, handle actions
        4. Return final result
        """
        video_id = video_info["video_id"]
        video_output_dir = os.path.join(output_dir, "videos", video_id)
        os.makedirs(video_output_dir, exist_ok=True)
        
        # Setup video logger
        logger = create_video_logger(video_id, video_output_dir)
        logger.info(f"=== Processing video: {video_id} ===")
        logger.info(f"Question: {video_info['question']}")
        
        # Extract choices
        choices = []
        for i in range(5):
            option_key = f"option {i}"
            if option_key in video_info:
                choices.append(video_info[option_key])
        
        # Load video frames
        video_dir = self.config.get("video_dir", "data/EgoSchema_test/videos")
        video_path = os.path.join(video_dir, f"{video_id}.mp4")
        
        frames = get_video_frames(video_path, interval=30)
        if frames is None:
            logger.error(f"Cannot load video: {video_path}")
            return self._create_error_result(video_info, "Video load failed")
        
        logger.info(f"Loaded {len(frames)} frames from video")
        
        # Initialize Memory
        memory = Memory(video_frames=frames, video_id=video_id)
        
        # Initialize Caption Generator
        caption_gen = CaptionGenerator(model=self.viewer_model, logger=logger)
        
        # Sample initial frames and generate captions
        initial_indices = sample_frame_indices(len(frames), self.initial_frames)
        initial_frames = [frames[i] for i in initial_indices]
        initial_captions = caption_gen.generate_captions(initial_frames, initial_indices)
        
        # Add initial captions to memory
        for idx, caption in zip(initial_indices, initial_captions):
            memory.add_frame(idx, caption)
        
        logger.info(f"Initial frames: {initial_indices}")
        logger.info(f"Initial memory:\n{memory.format_for_solver()}")
        
        # Initialize Solver
        solver = Solver(model=self.scheduler_model, logger=logger)
        
        # Format question with choices
        formatted_question = video_info["question"]
        if choices:
            choice_lines = [f"{i}. {choice}" for i, choice in enumerate(choices)]
            formatted_question = f"{video_info['question']}\n\nChoices:\n" + "\n".join(choice_lines)
        
        # Send initial message to Solver
        initial_message = f"""Question: {formatted_question}

{memory.format_for_solver()}

Note: When requesting more frames, you can specify a frame range using start_frame and end_frame.
Frame indices range from 0 to {len(frames) - 1}.

Based on the frame captions above, decide whether to:
1. Request more frames (if you need more information) - you can specify a range to focus on specific parts of the video
2. Answer the question (if you have enough information)
"""
        
        logger.info("=== Sending initial message to Solver ===")
        
        try:
            thinking, action = solver.send_message(initial_message)
        except Exception as e:
            logger.error(f"Failed to send initial message: {e}")
            return self._create_error_result(video_info, f"Solver initialization failed: {e}")
        
        # Main algorithm loop
        step = 0
        best_answer = None
        best_explanation = ""
        best_confidence = 0
        
        conversation_history = []
        answers_history = []  # Track all answers across rounds
        
        while step < self.max_steps:
            step += 1
            logger.info(f"=== Step {step}/{self.max_steps} ===")
            
            try:
                # Handle JSON parse error
                if action is None:
                    logger.warning("JSON parse failed, sending error message")
                    error_msg = (
                        "System Error: Invalid JSON format in your response. "
                        "Please output a valid JSON block with 'action_type' and 'action_payload'. "
                        "The JSON must be enclosed in ```json and ``` markers."
                    )
                    thinking, action = solver.send_error_message(error_msg)
                    
                    if action is None:
                        logger.error("Max parse errors reached, using fallback")
                        break
                    continue
                
                action_type = action.get("action_type")
                payload = action.get("action_payload", {})
                
                # Record action in history
                conversation_history.append({
                    "step": step,
                    "action_type": action_type,
                    "thinking": thinking[:200] + "..." if len(thinking) > 200 else thinking
                })
                
                # Branch on action type
                if action_type == "retrieve_more_frames":
                    # If this is the last step, force the solver to answer instead
                    if step >= self.max_steps:
                        logger.info("Last step: forcing Solver to answer instead of retrieving more frames")
                        force_msg = (
                            "System Notification: Maximum steps reached. No more frame retrievals allowed. "
                            "Please provide your best answer now based on available information. "
                            "You MUST select an answer (0, 1, 2, 3, or 4) even if uncertain."
                        )
                        thinking, action = solver.send_message(force_msg)
                        
                        # Process the forced answer immediately
                        if action and action.get("action_type") == "answer_question":
                            payload = action.get("action_payload", {})
                            answer = payload.get("final_answer")
                            explanation = payload.get("explanation", "")
                            
                            # Validate answer
                            valid_answer = None
                            if answer is not None:
                                try:
                                    valid_answer = int(answer)
                                    if valid_answer < 0 or valid_answer > 4:
                                        valid_answer = None
                                except (ValueError, TypeError):
                                    valid_answer = None
                            
                            if valid_answer is not None:
                                answers_history.append(valid_answer)
                                logger.info(f"Forced answer: {valid_answer}")
                                if best_answer is None or not isinstance(best_answer, int):
                                    best_answer = valid_answer
                                    best_explanation = explanation
                                    best_confidence = 1
                        break  # Exit the loop after forcing answer
                    
                    # Extract parameters
                    count = payload.get("count", 3)
                    start_frame = payload.get("start_frame", 0)
                    end_frame = payload.get("end_frame", None)
                    focus = payload.get("focus", "")
                    
                    logger.info(f"Action: retrieve_more_frames (count={count}, range=[{start_frame}, {end_frame}], focus={focus})")
                    
                    # Sample frames within the specified range
                    new_indices = sample_frame_indices_in_range(
                        total_frames=len(frames),
                        num_samples=count,
                        start_frame=start_frame,
                        end_frame=end_frame,
                        sampled_indices=memory.sampled_indices
                    )
                    
                    if not new_indices:
                        # No more frames available in the specified range
                        if end_frame is not None:
                            notification = (
                                f"System Notification: No more frames available in range [{start_frame}, {end_frame}]. "
                                "All frames in this range have been analyzed. "
                                "Try a different range or provide your answer based on current information."
                            )
                        else:
                            notification = (
                                "System Notification: No more frames available to retrieve. "
                                "All frames have been analyzed. Please provide your answer based on current information."
                            )
                        logger.info(f"No more frames available in range [{start_frame}, {end_frame}]")
                    else:
                        # Generate captions for new frames
                        new_frames = [frames[i] for i in new_indices]
                        new_captions = caption_gen.generate_captions(new_frames, new_indices)
                        
                        # Add to memory and get notification
                        notification = memory.add_frames(new_indices, new_captions)
                        logger.info(f"Retrieved frames: {new_indices} (from range [{start_frame}, {end_frame}])")
                    
                    # Send notification to Solver
                    thinking, action = solver.send_frame_notification(notification)
                    
                elif action_type == "answer_question":
                    # Extract answer and explanation
                    answer = payload.get("final_answer")
                    explanation = payload.get("explanation", "")
                    
                    # Validate answer is a valid integer 0-4
                    valid_answer = None
                    if answer is not None:
                        try:
                            valid_answer = int(answer)
                            if valid_answer < 0 or valid_answer > 4:
                                valid_answer = None
                        except (ValueError, TypeError):
                            valid_answer = None
                    
                    if valid_answer is None:
                        # Invalid answer - ask for a valid one
                        logger.warning(f"Invalid answer: {answer}, requesting valid answer")
                        error_msg = (
                            f"System Error: Your answer '{answer}' is invalid. "
                            "For multiple choice questions, final_answer must be an integer from 0 to 4. "
                            "Please provide a valid answer based on available information."
                        )
                        thinking, action = solver.send_error_message(error_msg)
                        continue
                    
                    answer = valid_answer
                    
                    # Track answer in history
                    answers_history.append(answer)
                    
                    logger.info(f"Action: answer_question (answer={answer})")
                    logger.info(f"Explanation: {explanation[:200]}..." if len(explanation) > 200 else f"Explanation: {explanation}")
                    
                    # Track best answer
                    if best_answer is None:
                        best_answer = answer
                        best_explanation = explanation
                    
                    # Call Checker (stateless)
                    checker_result = evaluate_answer(
                        question=video_info["question"],
                        memory=memory,
                        answer=answer,
                        explanation=explanation,
                        model=self.checker_model,
                        choices=choices,
                        logger=logger
                    )
                    
                    confidence = checker_result.get("confidence_score", 2)
                    feedback = checker_result.get("feedback", "")
                    
                    logger.info(f"Checker result: confidence={confidence}, feedback={feedback}")
                    
                    # Check confidence threshold
                    if confidence >= self.confidence_threshold:
                        # SUCCESS - return result
                        logger.info(f"=== SUCCESS: Confidence {confidence} >= {self.confidence_threshold} ===")
                        
                        result = self._create_success_result(
                            video_info=video_info,
                            answer=answer,
                            explanation=explanation,
                            confidence=confidence,
                            steps=step,
                            memory=memory,
                            conversation_history=conversation_history,
                            answers_history=answers_history
                        )
                        
                        # Save outputs
                        self._save_video_outputs(result, video_output_dir, memory)
                        return result
                    
                    # Confidence low - update best and send feedback
                    if confidence > best_confidence:
                        best_answer = answer
                        best_explanation = explanation
                        best_confidence = confidence
                    
                    # Format and send feedback to Solver
                    feedback_msg = format_feedback_message(confidence, feedback)
                    logger.info(f"Sending feedback to Solver: {feedback_msg[:200]}...")
                    
                    thinking, action = solver.send_feedback(feedback_msg)
                
                else:
                    # Unknown action type
                    logger.warning(f"Unknown action_type: {action_type}")
                    error_msg = (
                        f"System Error: Unknown action_type '{action_type}'. "
                        "Valid options are 'retrieve_more_frames' or 'answer_question'."
                    )
                    thinking, action = solver.send_error_message(error_msg)
            
            except Exception as e:
                logger.error(f"Error in main loop step {step}: {e}")
                # Try to continue with fallback
                break
        
        # Process any pending answer_question action that came after last frame retrieval
        if action is not None and action.get("action_type") == "answer_question":
            logger.info("=== Processing pending answer after max steps ===")
            payload = action.get("action_payload", {})
            answer = payload.get("final_answer")
            explanation = payload.get("explanation", "")
            
            # Validate answer is a valid choice (0-4)
            if answer is not None:
                try:
                    answer_int = int(answer)
                    if 0 <= answer_int <= 4:
                        # Track this answer
                        answers_history.append(answer_int)
                        logger.info(f"Pending answer: {answer_int}")
                        
                        # Update best if this is our first valid answer
                        if best_answer is None or not isinstance(best_answer, int) or best_answer < 0 or best_answer > 4:
                            best_answer = answer_int
                            best_explanation = explanation
                            best_confidence = 1  # Low confidence but valid
                except (ValueError, TypeError):
                    logger.warning(f"Invalid answer format: {answer}")
        
        # Max steps reached - return best answer
        logger.info(f"=== Max steps ({self.max_steps}) reached ===")
        
        # Determine validity: valid if we have a proper answer (0-4), not random fallback
        is_valid_answer = (
            best_answer is not None and 
            isinstance(best_answer, int) and 
            0 <= best_answer <= 4 and
            len(answers_history) > 0  # At least one answer was submitted
        )
        
        if best_answer is None or not isinstance(best_answer, int) or best_answer < 0 or best_answer > 4:
            best_answer = random.randint(0, 4)
            is_valid_answer = False
            logger.warning(f"No valid answer found, using random fallback: {best_answer}")
        
        result = self._create_success_result(
            video_info=video_info,
            answer=best_answer,
            explanation=best_explanation,
            confidence=best_confidence,
            steps=step,
            memory=memory,
            conversation_history=conversation_history,
            answers_history=answers_history,
            is_valid=is_valid_answer
        )
        
        self._save_video_outputs(result, video_output_dir, memory)
        return result
    
    def _create_success_result(self, video_info: Dict, answer: Any, explanation: str,
                               confidence: int, steps: int, memory: Memory,
                               conversation_history: List,
                               answers_history: List = None,
                               is_valid: bool = True) -> Dict:
        """Create successful result dictionary."""
        return {
            "video_id": video_info["video_id"],
            "question": video_info["question"],
            "final_answer": answer,
            "explanation": explanation,
            "confidence_score": confidence,
            "truth": video_info.get("truth"),
            "total_steps": steps,
            "frame_count": len(memory),
            "is_valid": is_valid,
            "answers_history": answers_history or [],
            "conversation_history": conversation_history,
            "memory_state": memory.to_dict()
        }
    
    def _create_error_result(self, video_info: Dict, error: str) -> Dict:
        """Create error result dictionary."""
        fallback = random.randint(0, 4)
        return {
            "video_id": video_info["video_id"],
            "question": video_info.get("question", ""),
            "final_answer": fallback,
            "explanation": f"Error: {error}",
            "confidence_score": 0,
            "truth": video_info.get("truth"),
            "total_steps": 0,
            "frame_count": 0,
            "is_valid": False,
            "error": error,
            "answers_history": [],
            "conversation_history": [],
        }
    
    def _save_video_outputs(self, result: Dict, output_dir: str, memory: Memory):
        """Save video-specific outputs."""
        # Save result JSON
        result_file = os.path.join(output_dir, "result.json")
        with open(result_file, 'w') as f:
            # Create serializable version (remove numpy arrays from memory)
            serializable_result = result.copy()
            if "memory_state" in serializable_result:
                serializable_result["memory_state"] = memory.to_dict()
            json.dump(serializable_result, f, indent=2)
        
        # Save frames
        memory.save_frames(output_dir)
    
    def _save_global_results(self, results: List[Dict], output_dir: str):
        """Save global result files."""
        if not results:
            return
        
        if hasattr(self, '_eval_stats'):
            stats = self._eval_stats
        else:
            stats = EvalStatistics()
            for result in results:
                stats.update(result)
        
        stats_dict = stats.to_dict()
        
        # 1. Save result.json
        video_results = {}
        for result in results:
            video_id = result.get("video_id", "unknown")
            video_result = {
                "question": result.get("question", ""),
                "final_answer": result.get("final_answer"),
                "explanation": result.get("explanation", ""),
                "confidence_score": result.get("confidence_score", 0),
                "truth": result.get("truth"),
                "total_steps": result.get("total_steps", 0),
                "frame_count": result.get("frame_count", 0),
                "is_valid": result.get("is_valid", True),
                "answers_history": result.get("answers_history", []),
            }
            # Include error field if present
            if "error" in result:
                video_result["error"] = result["error"]
            video_results[video_id] = video_result
        
        result_output = {
            "_summary": stats_dict,
            "results": video_results
        }
        
        result_file = os.path.join(output_dir, "result.json")
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result_output, f, indent=2)
        
        # 2. Save metrics.csv
        metrics_file = os.path.join(output_dir, "metrics.csv")
        with open(metrics_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            writer.writerow(["total_videos", stats_dict["total"]])
            writer.writerow(["valid_videos", stats_dict["valid"]])
            writer.writerow(["invalid_videos", stats_dict["invalid"]])
            writer.writerow(["correct_answers", stats_dict["correct"]])
            writer.writerow(["accuracy", f"{stats_dict['accuracy']:.4f}"])
            writer.writerow(["first_round_correct", stats_dict["first_round_correct"]])
            writer.writerow(["first_round_accuracy", f"{stats_dict['first_round_accuracy']:.4f}"])
            writer.writerow(["improved_cases", stats_dict["improved_cases"]])
            writer.writerow(["degraded_cases", stats_dict["degraded_cases"]])
            writer.writerow(["failed_cases", stats_dict["failed_cases"]])
            writer.writerow(["maintained_cases", stats_dict["maintained_cases"]])
            writer.writerow(["improvement_rate", f"{stats_dict['improvement_rate']:.4f}"])
            writer.writerow(["avg_steps", f"{stats_dict['avg_steps']:.2f}"])
            writer.writerow(["avg_frames", f"{stats_dict['avg_frames']:.2f}"])
        
        # 3. Save summary.txt
        summary_file = os.path.join(output_dir, "summary.txt")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("VideoAgent Multi-Agent Evaluation Summary\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Output Directory: {output_dir}\n\n")
            
            f.write("-" * 70 + "\n")
            f.write("Overall Results\n")
            f.write("-" * 70 + "\n")
            f.write(f"Total Videos:       {stats_dict['total']}\n")
            f.write(f"Valid Videos:       {stats_dict['valid']}\n")
            f.write(f"Invalid Videos:     {stats_dict['invalid']}\n")
            f.write(f"Correct Answers:    {stats_dict['correct']}\n\n")
            
            f.write("-" * 70 + "\n")
            f.write("Accuracy Metrics\n")
            f.write("-" * 70 + "\n")
            f.write(f"Final Accuracy:         {stats_dict['accuracy']:.2%} ({stats_dict['correct']}/{stats_dict['valid']})\n")
            f.write(f"First Round Accuracy:   {stats_dict['first_round_accuracy']:.2%} ({stats_dict['first_round_correct']}/{stats_dict['valid']})\n")
            acc_delta = stats_dict['accuracy'] - stats_dict['first_round_accuracy']
            delta_sign = "+" if acc_delta >= 0 else ""
            f.write(f"Accuracy Improvement:   {delta_sign}{acc_delta:.2%}\n\n")
            
            f.write("-" * 70 + "\n")
            f.write("Case Type Analysis\n")
            f.write("-" * 70 + "\n")
            f.write(f"MAINTAINED (Correct -> Correct):  {stats_dict['maintained_cases']}\n")
            f.write(f"IMPROVED   (Wrong -> Correct):    {stats_dict['improved_cases']}\n")
            f.write(f"DEGRADED   (Correct -> Wrong):    {stats_dict['degraded_cases']}\n")
            f.write(f"FAILED     (Wrong -> Wrong):      {stats_dict['failed_cases']}\n\n")
            
            f.write("-" * 70 + "\n")
            f.write("Improvement Analysis\n")
            f.write("-" * 70 + "\n")
            initially_wrong = stats_dict['improved_cases'] + stats_dict['failed_cases']
            f.write(f"Initially Wrong:        {initially_wrong}\n")
            f.write(f"Successfully Improved:  {stats_dict['improved_cases']}\n")
            f.write(f"Improvement Rate:       {stats_dict['improvement_rate']:.2%}\n\n")
            
            f.write("-" * 70 + "\n")
            f.write("Resource Usage\n")
            f.write("-" * 70 + "\n")
            f.write(f"Avg Steps/Video:    {stats_dict['avg_steps']:.2f}\n")
            f.write(f"Avg Frames/Video:   {stats_dict['avg_frames']:.1f}\n\n")
            
            if stats_dict['steps_distribution']:
                f.write("-" * 70 + "\n")
                f.write("Steps Distribution\n")
                f.write("-" * 70 + "\n")
                for steps, count in sorted(stats_dict['steps_distribution'].items()):
                    pct = count / stats_dict['valid'] * 100 if stats_dict['valid'] > 0 else 0
                    f.write(f"  {steps} step(s): {count} videos ({pct:.1f}%)\n")
                f.write("\n")
            
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
            f.write(f"first_round_accuracy: {stats_dict['first_round_accuracy']}\n")
            f.write(f"improved_cases: {stats_dict['improved_cases']}\n")
            f.write(f"degraded_cases: {stats_dict['degraded_cases']}\n")
            f.write(f"improvement_rate: {stats_dict['improvement_rate']}\n")
            f.write(f"mean_frames: {stats_dict['avg_frames']}\n")
            f.write(f"mean_steps: {stats_dict['avg_steps']}\n")
            f.write(f"invalid_videos: {stats_dict['invalid']}\n")
