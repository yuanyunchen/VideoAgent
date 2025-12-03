"""
Checker Agent for VideoAgent.
Stateless evaluator that assesses answer confidence and provides feedback.
"""

import logging
from typing import Dict, Any, Optional

from video_agent.utils.api import get_llm_response
from video_agent.utils.parsing import extract_json_from_response
from video_agent.core.memory import Memory


# System prompt for Checker agent
CHECKER_SYSTEM_PROMPT = """You are a strict examiner evaluating video question-answering results.

Your task is to assess the confidence level of an answer based on:
1. The available video frame captions (Context)
2. The question being asked
3. The provided answer and explanation

## Confidence Scoring (1-10)

- **Score 1-2**: The answer is clearly wrong or completely unsupported
- **Score 3-4**: The answer has major issues or contradicts evidence
- **Score 5-6**: The answer is plausible but lacks sufficient support
- **Score 7-8**: The answer is well-supported with minor uncertainties
- **Score 9-10**: The answer is strongly supported with comprehensive evidence

## Evaluation Criteria

1. **Evidence Alignment**: Does the answer align with the frame captions?
2. **Reasoning Quality**: Is the explanation logical and well-reasoned?
3. **Completeness**: Does the answer fully address the question?
4. **Specificity**: Does the answer use specific evidence from the captions?

## Output Format

You must output your evaluation as a JSON block:

```json
{
  "confidence_score": 7,
  "feedback": "Specific feedback explaining the score and suggestions for improvement"
}
```

## Important Rules

1. Be critical but fair - only give high scores (8+) when truly deserved
2. If score < 8, provide specific, actionable feedback for improvement
3. Feedback should mention what information is missing or what could strengthen the answer
4. Consider whether the answer makes reasonable inferences from available evidence
"""


def evaluate_answer(
    question: str,
    memory: Memory,
    answer: Any,
    explanation: str,
    model: str,
    choices: Optional[list] = None,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Stateless evaluation of an answer.
    
    This function creates a fresh LLM call each time without maintaining any state.
    It evaluates the answer based on the question, memory context, and explanation.
    
    Args:
        question: The original question
        memory: Memory instance containing frame captions
        answer: The answer provided by Solver
        explanation: The explanation for the answer
        model: Model name for LLM call
        choices: Optional list of multiple choice options
        logger: Optional logger for debugging
        
    Returns:
        Dictionary with 'confidence_score' (int 1-5) and 'feedback' (str)
    """
    # Format the context from memory
    context = memory.format_for_solver()
    
    # Format question with choices if available
    formatted_question = question
    if choices:
        choice_lines = [f"{i}. {choice}" for i, choice in enumerate(choices)]
        formatted_question = f"{question}\n\nChoices:\n" + "\n".join(choice_lines)
    
    # Construct evaluation prompt
    user_prompt = f"""Please evaluate the following video question-answering result.

## Context (Frame Captions)
{context}

## Question
{formatted_question}

## Answer
{answer}

## Explanation
{explanation}

---

Evaluate the answer's confidence level (1-5) and provide specific feedback.
Output your evaluation as a JSON block with 'confidence_score' and 'feedback' fields.
"""
    
    if logger:
        logger.info("=== Checker Evaluation ===")
        logger.info(f"Question: {question}")
        logger.info(f"Answer: {answer}")
        logger.info(f"Explanation: {explanation[:200]}..." if len(explanation) > 200 else f"Explanation: {explanation}")
    
    try:
        response = get_llm_response(
            model=model,
            query=user_prompt,
            system_prompt=CHECKER_SYSTEM_PROMPT,
            logger=logger
        )
        
        # Parse response
        _, result_json = extract_json_from_response(response)
        
        if result_json and "confidence_score" in result_json:
            confidence = int(result_json.get("confidence_score", 5))
            feedback = result_json.get("feedback", "No specific feedback provided.")
            
            # Ensure confidence is in valid range (1-10)
            confidence = max(1, min(10, confidence))
            
            if logger:
                logger.info(f"Checker Result: confidence={confidence}, feedback={feedback}")
            
            return {
                "confidence_score": confidence,
                "feedback": feedback
            }
        else:
            if logger:
                logger.warning("Failed to parse Checker response, using default values")
            return _default_result()
            
    except Exception as e:
        if logger:
            logger.error(f"Checker evaluation failed: {e}")
        return _default_result()


def _default_result() -> Dict[str, Any]:
    """Return default result when evaluation fails."""
    return {
        "confidence_score": 5,
        "feedback": "Unable to evaluate the answer. Please reconsider with more evidence."
    }


def format_feedback_message(confidence_score: int, feedback: str) -> str:
    """
    Format checker feedback as a message for Solver.
    
    Args:
        confidence_score: Confidence score (1-5)
        feedback: Feedback text
        
    Returns:
        Formatted feedback message
    """
    return (
        f"System Feedback: Your answer received a confidence score of {confidence_score}/5.\n"
        f"Reason: {feedback}\n"
        f"Please reconsider your answer based on this feedback. "
        f"You may choose to retrieve more frames or provide a revised answer."
    )

