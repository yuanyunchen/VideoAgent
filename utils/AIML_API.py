"""
AIML API module for VideoAgent.
Provides interfaces for text and image-based LLM interactions.
"""

import base64
import cv2
import numpy as np
from openai import OpenAI
from typing import Union, List, Optional

try:
    # Try relative imports first (when used as package)
    from .config import AIML_API_KEY, AIML_BASE_URL, LLM_TEMPERATURE, LLM_MAX_TOKENS, DEFAULT_SCHEDULER_MODEL
except ImportError:
    # Fall back to absolute imports (when run directly)
    from utils.config import AIML_API_KEY, AIML_BASE_URL, LLM_TEMPERATURE, LLM_MAX_TOKENS, DEFAULT_SCHEDULER_MODEL


class AIMLClient:
    """
    Client for interacting with AIML API for text and image processing.
    """
    
    def __init__(self, api_key: str = AIML_API_KEY, base_url: str = AIML_BASE_URL):
        """
        Initialize AIML client.
        
        Args:
            api_key: AIML API key
            base_url: AIML API base URL
        """
        self.api = OpenAI(api_key=api_key, base_url=base_url)
        self.temperature = LLM_TEMPERATURE
        self.max_tokens = LLM_MAX_TOKENS
    
    def get_text_response(self, user_prompt: str, model_name: str = DEFAULT_SCHEDULER_MODEL,
                         system_prompt: str = "", json_format: bool = False) -> str:
        """
        Get text-only response from LLM.
        
        Args:
            user_prompt: User message content
            model_name: Model identifier
            system_prompt: System message content
            json_format: Whether to request JSON formatted response
            
        Returns:
            LLM response content
        """
        completion = self.api.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
            response_format={"type": "json_object"} if json_format else {"type": "text"},
            max_tokens=self.max_tokens
        )
        return completion.choices[0].message.content
    
    def get_image_response(self, user_prompt: str, images: Union[str, np.ndarray, bytes, List],
                          model_name: str = DEFAULT_SCHEDULER_MODEL, system_prompt: str = "",
                          json_format: bool = False) -> str:
        """
        Get response from LLM with image inputs.
        
        Args:
            user_prompt: User message content
            images: Image(s) as path, numpy array, bytes, or list of these
            model_name: Model identifier
            system_prompt: System message content
            json_format: Whether to request JSON formatted response
            
        Returns:
            LLM response content
        """
        if not isinstance(images, list):
            images = [images]
        
        base64_images = [self._image_to_base64(image) for image in images]
        
        # Prepare message content
        content = [{"type": "text", "text": user_prompt}]
        for base64_image in base64_images:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            })
        
        completion = self.api.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content}
            ],
            temperature=self.temperature,
            response_format={"type": "json_object"} if json_format else {"type": "text"},
            max_tokens=self.max_tokens
        )
        
        return completion.choices[0].message.content
    
    def _image_to_base64(self, image: Union[str, np.ndarray, bytes]) -> str:
        """
        Convert image to base64 string.
        
        Args:
            image: Image as file path, numpy array, or bytes
            
        Returns:
            Base64 encoded image string
            
        Raises:
            ValueError: If image type is not supported
        """
        if isinstance(image, str):
            # File path
            with open(image, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode("utf-8")
        elif isinstance(image, np.ndarray):
            # Numpy array
            _, buffer = cv2.imencode('.jpg', image)
            image_bytes = buffer.tobytes()
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
        elif isinstance(image, bytes):
            # Bytes
            base64_image = base64.b64encode(image).decode("utf-8")
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        return base64_image


# Global client instance for backward compatibility
_global_client = AIMLClient()

# Global retry configuration
RETRY_DELAY = 30  # seconds to wait between retries for rate limit errors
MAX_RETRY_TIME = 3600  # maximum total time to spend retrying （1h)

def get_llm_response(model: str, query: str, images: Optional[List] = None, 
                    temperature: Optional[float] = None, max_tokens: Optional[int] = None, logger=None) -> str:
    """
    Unified LLM response function with rate limit retry logic.
    
    Args:
        model: Model name
        query: Query text
        images: Optional list of images
        temperature: Temperature override
        max_tokens: Max tokens override
        
    Returns:
        LLM response text
    """
    import time
    from datetime import datetime
    
    # Update client settings if overrides provided
    if temperature is not None:
        _global_client.temperature = temperature
    if max_tokens is not None:
        _global_client.max_tokens = max_tokens
    
    # Log input if logger provided
    if logger:
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        logger.info(f"[{timestamp}] === LLM INPUT ===")
        logger.info(f"Model: {model}")
        logger.info(f"Has Images: {len(images) if images else 0}")
        logger.info(f"Query: {query}")  # Full query, no truncation
    
    # Retry logic for rate limit errors
    start_time = time.time()
    retry_count = 0
    
    while True:
        try:
            # Get response
            if images:
                response = _global_client.get_image_response(query, images, model)
            else:
                response = _global_client.get_text_response(query, model)
            
            # Log output if logger provided
            if logger:
                timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                logger.info(f"[{timestamp}] === LLM OUTPUT ===")
                logger.info(f"Response: {response}")  # Full response, no truncation
                logger.info(f"Response Length: {len(response)} characters")
                if retry_count > 0:
                    logger.info(f"Succeeded after {retry_count} retries")
                logger.info(f"=== END LLM CALL ===\n")
            
            return response
            
        except Exception as e:
            error_str = str(e)
            
            # Check if this is a rate limit error (429)
            is_rate_limit_error = ("Rate limit exceeded" in error_str or "rate-limited" in error_str)
            
            if is_rate_limit_error:
                elapsed_time = time.time() - start_time
                
                # Check if we've exceeded maximum retry time
                if elapsed_time >= MAX_RETRY_TIME:
                    if logger:
                        logger.error(f"Rate limit retry timeout after {elapsed_time:.1f}s, giving up")
                    raise e
                
                retry_count += 1
                if logger:
                    logger.warning(f"Rate limit hit (attempt {retry_count}), waiting {RETRY_DELAY}s before retry...")
                    logger.warning(f"Error: {error_str}")
                
                time.sleep(RETRY_DELAY)
                continue
            else:
                # Not a rate limit error, re-raise immediately
                if logger:
                    logger.error(f"Non-rate-limit error: {error_str}")
                raise e

def get_text_response(user_prompt: str, model_name: str = DEFAULT_SCHEDULER_MODEL,
                     system_prompt: str = "", json_format: bool = False) -> str:
    """Legacy function for backward compatibility."""
    return _global_client.get_text_response(user_prompt, model_name, system_prompt, json_format)

def get_image_response(user_prompt: str, image_or_paths: Union[str, np.ndarray, bytes, List],
                      model_name: str = DEFAULT_SCHEDULER_MODEL, system_prompt: str = "",
                      json_format: bool = False) -> str:
    """Legacy function for backward compatibility."""
    return _global_client.get_image_response(user_prompt, image_or_paths, model_name, system_prompt, json_format)

def image_to_bytes_str(image_or_path: Union[str, np.ndarray, bytes]) -> str:
    """Legacy function for backward compatibility."""
    return _global_client._image_to_base64(image_or_path)

