"""
AIML API module for VideoAgent.
Provides interfaces for text and image-based LLM interactions.
"""

import base64
import time
import cv2
import numpy as np
from openai import OpenAI
from typing import Union, List, Optional, Dict
from datetime import datetime

from video_agent.utils.config import (
    AIML_API_KEY, AIML_BASE_URL, LLM_TEMPERATURE, LLM_MAX_TOKENS, DEFAULT_SCHEDULER_MODEL
)

DEFAULT_SYSTEM_PROMPT = ""


class AIMLClient:
    """Client for interacting with AIML API for text and image processing."""
    
    def __init__(self, api_key: str = None, base_url: str = None):
        """
        Initialize AIML client.
        
        Args:
            api_key: AIML API key (defaults to env var)
            base_url: AIML API base URL (defaults to env var)
        """
        self.api_key = api_key or AIML_API_KEY
        self.base_url = base_url or AIML_BASE_URL
        self.api = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.temperature = LLM_TEMPERATURE if LLM_TEMPERATURE is not None else 0.7
        self.max_tokens = LLM_MAX_TOKENS if LLM_MAX_TOKENS is not None else 10000
    
    def get_text_response(self, user_prompt: str, model_name: str = DEFAULT_SCHEDULER_MODEL,
                         system_prompt: str = "", json_format: bool = False) -> str:
        """Get text-only response from LLM."""
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
    
    def get_chat_response(self, messages: List[Dict], model_name: str = DEFAULT_SCHEDULER_MODEL,
                         json_format: bool = False) -> str:
        """
        Get response from LLM with full chat history.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            model_name: Model identifier
            json_format: Whether to request JSON formatted response
            
        Returns:
            LLM response content
        """
        completion = self.api.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=self.temperature,
            response_format={"type": "json_object"} if json_format else {"type": "text"},
            max_tokens=self.max_tokens
        )
        return completion.choices[0].message.content
    
    def get_image_response(self, user_prompt: str, images: Union[str, np.ndarray, bytes, List],
                          model_name: str = DEFAULT_SCHEDULER_MODEL, system_prompt: str = "",
                          json_format: bool = False) -> str:
        """Get response from LLM with image inputs."""
        if not isinstance(images, list):
            images = [images]
        
        base64_images = [self._image_to_base64(image) for image in images]
        
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
        """Convert image to base64 string."""
        if isinstance(image, str):
            with open(image, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode("utf-8")
        elif isinstance(image, np.ndarray):
            _, buffer = cv2.imencode('.jpg', image)
            image_bytes = buffer.tobytes()
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
        elif isinstance(image, bytes):
            base64_image = base64.b64encode(image).decode("utf-8")
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        return base64_image


# Global client instance
_global_client = None


def _get_client():
    """Get or create global client instance."""
    global _global_client
    if _global_client is None:
        _global_client = AIMLClient()
    return _global_client


# Retry configuration
RETRY_DELAY = 30
MAX_RETRY_TIME = 3600


def get_llm_response(model: str, query: str, images: Optional[List] = None, 
                    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
                    temperature: Optional[float] = None, max_tokens: Optional[int] = None, 
                    json_format: bool = False, logger=None) -> str:
    """
    Unified LLM response function with rate limit retry logic.
    """
    client = _get_client()
    
    if temperature is not None:
        client.temperature = temperature
    if max_tokens is not None:
        client.max_tokens = max_tokens
    
    if logger:
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        logger.info(f"[{timestamp}] === LLM INPUT ===")
        logger.info(f"Model: {model}")
        logger.info(f"System Prompt: {system_prompt[:200]}..." if len(system_prompt) > 200 else f"System Prompt: {system_prompt}")
        logger.info(f"Has Images: {len(images) if images else 0}")
        logger.info(f"Query: {query[:500]}..." if len(query) > 500 else f"Query: {query}")
    
    start_time = time.time()
    retry_count = 0
    
    while True:
        try:
            if images:
                response = client.get_image_response(query, images, model, system_prompt, json_format)
            else:
                response = client.get_text_response(query, model, system_prompt, json_format)
            
            if logger:
                timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                logger.info(f"[{timestamp}] === LLM OUTPUT ===")
                logger.info(f"Response: {response[:1000]}..." if len(response) > 1000 else f"Response: {response}")
                logger.info(f"Response Length: {len(response)} characters")
                if retry_count > 0:
                    logger.info(f"Succeeded after {retry_count} retries")
                logger.info(f"=== END LLM CALL ===\n")
            
            return response
            
        except Exception as e:
            error_str = str(e)
            is_rate_limit_error = ("Rate limit exceeded" in error_str or "rate-limited" in error_str)
            
            if is_rate_limit_error:
                elapsed_time = time.time() - start_time
                
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
                if logger:
                    logger.error(f"Non-rate-limit error: {error_str}")
                raise e


def get_chat_response(model: str, messages: List[Dict], 
                     temperature: Optional[float] = None, max_tokens: Optional[int] = None, 
                     json_format: bool = False, logger=None) -> str:
    """
    Get LLM response with full chat history (for stateful conversation).
    """
    client = _get_client()
    
    if temperature is not None:
        client.temperature = temperature
    if max_tokens is not None:
        client.max_tokens = max_tokens
    
    if logger:
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        logger.info(f"[{timestamp}] === LLM CHAT INPUT ===")
        logger.info(f"Model: {model}")
        logger.info(f"Message Count: {len(messages)}")
        if messages:
            last_msg = messages[-1]
            content = last_msg.get('content', '')
            logger.info(f"Last Message ({last_msg.get('role')}): {content[:500]}..." if len(content) > 500 else f"Last Message ({last_msg.get('role')}): {content}")
    
    start_time = time.time()
    retry_count = 0
    
    while True:
        try:
            response = client.get_chat_response(messages, model, json_format)
            
            if logger:
                timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                logger.info(f"[{timestamp}] === LLM CHAT OUTPUT ===")
                logger.info(f"Response: {response[:1000]}..." if len(response) > 1000 else f"Response: {response}")
                logger.info(f"Response Length: {len(response)} characters")
                if retry_count > 0:
                    logger.info(f"Succeeded after {retry_count} retries")
                logger.info(f"=== END LLM CHAT CALL ===\n")
            
            return response
            
        except Exception as e:
            error_str = str(e)
            is_rate_limit_error = ("Rate limit exceeded" in error_str or "rate-limited" in error_str)
            
            if is_rate_limit_error:
                elapsed_time = time.time() - start_time
                
                if elapsed_time >= MAX_RETRY_TIME:
                    if logger:
                        logger.error(f"Rate limit retry timeout after {elapsed_time:.1f}s, giving up")
                    raise e
                
                retry_count += 1
                if logger:
                    logger.warning(f"Rate limit hit (attempt {retry_count}), waiting {RETRY_DELAY}s before retry...")
                
                time.sleep(RETRY_DELAY)
                continue
            else:
                if logger:
                    logger.error(f"Non-rate-limit error: {error_str}")
                raise e


def get_text_response(user_prompt: str, model_name: str = DEFAULT_SCHEDULER_MODEL,
                     system_prompt: str = "", json_format: bool = False) -> str:
    """Legacy function for backward compatibility."""
    return _get_client().get_text_response(user_prompt, model_name, system_prompt, json_format)


def get_image_response(user_prompt: str, image_or_paths: Union[str, np.ndarray, bytes, List],
                      model_name: str = DEFAULT_SCHEDULER_MODEL, system_prompt: str = "",
                      json_format: bool = False) -> str:
    """Legacy function for backward compatibility."""
    return _get_client().get_image_response(user_prompt, image_or_paths, model_name, system_prompt, json_format)


def image_to_bytes_str(image_or_path: Union[str, np.ndarray, bytes]) -> str:
    """Legacy function for backward compatibility."""
    return _get_client()._image_to_base64(image_or_path)
