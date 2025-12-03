"""
Cache management for VideoAgent.
"""

import os
import pickle
from typing import Dict


class CacheManager:
    """Simple cache manager for LLM and CLIP responses."""
    
    def __init__(self, llm_cache_file: str, clip_cache_file: str, use_cache: bool = True):
        """
        Initialize cache manager.
        
        Args:
            llm_cache_file: Path to LLM cache file
            clip_cache_file: Path to CLIP cache file
            use_cache: Whether to use caching
        """
        self.llm_cache_file = llm_cache_file
        self.clip_cache_file = clip_cache_file
        self.use_cache = use_cache
        
        # Load caches
        self.llm_cache = self._load_cache(llm_cache_file) if use_cache else {}
        self.clip_cache = self._load_cache(clip_cache_file) if use_cache else {}
    
    def _load_cache(self, cache_file: str) -> Dict:
        """Load cache from file."""
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except (pickle.PickleError, EOFError):
                pass
        return {}
    
    def save_caches(self):
        """Save caches to files."""
        if not self.use_cache:
            return
            
        os.makedirs(os.path.dirname(self.llm_cache_file), exist_ok=True)
        os.makedirs(os.path.dirname(self.clip_cache_file), exist_ok=True)
        
        with open(self.llm_cache_file, 'wb') as f:
            pickle.dump(self.llm_cache, f)
        
        with open(self.clip_cache_file, 'wb') as f:
            pickle.dump(self.clip_cache, f)
    
    def get_llm_cache(self, key: str):
        """Get value from LLM cache."""
        return self.llm_cache.get(key)
    
    def set_llm_cache(self, key: str, value):
        """Set value in LLM cache."""
        if self.use_cache:
            self.llm_cache[key] = value
    
    def get_clip_cache(self, key: str):
        """Get value from CLIP cache."""
        return self.clip_cache.get(key)
    
    def set_clip_cache(self, key: str, value):
        """Set value in CLIP cache."""
        if self.use_cache:
            self.clip_cache[key] = value

