"""
Tool Result Cache for VideoAgent

Provides caching for expensive tool operations:
- Caption tools (OmniCaptioner / API Captioning)
- InternVideo Description

Cache is stored as JSON files to persist across sessions.
"""

import os
import json
import hashlib
import logging
from typing import Dict, Any, Optional
from datetime import datetime


# Default cache directory
DEFAULT_TOOL_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "cache", "tool_cache"
)


class ToolCache:
    """
    Cache manager for tool results.
    
    Supports caching for:
    - caption: key = (video_hash, frame_idx, model, detail_level)
    - internvideo_description: key = (video_hash, start_frame, end_frame)
    """
    
    def __init__(
        self,
        cache_dir: str = None,
        enabled: bool = True,
        logger: logging.Logger = None,
    ):
        """
        Initialize ToolCache.
        
        Args:
            cache_dir: Directory to store cache files
            enabled: Whether caching is enabled
            logger: Logger instance
        """
        self.cache_dir = cache_dir or DEFAULT_TOOL_CACHE_DIR
        self.enabled = enabled
        self.logger = logger or logging.getLogger(__name__)
        
        # In-memory cache (dict of dicts, per video)
        self._caption_cache: Dict[str, Dict[str, Any]] = {}
        self._description_cache: Dict[str, Dict[str, Any]] = {}
        
        # Stats
        self._hits = 0
        self._misses = 0
        
        if enabled:
            os.makedirs(self.cache_dir, exist_ok=True)
            self.logger.info(f"ToolCache initialized at: {self.cache_dir}")
    
    def _get_video_hash(self, video_path: str) -> str:
        """Get a short hash for the video path."""
        abs_path = os.path.abspath(video_path)
        return hashlib.md5(abs_path.encode()).hexdigest()[:12]
    
    def _get_video_name(self, video_path: str) -> str:
        """Get video name without extension."""
        return os.path.splitext(os.path.basename(video_path))[0]
    
    def _get_caption_cache_path(self, video_path: str) -> str:
        """Get cache file path for captions."""
        video_hash = self._get_video_hash(video_path)
        video_name = self._get_video_name(video_path)
        return os.path.join(self.cache_dir, "captions", f"{video_name}_{video_hash}.json")
    
    def _get_description_cache_path(self, video_path: str) -> str:
        """Get cache file path for descriptions."""
        video_hash = self._get_video_hash(video_path)
        video_name = self._get_video_name(video_path)
        return os.path.join(self.cache_dir, "descriptions", f"{video_name}_{video_hash}.json")
    
    def _make_caption_key(
        self,
        frame_idx: int,
        model: str,
        detail_level: str,
    ) -> str:
        """Create cache key for caption."""
        return f"{frame_idx}_{model}_{detail_level}"
    
    def _make_description_key(
        self,
        start_frame: int,
        end_frame: int,
    ) -> str:
        """Create cache key for description."""
        return f"{start_frame}_{end_frame}"
    
    def _load_caption_cache(self, video_path: str) -> Dict[str, Any]:
        """Load caption cache for a video from disk."""
        video_hash = self._get_video_hash(video_path)
        
        # Check in-memory cache first
        if video_hash in self._caption_cache:
            return self._caption_cache[video_hash]
        
        # Load from disk
        cache_path = self._get_caption_cache_path(video_path)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                self._caption_cache[video_hash] = cache_data
                return cache_data
            except Exception as e:
                self.logger.warning(f"Failed to load caption cache: {e}")
        
        # Initialize empty cache
        cache_data = {"video_path": video_path, "captions": {}}
        self._caption_cache[video_hash] = cache_data
        return cache_data
    
    def _save_caption_cache(self, video_path: str) -> None:
        """Save caption cache to disk."""
        video_hash = self._get_video_hash(video_path)
        if video_hash not in self._caption_cache:
            return
        
        cache_path = self._get_caption_cache_path(video_path)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        
        try:
            cache_data = self._caption_cache[video_hash]
            cache_data["last_updated"] = datetime.now().isoformat()
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.warning(f"Failed to save caption cache: {e}")
    
    def _load_description_cache(self, video_path: str) -> Dict[str, Any]:
        """Load description cache for a video from disk."""
        video_hash = self._get_video_hash(video_path)
        
        # Check in-memory cache first
        if video_hash in self._description_cache:
            return self._description_cache[video_hash]
        
        # Load from disk
        cache_path = self._get_description_cache_path(video_path)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                self._description_cache[video_hash] = cache_data
                return cache_data
            except Exception as e:
                self.logger.warning(f"Failed to load description cache: {e}")
        
        # Initialize empty cache
        cache_data = {"video_path": video_path, "descriptions": {}}
        self._description_cache[video_hash] = cache_data
        return cache_data
    
    def _save_description_cache(self, video_path: str) -> None:
        """Save description cache to disk."""
        video_hash = self._get_video_hash(video_path)
        if video_hash not in self._description_cache:
            return
        
        cache_path = self._get_description_cache_path(video_path)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        
        try:
            cache_data = self._description_cache[video_hash]
            cache_data["last_updated"] = datetime.now().isoformat()
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.warning(f"Failed to save description cache: {e}")
    
    # ========== Caption Cache API ==========
    
    def get_caption(
        self,
        video_path: str,
        frame_idx: int,
        model: str,
        detail_level: str,
    ) -> Optional[str]:
        """
        Get cached caption for a specific frame.
        
        Args:
            video_path: Path to video file
            frame_idx: Frame index
            model: Model name/path used for captioning
            detail_level: Caption detail level (short/medium/long)
        
        Returns:
            Cached caption string or None if not cached
        """
        if not self.enabled:
            return None
        
        cache_data = self._load_caption_cache(video_path)
        key = self._make_caption_key(frame_idx, model, detail_level)
        
        caption = cache_data.get("captions", {}).get(key)
        
        if caption is not None:
            self._hits += 1
            self.logger.debug(f"Caption cache HIT: frame={frame_idx}, model={model}, level={detail_level}")
        else:
            self._misses += 1
        
        return caption
    
    def set_caption(
        self,
        video_path: str,
        frame_idx: int,
        model: str,
        detail_level: str,
        caption: str,
    ) -> None:
        """
        Cache a caption for a specific frame.
        
        Args:
            video_path: Path to video file
            frame_idx: Frame index
            model: Model name/path used for captioning
            detail_level: Caption detail level
            caption: Caption text to cache
        """
        if not self.enabled:
            return
        
        cache_data = self._load_caption_cache(video_path)
        key = self._make_caption_key(frame_idx, model, detail_level)
        
        if "captions" not in cache_data:
            cache_data["captions"] = {}
        
        cache_data["captions"][key] = caption
        self._save_caption_cache(video_path)
        
        self.logger.debug(f"Caption cached: frame={frame_idx}, model={model}, level={detail_level}")
    
    def get_captions_batch(
        self,
        video_path: str,
        frame_indices: list,
        model: str,
        detail_level: str,
    ) -> Dict[int, Optional[str]]:
        """
        Get cached captions for multiple frames.
        
        Args:
            video_path: Path to video file
            frame_indices: List of frame indices
            model: Model name/path
            detail_level: Caption detail level
        
        Returns:
            Dict mapping frame_idx to caption (None if not cached)
        """
        if not self.enabled:
            return {idx: None for idx in frame_indices}
        
        cache_data = self._load_caption_cache(video_path)
        captions_cache = cache_data.get("captions", {})
        
        results = {}
        for idx in frame_indices:
            key = self._make_caption_key(idx, model, detail_level)
            results[idx] = captions_cache.get(key)
            
            if results[idx] is not None:
                self._hits += 1
            else:
                self._misses += 1
        
        cached_count = sum(1 for v in results.values() if v is not None)
        self.logger.debug(f"Caption batch lookup: {cached_count}/{len(frame_indices)} cached")
        
        return results
    
    def set_captions_batch(
        self,
        video_path: str,
        captions: Dict[int, str],
        model: str,
        detail_level: str,
    ) -> None:
        """
        Cache multiple captions at once.
        
        Args:
            video_path: Path to video file
            captions: Dict mapping frame_idx to caption
            model: Model name/path
            detail_level: Caption detail level
        """
        if not self.enabled or not captions:
            return
        
        cache_data = self._load_caption_cache(video_path)
        
        if "captions" not in cache_data:
            cache_data["captions"] = {}
        
        for idx, caption in captions.items():
            key = self._make_caption_key(idx, model, detail_level)
            cache_data["captions"][key] = caption
        
        self._save_caption_cache(video_path)
        self.logger.debug(f"Caption batch cached: {len(captions)} captions")
    
    # ========== InternVideo Description Cache API ==========
    
    def get_description(
        self,
        video_path: str,
        start_frame: int,
        end_frame: int,
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached description for a video segment.
        
        Args:
            video_path: Path to video file
            start_frame: Start frame index
            end_frame: End frame index
        
        Returns:
            Cached result dict or None if not cached
        """
        if not self.enabled:
            return None
        
        cache_data = self._load_description_cache(video_path)
        key = self._make_description_key(start_frame, end_frame)
        
        result = cache_data.get("descriptions", {}).get(key)
        
        if result is not None:
            self._hits += 1
            self.logger.debug(f"Description cache HIT: start={start_frame}, end={end_frame}")
        else:
            self._misses += 1
        
        return result
    
    def set_description(
        self,
        video_path: str,
        start_frame: int,
        end_frame: int,
        result: Dict[str, Any],
    ) -> None:
        """
        Cache a description for a video segment.
        
        Args:
            video_path: Path to video file
            start_frame: Start frame index
            end_frame: End frame index
            result: Result dict to cache (should contain 'description' key)
        """
        if not self.enabled:
            return
        
        cache_data = self._load_description_cache(video_path)
        key = self._make_description_key(start_frame, end_frame)
        
        if "descriptions" not in cache_data:
            cache_data["descriptions"] = {}
        
        # Store the serializable parts of the result
        cache_result = {
            "description": result.get("description", ""),
            "start_frame": start_frame,
            "end_frame": end_frame,
            "cached_at": datetime.now().isoformat(),
        }
        
        cache_data["descriptions"][key] = cache_result
        self._save_description_cache(video_path)
        
        self.logger.debug(f"Description cached: start={start_frame}, end={end_frame}")
    
    # ========== Statistics ==========
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        
        return {
            "hits": self._hits,
            "misses": self._misses,
            "total_lookups": total,
            "hit_rate": f"{hit_rate:.1%}",
            "enabled": self.enabled,
            "cache_dir": self.cache_dir,
        }
    
    def log_stats(self) -> None:
        """Log cache statistics."""
        stats = self.get_stats()
        self.logger.info(
            f"ToolCache stats: hits={stats['hits']}, misses={stats['misses']}, "
            f"hit_rate={stats['hit_rate']}"
        )
    
    def clear_cache(self, video_path: str = None) -> None:
        """
        Clear cache.
        
        Args:
            video_path: If provided, only clear cache for this video.
                       If None, clear all caches.
        """
        if video_path:
            video_hash = self._get_video_hash(video_path)
            
            # Clear in-memory
            self._caption_cache.pop(video_hash, None)
            self._description_cache.pop(video_hash, None)
            
            # Clear on-disk
            caption_path = self._get_caption_cache_path(video_path)
            desc_path = self._get_description_cache_path(video_path)
            
            for path in [caption_path, desc_path]:
                if os.path.exists(path):
                    os.remove(path)
            
            self.logger.info(f"Cleared cache for video: {video_path}")
        else:
            # Clear all
            self._caption_cache.clear()
            self._description_cache.clear()
            
            # Clear directories
            import shutil
            for subdir in ["captions", "descriptions"]:
                dir_path = os.path.join(self.cache_dir, subdir)
                if os.path.exists(dir_path):
                    shutil.rmtree(dir_path)
            
            self.logger.info("Cleared all tool caches")


# Global cache instance (lazy initialized)
_tool_cache: Optional[ToolCache] = None


def get_tool_cache(
    cache_dir: str = None,
    enabled: bool = True,
    logger: logging.Logger = None,
) -> ToolCache:
    """
    Get or create global ToolCache instance.
    
    Args:
        cache_dir: Cache directory (only used on first call)
        enabled: Whether caching is enabled
        logger: Logger instance
    
    Returns:
        ToolCache instance
    """
    global _tool_cache
    
    if _tool_cache is None:
        _tool_cache = ToolCache(
            cache_dir=cache_dir,
            enabled=enabled,
            logger=logger,
        )
    
    return _tool_cache





















