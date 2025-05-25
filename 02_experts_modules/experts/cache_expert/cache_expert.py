"""
Cache Expert Module for EzioFilhoUnified Trading System

This expert provides caching capabilities for all other experts in the system,
allowing them to store and retrieve data with time-to-live (TTL) functionality.
"""

import os
import json
import time
import logging
import pathlib
from typing import Any, Dict, Optional, Union
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("CacheExpert")

# Try to import diskcache, fall back to pure Python implementation if not available
try:
    import diskcache
    DISKCACHE_AVAILABLE = True
    logger.info("Using diskcache backend for caching")
except ImportError:
    DISKCACHE_AVAILABLE = False
    logger.warning("diskcache not available, falling back to JSON-based caching")


class CacheExpert:
    """
    Expert class that provides caching functionality for the EzioFilhoUnified system.

    Supports both memory and disk-based caching with TTL (Time-To-Live) functionality.
    """

    def __init__(self, cache_dir: str = "./data/shared_cache", use_diskcache: bool = True):
        self.cache_dir = pathlib.Path(cache_dir)
        self.use_diskcache = use_diskcache and DISKCACHE_AVAILABLE

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Cache directory set to: {self.cache_dir.absolute()}")

        if self.use_diskcache:
            self.cache = diskcache.Cache(str(self.cache_dir))
            logger.info("Initialized diskcache backend")
        else:
            self.memory_cache: Dict[str, Dict] = {}
            logger.info("Initialized JSON file-based cache backend")

    def process(self, query: str, context: Dict[str, Any]) -> tuple[Any, float]:
        key = self._generate_cache_key(query, context)
        result = self.get_from_cache(key)
        if result is not None:
            return result, 1.0  # Full confidence on hit
        return None, 0.0

    def set(self, namespace: str, key: str, value: Any, ttl: int = 3600) -> None:
        full_key = f"{namespace}:{key}"
        self.set_to_cache(full_key, value, ttl=ttl)

    def get_from_cache(self, key: str) -> Any:
        if not key:
            logger.warning("Empty key provided to get_from_cache")
            return None

        if self.use_diskcache:
            value = self.cache.get(key, default=None)
            logger.info(f"Cache {'hit' if value is not None else 'miss'} for key: {key}")
            return value
        else:
            self._cleanup_expired()

            if key in self.memory_cache:
                if self.memory_cache[key].get('expiry', float('inf')) > time.time():
                    return self.memory_cache[key]['value']
                else:
                    del self.memory_cache[key]

            cache_file = self._get_cache_file_path(key)
            if cache_file.exists():
                try:
                    with open(cache_file, 'r') as f:
                        data = json.load(f)
                    if data.get('expiry', float('inf')) > time.time():
                        self.memory_cache[key] = data
                        return data['value']
                    else:
                        cache_file.unlink(missing_ok=True)
                except (json.JSONDecodeError, IOError) as e:
                    logger.warning(f"Error reading cache file for key {key}: {e}")

            return None

    def set_to_cache(self, key: str, value: Any, ttl: int = 3600) -> None:
        if not key:
            logger.warning("Empty key provided to set_to_cache")
            return

        expiry = time.time() + ttl

        if self.use_diskcache:
            self.cache.set(key, value, expire=ttl)
            logger.info(f"Stored in diskcache with key: {key}, TTL: {ttl}s")
        else:
            self.memory_cache[key] = {
                'value': value,
                'expiry': expiry,
                'created_at': time.time()
            }
            cache_file = self._get_cache_file_path(key)
            try:
                with open(cache_file, 'w') as f:
                    json.dump({
                        'value': value,
                        'expiry': expiry,
                        'created_at': time.time()
                    }, f)
            except (TypeError, IOError) as e:
                logger.warning(f"Failed to write to cache file for key {key}: {e}")

    def delete_cache(self, key: str) -> None:
        if not key:
            logger.warning("Empty key provided to delete_cache")
            return

        if self.use_diskcache:
            if key in self.cache:
                del self.cache[key]
        else:
            if key in self.memory_cache:
                del self.memory_cache[key]
            cache_file = self._get_cache_file_path(key)
            if cache_file.exists():
                try:
                    cache_file.unlink()
                except IOError as e:
                    logger.warning(f"Error deleting cache file for key {key}: {e}")

    def clear_all_cache(self) -> None:
        if self.use_diskcache:
            self.cache.clear()
        else:
            self.memory_cache.clear()
            for cache_file in self.cache_dir.glob("*.cache"):
                try:
                    cache_file.unlink()
                except IOError as e:
                    logger.warning(f"Error clearing cache file: {e}")

    def _generate_cache_key(self, query: str, context: Dict[str, Any]) -> str:
        key_base = f"{query}:{json.dumps(context, sort_keys=True)}"
        return ''.join(c if c.isalnum() else '_' for c in key_base)

    def _get_cache_file_path(self, key: str) -> pathlib.Path:
        safe_key = ''.join(c if c.isalnum() else '_' for c in key)
        return self.cache_dir / f"{safe_key}.cache"

    def _cleanup_expired(self) -> None:
        now = time.time()
        expired = [k for k, v in self.memory_cache.items() if v.get('expiry', float('inf')) <= now]
        for k in expired:
            del self.memory_cache[k]


if __name__ == "__main__":
    logger.info("Running CacheExpert as standalone test")
    cache = CacheExpert(use_diskcache=False)
    cache.set_to_cache("test", {"value": 42}, ttl=2)
    print("Cached value:", cache.get_from_cache("test"))
    time.sleep(3)
    print("Expired value:", cache.get_from_cache("test"))
