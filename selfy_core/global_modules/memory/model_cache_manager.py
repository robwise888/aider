"""
Model cache manager for Windows environments.

This module provides functionality for managing model caches on Windows,
where the Hugging Face cache system has limitations due to symlink restrictions.
"""

import os
import glob
import shutil
import time
import hashlib
from typing import Optional, List, Dict, Any
import json
import logging

from selfy_core.global_modules.logging import get_logger

# Set up logger
logger = get_logger(__name__)


class ModelCacheManager:
    """
    Manager for model caches on Windows environments.

    This class provides functionality for managing model caches on Windows,
    where the Hugging Face cache system has limitations due to symlink restrictions.
    It implements a custom caching mechanism that works around these limitations.
    """

    def __init__(self, cache_dir: str = None):
        """
        Initialize the model cache manager.

        Args:
            cache_dir: The directory to use for caching. If None, a default directory is used.
        """
        if cache_dir is None:
            # Use default cache directory in the project root
            self.cache_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
                "data", "model_cache"
            )
        else:
            self.cache_dir = cache_dir

        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)

        # Create a metadata file to track cached models
        self.metadata_file = os.path.join(self.cache_dir, "cache_metadata.json")
        self.metadata = self._load_metadata()

        # Set environment variable to disable symlinks warning
        os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

        logger.info(f"Initialized ModelCacheManager with cache_dir={self.cache_dir}")

    def _load_metadata(self) -> Dict[str, Any]:
        """
        Load metadata from the metadata file.

        Returns:
            The metadata as a dictionary
        """
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load metadata: {e}", exc_info=True)
                return {"models": {}, "last_cleanup": 0}
        else:
            return {"models": {}, "last_cleanup": 0}

    def _save_metadata(self):
        """Save metadata to the metadata file."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}", exc_info=True)

    def is_model_cached(self, model_name: str) -> bool:
        """
        Check if a model is cached.

        Args:
            model_name: The name of the model to check

        Returns:
            True if the model is cached, False otherwise
        """
        # Check if the model is in the metadata
        if model_name in self.metadata["models"]:
            # Check if the model directory exists
            model_dir = self.metadata["models"][model_name]["directory"]
            if os.path.exists(model_dir):
                return True
            else:
                # Remove from metadata if directory doesn't exist
                del self.metadata["models"][model_name]
                self._save_metadata()
                return False

        # Check if the model directory exists using the Hugging Face pattern
        model_dir_pattern = os.path.join(
            self.cache_dir, f"models--sentence-transformers--{model_name.replace('/', '--')}"
        )
        model_dirs = glob.glob(model_dir_pattern)
        
        if len(model_dirs) > 0:
            # Add to metadata
            self.metadata["models"][model_name] = {
                "directory": model_dirs[0],
                "last_used": time.time(),
                "size_bytes": self._get_directory_size(model_dirs[0])
            }
            self._save_metadata()
            return True
            
        return False

    def _get_directory_size(self, directory: str) -> int:
        """
        Get the size of a directory in bytes.

        Args:
            directory: The directory to get the size of

        Returns:
            The size of the directory in bytes
        """
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(directory):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                total_size += os.path.getsize(file_path)
        return total_size

    def update_model_usage(self, model_name: str):
        """
        Update the last used timestamp for a model.

        Args:
            model_name: The name of the model to update
        """
        if model_name in self.metadata["models"]:
            self.metadata["models"][model_name]["last_used"] = time.time()
            self._save_metadata()

    def cleanup_cache(self, max_size_bytes: int = 1024 * 1024 * 1024, min_age_days: int = 30):
        """
        Clean up the cache by removing old models.

        Args:
            max_size_bytes: The maximum size of the cache in bytes
            min_age_days: The minimum age of models to consider for removal in days
        """
        # Only run cleanup once a day
        if time.time() - self.metadata.get("last_cleanup", 0) < 86400:
            return

        try:
            # Get total cache size
            total_size = sum(model["size_bytes"] for model in self.metadata["models"].values())
            
            # If cache is smaller than max size, no need to clean up
            if total_size <= max_size_bytes:
                self.metadata["last_cleanup"] = time.time()
                self._save_metadata()
                return
                
            # Sort models by last used time
            models = [(name, data) for name, data in self.metadata["models"].items()]
            models.sort(key=lambda x: x[1]["last_used"])
            
            # Remove old models until cache size is below max size
            for name, data in models:
                # Skip models used in the last min_age_days
                if time.time() - data["last_used"] < min_age_days * 86400:
                    continue
                    
                # Remove model directory
                if os.path.exists(data["directory"]):
                    shutil.rmtree(data["directory"])
                    
                # Remove from metadata
                del self.metadata["models"][name]
                
                # Update total size
                total_size -= data["size_bytes"]
                
                logger.info(f"Removed old model from cache: {name}")
                
                # Stop if cache size is below max size
                if total_size <= max_size_bytes:
                    break
                    
            self.metadata["last_cleanup"] = time.time()
            self._save_metadata()
        except Exception as e:
            logger.error(f"Failed to clean up cache: {e}", exc_info=True)


# Global instance
_model_cache_manager_instance = None


def get_model_cache_manager() -> ModelCacheManager:
    """
    Get the model cache manager instance.

    Returns:
        The model cache manager instance
    """
    global _model_cache_manager_instance
    
    if _model_cache_manager_instance is None:
        _model_cache_manager_instance = ModelCacheManager()
        
    return _model_cache_manager_instance
