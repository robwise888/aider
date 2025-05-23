"""
Embedding service for the Selfy agent.

This module provides functionality for generating vector embeddings from text.
It supports multiple embedding models and providers, with a focus on efficiency
and quality for semantic search.
"""

import time
import os
import sys
import platform
from typing import List, Dict, Any, Optional, Union

import numpy as np

from selfy_core.global_modules.config import get as config_get
from selfy_core.global_modules.logging import get_logger
from selfy_core.global_modules.llm_wrapper import create_default_provider
from selfy_core.global_modules.memory.model_cache_manager import get_model_cache_manager

# Set up logger
logger = get_logger(__name__)


class EmbeddingService:
    """
    Service for generating vector embeddings from text.

    This class provides functionality for generating embeddings from text strings.
    It supports multiple embedding models and providers, with a focus on efficiency
    and quality for semantic search.
    """

    def __init__(self):
        """Initialize the embedding service."""
        # Get configuration
        self.embedding_model = config_get('memory.embedding_service.model', 'all-MiniLM-L6-v2')
        self.embedding_provider = config_get('memory.embedding_service.provider', 'sentence_transformers')
        self.embedding_dimension = config_get('memory.embedding_service.dimension', 384)
        self.use_cache = config_get('memory.embedding_service.use_cache', True)
        self.use_gpu = config_get('memory.embedding_service.use_gpu', False)

        # Initialize embedding model
        self.model = None
        self.cache: Dict[str, List[float]] = {}

        logger.info(f"Initialized EmbeddingService with model={self.embedding_model}, "
                   f"provider={self.embedding_provider}, dimension={self.embedding_dimension}")

    def setup(self) -> bool:
        """
        Set up the embedding service.

        Returns:
            True if setup was successful, False otherwise
        """
        try:
            # Reload configuration in case it changed
            self.embedding_model = config_get('memory.embedding_service.model', 'all-MiniLM-L6-v2')
            self.embedding_provider = config_get('memory.embedding_service.provider', 'sentence_transformers')
            self.embedding_dimension = config_get('memory.embedding_service.dimension', 384)
            self.use_cache = config_get('memory.embedding_service.use_cache', True)
            self.use_gpu = config_get('memory.embedding_service.use_gpu', False)

            # Initialize the embedding model based on the provider
            if self.embedding_provider == 'sentence_transformers':
                try:
                    from sentence_transformers import SentenceTransformer

                    # Implementation with improved Windows-compatible model cache manager
                    logger.info(f"Loading SentenceTransformer model: {self.embedding_model}")
                    start_time = time.time()

                    # Get the model cache manager
                    cache_manager = get_model_cache_manager()
                    cache_dir = cache_manager.cache_dir

                    logger.info(f"Using model cache directory: {cache_dir}")

                    # Check if model is already cached
                    is_cached = cache_manager.is_model_cached(self.embedding_model)

                    # Log cache status with clear indication
                    if is_cached:
                        logger.info(f"Model found in cache - loading from {cache_dir}")
                        if sys.stdout.isatty():
                            print(f"\rModel found in cache - loading from local storage...", end="")
                    else:
                        logger.info(f"Model not found in cache - will download from Hugging Face")
                        if sys.stdout.isatty():
                            print(f"\rModel not found in cache - will download from Hugging Face...", end="")

                    # Initialize with timeout protection
                    try:
                        # Set a timeout for model loading (30 seconds)
                        if sys.stdout.isatty():
                            print(f"\rInitializing SentenceTransformer model...", end="")

                        # Check if GPU should be used
                        device = None
                        if self.use_gpu:
                            try:
                                import torch
                                if torch.cuda.is_available():
                                    device = "cuda"
                                    logger.info(f"Using GPU for SentenceTransformer model (CUDA available)")
                                    logger.info(f"CUDA version: {torch.version.cuda}")
                                    logger.info(f"CUDA device count: {torch.cuda.device_count()}")
                                    logger.info(f"Current CUDA device: {torch.cuda.current_device()}")
                                    logger.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")

                                    # Log memory usage
                                    allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
                                    reserved = torch.cuda.memory_reserved() / (1024 ** 2)    # MB
                                    logger.info(f"CUDA Memory: Allocated={allocated:.2f}MB, Reserved={reserved:.2f}MB")

                                    if sys.stdout.isatty():
                                        print(f"\rUsing GPU ({torch.cuda.get_device_name(0)}) for SentenceTransformer model...", end="")
                                else:
                                    logger.info("GPU requested but CUDA not available, falling back to CPU")
                                    logger.info(f"PyTorch version: {torch.__version__}")
                                    device = "cpu"
                            except ImportError:
                                logger.warning("PyTorch not available, cannot use GPU")
                                device = "cpu"
                        else:
                            device = "cpu"
                            logger.info("Using CPU for SentenceTransformer model (GPU not requested)")

                        # Initialize the model with the specified device
                        self.model = SentenceTransformer(self.embedding_model, cache_folder=cache_dir, device=device)

                        # Log the actual device being used
                        actual_device = self.model.device
                        logger.info(f"SentenceTransformer model initialized on device: {actual_device}")

                        if sys.stdout.isatty():
                            print(f"\rSentenceTransformer model initialized successfully on {actual_device}.", end="")
                    except Exception as e:
                        logger.error(f"Error loading model with cache: {e}")
                        # Fallback to loading without cache
                        logger.info("Falling back to loading without cache")
                        if sys.stdout.isatty():
                            print(f"\rFalling back to loading without cache...", end="")

                        # Try again without cache but still with device preference
                        try:
                            device = "cuda" if self.use_gpu else "cpu"
                            self.model = SentenceTransformer(self.embedding_model, device=device)
                            logger.info(f"Model loaded without cache on device: {self.model.device}")
                        except Exception:
                            # Final fallback - no cache, CPU only
                            logger.warning("Falling back to CPU-only model without cache")
                            self.model = SentenceTransformer(self.embedding_model)

                    load_time = time.time() - start_time
                    logger.info(f"SentenceTransformer model loaded in {load_time:.2f}s")

                    if sys.stdout.isatty():
                        print(f"\rSentenceTransformer model loaded in {load_time:.2f}s                ", end="\n")

                    # Update model usage in cache manager
                    cache_manager.update_model_usage(self.embedding_model)

                    # Run cache cleanup in background (will only run once a day)
                    cache_manager.cleanup_cache()

                    logger.info(f"Loaded SentenceTransformer model: {self.embedding_model} (from cache: {is_cached}, load time: {load_time:.2f}s)")
                except ImportError:
                    logger.warning("SentenceTransformer not available. Using fallback embedding method.")
                    self.model = None
            elif self.embedding_provider == 'llm':
                # Use the LLM wrapper for embeddings
                try:
                    self.model = create_default_provider()
                    if self.model is None:
                        logger.error("Failed to create LLM provider for embeddings")
                        return False
                    logger.info("Using LLM provider for embeddings")
                except Exception as e:
                    logger.error(f"Failed to create LLM provider for embeddings: {e}", exc_info=True)
                    self.model = None
            else:
                logger.warning(f"Unknown embedding provider: {self.embedding_provider}. Using fallback method.")
                self.model = None

            logger.info(f"Set up EmbeddingService with model={self.embedding_model}, "
                       f"provider={self.embedding_provider}, dimension={self.embedding_dimension}")
            return True
        except Exception as e:
            logger.error(f"Failed to set up EmbeddingService: {e}", exc_info=True)
            return False

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate an embedding for a text string.

        Args:
            text: The text to generate an embedding for

        Returns:
            The embedding as a list of floats
        """
        # Check if the embedding is already in the cache
        if self.use_cache and text in self.cache:
            logger.debug("Using cached embedding")
            return self.cache[text]

        start_time = time.time()

        try:
            # Generate embedding based on the provider
            if self.model is not None:
                if self.embedding_provider == 'sentence_transformers':
                    # Use SentenceTransformer
                    embedding = self.model.encode(text)
                    embedding = embedding.tolist()
                elif self.embedding_provider == 'llm':
                    # Use LLM wrapper
                    embedding = self.model.get_embedding(text, model=self.embedding_model)
                else:
                    # Fallback to simple embedding
                    embedding = self._generate_simple_embedding(text)
            else:
                # Fallback to simple embedding
                embedding = self._generate_simple_embedding(text)

            # Cache the embedding if enabled
            if self.use_cache:
                self.cache[text] = embedding

            generation_time_ms = (time.time() - start_time) * 1000
            logger.debug(f"Generated embedding in {generation_time_ms:.2f}ms")

            return embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}", exc_info=True)
            # Return a zero embedding as fallback
            return [0.0] * self.embedding_dimension

    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of text strings.

        Args:
            texts: The texts to generate embeddings for

        Returns:
            List of embeddings, each as a list of floats
        """
        # Check which texts are already in the cache
        if self.use_cache:
            uncached_texts = []
            uncached_indices = []

            for i, text in enumerate(texts):
                if text not in self.cache:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
        else:
            uncached_texts = texts
            uncached_indices = list(range(len(texts)))

        # If all texts are cached, return from cache
        if not uncached_texts:
            logger.debug(f"Using cached embeddings for all {len(texts)} texts")
            return [self.cache[text] for text in texts]

        start_time = time.time()

        try:
            # Generate embeddings based on the provider
            if self.model is not None:
                if self.embedding_provider == 'sentence_transformers':
                    # Use SentenceTransformer
                    embeddings = self.model.encode(uncached_texts)
                    embeddings = embeddings.tolist()
                elif self.embedding_provider == 'llm':
                    # Use LLM wrapper (one by one, as batch might not be supported)
                    embeddings = []
                    for text in uncached_texts:
                        embedding = self.model.get_embedding(text, model=self.embedding_model)
                        embeddings.append(embedding)
                else:
                    # Fallback to simple embedding
                    embeddings = [self._generate_simple_embedding(text) for text in uncached_texts]
            else:
                # Fallback to simple embedding
                embeddings = [self._generate_simple_embedding(text) for text in uncached_texts]

            # Cache the embeddings if enabled
            if self.use_cache:
                for i, text in enumerate(uncached_texts):
                    self.cache[text] = embeddings[i]

            # Create the result list with cached and newly generated embeddings
            result = []
            uncached_idx = 0

            for i in range(len(texts)):
                if i in uncached_indices:
                    result.append(embeddings[uncached_idx])
                    uncached_idx += 1
                else:
                    result.append(self.cache[texts[i]])

            generation_time_ms = (time.time() - start_time) * 1000
            logger.debug(f"Generated {len(uncached_texts)} embeddings in {generation_time_ms:.2f}ms")

            return result
        except Exception as e:
            logger.error(f"Failed to generate embeddings batch: {e}", exc_info=True)
            # Return zero embeddings as fallback
            return [[0.0] * self.embedding_dimension for _ in range(len(texts))]

    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embeddings.

        Returns:
            The dimension of the embeddings
        """
        return self.embedding_dimension

    def _generate_simple_embedding(self, text: str) -> List[float]:
        """
        Generate a simple embedding for a text string.

        This is a fallback method when the main embedding model is not available.
        It generates a simple embedding based on character frequencies.

        Args:
            text: The text to generate an embedding for

        Returns:
            The embedding as a list of floats
        """
        # Create a simple embedding based on character frequencies
        # This is just a fallback and not meant for production use
        char_counts = {}

        for char in text.lower():
            if char in char_counts:
                char_counts[char] += 1
            else:
                char_counts[char] = 1

        # Create a fixed-size vector
        embedding = [0.0] * self.embedding_dimension

        # Fill the vector with character frequencies
        for i, char in enumerate(sorted(char_counts.keys())):
            if i < self.embedding_dimension:
                embedding[i] = char_counts[char] / len(text)

        # Normalize the vector
        norm = sum(x * x for x in embedding) ** 0.5
        if norm > 0:
            embedding = [x / norm for x in embedding]

        logger.warning("Used simple fallback embedding method")
        return embedding


# Global instance
_embedding_service_instance = None


def setup_embedding_service() -> bool:
    """
    Set up the embedding service.

    Returns:
        True if setup was successful, False otherwise
    """
    global _embedding_service_instance

    try:
        logger.info("Starting embedding service setup")
        start_time = time.time()

        # Check if we're running in a terminal that supports progress bars
        use_progress_bar = sys.stdout.isatty()

        if _embedding_service_instance is None:
            logger.info("Creating new EmbeddingService instance")
            _embedding_service_instance = EmbeddingService()
            logger.info("EmbeddingService instance created")

        # Use the progress bar for the setup process
        if use_progress_bar:
            try:
                from selfy_core.global_modules.utils.progress_bar import IndeterminateProgressBar

                # Create and start the progress bar
                progress = IndeterminateProgressBar(
                    prefix="Loading sentence transformer model",
                    suffix="This may take a moment...",
                    show_time=True
                )
                progress.start()

                # Call setup
                result = _embedding_service_instance.setup()

                # Stop the progress bar
                setup_time = time.time() - start_time
                progress.stop(suffix=f"Model loaded successfully in {setup_time:.2f}s" if result else "Failed to load model")

            except ImportError:
                # Fall back to normal setup if progress bar is not available
                logger.info("About to call setup() on embedding service")
                result = _embedding_service_instance.setup()
                setup_time = time.time() - start_time
                logger.info(f"Embedding service setup completed in {setup_time:.2f}s with result: {result}")
        else:
            # Normal setup without progress bar
            logger.info("About to call setup() on embedding service")
            result = _embedding_service_instance.setup()
            setup_time = time.time() - start_time
            logger.info(f"Embedding service setup completed in {setup_time:.2f}s with result: {result}")

        return result
    except Exception as e:
        logger.error(f"Failed to set up embedding service: {e}", exc_info=True)
        return False


def get_embedding_service() -> Optional[EmbeddingService]:
    """
    Get the embedding service instance.

    Returns:
        The embedding service instance, or None if not set up
    """
    global _embedding_service_instance

    if _embedding_service_instance is None:
        logger.warning("Embedding service not initialized")

    return _embedding_service_instance
