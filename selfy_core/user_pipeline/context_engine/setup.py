"""
Setup functions for the context engine module.

This module provides functions to set up and access the context engine module.
"""

from typing import Optional, Any

# Set up logger
try:
    from selfy_core.global_modules.logging import get_logger
    logger = get_logger(__name__)
except ImportError:
    # Fall back to standard logging
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

# Import configuration
try:
    from selfy_core.global_modules.config import get as config_get
except ImportError:
    logger.warning("Could not import config, using default values")
    
    def config_get(key, default=None):
        """Mock config_get function."""
        return default

# Import the ContextEngine from the bridge
from selfy_core.user_pipeline.context_engine.bridge import ContextEngine

# Import dependencies
try:
    from selfy_core.global_modules.llm_wrapper import get_llm_provider
    from selfy_core.global_modules.capability_manifest import get_capability_manifest
    from selfy_core.user_pipeline.identity import get_identity_filter
except ImportError:
    logger.warning("Could not import dependencies, using mock versions")
    
    def get_llm_provider():
        """Get a mock LLM provider."""
        return None
    
    def get_capability_manifest():
        """Get a mock capability manifest."""
        return None
    
    def get_identity_filter():
        """Get a mock identity filter."""
        return None

# Global instance
_context_engine_instance = None

def setup_context_engine() -> bool:
    """
    Set up the context engine module.
    
    This function initializes the context engine module by creating an instance of
    ContextEngine and setting it as the global instance.
    
    Returns:
        True if the context engine module was set up successfully, False otherwise
    """
    global _context_engine_instance
    
    try:
        logger.info("Setting up context engine module...")
        
        # Get dependencies
        llm_wrapper = get_llm_provider()
        capability_manifest = get_capability_manifest()
        identity_filter = get_identity_filter()
        
        # Get configuration
        cache_size = config_get('context_engine.cache_size', 100)
        
        # Create a context engine
        _context_engine_instance = ContextEngine(
            llm_wrapper=llm_wrapper,
            capability_manifest=capability_manifest,
            identity_filter=identity_filter
        )
        
        logger.info("Context engine module set up successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to set up context engine module: {e}", exc_info=True)
        return False

def get_context_engine() -> Optional[Any]:
    """
    Get the context engine instance.
    
    Returns:
        The context engine instance, or None if it has not been set up
    """
    global _context_engine_instance
    
    if _context_engine_instance is None:
        logger.warning("Context engine has not been set up")
    
    return _context_engine_instance
