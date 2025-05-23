"""
Setup functions for the input handling module.

This module provides functions to set up and access the input handling module.
"""

from typing import Optional

# Set up logger
try:
    from selfy_core.global_modules.logging import get_logger
    logger = get_logger(__name__)
except ImportError:
    # Fall back to standard logging
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

# Import the InputProcessor from the bridge
from selfy_core.user_pipeline.input_handling.bridge import InputProcessor

# Global instance
_input_processor_instance = None

def setup_input_handling() -> bool:
    """
    Set up the input handling module.
    
    This function initializes the input handling module by creating an instance of
    InputProcessor and setting it as the global instance.
    
    Returns:
        True if the input handling module was set up successfully, False otherwise
    """
    global _input_processor_instance
    
    try:
        logger.info("Setting up input handling module...")
        
        # Create an input processor
        _input_processor_instance = InputProcessor()
        
        logger.info("Input handling module set up successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to set up input handling module: {e}", exc_info=True)
        return False

def get_input_processor() -> Optional[InputProcessor]:
    """
    Get the input processor instance.
    
    Returns:
        The input processor instance, or None if it has not been set up
    """
    global _input_processor_instance
    
    if _input_processor_instance is None:
        logger.warning("Input processor has not been set up")
    
    return _input_processor_instance
