"""
Setup functions for the output handling module.

This module provides functions to set up and access the output handling module.
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

# Import the OutputProcessor from the bridge
from selfy_core.user_pipeline.output_handling.bridge import OutputProcessor

# Import the identity filter
try:
    from selfy_core.user_pipeline.identity import get_identity_filter
except ImportError:
    logger.warning("Could not import identity filter, using mock version")
    
    class MockIdentityFilter:
        """Mock identity filter for testing."""
        def filter_output(self, text, context_type=None):
            """Filter output text."""
            class MockResult:
                status = 'allowed'
                output_text = text
                reason = None
            return MockResult()
    
    def get_identity_filter():
        """Get a mock identity filter."""
        return MockIdentityFilter()

# Global instance
_output_processor_instance = None

def setup_output_handling() -> bool:
    """
    Set up the output handling module.
    
    This function initializes the output handling module by creating an instance of
    OutputProcessor and setting it as the global instance.
    
    Returns:
        True if the output handling module was set up successfully, False otherwise
    """
    global _output_processor_instance
    
    try:
        logger.info("Setting up output handling module...")
        
        # Get the identity filter
        identity_filter = get_identity_filter()
        if identity_filter is None:
            logger.warning("Identity filter not available, using mock version")
            identity_filter = MockIdentityFilter()
        
        # Create an output processor
        _output_processor_instance = OutputProcessor(identity_filter=identity_filter)
        
        logger.info("Output handling module set up successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to set up output handling module: {e}", exc_info=True)
        return False

def get_output_processor() -> Optional[OutputProcessor]:
    """
    Get the output processor instance.
    
    Returns:
        The output processor instance, or None if it has not been set up
    """
    global _output_processor_instance
    
    if _output_processor_instance is None:
        logger.warning("Output processor has not been set up")
    
    return _output_processor_instance
