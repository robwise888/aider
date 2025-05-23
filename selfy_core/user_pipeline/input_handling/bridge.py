"""
Input Handler Bridge for Selfy.

This module provides a bridge to the input handling implementation.
It imports and re-exports the InputProcessor from the input_handling module.
"""

import logging
from typing import Dict, Any, Optional, List

# Set up logger
try:
    from selfy_core.global_modules.logging import get_logger
    logger = get_logger(__name__)
except ImportError:
    # Fall back to standard logging
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

# Try to import from input_handling implementation
try:
    from selfy_core.user_pipeline.input_handling.processor import InputProcessor
    logger.info("Using production input_handling implementation")
except ImportError:
    # Fall back to a mock implementation
    logger.warning("Could not import InputProcessor, using mock implementation")
    
    class InputProcessor:
        """
        Mock implementation of the InputProcessor.
        
        This is used as a fallback when the real implementation is not available.
        """
        
        def __init__(self):
            """Initialize the mock input processor."""
            logger.info("Initialized mock InputProcessor")
        
        def process_input(self, 
                        raw_input: str, 
                        user_id: str, 
                        session_id: str,
                        conversation_history: Optional[List[Any]] = None) -> Dict[str, Any]:
            """
            Process user input.
            
            Args:
                raw_input: The raw text input received from the user
                user_id: Identifier for the user
                session_id: Identifier for the current conversation session
                conversation_history: Optional pre-existing conversation history
                
            Returns:
                A ProcessedInput object containing the validated input, conversation history,
                and metadata
            """
            logger.info(f"Processing input: {raw_input[:100]}...")
            
            # Create a minimal result
            from dataclasses import dataclass
            
            @dataclass
            class ProcessedInput:
                validated_input: str
                conversation_history: List[Any]
                input_metadata: Dict[str, Any]
            
            return ProcessedInput(
                validated_input=raw_input,
                conversation_history=conversation_history or [],
                input_metadata={
                    "user_id": user_id,
                    "session_id": session_id
                }
            )

# Export the InputProcessor
__all__ = ["InputProcessor"]
