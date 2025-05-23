"""
Context Engine Bridge for Selfy.

This module provides a bridge to the context engine implementation.
It imports and re-exports the ContextEngine from the context_engine module.
"""

import logging
import time
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

# Try to import from context_engine implementation
try:
    # First try to import from the new V2 implementation
    from selfy_core.global_modules.config import get as config_get
    use_v2 = config_get('features.use_context_engine_v2', True)
    
    if use_v2:
        logger.info("Using Context Engine V2 implementation")
        from selfy_core.user_pipeline.context_engine.core import ContextEngineV2 as ContextEngine
        from selfy_core.user_pipeline.context_engine.data_structures import ContextEngineResult
    else:
        logger.info("Using Context Engine Adapter implementation")
        from selfy_core.user_pipeline.context_engine.adapter import ContextEngineAdapter as ContextEngine
        from selfy_core.user_pipeline.context_engine.adapter import ContextEngineResult
except ImportError:
    # Fall back to a mock implementation
    logger.warning("Could not import ContextEngine, using mock implementation")
    
    class ContextEngineResult:
        """
        Mock implementation of the ContextEngineResult.
        
        This is used to represent the result of context engine processing.
        """
        
        def __init__(self, 
                   response_text: str, 
                   capabilities_used: Optional[List[str]] = None,
                   execution_steps: Optional[List[Dict[str, Any]]] = None,
                   metadata: Optional[Dict[str, Any]] = None):
            """
            Initialize the context engine result.
            
            Args:
                response_text: The response text
                capabilities_used: List of capabilities used
                execution_steps: List of execution steps
                metadata: Additional metadata
            """
            self.response_text = response_text
            self.capabilities_used = capabilities_used or []
            self.execution_steps = execution_steps or []
            self.metadata = metadata or {}
    
    class ContextEngine:
        """
        Mock implementation of the ContextEngine.
        
        This is used as a fallback when the real implementation is not available.
        """
        
        def __init__(self, llm_wrapper=None, capability_manifest=None, identity_filter=None):
            """
            Initialize the mock context engine.
            
            Args:
                llm_wrapper: The LLM wrapper to use
                capability_manifest: The capability manifest to use
                identity_filter: The identity filter to use
            """
            self.llm_wrapper = llm_wrapper
            self.capability_manifest = capability_manifest
            self.identity_filter = identity_filter
            self._request_count = 0
            logger.info("Initialized mock ContextEngine")
        
        def process(self, processed_input):
            """
            Process a user request.
            
            Args:
                processed_input: The processed input from the input handler
                
            Returns:
                A ContextEngineResult object
            """
            # Track request count for metrics
            self._request_count += 1
            
            # Extract data from ProcessedInput
            user_input = processed_input.validated_input
            conversation_history = processed_input.conversation_history
            metadata = processed_input.input_metadata
            
            logger.info(f"Processing request #{self._request_count}: {user_input[:50]}...")
            
            # Generate a simple response
            response = f"You said: {user_input}"
            
            # Create a result
            result = ContextEngineResult(
                response_text=response,
                capabilities_used=[],
                execution_steps=[],
                metadata={
                    "processing_time_ms": 100,
                    "request_count": self._request_count,
                    "timestamp": time.time()
                }
            )
            
            return result

# Export the ContextEngine and ContextEngineResult
__all__ = ["ContextEngine", "ContextEngineResult"]
