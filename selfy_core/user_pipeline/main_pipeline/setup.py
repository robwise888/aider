"""
Setup functions for the main pipeline module.

This module provides functions to set up and access the main pipeline module.
"""

import time
import uuid
from typing import Optional, Dict, Any

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

# Import dependencies
try:
    from selfy_core.user_pipeline.input_handling import get_input_processor
    from selfy_core.user_pipeline.context_engine import get_context_engine
    from selfy_core.user_pipeline.identity import get_identity_filter
    from selfy_core.user_pipeline.output_handling import get_output_processor
except ImportError:
    logger.warning("Could not import dependencies")

# Mock pipeline classes for testing
class PipelineResult:
    """
    Result of pipeline processing.
    
    This class represents the result of processing a request through the pipeline.
    """
    
    def __init__(self, 
                status: str, 
                final_response: str, 
                original_response: str = None,
                capabilities_used: list = None,
                execution_steps: list = None,
                processing_time_ms: float = None,
                session_id: str = None,
                metadata: Dict[str, Any] = None):
        """
        Initialize the pipeline result.
        
        Args:
            status: The status of the processing ('success' or 'error')
            final_response: The final response to the user
            original_response: The original response before processing
            capabilities_used: List of capabilities used
            execution_steps: List of execution steps
            processing_time_ms: Processing time in milliseconds
            session_id: The session ID
            metadata: Additional metadata
        """
        self.status = status
        self.final_response = final_response
        self.original_response = original_response or final_response
        self.capabilities_used = capabilities_used or []
        self.execution_steps = execution_steps or []
        self.processing_time_ms = processing_time_ms or 0
        self.session_id = session_id
        self.metadata = metadata or {}

class UserPipeline:
    """
    Main pipeline for the User Chat Pipeline.
    
    This class orchestrates the flow of information through the Input Handling,
    Context Engine, Identity, and Output Handling modules.
    """
    
    def __init__(self, 
                input_processor=None, 
                context_engine=None, 
                identity_filter=None, 
                output_processor=None):
        """
        Initialize the user pipeline.
        
        Args:
            input_processor: The input processor to use
            context_engine: The context engine to use
            identity_filter: The identity filter to use
            output_processor: The output processor to use
        """
        self.input_processor = input_processor
        self.context_engine = context_engine
        self.identity_filter = identity_filter
        self.output_processor = output_processor
        
        # Active conversations
        self.active_conversations = {}
        
        logger.info("UserPipeline initialized")
    
    def process_request(self, 
                       user_input: str, 
                       user_id: str = 'default_user', 
                       session_id: Optional[str] = None,
                       requested_format: Optional[str] = None,
                       context: Optional[Dict[str, Any]] = None) -> PipelineResult:
        """
        Process a user request through the pipeline.
        
        Args:
            user_input: The raw user input
            user_id: The user ID
            session_id: The session ID (will be generated if not provided)
            requested_format: The requested output format
            context: Additional context for processing
            
        Returns:
            A PipelineResult object containing the processing result
        """
        start_time = time.time()
        
        # Generate a session ID if not provided
        if session_id is None:
            session_id = str(uuid.uuid4())
            logger.info(f"Generated new session ID: {session_id}")
        
        # Prepare context
        if context is None:
            context = {}
        
        try:
            # Step 1: Process input
            processed_input = self.input_processor.process_input(
                raw_input=user_input,
                user_id=user_id,
                session_id=session_id
            )
            
            # Step 2: Process with context engine
            context_engine_result = self.context_engine.process(processed_input)
            
            # Step 3: Process output
            output_result = self.output_processor.process_output(
                response_text=context_engine_result.response_text,
                requested_format=requested_format,
                context=context
            )
            
            # Create and return the pipeline result
            return PipelineResult(
                status='success',
                final_response=output_result.formatted_output,
                original_response=context_engine_result.response_text,
                capabilities_used=context_engine_result.capabilities_used,
                execution_steps=context_engine_result.execution_steps,
                processing_time_ms=(time.time() - start_time) * 1000,
                session_id=session_id,
                metadata={
                    'user_id': user_id,
                    'session_id': session_id,
                    'timestamp': time.time()
                }
            )
        except Exception as e:
            logger.error(f"Error processing request: {e}", exc_info=True)
            
            # Get error message from config
            error_message = config_get(
                'pipeline.main.error_messages.unexpected',
                "I'm sorry, an unexpected error occurred. Please try again."
            )
            
            # Create and return an error result
            return PipelineResult(
                status='error',
                final_response=error_message,
                processing_time_ms=(time.time() - start_time) * 1000,
                session_id=session_id,
                metadata={
                    'user_id': user_id,
                    'session_id': session_id,
                    'timestamp': time.time(),
                    'error': str(e)
                }
            )

# Global instance
_user_pipeline_instance = None

def setup_user_pipeline() -> bool:
    """
    Set up the user pipeline.
    
    This function initializes the user pipeline by creating an instance of
    UserPipeline and setting it as the global instance.
    
    Returns:
        True if the user pipeline was set up successfully, False otherwise
    """
    global _user_pipeline_instance
    
    try:
        logger.info("Setting up user pipeline...")
        
        # Get dependencies
        input_processor = get_input_processor()
        if input_processor is None:
            logger.error("Input processor not available")
            return False
        
        context_engine = get_context_engine()
        if context_engine is None:
            logger.error("Context engine not available")
            return False
        
        identity_filter = get_identity_filter()
        if identity_filter is None:
            logger.error("Identity filter not available")
            return False
        
        output_processor = get_output_processor()
        if output_processor is None:
            logger.error("Output processor not available")
            return False
        
        # Create the user pipeline
        _user_pipeline_instance = UserPipeline(
            input_processor=input_processor,
            context_engine=context_engine,
            identity_filter=identity_filter,
            output_processor=output_processor
        )
        
        logger.info("User pipeline set up successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to set up user pipeline: {e}", exc_info=True)
        return False

def get_user_pipeline() -> Optional[UserPipeline]:
    """
    Get the user pipeline instance.
    
    Returns:
        The user pipeline instance, or None if it has not been set up
    """
    global _user_pipeline_instance
    
    if _user_pipeline_instance is None:
        logger.warning("User pipeline has not been set up")
    
    return _user_pipeline_instance
