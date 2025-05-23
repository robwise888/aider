"""
Main pipeline module for the User Chat Pipeline.

This module provides the UserChatPipeline class, which orchestrates the flow of
user requests through the various components of the pipeline.
"""

import time
import uuid
from typing import Dict, List, Any, Optional, Union

from selfy_core.global_modules.config import get as config_get
from selfy_core.global_modules.logging import get_logger
from selfy_core.global_modules.memory import (
    MemoryItem, MemoryItemType, setup_memory_core, get_memory_system
)
from selfy_core.global_modules.llm_wrapper import (
    setup_llm_providers, create_default_provider
)
from selfy_core.global_modules.capability_manifest import (
    setup_capability_manifest, get_registry
)

from selfy_core.user_pipeline.input_handling import (
    ProcessedInput, setup_input_processor, get_input_processor
)
from selfy_core.user_pipeline.context_engine import (
    ContextEngineResult, setup_context_engine, get_context_engine
)
from selfy_core.user_pipeline.output_handling import (
    OutputResult, setup_output_processor, get_output_processor
)
from selfy_core.user_pipeline.identity import (
    setup_identity_system, get_identity_manager, get_identity_filter
)

# Set up logger
logger = get_logger(__name__)


class UserChatPipeline:
    """
    Main pipeline for the User Chat Pipeline.

    This class orchestrates the flow of user requests through the various
    components of the pipeline, including input handling, context engine,
    and output handling.
    """

    def __init__(self):
        """Initialize the user chat pipeline."""
        logger.info("Initializing UserChatPipeline")
        self.initialized = False

    def setup(self) -> bool:
        """
        Set up the user chat pipeline.

        This method initializes all the components of the pipeline.

        Returns:
            True if setup was successful, False otherwise
        """
        try:
            logger.info("Setting up UserChatPipeline")

            # Set up global modules
            logger.info("Setting up global modules")

            # Setup LLM providers
            logger.info("DEBUG: About to set up LLM providers")
            start_time = time.time()
            llm_result = setup_llm_providers()
            llm_setup_time = time.time() - start_time
            logger.info(f"DEBUG: LLM providers setup completed in {llm_setup_time:.2f}s with result: {llm_result}")

            if not llm_result:
                logger.error("Failed to set up LLM providers")
                return False

            # Setup memory core
            logger.info("DEBUG: About to set up memory core")
            start_time = time.time()
            memory_result = setup_memory_core()
            memory_setup_time = time.time() - start_time
            logger.info(f"DEBUG: Memory core setup completed in {memory_setup_time:.2f}s with result: {memory_result}")

            if not memory_result:
                logger.error("Failed to set up memory core")
                return False

            # Setup capability manifest
            logger.info("DEBUG: About to set up capability manifest")
            start_time = time.time()
            capability_result = setup_capability_manifest()
            capability_setup_time = time.time() - start_time
            logger.info(f"DEBUG: Capability manifest setup completed in {capability_setup_time:.2f}s with result: {capability_result}")

            if not capability_result:
                logger.error("Failed to set up capability manifest")
                return False

            # Set up identity system
            logger.info("Setting up identity system")
            if not setup_identity_system():
                logger.error("Failed to set up identity system")
                return False

            # Get identity filter
            identity_filter = get_identity_filter()
            if not identity_filter:
                logger.error("Failed to get identity filter")
                return False

            # Set up LLM wrapper
            logger.info("Setting up LLM wrapper")
            llm_wrapper = create_default_provider()
            if not llm_wrapper:
                logger.error("Failed to create default LLM provider")
                return False

            # Get capability manifest
            logger.info("Getting capability manifest")
            capability_manifest = get_registry()
            if not capability_manifest:
                logger.error("Failed to get capability manifest")
                return False

            # Set up pipeline components
            logger.info("Setting up pipeline components")
            if not setup_input_processor():
                logger.error("Failed to set up input processor")
                return False

            # Get memory system
            memory_system = get_memory_system()
            if not memory_system:
                logger.error("Failed to get memory system")
                return False

            if not setup_context_engine(
                llm_wrapper=llm_wrapper,
                capability_manifest=capability_manifest,
                identity_filter=identity_filter,
                memory_system=memory_system
            ):
                logger.error("Failed to set up context engine")
                return False

            if not setup_output_processor(identity_filter=identity_filter):
                logger.error("Failed to set up output processor")
                return False

            self.initialized = True
            logger.info("UserChatPipeline set up successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to set up UserChatPipeline: {e}", exc_info=True)
            return False

    def process_request(self,
                       user_input: str,
                       user_id: str,
                       session_id: str,
                       output_format: str = "text") -> Dict[str, Any]:
        """
        Process a user request.

        This method orchestrates the flow of a user request through the pipeline.

        Args:
            user_input: The raw user input
            user_id: The user ID
            session_id: The session ID
            output_format: The desired output format (text, markdown, json)

        Returns:
            Dictionary containing the response and metadata
        """
        start_time = time.time()
        logger.info(f"Processing request for user_id={user_id}, session_id={session_id}")

        # Check if pipeline is initialized
        if not self.initialized:
            logger.error("UserChatPipeline not initialized")
            return {
                "status": "error",
                "error": "Pipeline not initialized",
                "response": "I'm sorry, but I'm not fully initialized yet. Please try again later."
            }

        try:
            # Get pipeline components
            input_processor = get_input_processor()
            context_engine = get_context_engine()
            output_processor = get_output_processor()
            memory_system = get_memory_system()

            if not input_processor or not context_engine or not output_processor or not memory_system:
                logger.error("Pipeline components not initialized")
                return {
                    "status": "error",
                    "error": "Pipeline components not initialized",
                    "response": "I'm sorry, but I'm not fully initialized yet. Please try again later."
                }

            # Step 1: Process input
            logger.info("Step 1: Processing input")
            processed_input = input_processor.process_input(
                raw_input=user_input,
                user_id=user_id,
                session_id=session_id
            )

            # Step 2: Process through context engine
            logger.info("Step 2: Processing through context engine")

            # Process request normally
            context_result = context_engine.process_request(processed_input)

            # Step 3: Process output
            logger.info("Step 3: Processing output")
            output_result = output_processor.process_output(
                response_text=context_result.response_text,
                requested_format=output_format,
                context={
                    "user_id": user_id,
                    "session_id": session_id,
                    "capabilities_used": context_result.capabilities_used
                }
            )

            # Step 4: Store assistant response in memory
            logger.info("Step 4: Storing assistant response in memory")
            self._store_assistant_response(
                response=output_result.formatted_output,
                user_id=user_id,
                session_id=session_id
            )

            # Step 5: Prepare result
            processing_time = time.time() - start_time
            logger.info(f"Request processed in {processing_time:.4f} seconds")

            return {
                "status": "success" if output_result.status == "success" else "warning",
                "response": output_result.formatted_output,
                "capabilities_used": context_result.capabilities_used,
                "execution_steps": context_result.execution_steps,
                "processing_time": processing_time,
                "metadata": {
                    "input_metadata": processed_input.input_metadata,
                    "context_metadata": context_result.metadata,
                    "output_metadata": {
                        "format_used": output_result.format_used,
                        "validation_status": output_result.validation_status,
                        "identity_filter_status": output_result.identity_filter_status
                    }
                }
            }
        except Exception as e:
            logger.error(f"Error processing request: {e}", exc_info=True)
            processing_time = time.time() - start_time

            return {
                "status": "error",
                "error": str(e),
                "response": "I'm sorry, but I encountered an error while processing your request.",
                "processing_time": processing_time
            }

    def _store_assistant_response(self, response: str, user_id: str, session_id: str) -> None:
        """
        Store assistant response in memory.

        Args:
            response: The assistant response
            user_id: The user ID
            session_id: The session ID
        """
        try:
            # Get memory system
            memory_system = get_memory_system()
            if not memory_system:
                logger.error("Memory system not initialized")
                return

            # Create memory item
            memory_item = MemoryItem(
                id=str(uuid.uuid4()),
                timestamp=time.time(),
                type=MemoryItemType.CONVERSATION_TURN,
                content=response,
                metadata={
                    "user_id": user_id,
                    "session_id": session_id,
                    "role": "assistant"
                }
            )

            # Add to memory
            memory_system.add_memory(memory_item)
            logger.debug(f"Stored assistant response in memory for session_id={session_id}")
        except Exception as e:
            logger.error(f"Failed to store assistant response: {e}", exc_info=True)


# Global instance
_pipeline_instance = None


def setup_pipeline() -> bool:
    """
    Set up the user chat pipeline.

    Returns:
        True if setup was successful, False otherwise
    """
    global _pipeline_instance

    try:
        if _pipeline_instance is None:
            _pipeline_instance = UserChatPipeline()

        return _pipeline_instance.setup()
    except Exception as e:
        logger.error(f"Failed to set up pipeline: {e}", exc_info=True)
        return False


def get_pipeline() -> Optional[UserChatPipeline]:
    """
    Get the user chat pipeline instance.

    Returns:
        The user chat pipeline instance, or None if not set up
    """
    global _pipeline_instance

    if _pipeline_instance is None:
        logger.warning("User chat pipeline not initialized")

    return _pipeline_instance
