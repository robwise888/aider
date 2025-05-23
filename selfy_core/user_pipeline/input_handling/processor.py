"""
Input Processor for the User Chat Pipeline.

This module provides the InputProcessor class, which is responsible for
validating user input, retrieving conversation history from memory,
storing the current input turn to memory, and packaging the validated
input and history for the next pipeline stage.
"""

import time
from typing import Dict, Any, List, Optional

from selfy_core.global_modules.config import get as config_get
from selfy_core.global_modules.logging import get_logger
from selfy_core.user_pipeline.input_handling.data_structures import ProcessedInput, InputValidationError
from selfy_core.user_pipeline.input_handling.memory_integration import get_recent_conversation, add_memory_item

# Set up logger
logger = get_logger(__name__)


class InputProcessor:
    """
    Encapsulates the logic for handling user input.

    The InputProcessor is responsible for:
    1. Validating input (non-empty, length)
    2. Retrieving recent conversation history from memory
    3. Storing the current input turn to memory
    4. Packaging the validated input and history for the next pipeline stage
    """

    def __init__(self):
        """
        Initialize the input processor.

        Loads configuration values needed for input processing.
        """
        # Load configuration
        self.max_input_length = config_get('pipeline.input_handling.max_input_length', 2048)
        self.max_history_turns = config_get('pipeline.input_handling.history.max_turns', 10)

        logger.info(f"InputProcessor initialized with max_input_length={self.max_input_length}, "
                   f"max_history_turns={self.max_history_turns}")

    def process_input(self, raw_input: str, user_id: str, session_id: str, conversation_history: Optional[List[Any]] = None) -> ProcessedInput:
        """
        Process user input.

        This method takes raw user text and session identifiers, performs validation,
        interacts with the memory system for history and storage, and returns a
        structured output.

        Args:
            raw_input: The raw text input received from the user
            user_id: Identifier for the user
            session_id: Identifier for the current conversation session
            conversation_history: Optional pre-existing conversation history. If provided,
                                 this will be used instead of retrieving from memory.

        Returns:
            A ProcessedInput object containing the validated input, conversation history,
            and metadata

        Raises:
            InputValidationError: If the input fails validation checks (empty, too long)
        """
        start_time = time.time()
        logger.info(f"Processing input for user_id={user_id}, session_id={session_id}")

        # Validate input
        validated_input = self._validate_input(raw_input)

        # Get conversation history from memory if not provided
        if conversation_history is None:
            conversation_history = self._get_conversation_history(session_id)

        # Store input to memory
        self._store_input_to_memory(validated_input, user_id, session_id)

        # Create metadata
        input_metadata = {
            'user_id': user_id,
            'session_id': session_id,
            'timestamp': time.time(),
            'processing_time': time.time() - start_time
        }

        # Create processed input
        processed_input = ProcessedInput(
            validated_input=validated_input,
            conversation_history=conversation_history,
            input_metadata=input_metadata
        )

        logger.info(f"Input processed in {time.time() - start_time:.4f} seconds")
        return processed_input

    def _validate_input(self, raw_input: str) -> str:
        """
        Validate user input.

        Performs basic validation checks on the input:
        - Non-empty
        - Maximum length

        Args:
            raw_input: The raw text input received from the user

        Returns:
            The validated and preprocessed input

        Raises:
            InputValidationError: If the input fails validation checks
        """
        # Check if input is empty
        if not raw_input or raw_input.strip() == '':
            error_message = "Input cannot be empty"
            logger.warning(error_message)
            raise InputValidationError(error_message)

        # Check input length
        if len(raw_input) > self.max_input_length:
            error_message = f"Input exceeds maximum length of {self.max_input_length} characters"
            logger.warning(error_message)
            raise InputValidationError(error_message)

        # Preprocess input (whitespace trimming)
        validated_input = raw_input.strip()

        logger.debug(f"Input validated successfully: {validated_input[:50]}...")
        return validated_input

    def _get_conversation_history(self, session_id: str) -> List[Any]:
        """
        Get conversation history from memory.

        Retrieves recent conversation history from the memory system.

        Args:
            session_id: Identifier for the current conversation session

        Returns:
            List of conversation turns from memory
        """
        logger.debug(f"Getting conversation history for session_id={session_id}")
        return get_recent_conversation(session_id, self.max_history_turns)

    def _store_input_to_memory(self, validated_input: str, user_id: str, session_id: str) -> None:
        """
        Store input to memory.

        Stores the current validated user input turn to memory.

        Args:
            validated_input: The validated user input
            user_id: Identifier for the user
            session_id: Identifier for the current conversation session
        """
        logger.debug(f"Storing input to memory for user_id={user_id}, session_id={session_id}")
        add_memory_item(validated_input, user_id, session_id)


# Global instance
_input_processor_instance = None


def setup_input_processor() -> bool:
    """
    Set up the input processor.

    Returns:
        True if setup was successful, False otherwise
    """
    global _input_processor_instance

    try:
        _input_processor_instance = InputProcessor()
        logger.info("Input processor set up successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to set up input processor: {e}")
        return False


def get_input_processor() -> Optional[InputProcessor]:
    """
    Get the input processor instance.

    Returns:
        The input processor instance, or None if not set up
    """
    global _input_processor_instance

    if _input_processor_instance is None:
        logger.warning("Input processor not initialized")

    return _input_processor_instance
