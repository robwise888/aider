"""
Input Handling module for the User Chat Pipeline.

This module performs initial processing of user text input for the User Chat Pipeline.
It validates input (non-empty, length), retrieves recent conversation history from memory,
stores the current input turn to memory, and packages the validated input and history
for the next pipeline stage (Context Engine).
"""

from selfy_core.user_pipeline.input_handling.processor import (
    InputProcessor, setup_input_processor, get_input_processor
)
from selfy_core.user_pipeline.input_handling.data_structures import (
    ProcessedInput, InputValidationError
)

__all__ = [
    'InputProcessor',
    'setup_input_processor',
    'get_input_processor',
    'ProcessedInput',
    'InputValidationError'
]