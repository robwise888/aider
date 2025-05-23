"""
Data structures for the Input Handling module.

This module defines the data structures used by the Input Handling module.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any


@dataclass
class ProcessedInput:
    """
    Standardized output structure for the input handling module.
    
    Attributes:
        validated_input: The user input text after validation and preprocessing (whitespace trimming)
        conversation_history: The recent conversation history fetched from memory
        input_metadata: Dictionary containing key context identifiers
    """
    validated_input: str
    conversation_history: List[Any]  # Will be List[MemoryItem] when memory integration is complete
    input_metadata: Dict[str, Any] = field(default_factory=dict)


class InputValidationError(Exception):
    """
    Raised when input fails validation rules.
    
    This exception is raised when the input fails validation checks (empty, too long).
    It carries an informative message about the validation failure.
    """
    pass
