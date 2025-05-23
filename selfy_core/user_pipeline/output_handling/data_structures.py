"""
Data structures for the Output Handling module.

This module defines the data structures used by the Output Handling module.
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional


@dataclass
class OutputResult:
    """
    Standardized return object for the output handling module.
    
    Attributes:
        status: The status of the output processing ('success' or 'error')
        formatted_output: The final, validated, formatted response text
        original_response: The response text received before processing
        identity_filter_status: Status from IdentityFilter ('allowed', 'modified', 'blocked')
        validation_status: Status from OutputValidator ('valid', 'invalid')
        format_used: The actual format applied ('text', 'markdown', etc.)
        error_message: Description if status is 'error'
    """
    status: str
    formatted_output: Optional[str]
    original_response: str
    identity_filter_status: str
    validation_status: str
    format_used: str
    error_message: Optional[str] = None


@dataclass
class ValidationResult:
    """
    Result of output validation.
    
    Attributes:
        status: The status of the validation ('valid' or 'invalid')
        issues: List of validation issues
        sanitized_output: The sanitized output if available
    """
    status: str
    issues: List[str]
    sanitized_output: Optional[str] = None


class OutputValidationError(Exception):
    """
    Raised when output fails validation rules.
    
    This exception is raised when the output fails validation checks.
    It carries an informative message about the validation failure.
    """
    pass
