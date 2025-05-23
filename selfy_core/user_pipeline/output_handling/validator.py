"""
Output Validator for the Output Handling module.

This module provides the OutputValidator class, which is responsible for
validating output before delivery to ensure quality and safety.
"""

import re
import time
from typing import Dict, List, Any, Optional, Tuple

from selfy_core.global_modules.config import get as config_get
from selfy_core.global_modules.logging import get_logger
from selfy_core.user_pipeline.output_handling.data_structures import ValidationResult

# Set up logger
logger = get_logger(__name__)


class OutputValidator:
    """
    Validates output before delivery to ensure quality and safety.

    The OutputValidator is responsible for:
    1. Checking output for basic validation rules (non-empty, length)
    2. Ensuring output meets basic safety guidelines
    3. Providing sanitized output when possible
    """

    def __init__(self):
        """
        Initialize the output validator.

        Loads configuration values needed for validation.
        """
        # Load configuration
        self.min_output_length = config_get('pipeline.output_handling.validation.min_length', 1)
        self.max_output_length = config_get('pipeline.output_handling.validation.max_length', 8192)
        self.enable_output_sanitization = config_get('pipeline.output_handling.validation.enable_sanitization', True)

        # Load safety patterns from configuration
        config_safety_patterns = config_get('pipeline.output_handling.validation.safety_patterns', [])
        
        # Default safety patterns
        default_safety_patterns = [
            (r"(?i)\b(I cannot|I'm unable to|I am unable to|I'm not able to|I am not able to) (assist|help|provide|answer|respond)", "refusal"),
            (r"(?i)\b(I apologize|I'm sorry|I am sorry), (but|however) (I cannot|I'm unable to|I am unable to|I'm not able to|I am not able to)", "refusal"),
            (r"(?i)\b(as an AI|as an artificial intelligence|as a language model|as an assistant)", "self-reference"),
            (r"(?i)\b(my training|my programming|my knowledge|my data|my capabilities)", "self-reference"),
            (r"(?i)\b(OpenAI|Anthropic|Claude|GPT|ChatGPT)", "provider-reference")
        ]
        
        # Combine default and config patterns
        self.safety_patterns = []
        for pattern, issue_type in default_safety_patterns + config_safety_patterns:
            try:
                compiled_pattern = re.compile(pattern)
                self.safety_patterns.append((compiled_pattern, issue_type))
            except re.error as e:
                logger.warning(f"Invalid safety pattern '{pattern}': {e}")
        
        logger.info(f"OutputValidator initialized with {len(self.safety_patterns)} safety patterns")

    def validate(self, text: str) -> ValidationResult:
        """
        Validate output.

        Args:
            text: The output to validate

        Returns:
            A ValidationResult object containing the validation result
        """
        issues = []

        # Check if output is None or empty
        if text is None or text.strip() == '':
            issues.append("Output cannot be empty")
            return ValidationResult(status='invalid', issues=issues, sanitized_output=None)

        # Check output length
        if len(text) < self.min_output_length:
            issues.append(f"Output must be at least {self.min_output_length} characters")
            return ValidationResult(status='invalid', issues=issues, sanitized_output=None)

        if len(text) > self.max_output_length:
            issues.append(f"Output exceeds maximum length of {self.max_output_length} characters")
            sanitized_output = text[:self.max_output_length] if self.enable_output_sanitization else None
            return ValidationResult(status='invalid', issues=issues, sanitized_output=sanitized_output)

        # Check safety patterns
        for pattern, issue_type in self.safety_patterns:
            if re.search(pattern, text):
                issues.append(f"Output has safety issue: {issue_type}")
                sanitized_output = self.sanitize(text, issue_type) if self.enable_output_sanitization else None
                if sanitized_output and sanitized_output != text:
                    logger.warning(f"Output sanitized for safety issue: {issue_type}")
                    return ValidationResult(status='invalid', issues=issues, sanitized_output=sanitized_output)
                return ValidationResult(status='invalid', issues=issues, sanitized_output=None)

        # Output is valid
        return ValidationResult(status='valid', issues=[], sanitized_output=None)

    def sanitize(self, text: str, issue_type: str) -> Optional[str]:
        """
        Sanitize output based on the issue type.

        Args:
            text: The output to sanitize
            issue_type: The type of issue to sanitize

        Returns:
            The sanitized output, or None if sanitization is not possible
        """
        if not self.enable_output_sanitization:
            return None

        # Sanitize based on issue type
        if issue_type == 'refusal':
            # Replace refusal with a more helpful response
            sanitized = re.sub(
                r"(?i)\b(I cannot|I'm unable to|I am unable to|I'm not able to|I am not able to) (assist|help|provide|answer|respond).*",
                "I'll do my best to help with that. ",
                text
            )
            return sanitized
        elif issue_type == 'self-reference':
            # Remove self-references
            sanitized = re.sub(
                r"(?i)\b(as an AI|as an artificial intelligence|as a language model|as an assistant).*?[,.] ",
                "",
                text
            )
            sanitized = re.sub(
                r"(?i)\b(my training|my programming|my knowledge|my data|my capabilities).*?[,.] ",
                "",
                sanitized
            )
            return sanitized
        elif issue_type == 'provider-reference':
            # Remove provider references
            sanitized = re.sub(
                r"(?i)\b(OpenAI|Anthropic|Claude|GPT|ChatGPT)",
                "I",
                text
            )
            return sanitized
        else:
            # Unknown issue type, cannot sanitize
            return None
