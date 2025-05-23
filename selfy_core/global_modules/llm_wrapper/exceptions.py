"""
Exceptions for the LLM Wrapper module.

This module defines the exception hierarchy for LLM-related errors.
"""


class LLMError(Exception):
    """Base exception for all LLM-related errors."""
    def __init__(self, message: str, status_code: int = None, response: str = None):
        super().__init__(message)
        self.status_code = status_code
        self.response = response


class LLMConfigurationError(LLMError):
    """Raised when there is an error in the LLM configuration."""
    pass


class LLMConnectionError(LLMError):
    """Raised when there is an error connecting to the LLM provider."""
    pass


class LLMAPIError(LLMError):
    """Raised when there is an error in the LLM API call."""
    pass


class LLMRateLimitError(LLMError):
    """Raised when the LLM provider rate limit is exceeded."""
    pass


class LLMAuthenticationError(LLMError):
    """Raised when there is an authentication error with the LLM provider."""
    pass


class LLMTimeoutError(LLMError):
    """Raised when the LLM provider request times out."""
    pass


class LLMTokenLimitError(LLMError):
    """Raised when the token limit is exceeded."""
    pass


class LLMProviderError(LLMError):
    """Raised when there is an error with the LLM provider."""
    pass


def get_user_friendly_message(error: Exception) -> str:
    """
    Get a user-friendly error message for an LLM error.
    
    Args:
        error: The error to get a message for
        
    Returns:
        A user-friendly error message
    """
    if isinstance(error, LLMConfigurationError):
        return "There was an issue with the LLM configuration. Please check the logs."
    elif isinstance(error, LLMConnectionError):
        return "There was an issue connecting to the LLM provider. Please try again later."
    elif isinstance(error, LLMAPIError):
        return "There was an issue with the LLM API call. Please try again."
    elif isinstance(error, LLMRateLimitError):
        return "The LLM provider rate limit has been exceeded. Please try again later."
    elif isinstance(error, LLMAuthenticationError):
        return "There was an authentication issue with the LLM provider. Please check your API key."
    elif isinstance(error, LLMTimeoutError):
        return "The LLM provider request timed out. Please try again later."
    elif isinstance(error, LLMTokenLimitError):
        return "The token limit has been exceeded. Please try a shorter prompt."
    elif isinstance(error, LLMProviderError):
        return "There was an issue with the LLM provider. Please try again."
    else:
        return "There was an unexpected error with the LLM. Please try again."
