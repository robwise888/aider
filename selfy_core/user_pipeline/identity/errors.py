"""
Error classes for the Identity System.

This module defines the error classes used by the Identity System.
"""


class IdentityError(Exception):
    """Base exception for all identity-related errors."""
    pass


class IdentityConfigError(IdentityError):
    """Raised when there is an error in the identity configuration."""
    pass


class IdentityFilterError(IdentityError):
    """Raised when there is an error during identity filtering."""
    pass


class IdentityLLMError(IdentityError):
    """Raised when there is an error during LLM-based identity checks."""
    pass


class IdentityNotInitializedError(IdentityError):
    """Raised when the identity system is not initialized."""
    pass


def get_user_friendly_message(error: Exception) -> str:
    """
    Get a user-friendly error message for an identity error.
    
    Args:
        error: The error to get a message for
        
    Returns:
        A user-friendly error message
    """
    if isinstance(error, IdentityConfigError):
        return "There was an issue with the identity configuration. Please check the logs."
    elif isinstance(error, IdentityFilterError):
        return "There was an issue with identity filtering. Please try again."
    elif isinstance(error, IdentityLLMError):
        return "There was an issue with the identity check. Please try again."
    elif isinstance(error, IdentityNotInitializedError):
        return "The identity system is not initialized. Please try again later."
    else:
        return "There was an unexpected error in the identity system. Please try again."
