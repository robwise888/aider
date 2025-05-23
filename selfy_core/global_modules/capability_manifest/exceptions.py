"""
Exceptions for the capability manifest module.

This module defines the exception hierarchy for capability-related errors.
"""


class CapabilityError(Exception):
    """Base exception for all capability-related errors."""
    def __init__(self, message, details=None):
        super().__init__(message)
        self.details = details or {}


class CapabilityNotFoundError(CapabilityError):
    """Raised when a capability is not found."""
    pass


class ParameterError(CapabilityError):
    """Raised when there is an error with capability parameters."""
    pass


class ExecutionError(CapabilityError):
    """Raised when there is an error during capability execution."""
    pass


class ConfigurationError(CapabilityError):
    """Raised when there is an error in the capability configuration."""
    pass


class PersistenceError(CapabilityError):
    """Raised when there is an error with capability persistence."""
    pass


class ValidationError(CapabilityError):
    """Raised when a capability definition is invalid."""
    pass


class RegistrationError(CapabilityError):
    """Raised when there is an error registering a capability."""
    pass


class DiscoveryError(CapabilityError):
    """Raised when there is an error discovering capabilities."""
    pass


def get_user_friendly_message(error: Exception) -> str:
    """
    Get a user-friendly error message for a capability error.

    Args:
        error: The error to get a message for

    Returns:
        A user-friendly error message
    """
    if isinstance(error, CapabilityNotFoundError):
        return "The requested capability was not found. Please check the capability name."
    elif isinstance(error, ParameterError):
        return "There was an issue with the capability parameters. Please check the parameters."
    elif isinstance(error, ExecutionError):
        return "There was an issue executing the capability. Please try again."
    elif isinstance(error, ConfigurationError):
        return "There was an issue with the capability configuration. Please check the logs."
    elif isinstance(error, PersistenceError):
        return "There was an issue with capability persistence. Please check the logs."
    elif isinstance(error, ValidationError):
        return "The capability definition is invalid. Please check the capability format."
    elif isinstance(error, RegistrationError):
        return "There was an issue registering the capability. Please check the logs."
    elif isinstance(error, DiscoveryError):
        return "There was an issue discovering capabilities. Please check the logs."
    else:
        return "There was an unexpected error with the capability. Please try again."
