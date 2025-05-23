"""
Exceptions for the configuration module.

This module defines custom exceptions for the configuration system.
"""


class ConfigurationError(Exception):
    """
    Exception raised for configuration errors.

    This exception is raised when there's an issue with the configuration,
    such as invalid values, missing required fields, or schema validation failures.
    """
    pass
