"""
Error handling module for the Selfy agent.

This package provides error handling and recovery mechanisms for the Selfy agent.
"""

from selfy_core.global_modules.error_handling.error_manager import (
    Error, ErrorCategory, ErrorSeverity, ErrorRecoveryStrategy,
    ErrorManager, setup_error_manager, get_error_manager
)

__all__ = [
    'Error',
    'ErrorCategory',
    'ErrorSeverity',
    'ErrorRecoveryStrategy',
    'ErrorManager',
    'setup_error_manager',
    'get_error_manager'
]
