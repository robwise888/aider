"""
Logging module for the Selfy agent.

This module provides a centralized logging system for the Selfy agent.
It enables consistent log formatting, component-specific log levels,
automatic log rotation based on size, and specialized logging methods
for common scenarios.
"""

from .logging_system import (
    # Core functions
    setup_logging,
    get_logger,

    # Custom log levels
    LogLevel,

    # Enhanced logging utilities
    ContextAdapter,
    log_performance,
    log_error,
    log_llm_call,
)

__all__ = [
    'setup_logging',
    'get_logger',
    'LogLevel',
    'ContextAdapter',
    'log_performance',
    'log_error',
    'log_llm_call',
]