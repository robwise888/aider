"""
Logging system for the Selfy agent.

This module provides a centralized logging system for the Selfy agent.
"""

import os
import sys
import time
import json
import logging
import logging.handlers
from enum import IntEnum
from typing import Any, Dict, List, Optional, Union

from selfy_core.global_modules.config import get as config_get


class LogLevel(IntEnum):
    """
    Custom log levels for the Selfy agent.

    These levels extend the standard logging levels to provide more granularity.
    """
    TRACE = 5       # More detailed than DEBUG
    VERBOSE = 15    # Between DEBUG and INFO
    SUCCESS = 25    # Between INFO and WARNING

    # Standard levels for reference
    # DEBUG = 10
    # INFO = 20
    # WARNING = 30
    # ERROR = 40
    # CRITICAL = 50


class ColoredFormatter(logging.Formatter):
    """
    Formatter that adds color to console output.

    This formatter adds ANSI color codes to the log level only, not the entire message.
    """

    # ANSI color codes
    COLORS = {
        'TRACE': '\033[36;1m',     # Bright cyan
        'DEBUG': '\033[36m',       # Cyan
        'VERBOSE': '\033[34;1m',   # Bright blue
        'INFO': '\033[32m',        # Green
        'SUCCESS': '\033[32;1m',   # Bright green
        'WARNING': '\033[33m',     # Yellow
        'ERROR': '\033[31m',       # Red
        'CRITICAL': '\033[31;1m',  # Bright red
        'RESET': '\033[0m'         # Reset
    }

    def format(self, record):
        """Format the log record with color only for the level name."""
        # Store the original levelname
        original_levelname = record.levelname

        # Color the levelname if it's in our color map
        if original_levelname in self.COLORS:
            record.levelname = f"{self.COLORS[original_levelname]}{original_levelname}{self.COLORS['RESET']}"

        # Format the record with the colored levelname
        formatted_message = super().format(record)

        # Restore the original levelname for other handlers
        record.levelname = original_levelname

        return formatted_message


class StructuredFormatter(logging.Formatter):
    """
    Formatter that outputs logs in JSON format.

    This formatter converts log records to JSON objects for structured logging.
    """

    def format(self, record):
        """Format the log record as JSON."""
        # Create a dictionary with the log record data
        log_data = {
            'timestamp': self.formatTime(record, self.datefmt),
            'level': record.levelname,
            'name': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }

        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)

        # Add context if present
        if hasattr(record, 'context'):
            log_data['context'] = record.context

        # Add any extra attributes
        for key, value in record.__dict__.items():
            if key not in ['args', 'asctime', 'created', 'exc_info', 'exc_text', 'filename',
                          'funcName', 'id', 'levelname', 'levelno', 'lineno', 'module',
                          'msecs', 'message', 'msg', 'name', 'pathname', 'process',
                          'processName', 'relativeCreated', 'stack_info', 'thread',
                          'threadName', 'context']:
                try:
                    # Try to serialize the value to JSON
                    json.dumps({key: value})
                    log_data[key] = value
                except (TypeError, OverflowError):
                    # If the value can't be serialized, convert it to a string
                    log_data[key] = str(value)

        # Return the JSON string
        return json.dumps(log_data)


class ContextAdapter(logging.LoggerAdapter):
    """
    Adapter that adds context to log messages.

    This adapter adds context information to log messages, such as session ID,
    user ID, or other contextual information.
    """

    def process(self, msg, kwargs):
        """Process the log message to add context."""
        # Add context to the message
        if self.extra and 'context' in self.extra and self.extra['context']:
            msg = f"[{self.extra['context']}] {msg}"

        # Add context to the record
        if 'extra' not in kwargs:
            kwargs['extra'] = {}

        if self.extra:
            for key, value in self.extra.items():
                kwargs['extra'][key] = value

        return msg, kwargs


def setup_logging(level=None, format=None, file=None, max_size=None, backup_count=None):
    """
    Set up the logging system.

    Args:
        level: The log level (default: from config)
        format: The log format (default: from config)
        file: The log file path (default: from config)
        max_size: The maximum log file size in bytes (default: from config)
        backup_count: The number of backup files to keep (default: from config)

    Returns:
        True if setup was successful, False otherwise
    """
    try:
        # Get configuration values
        level = level or config_get('logging.level', 'INFO')
        format = format or config_get('logging.format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file = file or config_get('logging.file', 'logs/selfy.log')
        max_size = max_size or config_get('logging.max_size', 10485760)  # 10 MB
        backup_count = backup_count or config_get('logging.backup_count', 5)

        # Convert level string to int
        if isinstance(level, str):
            level = getattr(logging, level.upper(), logging.INFO)

        # Register custom log levels
        logging.addLevelName(LogLevel.TRACE, 'TRACE')
        logging.addLevelName(LogLevel.VERBOSE, 'VERBOSE')
        logging.addLevelName(LogLevel.SUCCESS, 'SUCCESS')

        # Add custom log methods to Logger class
        def trace(self, message, *args, **kwargs):
            self.log(LogLevel.TRACE, message, *args, **kwargs)

        def verbose(self, message, *args, **kwargs):
            self.log(LogLevel.VERBOSE, message, *args, **kwargs)

        def success(self, message, *args, **kwargs):
            self.log(LogLevel.SUCCESS, message, *args, **kwargs)

        # Add methods if they don't exist
        if not hasattr(logging.Logger, 'trace'):
            logging.Logger.trace = trace

        if not hasattr(logging.Logger, 'verbose'):
            logging.Logger.verbose = verbose

        if not hasattr(logging.Logger, 'success'):
            logging.Logger.success = success

        # Configure the root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(level)

        # Remove existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_formatter = ColoredFormatter(format)
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)

        # Create file handler
        if file:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file), exist_ok=True)

            # Create rotating file handler
            file_handler = logging.handlers.RotatingFileHandler(
                file,
                maxBytes=max_size,
                backupCount=backup_count
            )
            file_handler.setLevel(level)
            file_formatter = logging.Formatter(format)
            file_handler.setFormatter(file_formatter)
            root_logger.addHandler(file_handler)

        # Create JSON handler for structured logging
        json_file = config_get('logging.json_file', 'logs/selfy_structured.json')
        if json_file:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(json_file), exist_ok=True)

            # Create rotating file handler
            json_handler = logging.handlers.RotatingFileHandler(
                json_file,
                maxBytes=max_size,
                backupCount=backup_count
            )
            json_handler.setLevel(level)
            json_formatter = StructuredFormatter()
            json_handler.setFormatter(json_formatter)
            root_logger.addHandler(json_handler)

        # Create error log handler
        error_file = config_get('logging.error_file', 'logs/errors.log')
        if error_file:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(error_file), exist_ok=True)

            # Create rotating file handler
            error_handler = logging.handlers.RotatingFileHandler(
                error_file,
                maxBytes=max_size,
                backupCount=backup_count
            )
            error_handler.setLevel(logging.ERROR)
            error_formatter = logging.Formatter(format)
            error_handler.setFormatter(error_formatter)
            root_logger.addHandler(error_handler)

        # Set component-specific log levels
        component_levels = config_get('logging.levels', {})
        for component, component_level in component_levels.items():
            # Convert level string to int
            if isinstance(component_level, str):
                component_level = getattr(logging, component_level.upper(), logging.INFO)

            # Set the level
            logging.getLogger(component).setLevel(component_level)

        # Log setup completion
        logging.getLogger(__name__).info("Logging system initialized")
        return True
    except Exception as e:
        # Fall back to basic logging
        logging.basicConfig(level=logging.INFO)
        logging.getLogger(__name__).error(f"Failed to set up logging: {e}")
        return False


def get_logger(name, context=None):
    """
    Get a logger with the specified name and optional context.

    Args:
        name: The logger name
        context: Optional context to add to log messages

    Returns:
        A logger instance
    """
    logger = logging.getLogger(name)

    if context:
        return ContextAdapter(logger, {'context': context})

    return logger


def log_performance(logger, operation, start_time, **kwargs):
    """
    Log performance metrics for an operation.

    Args:
        logger: The logger to use
        operation: The operation name
        start_time: The start time of the operation
        **kwargs: Additional context to include in the log
    """
    duration_ms = (time.time() - start_time) * 1000
    logger.info(f"Performance: {operation} completed in {duration_ms:.2f}ms", extra={
        'duration_ms': duration_ms,
        'operation': operation,
        **kwargs
    })


def log_error(logger, error, operation=None, **kwargs):
    """
    Log an error with context.

    Args:
        logger: The logger to use
        error: The error to log
        operation: The operation that failed
        **kwargs: Additional context to include in the log
    """
    if operation:
        logger.error(f"Error in {operation}: {error}", exc_info=True, extra={
            'error': str(error),
            'error_type': type(error).__name__,
            'operation': operation,
            **kwargs
        })
    else:
        logger.error(f"Error: {error}", exc_info=True, extra={
            'error': str(error),
            'error_type': type(error).__name__,
            **kwargs
        })


def log_llm_call(logger, provider, model, prompt_tokens, completion_tokens, duration_ms, **kwargs):
    """
    Log an LLM call with metrics.

    Args:
        logger: The logger to use
        provider: The LLM provider
        model: The model used
        prompt_tokens: The number of prompt tokens
        completion_tokens: The number of completion tokens
        duration_ms: The duration of the call in milliseconds
        **kwargs: Additional context to include in the log
    """
    total_tokens = prompt_tokens + completion_tokens
    logger.info(f"LLM call: {provider}/{model} - {total_tokens} tokens in {duration_ms:.2f}ms", extra={
        'provider': provider,
        'model': model,
        'prompt_tokens': prompt_tokens,
        'completion_tokens': completion_tokens,
        'total_tokens': total_tokens,
        'duration_ms': duration_ms,
        **kwargs
    })
