"""
Error handling utilities for LLM wrapper.

This module provides decorators and utilities for handling errors in LLM API calls,
with sophisticated retry logic, error categorization, and monitoring.
"""

import logging
import time
import functools
import traceback
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union, cast

from .exceptions import (
    LLMError,
    LLMConnectionError,
    LLMTimeoutError,
    LLMAPIError,
    LLMRateLimitError,
    LLMAuthenticationError,
    LLMTokenLimitError,
    LLMProviderError
)

# Configure logger
logger = logging.getLogger(__name__)

# Type variable for function
F = TypeVar('F', bound=Callable[..., Any])

# Error metrics tracking
error_metrics: Dict[str, Dict[str, int]] = {
    'total_errors': {},
    'error_types': {},
    'providers': {}
}

def reset_error_metrics():
    """Reset all error metrics."""
    error_metrics['total_errors'] = {}
    error_metrics['error_types'] = {}
    error_metrics['providers'] = {}

def get_error_metrics() -> Dict[str, Dict[str, int]]:
    """Get current error metrics."""
    return error_metrics

def track_error(error: Exception, provider: Optional[str] = None):
    """
    Track an error in the metrics.
    
    Args:
        error: The exception that occurred
        provider: The LLM provider name (if applicable)
    """
    # Track total errors
    error_type = type(error).__name__
    error_metrics['total_errors'][error_type] = error_metrics['total_errors'].get(error_type, 0) + 1
    
    # Track error types
    if isinstance(error, LLMError):
        category = 'llm'
        if isinstance(error, LLMConnectionError):
            category = 'connection'
        elif isinstance(error, LLMTimeoutError):
            category = 'timeout'
        elif isinstance(error, LLMRateLimitError):
            category = 'rate_limit'
        elif isinstance(error, LLMAuthenticationError):
            category = 'authentication'
        elif isinstance(error, LLMTokenLimitError):
            category = 'token_limit'
        elif isinstance(error, LLMAPIError):
            category = 'api'
        elif isinstance(error, LLMProviderError):
            category = 'provider'
        
        error_metrics['error_types'][category] = error_metrics['error_types'].get(category, 0) + 1
    
    # Track provider errors
    if provider:
        error_metrics['providers'][provider] = error_metrics['providers'].get(provider, 0) + 1

def categorize_error(error: Exception) -> str:
    """
    Categorize an error based on its type and content.
    
    Args:
        error: The exception to categorize
        
    Returns:
        A string category for the error
    """
    error_str = str(error).lower()
    
    # Check for specific error patterns
    if isinstance(error, LLMError):
        if isinstance(error, LLMConnectionError):
            return "connection"
        elif isinstance(error, LLMTimeoutError):
            return "timeout"
        elif isinstance(error, LLMRateLimitError):
            return "rate_limit"
        elif isinstance(error, LLMAuthenticationError):
            return "authentication"
        elif isinstance(error, LLMTokenLimitError):
            return "token_limit"
        elif isinstance(error, LLMAPIError):
            return "api"
        elif isinstance(error, LLMProviderError):
            return "provider"
        return "llm"
    
    # Check error string patterns
    if "connection" in error_str or "network" in error_str:
        return "connection"
    elif "timeout" in error_str or "timed out" in error_str:
        return "timeout"
    elif "rate limit" in error_str or "too many requests" in error_str or "429" in error_str:
        return "rate_limit"
    elif "authentication" in error_str or "auth" in error_str or "key" in error_str or "401" in error_str:
        return "authentication"
    elif "token" in error_str and ("limit" in error_str or "exceed" in error_str):
        return "token_limit"
    elif "context" in error_str and ("length" in error_str or "window" in error_str):
        return "context_length"
    
    # Default category
    return "unknown"

def is_transient_error(error: Exception) -> bool:
    """
    Determine if an error is likely transient and can be retried.
    
    Args:
        error: The exception to check
        
    Returns:
        True if the error is likely transient, False otherwise
    """
    category = categorize_error(error)
    
    # These categories are typically transient
    transient_categories = [
        "connection", "timeout", "rate_limit", 
        "server_error", "unknown"
    ]
    
    return category in transient_categories

def retry_with_exponential_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    retry_on: Optional[List[Type[Exception]]] = None,
    retry_if_func: Optional[Callable[[Exception], bool]] = None
) -> Callable[[F], F]:
    """
    Decorator for retrying functions with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        backoff_factor: Factor by which to increase delay after each retry
        retry_on: List of exception types to retry on
        retry_if_func: Function that takes an exception and returns True if it should be retried
        
    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            delay = initial_delay
            
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    # Track the error
                    provider = kwargs.get('provider_name', getattr(args[0], 'name', None) if args else None)
                    track_error(e, provider)
                    
                    # Get function name for better error messages
                    func_name = func.__qualname__
                    
                    # Check if we should retry
                    should_retry = False
                    if retries < max_retries:
                        if retry_on and isinstance(e, tuple(retry_on)):
                            should_retry = True
                        elif retry_if_func and retry_if_func(e):
                            should_retry = True
                        elif retry_if_func is None and retry_on is None and is_transient_error(e):
                            # Default behavior if no criteria specified
                            should_retry = True
                    
                    if should_retry:
                        retries += 1
                        # Calculate next delay with exponential backoff
                        current_delay = min(delay * (backoff_factor ** (retries - 1)), max_delay)
                        
                        # Log the retry
                        logger.warning(
                            f"Error in {func_name} (attempt {retries}/{max_retries+1}): {str(e)}. "
                            f"Retrying in {current_delay:.1f}s..."
                        )
                        
                        # Wait before retrying
                        time.sleep(current_delay)
                        continue
                    
                    # Log the error with traceback
                    logger.error(f"Error in {func_name} after {retries} retries: {str(e)}")
                    logger.debug(f"Error traceback:\n{traceback.format_exc()}")
                    
                    # Re-raise the exception
                    raise
        
        return cast(F, wrapper)
    
    return decorator
