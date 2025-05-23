"""
LLM Wrapper module for the Selfy agent.

This module provides a unified interface for interacting with different LLM providers.
It abstracts provider-specific implementations, integrates with the configuration system
for settings (API keys, models), includes token counting utilities, and tracks token usage.
"""

# Core components
from .base_llm_provider import BaseLLMProvider
from .response import LLMResponse
from .setup import setup_llm_providers, is_setup_complete, is_provider_enabled, get_provider_config

# Factory functions
from .llm_factory import (
    create_provider,
    create_default_provider,
    create_local_provider,
    create_cloud_provider,
    register_provider,
    get_available_providers,
    get_llm_provider,
    clear_provider_cache
)

# Provider implementations
from .groq_provider import GroqProvider
from .ollama_provider import OllamaProvider

# Token utilities
from .token_utils import (
    count_tokens,
    count_message_tokens,
    check_token_limit,
    truncate_text_to_token_limit,
    truncate_messages_to_token_limit,
    TokenTracker
)

# Logging helpers
from .logging_helpers import log_llm_call, get_token_tracker

# HTTP and error handling utilities
from .http_connection_manager import HTTPConnectionManager
from .error_handling import (
    retry_with_exponential_backoff,
    categorize_error,
    is_transient_error,
    track_error,
    get_error_metrics,
    reset_error_metrics
)

# Exceptions
from .exceptions import (
    LLMError,
    LLMConfigurationError,
    LLMConnectionError,
    LLMAPIError,
    LLMRateLimitError,
    LLMAuthenticationError,
    LLMTimeoutError,
    LLMTokenLimitError,
    LLMProviderError
)

# Global token tracker instance
token_tracker = get_token_tracker()

__all__ = [
    # Provider classes
    'BaseLLMProvider',
    'GroqProvider',
    'OllamaProvider',

    # Setup functions
    'setup_llm_providers',
    'is_setup_complete',
    'is_provider_enabled',
    'get_provider_config',

    # Factory functions
    'create_provider',
    'create_default_provider',
    'create_local_provider',
    'create_cloud_provider',
    'register_provider',
    'get_available_providers',
    'get_llm_provider',
    'clear_provider_cache',

    # Token utilities
    'count_tokens',
    'count_message_tokens',
    'check_token_limit',
    'truncate_text_to_token_limit',
    'truncate_messages_to_token_limit',
    'TokenTracker',
    'token_tracker',

    # Response class
    'LLMResponse',

    # Logging helpers
    'log_llm_call',
    'get_token_tracker',

    # HTTP and error handling utilities
    'HTTPConnectionManager',
    'retry_with_exponential_backoff',
    'categorize_error',
    'is_transient_error',
    'track_error',
    'get_error_metrics',
    'reset_error_metrics',

    # Exceptions
    'LLMError',
    'LLMConfigurationError',
    'LLMConnectionError',
    'LLMAPIError',
    'LLMRateLimitError',
    'LLMAuthenticationError',
    'LLMTimeoutError',
    'LLMTokenLimitError',
    'LLMProviderError'
]