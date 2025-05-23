"""
Token utilities for the LLM Wrapper module.

This module provides utilities for counting tokens, checking token limits,
and truncating text to fit within token limits.
"""

import re
import threading
import time
from typing import Dict, List, Any, Optional, Union

from selfy_core.global_modules.config import get as config_get
from selfy_core.global_modules.logging import get_logger

# Set up logger
logger = get_logger(__name__)


def count_tokens(text: str, model: str = None) -> int:
    """
    Count the number of tokens in a text string.

    This is a simple approximation based on word count. For more accurate
    token counting, use a provider-specific tokenizer.

    Args:
        text: The text to count tokens for
        model: The model to use for tokenization (optional)

    Returns:
        The approximate number of tokens
    """
    # Simple approximation: 1 token ~= 4 characters
    return len(text) // 4 + 1


def count_message_tokens(messages: List[Dict[str, str]], model: str = None) -> int:
    """
    Count the number of tokens in a list of messages.

    This is a simple approximation based on word count. For more accurate
    token counting, use a provider-specific tokenizer.

    Args:
        messages: The messages to count tokens for
        model: The model to use for tokenization (optional)

    Returns:
        The approximate number of tokens
    """
    total_tokens = 0
    for message in messages:
        content = message.get('content', '')
        total_tokens += count_tokens(content, model)
    return total_tokens


def check_token_limit(text: str, max_tokens: int, model: str = None) -> bool:
    """
    Check if a text string exceeds a token limit.

    Args:
        text: The text to check
        max_tokens: The maximum number of tokens allowed
        model: The model to use for tokenization (optional)

    Returns:
        True if the text is within the token limit, False otherwise
    """
    return count_tokens(text, model) <= max_tokens


def truncate_text_to_token_limit(text: str, max_tokens: int, model: str = None) -> str:
    """
    Truncate a text string to fit within a token limit.

    Args:
        text: The text to truncate
        max_tokens: The maximum number of tokens allowed
        model: The model to use for tokenization (optional)

    Returns:
        The truncated text
    """
    if check_token_limit(text, max_tokens, model):
        return text

    # Simple approximation: 1 token ~= 4 characters
    max_chars = max_tokens * 4
    if len(text) > max_chars:
        return text[:max_chars] + "..."
    return text


def truncate_messages_to_token_limit(messages: List[Dict[str, str]], max_tokens: int, model: str = None) -> List[Dict[str, str]]:
    """
    Truncate a list of messages to fit within a token limit.

    Args:
        messages: The messages to truncate
        max_tokens: The maximum number of tokens allowed
        model: The model to use for tokenization (optional)

    Returns:
        The truncated messages
    """
    if count_message_tokens(messages, model) <= max_tokens:
        return messages

    # Start removing messages from the beginning (oldest first)
    # but always keep the last message (current user query)
    truncated_messages = messages.copy()
    while len(truncated_messages) > 1 and count_message_tokens(truncated_messages, model) > max_tokens:
        truncated_messages.pop(0)

    # If we're still over the limit with just one message, truncate the content
    if len(truncated_messages) == 1 and count_message_tokens(truncated_messages, model) > max_tokens:
        content = truncated_messages[0].get('content', '')
        truncated_content = truncate_text_to_token_limit(content, max_tokens, model)
        truncated_messages[0]['content'] = truncated_content

    return truncated_messages


class TokenTracker:
    """
    Thread-safe token usage tracker.

    This class tracks token usage by provider, model, and time period.
    """

    def __init__(self):
        """Initialize the token tracker."""
        self.lock = threading.Lock()
        self.usage = {
            'total': {
                'prompt_tokens': 0,
                'completion_tokens': 0,
                'total_tokens': 0
            },
            'providers': {},
            'models': {},
            'time_periods': {
                'last_hour': {
                    'prompt_tokens': 0,
                    'completion_tokens': 0,
                    'total_tokens': 0,
                    'timestamp': time.time()
                },
                'last_day': {
                    'prompt_tokens': 0,
                    'completion_tokens': 0,
                    'total_tokens': 0,
                    'timestamp': time.time()
                }
            }
        }

    def track_usage(self, provider_name: str, model_name: str, prompt_tokens: int, completion_tokens: int):
        """
        Track token usage.

        Args:
            provider_name: The name of the provider
            model_name: The name of the model
            prompt_tokens: The number of prompt tokens
            completion_tokens: The number of completion tokens
        """
        with self.lock:
            # Update total usage
            self.usage['total']['prompt_tokens'] += prompt_tokens
            self.usage['total']['completion_tokens'] += completion_tokens
            self.usage['total']['total_tokens'] += prompt_tokens + completion_tokens

            # Update provider usage
            if provider_name not in self.usage['providers']:
                self.usage['providers'][provider_name] = {
                    'prompt_tokens': 0,
                    'completion_tokens': 0,
                    'total_tokens': 0
                }
            self.usage['providers'][provider_name]['prompt_tokens'] += prompt_tokens
            self.usage['providers'][provider_name]['completion_tokens'] += completion_tokens
            self.usage['providers'][provider_name]['total_tokens'] += prompt_tokens + completion_tokens

            # Update model usage
            if model_name not in self.usage['models']:
                self.usage['models'][model_name] = {
                    'prompt_tokens': 0,
                    'completion_tokens': 0,
                    'total_tokens': 0
                }
            self.usage['models'][model_name]['prompt_tokens'] += prompt_tokens
            self.usage['models'][model_name]['completion_tokens'] += completion_tokens
            self.usage['models'][model_name]['total_tokens'] += prompt_tokens + completion_tokens

            # Update time period usage
            current_time = time.time()
            self._update_time_period('last_hour', current_time, 3600, prompt_tokens, completion_tokens)
            self._update_time_period('last_day', current_time, 86400, prompt_tokens, completion_tokens)

    def _update_time_period(self, period_name: str, current_time: float, period_seconds: int, prompt_tokens: int, completion_tokens: int):
        """
        Update time period usage.

        Args:
            period_name: The name of the time period
            current_time: The current time
            period_seconds: The number of seconds in the time period
            prompt_tokens: The number of prompt tokens
            completion_tokens: The number of completion tokens
        """
        period = self.usage['time_periods'][period_name]
        if current_time - period['timestamp'] > period_seconds:
            # Reset if the period has elapsed
            period['prompt_tokens'] = prompt_tokens
            period['completion_tokens'] = completion_tokens
            period['total_tokens'] = prompt_tokens + completion_tokens
            period['timestamp'] = current_time
        else:
            # Add to the current period
            period['prompt_tokens'] += prompt_tokens
            period['completion_tokens'] += completion_tokens
            period['total_tokens'] += prompt_tokens + completion_tokens

    def get_usage(self) -> Dict[str, Any]:
        """
        Get token usage statistics.

        Returns:
            Dictionary containing token usage statistics
        """
        with self.lock:
            return self.usage.copy()

    def get_provider_usage(self, provider_name: str) -> Dict[str, int]:
        """
        Get token usage for a specific provider.

        Args:
            provider_name: The name of the provider

        Returns:
            Dictionary containing token usage statistics for the provider
        """
        with self.lock:
            return self.usage['providers'].get(provider_name, {
                'prompt_tokens': 0,
                'completion_tokens': 0,
                'total_tokens': 0
            }).copy()

    def get_model_usage(self, model_name: str) -> Dict[str, int]:
        """
        Get token usage for a specific model.

        Args:
            model_name: The name of the model

        Returns:
            Dictionary containing token usage statistics for the model
        """
        with self.lock:
            return self.usage['models'].get(model_name, {
                'prompt_tokens': 0,
                'completion_tokens': 0,
                'total_tokens': 0
            }).copy()

    def reset(self):
        """Reset all token usage statistics."""
        with self.lock:
            self.usage = {
                'total': {
                    'prompt_tokens': 0,
                    'completion_tokens': 0,
                    'total_tokens': 0
                },
                'providers': {},
                'models': {},
                'time_periods': {
                    'last_hour': {
                        'prompt_tokens': 0,
                        'completion_tokens': 0,
                        'total_tokens': 0,
                        'timestamp': time.time()
                    },
                    'last_day': {
                        'prompt_tokens': 0,
                        'completion_tokens': 0,
                        'total_tokens': 0,
                        'timestamp': time.time()
                    }
                }
            }


# Global token tracker instance
_token_tracker_instance = None


def get_token_tracker() -> TokenTracker:
    """
    Get the token tracker instance.

    Returns:
        The token tracker instance
    """
    global _token_tracker_instance
    if _token_tracker_instance is None:
        _token_tracker_instance = TokenTracker()
    return _token_tracker_instance
