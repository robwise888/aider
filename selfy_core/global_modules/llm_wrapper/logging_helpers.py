"""
Logging helpers for the LLM Wrapper module.

This module provides helper functions for logging LLM calls and tracking token usage.
"""

import logging
import time
from typing import Dict, Any, Optional

from .token_utils import get_token_tracker


def log_llm_call(
    logger: logging.Logger,
    provider_name: str,
    model_name: str,
    prompt_tokens: int,
    completion_tokens: int,
    duration: float,
    error: Optional[str] = None,
    streaming: bool = False
) -> None:
    """
    Log an LLM call with token usage and duration.

    Args:
        logger: The logger to use
        provider_name: The name of the provider
        model_name: The name of the model
        prompt_tokens: The number of prompt tokens
        completion_tokens: The number of completion tokens
        duration: The duration of the call in seconds
        error: Optional error message
        streaming: Whether this was a streaming call
    """
    total_tokens = prompt_tokens + completion_tokens
    stream_indicator = " (streaming)" if streaming else ""

    # Log the call
    if error:
        logger.warning(
            f"LLM call failed: {provider_name}/{model_name}{stream_indicator}, "
            f"Tokens: {prompt_tokens}+{completion_tokens}={total_tokens}, "
            f"Duration: {duration:.2f}s, Error: {error}"
        )
    else:
        logger.info(
            f"LLM call: {provider_name}/{model_name}{stream_indicator}, "
            f"Tokens: {prompt_tokens}+{completion_tokens}={total_tokens}, "
            f"Duration: {duration:.2f}s"
        )

    # Track token usage
    if not error:
        token_tracker = get_token_tracker()
        token_tracker.track_usage(provider_name, model_name, prompt_tokens, completion_tokens)
