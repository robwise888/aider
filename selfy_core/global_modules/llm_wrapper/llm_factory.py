"""
LLM Factory for the Selfy agent.

This module provides a factory for creating LLM provider instances.
It includes functions for creating specific providers, the default provider,
and specialized providers (local and cloud).
"""

import logging
import os
from typing import Dict, Any, Optional, Union, List, Type

from .base_llm_provider import BaseLLMProvider
from .groq_provider import GroqProvider
from .ollama_provider import OllamaProvider
from .setup import is_setup_complete, is_provider_enabled, get_provider_config
from .exceptions import LLMConfigurationError, LLMProviderError
from selfy_core.global_modules.config import get as config_get

# Set up logging
logger = logging.getLogger(__name__)

# Provider registry
_provider_registry = {
    'groq': GroqProvider,
    'ollama': OllamaProvider
}

# Provider instance cache
_provider_instances = {}


def register_provider(provider_name: str, provider_class: Type[BaseLLMProvider]) -> None:
    """
    Register a custom provider implementation.

    Args:
        provider_name: The name of the provider
        provider_class: The provider class

    Raises:
        LLMConfigurationError: If the provider is already registered
    """
    if provider_name in _provider_registry:
        raise LLMConfigurationError(f"Provider '{provider_name}' is already registered")

    _provider_registry[provider_name] = provider_class
    logger.info(f"Registered provider '{provider_name}'")


def get_available_providers() -> List[str]:
    """
    Get a list of available provider names.

    Returns:
        List of available provider names
    """
    return list(_provider_registry.keys())


def create_provider(provider_name: str, **kwargs) -> BaseLLMProvider:
    """
    Create a provider instance.

    Args:
        provider_name: The name of the provider
        **kwargs: Additional provider-specific arguments

    Returns:
        A provider instance

    Raises:
        LLMConfigurationError: If the provider is not found or not enabled
    """
    global _provider_instances

    # Check if setup is complete
    if not is_setup_complete():
        raise LLMConfigurationError("LLM wrapper system not set up. Call setup_llm_providers() first.")

    # Check if provider is registered
    if provider_name not in _provider_registry:
        raise LLMConfigurationError(f"Provider '{provider_name}' not found")

    # Check if provider is enabled
    if not is_provider_enabled(provider_name):
        raise LLMConfigurationError(f"Provider '{provider_name}' is not enabled")

    # Get provider configuration
    provider_config = get_provider_config(provider_name)
    if not provider_config:
        raise LLMConfigurationError(f"Provider '{provider_name}' configuration not found")

    # Special handling for Groq provider
    if provider_name == 'groq':
        # Check for API key in environment variable if not provided in kwargs or config
        import os
        env_api_key = os.environ.get('GROQ_API_KEY')
        if env_api_key and not kwargs.get('api_key') and not provider_config.get('api_key'):
            logger.debug("Using Groq API key from environment variable in create_provider")
            provider_config['api_key'] = env_api_key

    # Merge provider configuration with kwargs
    merged_kwargs = {**provider_config, **kwargs}

    # Create a cache key based on provider name and configuration
    # For simplicity, we'll use the provider name and model name as the cache key
    model_name = merged_kwargs.get('model_name', provider_config.get('default_model', 'default'))
    cache_key = f"{provider_name}:{model_name}"

    # Check if we already have a provider instance with this configuration
    if cache_key in _provider_instances:
        logger.debug(f"Reusing cached provider instance for '{cache_key}'")
        return _provider_instances[cache_key]

    # Create provider instance
    try:
        provider_class = _provider_registry[provider_name]

        # Create provider instance
        provider = provider_class(**merged_kwargs)

        # Cache the provider instance
        _provider_instances[cache_key] = provider

        logger.info(f"Created and cached provider '{provider_name}' with model '{provider.model_name}'")
        return provider
    except Exception as e:
        logger.error(f"Failed to create provider '{provider_name}': {e}", exc_info=True)
        raise LLMProviderError(f"Failed to create provider '{provider_name}': {e}")


def create_default_provider(**kwargs) -> BaseLLMProvider:
    """
    Create the default provider instance.

    Args:
        **kwargs: Additional provider-specific arguments

    Returns:
        A provider instance

    Raises:
        LLMConfigurationError: If the default provider is not found or not enabled
    """
    # Check if setup is complete
    if not is_setup_complete():
        raise LLMConfigurationError("LLM wrapper system not set up. Call setup_llm_providers() first.")

    # Get default provider
    default_provider = config_get('llm.default_provider', 'ollama')

    # Create provider instance
    return create_provider(default_provider, **kwargs)


def create_local_provider(**kwargs) -> BaseLLMProvider:
    """
    Create a local provider instance (Ollama).

    Args:
        **kwargs: Additional provider-specific arguments

    Returns:
        A provider instance

    Raises:
        LLMConfigurationError: If the local provider is not found or not enabled
    """
    # Check if setup is complete
    if not is_setup_complete():
        raise LLMConfigurationError("LLM wrapper system not set up. Call setup_llm_providers() first.")

    # Create provider instance
    return create_provider('ollama', **kwargs)


def create_cloud_provider(**kwargs) -> BaseLLMProvider:
    """
    Create a cloud provider instance (Groq).

    Args:
        **kwargs: Additional provider-specific arguments

    Returns:
        A provider instance

    Raises:
        LLMConfigurationError: If the cloud provider is not found or not enabled
    """
    # Check if setup is complete
    if not is_setup_complete():
        raise LLMConfigurationError("LLM wrapper system not set up. Call setup_llm_providers() first.")

    # Check for API key in environment variable if not provided in kwargs
    import os
    if 'api_key' not in kwargs:
        env_api_key = os.environ.get('GROQ_API_KEY')
        if env_api_key:
            kwargs['api_key'] = env_api_key
            logger.debug("Using Groq API key from environment variable in cloud_provider factory")

    # Create provider instance
    return create_provider('groq', **kwargs)


def get_llm_provider(provider_name: str = None, **kwargs) -> BaseLLMProvider:
    """
    Get an LLM provider instance.

    This is a convenience function that creates a provider instance based on the
    provider name. If no provider name is specified, it creates the default provider.

    Args:
        provider_name: The name of the provider (e.g., 'groq', 'ollama')
        **kwargs: Additional provider-specific arguments

    Returns:
        A provider instance

    Raises:
        LLMConfigurationError: If the provider is not found or not enabled
    """
    if provider_name is None:
        return create_default_provider(**kwargs)
    elif provider_name == 'cloud':
        return create_cloud_provider(**kwargs)
    elif provider_name == 'local':
        return create_local_provider(**kwargs)
    else:
        return create_provider(provider_name, **kwargs)


def clear_provider_cache(provider_name: str = None) -> None:
    """
    Clear the provider instance cache.

    Args:
        provider_name: The name of the provider to clear from the cache,
                      or None to clear all providers
    """
    global _provider_instances

    if provider_name is None:
        # Clear all providers
        logger.info("Clearing all provider instances from cache")
        _provider_instances.clear()
    else:
        # Clear only providers with the specified name
        keys_to_remove = [k for k in _provider_instances.keys() if k.startswith(f"{provider_name}:")]
        for key in keys_to_remove:
            logger.info(f"Removing provider instance '{key}' from cache")
            del _provider_instances[key]
