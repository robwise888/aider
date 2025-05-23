"""
Setup functions for the LLM Wrapper module.

This module provides functions for setting up the LLM wrapper system.
"""

import os
from typing import Dict, Any, Optional, List

from selfy_core.global_modules.config import get as config_get
from selfy_core.global_modules.logging import get_logger

# Set up logger
logger = get_logger(__name__)

# Global state
_setup_complete = False
_provider_configs = {}


def setup_llm_providers() -> bool:
    """
    Set up the LLM wrapper system.

    This function initializes the LLM wrapper system by loading provider
    configurations from the configuration system and starting the Ollama server
    if it's enabled.

    Returns:
        True if setup was successful, False otherwise
    """
    global _setup_complete, _provider_configs

    try:
        logger.info("Setting up LLM providers...")

        # Load provider configurations
        _provider_configs = {
            'groq': {
                'enabled': config_get('llm.groq.enabled', False),
                'api_key': config_get('llm.groq.api_key', os.environ.get('GROQ_API_KEY', '')),
                'default_model': config_get('llm.groq.default_model', 'llama3-70b-8192'),
                'max_retries': config_get('llm.groq.max_retries', 2),
                'retry_delay_seconds': config_get('llm.groq.retry_delay_seconds', 1.0),
                'timeout_seconds': config_get('llm.groq.timeout_seconds', 30)
            },
            'ollama': {
                'enabled': config_get('llm.ollama.enabled', True),
                'host': config_get('llm.ollama.host', 'http://localhost:11434'),
                'default_model': config_get('llm.ollama.default_model', 'llama3:8b'),
                'max_retries': config_get('llm.ollama.max_retries', 2),
                'retry_delay_seconds': config_get('llm.ollama.retry_delay_seconds', 1.0),
                'timeout_seconds': config_get('llm.ollama.timeout_seconds', 30)
            }
        }

        # Get default provider
        default_provider = config_get('llm.default_provider', 'ollama')
        if default_provider not in _provider_configs:
            logger.warning(f"Default provider '{default_provider}' not found in configuration. Using 'ollama'.")
            default_provider = 'ollama'

        # Check if default provider is enabled
        if not _provider_configs[default_provider]['enabled']:
            logger.warning(f"Default provider '{default_provider}' is disabled. Enabling it.")
            _provider_configs[default_provider]['enabled'] = True

        # Log provider configurations
        for provider_name, provider_config in _provider_configs.items():
            if provider_config['enabled']:
                logger.info(f"Provider '{provider_name}' enabled with model '{provider_config['default_model']}'")
            else:
                logger.info(f"Provider '{provider_name}' disabled")

        # Ollama server is now started earlier in the initialization process
        # in selfy_core/main.py to ensure it's ready before any connections are attempted

        _setup_complete = True
        logger.info("LLM providers set up successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to set up LLM providers: {e}", exc_info=True)
        return False


def is_setup_complete() -> bool:
    """
    Check if the LLM wrapper system is set up.

    Returns:
        True if the LLM wrapper system is set up, False otherwise
    """
    return _setup_complete


def is_provider_enabled(provider_name: str) -> bool:
    """
    Check if a provider is enabled.

    Args:
        provider_name: The name of the provider

    Returns:
        True if the provider is enabled, False otherwise
    """
    if not _setup_complete:
        logger.warning("LLM wrapper system not set up. Call setup_llm_providers() first.")
        return False

    if provider_name not in _provider_configs:
        logger.warning(f"Provider '{provider_name}' not found in configuration")
        return False

    return _provider_configs[provider_name]['enabled']


def get_provider_config(provider_name: str) -> Optional[Dict[str, Any]]:
    """
    Get the configuration for a provider.

    Args:
        provider_name: The name of the provider

    Returns:
        The provider configuration, or None if the provider is not found
    """
    if not _setup_complete:
        logger.warning("LLM wrapper system not set up. Call setup_llm_providers() first.")
        return None

    if provider_name not in _provider_configs:
        logger.warning(f"Provider '{provider_name}' not found in configuration")
        return None

    return _provider_configs[provider_name].copy()
