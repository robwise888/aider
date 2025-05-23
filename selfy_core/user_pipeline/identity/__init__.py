"""
Identity System for the User Chat Pipeline.

This module provides identity management and filtering for the User Chat Pipeline.
It defines the agent's identity, filters inputs and outputs for identity consistency,
and provides identity context for prompt injection.
"""

from typing import Optional

from selfy_core.user_pipeline.identity.manager import (
    IdentityManager, setup_identity_system, get_identity_manager
)
from selfy_core.user_pipeline.identity.filter import IdentityFilter

# Global instance
_identity_filter_instance = None

def get_identity_filter() -> Optional[IdentityFilter]:
    """
    Get the identity filter instance.

    Returns:
        The identity filter instance, or None if not set up
    """
    global _identity_filter_instance

    if _identity_filter_instance is None:
        try:
            # Get identity manager
            identity_manager = get_identity_manager()

            # Create identity filter
            _identity_filter_instance = IdentityFilter(identity_manager=identity_manager)
        except Exception as e:
            from selfy_core.global_modules.logging import get_logger
            logger = get_logger(__name__)
            logger.error(f"Failed to get identity filter: {e}")

    return _identity_filter_instance
from selfy_core.user_pipeline.identity.data_structures import (
    IdentityProfile, FilterResult
)
from selfy_core.user_pipeline.identity.errors import (
    IdentityError, IdentityConfigError, IdentityFilterError,
    IdentityLLMError, IdentityNotInitializedError, get_user_friendly_message
)

__all__ = [
    'IdentityManager',
    'setup_identity_system',
    'get_identity_manager',
    'IdentityFilter',
    'get_identity_filter',
    'IdentityProfile',
    'FilterResult',
    'IdentityError',
    'IdentityConfigError',
    'IdentityFilterError',
    'IdentityLLMError',
    'IdentityNotInitializedError',
    'get_user_friendly_message'
]