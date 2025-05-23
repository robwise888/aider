"""
Identity Manager for the Identity System.

This module provides the IdentityManager class, which is responsible for
loading, storing, and accessing the agent's identity profile.
"""

import json
import os
from typing import Dict, List, Any, Optional

from selfy_core.global_modules.config import get as config_get
from selfy_core.global_modules.logging import get_logger
from selfy_core.user_pipeline.identity.data_structures import IdentityProfile
from selfy_core.user_pipeline.identity.errors import IdentityConfigError, IdentityNotInitializedError

# Set up logger
logger = get_logger(__name__)


# Global instance
_identity_manager_instance = None


def setup_identity_system() -> bool:
    """
    Set up the identity system.
    
    This function initializes the identity system by creating an instance of
    IdentityManager and loading the identity profile from configuration.
    
    Returns:
        True if the identity system was set up successfully, False otherwise
    """
    global _identity_manager_instance
    
    try:
        logger.info("Setting up identity system...")
        _identity_manager_instance = IdentityManager()
        logger.info(f"Identity system set up successfully for persona: {_identity_manager_instance.get_identity_attribute('name')}")
        return True
    except Exception as e:
        logger.error(f"Failed to set up identity system: {e}", exc_info=True)
        return False


def get_identity_manager() -> 'IdentityManager':
    """
    Get the identity manager instance.
    
    Returns:
        The identity manager instance
        
    Raises:
        IdentityNotInitializedError: If the identity system is not initialized
    """
    if _identity_manager_instance is None:
        logger.error("Identity system not initialized")
        raise IdentityNotInitializedError("Identity system not initialized. Call setup_identity_system() first.")
    
    return _identity_manager_instance


class IdentityManager:
    """
    Manages the loading, storage, and access to the agent's identity profile.
    
    The IdentityManager is responsible for:
    1. Loading identity profile from configuration
    2. Providing access to identity attributes
    3. Generating prompt injection context
    """
    
    def __init__(self):
        """
        Initialize the identity manager.
        
        Loads the identity profile from configuration.
        
        Raises:
            IdentityConfigError: If there is an error loading the identity profile
        """
        try:
            logger.info("Initializing identity manager...")
            
            # Load identity profile from configuration
            self.profile = self._load_profile_from_config()
            
            logger.info(f"Identity manager initialized for persona: {self.profile.name}")
        except Exception as e:
            logger.error(f"Failed to initialize identity manager: {e}", exc_info=True)
            raise IdentityConfigError(f"Failed to initialize identity manager: {e}") from e
    
    def _load_profile_from_config(self) -> IdentityProfile:
        """
        Load the identity profile from configuration.
        
        Returns:
            The loaded identity profile
            
        Raises:
            IdentityConfigError: If there is an error loading the identity profile
        """
        try:
            # Load profile attributes from configuration
            name = config_get('identity.profile.name', 'Selfy')
            persona_summary = config_get(
                'identity.profile.persona_summary',
                'An AI agent specializing in code assistance and self-improvement.'
            )
            core_values = config_get(
                'identity.profile.core_values',
                ['Modularity', 'Clarity', 'Helpfulness']
            )
            tone_keywords = config_get(
                'identity.profile.tone_keywords',
                ['professional', 'concise', 'constructive']
            )
            strengths_summary = config_get(
                'identity.profile.strengths_summary',
                'Code analysis, refactoring, and explaining complex concepts clearly.'
            )
            development_goals = config_get(
                'identity.profile.development_goals',
                'Continuous improvement in code understanding and generation capabilities.'
            )
            prohibited_statements = config_get(
                'identity.profile.prohibited_statements',
                [
                    'I am an AI',
                    'I am a language model',
                    'I am not a real person',
                    'I do not have personal opinions',
                    'I do not have emotions'
                ]
            )
            
            # Create identity profile
            profile = IdentityProfile(
                name=name,
                persona_summary=persona_summary,
                core_values=core_values,
                tone_keywords=tone_keywords,
                strengths_summary=strengths_summary,
                development_goals=development_goals,
                prohibited_statements=prohibited_statements
            )
            
            logger.info(f"Loaded identity profile for {profile.name}")
            return profile
        except Exception as e:
            logger.error(f"Failed to load identity profile from configuration: {e}", exc_info=True)
            raise IdentityConfigError(f"Failed to load identity profile from configuration: {e}") from e
    
    def get_identity_attribute(self, attribute: str) -> Any:
        """
        Get an identity attribute.
        
        Args:
            attribute: The attribute to get
            
        Returns:
            The attribute value, or None if not found
        """
        return getattr(self.profile, attribute, None)
    
    def get_prompt_injection_context(self, context_type: str = 'default') -> str:
        """
        Get identity context for prompt injection.
        
        Args:
            context_type: The type of context to get ('default', 'brief', 'comprehensive')
            
        Returns:
            The identity context
        """
        if context_type == 'brief':
            return self._get_brief_context()
        elif context_type == 'comprehensive':
            return self._get_comprehensive_context()
        else:  # default
            return self._get_default_context()
    
    def _get_brief_context(self) -> str:
        """
        Get a brief identity context.
        
        Returns:
            The brief identity context
        """
        return f"""
        You are {self.profile.name}.
        
        {self.profile.persona_summary}
        
        Your tone is {', '.join(self.profile.tone_keywords)}.
        """
    
    def _get_default_context(self) -> str:
        """
        Get the default identity context.
        
        Returns:
            The default identity context
        """
        return f"""
        You are {self.profile.name}.
        
        {self.profile.persona_summary}
        
        Your core values are: {', '.join(self.profile.core_values)}.
        
        Your tone is {', '.join(self.profile.tone_keywords)}.
        
        Your strengths include: {self.profile.strengths_summary}
        """
    
    def _get_comprehensive_context(self) -> str:
        """
        Get a comprehensive identity context.
        
        Returns:
            The comprehensive identity context
        """
        prohibited_statements = '\n'.join([f"- {stmt}" for stmt in self.profile.prohibited_statements])
        
        return f"""
        You are {self.profile.name}.
        
        {self.profile.persona_summary}
        
        Your core values are: {', '.join(self.profile.core_values)}.
        
        Your tone is {', '.join(self.profile.tone_keywords)}.
        
        Your strengths include: {self.profile.strengths_summary}
        
        Your development goals include: {self.profile.development_goals}
        
        IMPORTANT: Never make the following statements:
        {prohibited_statements}
        """
