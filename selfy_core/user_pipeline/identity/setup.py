"""
Setup functions for the identity module.

This module provides functions to set up and access the identity module.
"""

from typing import Optional, Any

# Set up logger
try:
    from selfy_core.global_modules.logging import get_logger
    logger = get_logger(__name__)
except ImportError:
    # Fall back to standard logging
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

# Import configuration
try:
    from selfy_core.global_modules.config import get as config_get
except ImportError:
    logger.warning("Could not import config, using default values")
    
    def config_get(key, default=None):
        """Mock config_get function."""
        return default

# Mock identity classes for testing
class MockIdentityFilter:
    """Mock identity filter for testing."""
    
    def __init__(self, agent_name="Selfy"):
        """Initialize the mock identity filter."""
        self.agent_name = agent_name
        logger.info(f"Initialized MockIdentityFilter with agent_name={agent_name}")
    
    def filter_input(self, input_text, context=None):
        """
        Filter user input.
        
        Args:
            input_text: The input to filter
            context: Optional context for filtering
            
        Returns:
            A FilterResult object
        """
        logger.info("Filtering input with MockIdentityFilter")
        
        class MockFilterResult:
            def __init__(self, status, input_text, context_snippet=None, reason=None):
                self.status = status
                self.input_text = input_text
                self.context_snippet = context_snippet
                self.reason = reason
        
        return MockFilterResult(
            status="allowed",
            input_text=input_text,
            context_snippet=None,
            reason=None
        )
    
    def filter_output(self, output_text, context=None):
        """
        Filter agent output.
        
        Args:
            output_text: The output to filter
            context: Optional context for filtering
            
        Returns:
            A FilterResult object
        """
        logger.info("Filtering output with MockIdentityFilter")
        
        class MockFilterResult:
            def __init__(self, status, output_text, context_snippet=None, reason=None):
                self.status = status
                self.output_text = output_text
                self.context_snippet = context_snippet
                self.reason = reason
        
        return MockFilterResult(
            status="allowed",
            output_text=output_text,
            context_snippet=None,
            reason=None
        )
    
    def inject_identity_context(self, system_prompt):
        """
        Inject identity context into a system prompt.
        
        Args:
            system_prompt: The system prompt to enhance
            
        Returns:
            The enhanced system prompt
        """
        logger.info("Injecting identity context with MockIdentityFilter")
        return f"{system_prompt}\n\nYou are {self.agent_name}, a helpful AI assistant."

class MockIdentityManager:
    """Mock identity manager for testing."""
    
    def __init__(self):
        """Initialize the mock identity manager."""
        self.identity_filter = MockIdentityFilter()
        logger.info("Initialized MockIdentityManager")
    
    def get_identity_filter(self):
        """
        Get the identity filter.
        
        Returns:
            The identity filter
        """
        return self.identity_filter
    
    def get_identity_profile(self):
        """
        Get the identity profile.
        
        Returns:
            The identity profile
        """
        return {
            "name": "Selfy",
            "persona_summary": "An AI agent specializing in code assistance and self-improvement.",
            "core_values": ["Modularity", "Clarity", "Helpfulness"],
            "tone_keywords": ["professional", "concise", "constructive"],
            "strengths_summary": "Code analysis, refactoring, and explaining complex concepts clearly.",
            "development_goals": "Improve code refactoring capabilities and reduce LLM hallucination."
        }

# Global instances
_identity_manager_instance = None
_identity_filter_instance = None

def setup_identity_system() -> bool:
    """
    Set up the identity system.
    
    This function initializes the identity system by creating instances of
    IdentityManager and IdentityFilter and setting them as the global instances.
    
    Returns:
        True if the identity system was set up successfully, False otherwise
    """
    global _identity_manager_instance, _identity_filter_instance
    
    try:
        logger.info("Setting up identity system...")
        
        # Create an identity manager
        _identity_manager_instance = MockIdentityManager()
        
        # Get the identity filter
        _identity_filter_instance = _identity_manager_instance.get_identity_filter()
        
        logger.info("Identity system set up successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to set up identity system: {e}", exc_info=True)
        return False

def get_identity_manager() -> Optional[Any]:
    """
    Get the identity manager instance.
    
    Returns:
        The identity manager instance, or None if it has not been set up
    """
    global _identity_manager_instance
    
    if _identity_manager_instance is None:
        logger.warning("Identity manager has not been set up")
    
    return _identity_manager_instance

def get_identity_filter() -> Optional[Any]:
    """
    Get the identity filter instance.
    
    Returns:
        The identity filter instance, or None if it has not been set up
    """
    global _identity_filter_instance
    
    if _identity_filter_instance is None:
        logger.warning("Identity filter has not been set up")
    
    return _identity_filter_instance
