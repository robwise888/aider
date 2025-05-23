"""
Context Builder for the Context Engine.

This module provides the ContextBuilder class, which is responsible for
building appropriate contexts for LLM interactions.
"""

import logging
import json
from typing import Dict, List, Any, Optional

from selfy_core.global_modules.config import get as config_get
from selfy_core.user_pipeline.context_engine.utils.capability_utils import format_capabilities

class CustomJSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder that can handle complex objects.
    """
    def default(self, obj):
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        elif hasattr(obj, 'to_dict'):
            return obj.to_dict()
        return str(obj)

def safe_json_dumps(obj, indent=2):
    """
    Safely convert an object to a JSON string.
    
    Args:
        obj: The object to convert
        indent: The indentation level
        
    Returns:
        A JSON string representation of the object
    """
    return json.dumps(obj, indent=indent, cls=CustomJSONEncoder)

logger = logging.getLogger(__name__)

class ContextBuilder:
    """
    Builds appropriate contexts for LLM interactions.

    The ContextBuilder is responsible for:
    1. Building contexts for different types of requests
    2. Including relevant capabilities in contexts
    3. Including user preferences in contexts (placeholder for now)
    """

    def __init__(self, capability_manifest=None, identity_filter=None):
        """
        Initialize the context builder.

        Args:
            capability_manifest: The capability manifest to use for context building
            identity_filter: The identity filter to use for filtering prompts and responses
        """
        logger.info("Initializing ContextBuilder")

        self.capability_manifest = capability_manifest
        self.identity_filter = identity_filter

        logger.info("ContextBuilder initialized successfully")

    def build_context(self, request: str, analysis: Dict[str, Any], context_type: str = "default",
                     additional_context: Optional[Dict[str, Any]] = None) -> str:
        """
        Build a context for an LLM interaction.

        Args:
            request: The user's request
            analysis: The analysis of the request
            context_type: The type of context to build
            additional_context: Additional context to include

        Returns:
            The constructed context as a string
        """
        logger.info(f"Building {context_type} context for request: {request[:50]}...")

        # Route to the appropriate context-specific building method based on context_type
        if context_type == "capability_query":
            context = self._build_capability_query_context(request, analysis, additional_context)
        elif context_type == "action":
            context = self._build_action_context(request, analysis, additional_context)
        elif context_type == "code_generation":
            context = self._build_code_generation_context(request, analysis, additional_context)
        elif context_type == "error_recovery":
            context = self._build_error_recovery_context(request, analysis, additional_context)
        elif context_type == "alternative_plan":
            context = self._build_alternative_plan_context(request, analysis, additional_context)
        else:  # default
            context = self._build_default_context(request, analysis, additional_context)

        # Apply identity filter if available
        if self.identity_filter:
            context = self.identity_filter.filter_prompt(context)

        return context

    def _build_default_context(self,
                              request: str,
                              analysis: Dict[str, Any],
                              additional_context: Optional[Dict[str, Any]] = None) -> str:
        """
        Build a default context for an LLM interaction.

        Args:
            request: The user's request
            analysis: The analysis of the request
            additional_context: Additional context to include

        Returns:
            The constructed context as a string
        """
        # Get available capabilities
        available_capabilities = analysis.get("available_capabilities", [])
        capabilities_str = format_capabilities(available_capabilities)

        # Get parameters
        parameters = analysis.get("parameters", {})
        parameters_str = safe_json_dumps(parameters) if parameters else "No specific parameters identified."

        # Get user preferences if available
        user_preferences_str = self._get_user_preferences()

        # Build the context
        context = f"""
        User request: {request}

        Request analysis:
        - Description: {analysis.get('request_description', 'No description available.')}
        - Type: {analysis.get('request_type', 'unknown')}

        Available capabilities:
        {capabilities_str}

        Parameters:
        {parameters_str}
        """

        # Add user preferences if available
        if user_preferences_str:
            context += f"""
            User preferences:
            {user_preferences_str}
            """

        # Add additional context if available
        if additional_context:
            additional_context_str = safe_json_dumps(additional_context)
            context += f"""
            Additional context:
            {additional_context_str}
            """

        return context
        
    def _build_capability_query_context(self,
                                       request: str,
                                       analysis: Dict[str, Any],
                                       additional_context: Optional[Dict[str, Any]] = None) -> str:
        """
        Build a context for a capability query.

        Args:
            request: The user's request
            analysis: The analysis of the request
            additional_context: Additional context to include

        Returns:
            The constructed context as a string
        """
        # Get available capabilities
        available_capabilities = analysis.get("available_capabilities", [])
        capabilities_str = format_capabilities(available_capabilities)

        # Get parameters
        parameters = analysis.get("parameters", {})
        parameters_str = safe_json_dumps(parameters) if parameters else "No specific parameters identified."

        # Get user preferences if available
        user_preferences_str = self._get_user_preferences()

        # Build the context
        context = f"""
        User request: {request}

        The user is asking about available capabilities. This is a capability query.

        Available capabilities:
        {capabilities_str}

        Parameters:
        {parameters_str}
        """

        # Add user preferences if available
        if user_preferences_str:
            context += f"""
            User preferences:
            {user_preferences_str}
            """

        # Add additional context if available
        if additional_context:
            additional_context_str = safe_json_dumps(additional_context)
            context += f"""
            Additional context:
            {additional_context_str}
            """

        return context
        
    def _build_action_context(self,
                             request: str,
                             analysis: Dict[str, Any],
                             additional_context: Optional[Dict[str, Any]] = None) -> str:
        """
        Build a context for an action request.

        Args:
            request: The user's request
            analysis: The analysis of the request
            additional_context: Additional context to include

        Returns:
            The constructed context as a string
        """
        # Get available capabilities
        available_capabilities = analysis.get("available_capabilities", [])
        capabilities_str = format_capabilities(available_capabilities)

        # Get parameters
        parameters = analysis.get("parameters", {})
        parameters_str = safe_json_dumps(parameters) if parameters else "No specific parameters identified."

        # Get user preferences if available
        user_preferences_str = self._get_user_preferences()

        # Build the context
        context = f"""
        User request: {request}

        The user is asking you to perform an action. This is an action request.

        Available capabilities:
        {capabilities_str}

        Parameters:
        {parameters_str}
        """

        # Add user preferences if available
        if user_preferences_str:
            context += f"""
            User preferences:
            {user_preferences_str}
            """

        # Add additional context if available
        if additional_context:
            additional_context_str = safe_json_dumps(additional_context)
            context += f"""
            Additional context:
            {additional_context_str}
            """

        return context
