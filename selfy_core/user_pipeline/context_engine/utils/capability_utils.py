"""
Capability utility functions for the Context Engine.

This module provides utility functions for working with capabilities.
"""

import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

def format_capabilities(capabilities: List[Any]) -> str:
    """
    Format capabilities for inclusion in a context.

    Args:
        capabilities: The capabilities to format

    Returns:
        A formatted string representation of the capabilities
    """
    if not capabilities:
        return "No specific capabilities available for this request."

    formatted_capabilities = []
    for i, capability in enumerate(capabilities):
        # Handle both dictionary and object formats
        if isinstance(capability, dict):
            name = capability.get("name", f"Capability {i+1}")
            description = capability.get("description", "No description available")
            parameters = capability.get("parameters", {})
        else:
            # Assume it's a Capability object
            name = getattr(capability, "name", f"Capability {i+1}")
            description = getattr(capability, "description", "No description available")
            parameters = getattr(capability, "parameters", {})

        formatted_capability = f"- {name}: {description}"

        if parameters:
            formatted_capability += "\n  Parameters:"
            # Handle different parameter formats
            if isinstance(parameters, dict):
                for param_name, param_info in parameters.items():
                    if isinstance(param_info, dict):
                        param_type = param_info.get("type", "any")
                        param_description = param_info.get("description", "No description")
                    else:
                        # Assume it's an object
                        param_type = getattr(param_info, "type", "any")
                        param_description = getattr(param_info, "description", "No description")
                    formatted_capability += f"\n    - {param_name} ({param_type}): {param_description}"
            elif isinstance(parameters, list):
                for i, param in enumerate(parameters):
                    if isinstance(param, dict):
                        param_name = param.get("name", f"Param {i+1}")
                        param_type = param.get("type", "any")
                        param_description = param.get("description", "No description")
                    else:
                        # Assume it's an object
                        param_name = getattr(param, "name", f"Param {i+1}")
                        param_type = getattr(param, "type", "any")
                        param_description = getattr(param, "description", "No description")
                    formatted_capability += f"\n    - {param_name} ({param_type}): {param_description}"

        formatted_capabilities.append(formatted_capability)

    return "\n\n".join(formatted_capabilities)

def extract_capability_names(capabilities: List[Any]) -> List[str]:
    """
    Extract capability names from a list of capabilities.

    Args:
        capabilities: The capabilities to extract names from

    Returns:
        A list of capability names
    """
    result = []
    for i, capability in enumerate(capabilities):
        if isinstance(capability, dict):
            name = capability.get("name", f"Capability {i+1}")
        else:
            # Assume it's a Capability object
            name = getattr(capability, "name", f"Capability {i+1}")
        result.append(name)
    return result

def find_capability_by_name(capabilities: List[Any], name: str) -> Optional[Any]:
    """
    Find a capability by name in a list of capabilities.

    Args:
        capabilities: The capabilities to search
        name: The name to search for

    Returns:
        The capability, or None if not found
    """
    for capability in capabilities:
        if isinstance(capability, dict):
            if capability.get("name") == name:
                return capability
        else:
            # Assume it's a Capability object
            if getattr(capability, "name", None) == name:
                return capability
    return None

def validate_parameters(capability: Any, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate parameters against a capability's parameter schema.

    Args:
        capability: The capability to validate against
        parameters: The parameters to validate

    Returns:
        A dictionary of validation issues, or an empty dictionary if validation passed
    """
    issues = {}

    # Get capability parameters
    if isinstance(capability, dict):
        capability_parameters = capability.get("parameters", {})
    else:
        # Assume it's a Capability object
        capability_parameters = getattr(capability, "parameters", {})

    # Handle different parameter formats
    if isinstance(capability_parameters, dict):
        # Check for required parameters
        for param_name, param_info in capability_parameters.items():
            # Get required flag
            if isinstance(param_info, dict):
                required = param_info.get("required", False)
            else:
                # Assume it's an object
                required = getattr(param_info, "required", False)

            if required and param_name not in parameters:
                issues[param_name] = f"Required parameter '{param_name}' is missing"

        # Check parameter types
        for param_name, param_value in parameters.items():
            if param_name in capability_parameters:
                param_info = capability_parameters[param_name]

                # Get parameter type
                if isinstance(param_info, dict):
                    param_type = param_info.get("type", "any")
                else:
                    # Assume it's an object
                    param_type = getattr(param_info, "type", "any")

                # Basic type checking
                if param_type == "string" and not isinstance(param_value, str):
                    issues[param_name] = f"Parameter '{param_name}' should be a string"
                elif param_type == "number" and not isinstance(param_value, (int, float)):
                    issues[param_name] = f"Parameter '{param_name}' should be a number"
                elif param_type == "boolean" and not isinstance(param_value, bool):
                    issues[param_name] = f"Parameter '{param_name}' should be a boolean"
                elif param_type == "array" and not isinstance(param_value, list):
                    issues[param_name] = f"Parameter '{param_name}' should be an array"
                elif param_type == "object" and not isinstance(param_value, dict):
                    issues[param_name] = f"Parameter '{param_name}' should be an object"
    elif isinstance(capability_parameters, list):
        # For list-based parameters, we'll check if all required parameters are present
        required_params = []
        for param in capability_parameters:
            if isinstance(param, dict):
                param_name = param.get("name")
                required = param.get("required", False)
                if required and param_name:
                    required_params.append(param_name)
            else:
                # Assume it's an object
                param_name = getattr(param, "name", None)
                required = getattr(param, "required", False)
                if required and param_name:
                    required_params.append(param_name)

        # Check if all required parameters are present
        for param_name in required_params:
            if param_name not in parameters:
                issues[param_name] = f"Required parameter '{param_name}' is missing"

    return issues
