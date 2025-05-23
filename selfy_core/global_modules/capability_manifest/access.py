"""
Access functions for the capability manifest module.

This module provides functions for accessing capabilities in the registry.
"""

from typing import Dict, List, Any, Optional, Union

from selfy_core.global_modules.logging import get_logger
from .data_structures import Capability
from .exceptions import CapabilityError, CapabilityNotFoundError
from .setup import get_registry

# Set up logger
logger = get_logger(__name__)


def register_capability(capability_definition: Union[Capability, Dict[str, Any]]) -> bool:
    """
    Register a new capability or update an existing one.

    Args:
        capability_definition: The capability to register (Capability object or dictionary)

    Returns:
        True if registration was successful, False otherwise

    Raises:
        CapabilityError: If the capability definition is invalid
    """
    return get_registry().register_capability(capability_definition)


def unregister_capability(name: str) -> bool:
    """
    Unregister a capability.

    Args:
        name: The name of the capability to unregister

    Returns:
        True if unregistration was successful, False otherwise
    """
    return get_registry().unregister_capability(name)


def get_capability(name: str) -> Optional[Capability]:
    """
    Get a capability by name.

    Args:
        name: The name of the capability to get

    Returns:
        The capability, or None if not found
    """
    return get_registry().get_capability(name)


def list_capabilities(type: str = None, enabled: bool = None) -> List[Capability]:
    """
    List capabilities.

    Args:
        type: Filter by capability type (optional)
        enabled: Filter by enabled status (optional)

    Returns:
        List of capabilities
    """
    return get_registry().list_capabilities(type, enabled)


def find_capabilities(query: str, limit: int = None) -> List[Capability]:
    """
    Find capabilities matching a query.

    Args:
        query: The query to match
        limit: Maximum number of results to return (optional)

    Returns:
        List of matching capabilities
    """
    return get_registry().find_capabilities(query, limit)


def find_capabilities_by_function(function_name: str) -> List[Capability]:
    """
    Find capabilities by function name.

    Args:
        function_name: The function name to match

    Returns:
        List of matching capabilities
    """
    return get_registry().find_capabilities_by_function(function_name)


def find_capabilities_by_description(description: str) -> List[Capability]:
    """
    Find capabilities by description.

    Args:
        description: The description to match

    Returns:
        List of matching capabilities
    """
    return get_registry().find_capabilities_by_description(description)


def find_capabilities_by_category(category: str) -> List[Capability]:
    """
    Find capabilities by category.

    Args:
        category: The category to match

    Returns:
        List of matching capabilities
    """
    return get_registry().find_capabilities_by_category(category)


def get_capability_functions(name: str) -> List[Dict[str, Any]]:
    """
    Get functions for a capability.

    Args:
        name: The name of the capability

    Returns:
        List of functions

    Raises:
        CapabilityNotFoundError: If the capability is not found
    """
    capability = get_registry().get_capability(name)
    if not capability:
        raise CapabilityNotFoundError(f"Capability not found: {name}")

    return capability.metadata.get('functions', [])


def get_capability_examples(name: str) -> List[Dict[str, Any]]:
    """
    Get examples for a capability.

    Args:
        name: The name of the capability

    Returns:
        List of examples

    Raises:
        CapabilityNotFoundError: If the capability is not found
    """
    capability = get_registry().get_capability(name)
    if not capability:
        raise CapabilityNotFoundError(f"Capability not found: {name}")

    return capability.examples or []


def get_all_functions() -> List[Dict[str, Any]]:
    """
    Get all functions from all capabilities.

    Returns:
        List of functions
    """
    functions = []
    for capability in get_registry().list_capabilities():
        capability_functions = capability.metadata.get('functions', [])
        for function in capability_functions:
            function['capability'] = capability.name
            functions.append(function)
    return functions


def get_all_categories() -> List[str]:
    """
    Get all categories.

    Returns:
        List of all categories
    """
    return get_registry().get_all_categories()


def get_capabilities_summary() -> Dict[str, Any]:
    """
    Get a summary of all capabilities.

    Returns:
        Dictionary containing capability summary
    """
    return get_registry().get_capabilities_summary()


def get_capabilities_for_context() -> Dict[str, Any]:
    """
    Get capabilities formatted for context injection.

    Returns:
        Dictionary containing capabilities formatted for context injection
    """
    capabilities = list_capabilities()
    result = {
        'capabilities': []
    }

    for capability in capabilities:
        result['capabilities'].append({
            'name': capability.name,
            'description': capability.description,
            'type': capability.type,
            'category': capability.category,
            'parameters': capability.parameters,
            'returns': capability.returns
        })

    return result


def semantic_match_capabilities(query: str, limit: int = 5) -> List[Capability]:
    """
    Find capabilities using semantic matching.

    Args:
        query: The query to match
        limit: Maximum number of results to return

    Returns:
        List of matching capabilities
    """
    # For now, just use simple keyword matching
    # In the future, this could use embeddings or an LLM for better matching
    return find_capabilities(query, limit)


def update_capability(name: str, updates: Dict[str, Any]) -> bool:
    """
    Update a capability.

    Args:
        name: The name of the capability to update
        updates: Dictionary of updates to apply

    Returns:
        True if update was successful, False otherwise

    Raises:
        CapabilityNotFoundError: If the capability is not found
    """
    capability = get_registry().get_capability(name)
    if not capability:
        raise CapabilityNotFoundError(f"Capability not found: {name}")

    # Apply updates
    for key, value in updates.items():
        if hasattr(capability, key):
            setattr(capability, key, value)
        elif key in capability.metadata:
            capability.metadata[key] = value

    # Register the updated capability
    return get_registry().register_capability(capability)


def add_function_to_capability(name: str, function: Dict[str, Any]) -> bool:
    """
    Add a function to a capability.

    Args:
        name: The name of the capability
        function: The function to add

    Returns:
        True if addition was successful, False otherwise

    Raises:
        CapabilityNotFoundError: If the capability is not found
    """
    capability = get_registry().get_capability(name)
    if not capability:
        raise CapabilityNotFoundError(f"Capability not found: {name}")

    # Get functions
    functions = capability.metadata.get('functions', [])

    # Add function
    functions.append(function)
    capability.metadata['functions'] = functions

    # Register the updated capability
    return get_registry().register_capability(capability)


def remove_function_from_capability(name: str, function_name: str) -> bool:
    """
    Remove a function from a capability.

    Args:
        name: The name of the capability
        function_name: The name of the function to remove

    Returns:
        True if removal was successful, False otherwise

    Raises:
        CapabilityNotFoundError: If the capability is not found
    """
    capability = get_registry().get_capability(name)
    if not capability:
        raise CapabilityNotFoundError(f"Capability not found: {name}")

    # Get functions
    functions = capability.metadata.get('functions', [])

    # Remove function
    functions = [f for f in functions if f.get('name') != function_name]
    capability.metadata['functions'] = functions

    # Register the updated capability
    return get_registry().register_capability(capability)


# Enhanced search functions

def search_by_category(category: str, limit: int = None) -> List[Capability]:
    """
    Search capabilities by category.

    Args:
        category: The category to search for
        limit: Maximum number of results to return

    Returns:
        List of matching capabilities
    """
    results = find_capabilities_by_category(category)
    if limit is not None:
        results = results[:limit]
    return results


def search_by_tags(tags: List[str], limit: int = None) -> List[Capability]:
    """
    Search capabilities by tags.

    Args:
        tags: The tags to search for
        limit: Maximum number of results to return

    Returns:
        List of matching capabilities
    """
    results = []
    for capability in get_registry().list_capabilities():
        capability_tags = capability.metadata.get('tags', [])
        if any(tag in capability_tags for tag in tags):
            results.append(capability)

    if limit is not None:
        results = results[:limit]
    return results


def get_similar_capabilities(name: str, limit: int = 5) -> List[Capability]:
    """
    Get capabilities similar to a given capability.

    Args:
        name: The name of the capability
        limit: Maximum number of results to return

    Returns:
        List of similar capabilities

    Raises:
        CapabilityNotFoundError: If the capability is not found
    """
    capability = get_registry().get_capability(name)
    if not capability:
        raise CapabilityNotFoundError(f"Capability not found: {name}")

    # Get capabilities in the same category
    if capability.category:
        return find_capabilities_by_category(capability.category)[:limit]
    else:
        # Fall back to keyword search
        return find_capabilities(capability.name, limit)


def search_capabilities(query: str, categories: List[str] = None, types: List[str] = None, limit: int = None) -> List[Capability]:
    """
    Search capabilities with multiple filters.

    Args:
        query: The query to match
        categories: List of categories to filter by
        types: List of types to filter by
        limit: Maximum number of results to return

    Returns:
        List of matching capabilities
    """
    # Start with all capabilities
    results = get_registry().list_capabilities()

    # Filter by query
    if query:
        results = [c for c in results if query.lower() in c.name.lower() or query.lower() in c.description.lower()]

    # Filter by categories
    if categories:
        results = [c for c in results if c.category and c.category in categories]

    # Filter by types
    if types:
        results = [c for c in results if c.type in types]

    # Sort by relevance (name match first, then description match)
    if query:
        results.sort(key=lambda c: (
            0 if query.lower() in c.name.lower() else 1,
            0 if query.lower() in c.description.lower() else 1
        ))

    # Limit results
    if limit is not None:
        results = results[:limit]

    return results


def find_capabilities_by_memory(memory_type: str, limit: int = None) -> List[Capability]:
    """
    Find capabilities that can work with a specific memory type.

    Args:
        memory_type: The memory type to match
        limit: Maximum number of results to return

    Returns:
        List of matching capabilities
    """
    results = []
    for capability in get_registry().list_capabilities():
        memory_types = capability.metadata.get('memory_types', [])
        if memory_type in memory_types:
            results.append(capability)

    if limit is not None:
        results = results[:limit]
    return results
