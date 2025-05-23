"""
Capability Manifest module for the Selfy agent.

This module provides a registry for capabilities that Selfy can use to fulfill user requests.
It serves as the single source of truth for all agent capabilities.
"""

# Import data structures and exceptions
from .data_structures import Capability, CapabilityResult
from .exceptions import (
    CapabilityError, CapabilityNotFoundError, ParameterError,
    ExecutionError, ConfigurationError, PersistenceError
)

# Import setup functions
from .setup import (
    setup_capability_manifest, is_setup_complete, get_registry,
    save_manifest, load_manifest
)

# Import execution functions
from .execution import (
    execute_capability,
    execute_capability_with_backward_compatibility
)

# Discovery and conversion functions are part of the self-development pipeline
# but we now have them in production as well
from .discovery import discover_library_capabilities
from .semantic_matching import semantic_match_capabilities

# Import the registry class
from .capability_registry import CapabilityRegistry

# Import JSON utilities
from .json_utils import dumps as json_dumps, dump as json_dump, CapabilityJSONEncoder

# Import access functions
from .access import (
    register_capability,
    unregister_capability,
    get_capability,
    list_capabilities,
    find_capabilities,
    find_capabilities_by_function,
    find_capabilities_by_description,
    find_capabilities_by_category,
    get_capability_functions,
    get_capability_examples,
    get_all_functions,
    get_all_categories,
    get_capabilities_summary,
    get_capabilities_for_context,
    semantic_match_capabilities,
    update_capability,
    add_function_to_capability,
    remove_function_from_capability,
    # New search functions
    search_by_category,
    search_by_tags,
    get_similar_capabilities,
    search_capabilities,
    find_capabilities_by_memory
)

__all__ = [
    # Data structures
    'Capability',
    'CapabilityResult',

    # Exceptions
    'CapabilityError',
    'CapabilityNotFoundError',
    'ParameterError',
    'ExecutionError',
    'ConfigurationError',
    'PersistenceError',

    # Setup functions
    'setup_capability_manifest',
    'is_setup_complete',
    'get_registry',
    'save_manifest',
    'load_manifest',

    # Execution functions
    'execute_capability',
    'execute_capability_with_backward_compatibility',

    # Discovery and conversion functions
    'discover_library_capabilities',

    # Registry class
    'CapabilityRegistry',

    # Access functions
    'register_capability',
    'unregister_capability',
    'get_capability',
    'list_capabilities',
    'find_capabilities',
    'find_capabilities_by_function',
    'find_capabilities_by_description',
    'find_capabilities_by_category',
    'get_capability_functions',
    'get_capability_examples',
    'get_all_functions',
    'get_all_categories',
    'get_capabilities_summary',
    'get_capabilities_for_context',
    'semantic_match_capabilities',
    'update_capability',
    'add_function_to_capability',
    'remove_function_from_capability',

    # Enhanced search functions
    'search_by_category',
    'search_by_tags',
    'get_similar_capabilities',
    'search_capabilities',
    'find_capabilities_by_memory',

    # JSON utilities
    'json_dumps',
    'json_dump',
    'CapabilityJSONEncoder'
]