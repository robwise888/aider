"""
Capability Registry for the capability manifest module.

This module provides the CapabilityRegistry class, which is responsible for
storing and managing capabilities.
"""

import threading
from typing import Dict, List, Any, Optional, Union

from selfy_core.global_modules.logging import get_logger
from .data_structures import Capability
from .exceptions import CapabilityError, CapabilityNotFoundError

# Set up logger
logger = get_logger(__name__)


class CapabilityRegistry:
    """
    Registry for storing and managing capabilities.

    This class provides methods for registering, unregistering, and retrieving
    capabilities, as well as searching for capabilities by various criteria.
    """

    def __init__(self, manifest_path: str = None):
        """
        Initialize the capability registry.

        Args:
            manifest_path: The path to the manifest file
        """
        self.capabilities = {}
        self.manifest_path = manifest_path
        self.lock = threading.RLock()
        logger.info("Capability registry initialized")

    def register_capability(self, capability_definition: Union[Capability, Dict[str, Any]]) -> bool:
        """
        Register a new capability or update an existing one.

        Args:
            capability_definition: The capability to register (Capability object or dictionary)

        Returns:
            True if registration was successful, False otherwise

        Raises:
            CapabilityError: If the capability definition is invalid
        """
        try:
            # Convert to Capability object if necessary
            if isinstance(capability_definition, dict):
                try:
                    capability = Capability.from_dict(capability_definition)
                except Exception as e:
                    logger.error(f"Invalid capability format: {e}")
                    raise CapabilityError(f"Invalid capability format: {e}")
            else:
                capability = capability_definition

            # Validate the capability
            if not capability.name:
                error_msg = "Cannot register capability without a name"
                logger.error(error_msg)
                raise CapabilityError(error_msg)

            with self.lock:
                self.capabilities[capability.name] = capability
                # Use the new persistence function
                from .setup import save_manifest
                save_manifest(self.manifest_path)

            logger.info(f"Registered capability: {capability.name}")
            return True
        except Exception as e:
            logger.error(f"Failed to register capability: {e}")
            return False

    def unregister_capability(self, name: str) -> bool:
        """
        Unregister a capability.

        Args:
            name: The name of the capability to unregister

        Returns:
            True if unregistration was successful, False otherwise
        """
        try:
            with self.lock:
                if name in self.capabilities:
                    del self.capabilities[name]
                    # Use the new persistence function
                    from .setup import save_manifest
                    save_manifest(self.manifest_path)
                    logger.info(f"Unregistered capability: {name}")
                    return True
                else:
                    logger.warning(f"Capability not found: {name}")
                    return False
        except Exception as e:
            logger.error(f"Failed to unregister capability: {e}")
            return False

    def get_capability(self, name: str) -> Optional[Capability]:
        """
        Get a capability by name.

        Args:
            name: The name of the capability to get

        Returns:
            The capability, or None if not found
        """
        with self.lock:
            return self.capabilities.get(name)

    def list_capabilities(self, type: str = None, enabled: bool = None) -> List[Capability]:
        """
        List capabilities.

        Args:
            type: Filter by capability type (optional)
            enabled: Filter by enabled status (optional)

        Returns:
            List of capabilities
        """
        with self.lock:
            result = list(self.capabilities.values())

            # Filter by type if specified
            if type is not None:
                result = [c for c in result if c.type == type]

            # Filter by enabled status if specified
            if enabled is not None:
                result = [c for c in result if c.metadata.get('enabled', True) == enabled]

            return result

    def find_capabilities(self, query: str, limit: int = None) -> List[Capability]:
        """
        Find capabilities matching a query.

        Args:
            query: The query to match
            limit: Maximum number of results to return (optional)

        Returns:
            List of matching capabilities
        """
        with self.lock:
            # Simple search implementation
            query = query.lower()
            result = []

            for capability in self.capabilities.values():
                # Check if query matches name or description
                if query in capability.name.lower() or query in capability.description.lower():
                    result.append(capability)

            # Sort by relevance (name match first, then description match)
            result.sort(key=lambda c: (
                0 if query in c.name.lower() else 1,
                0 if query in c.description.lower() else 1
            ))

            # Limit results if specified
            if limit is not None:
                result = result[:limit]

            return result

    def find_capabilities_by_function(self, function_name: str) -> List[Capability]:
        """
        Find capabilities by function name.

        Args:
            function_name: The function name to match

        Returns:
            List of matching capabilities
        """
        with self.lock:
            result = []

            for capability in self.capabilities.values():
                # Check if function name matches
                functions = capability.metadata.get('functions', [])
                if functions and any(f.get('name') == function_name for f in functions):
                    result.append(capability)

            return result

    def find_capabilities_by_description(self, description: str) -> List[Capability]:
        """
        Find capabilities by description.

        Args:
            description: The description to match

        Returns:
            List of matching capabilities
        """
        with self.lock:
            # Simple search implementation
            description = description.lower()
            result = []

            for capability in self.capabilities.values():
                # Check if description matches
                if description in capability.description.lower():
                    result.append(capability)

            return result

    def find_capabilities_by_category(self, category: str) -> List[Capability]:
        """
        Find capabilities by category.

        Args:
            category: The category to match

        Returns:
            List of matching capabilities
        """
        with self.lock:
            result = []

            for capability in self.capabilities.values():
                # Check if category matches
                if capability.category and capability.category.lower() == category.lower():
                    result.append(capability)

            return result

    def get_all_categories(self) -> List[str]:
        """
        Get all categories.

        Returns:
            List of all categories
        """
        with self.lock:
            categories = set()

            for capability in self.capabilities.values():
                if capability.category:
                    categories.add(capability.category)

            return sorted(list(categories))

    def get_all_capabilities(self) -> List[Capability]:
        """
        Get all capabilities.

        Returns:
            List of all capabilities
        """
        with self.lock:
            return list(self.capabilities.values())

    def get_capabilities_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all capabilities.

        Returns:
            Dictionary containing capability summary
        """
        with self.lock:
            total = len(self.capabilities)
            by_type = {}
            by_category = {}

            for capability in self.capabilities.values():
                # Count by type
                if capability.type not in by_type:
                    by_type[capability.type] = 0
                by_type[capability.type] += 1

                # Count by category
                category = capability.category or 'uncategorized'
                if category not in by_category:
                    by_category[category] = 0
                by_category[category] += 1

            return {
                'total': total,
                'by_type': by_type,
                'by_category': by_category
            }
