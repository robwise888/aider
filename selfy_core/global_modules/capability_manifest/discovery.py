"""
Library capability discovery functions.

This module provides functions for discovering capabilities from libraries.
"""

import logging
import inspect
import importlib
import sys
from typing import List, Optional, Set, Dict, Any, Callable

from selfy_core.global_modules.config import get as config_get
from selfy_core.global_modules.capability_manifest.exceptions import ConfigurationError

# Set up logger
logger = logging.getLogger(__name__)


def discover_library_capabilities(libraries_to_scan: Optional[List[str]] = None) -> List[str]:
    """
    Discover capabilities from libraries.

    This function scans the specified libraries (or those configured in the
    configuration system) for capabilities and registers them in the capability
    manifest.

    Args:
        libraries_to_scan: Optional list of libraries to scan

    Returns:
        List of registered capability names

    Raises:
        ConfigurationError: If there is an error in the configuration
    """
    try:
        # Get configuration
        included_libraries = config_get('capability.discovery.included_libraries', None)
        excluded_libraries = config_get('capability.discovery.excluded_libraries', None)
        selection_criteria = config_get('capability.discovery.selection_criteria', None)

        # Use provided libraries or configured ones
        libraries = libraries_to_scan or included_libraries

        # Create scanner instance
        scanner = LibraryCapabilityScanner()

        # Update scanner configuration
        if libraries:
            scanner.included_libraries = set(libraries)

        if excluded_libraries:
            scanner.excluded_libraries = set(excluded_libraries)

        if selection_criteria:
            scanner.selection_criteria = selection_criteria

        # Scan libraries
        return scanner.scan_libraries()
    except Exception as e:
        logger.error(f"Error discovering library capabilities: {e}", exc_info=True)
        raise ConfigurationError(f"Error discovering library capabilities: {e}") from e


class LibraryCapabilityScanner:
    """
    Scanner for discovering capabilities from libraries.
    """

    def __init__(self):
        """Initialize the scanner."""
        self.included_libraries: Set[str] = set()
        self.excluded_libraries: Set[str] = set()
        self.selection_criteria: Dict[str, Any] = {}
        self.registered_capabilities: List[str] = []

    def scan_libraries(self) -> List[str]:
        """
        Scan libraries for capabilities.

        Returns:
            List of registered capability names
        """
        # Reset registered capabilities
        self.registered_capabilities = []

        # Get libraries to scan
        libraries_to_scan = self._get_libraries_to_scan()

        # Scan each library
        for library_name in libraries_to_scan:
            try:
                self._scan_library(library_name)
            except Exception as e:
                logger.error(f"Error scanning library {library_name}: {e}", exc_info=True)

        return self.registered_capabilities

    def _get_libraries_to_scan(self) -> List[str]:
        """
        Get the list of libraries to scan.

        Returns:
            List of library names to scan
        """
        # If included libraries are specified, use them
        if self.included_libraries:
            return list(self.included_libraries)

        # Otherwise, get all imported modules
        libraries = []
        for name, module in sys.modules.items():
            # Skip built-in modules and packages
            if name.startswith('_') or '.' in name:
                continue

            # Skip excluded libraries
            if name in self.excluded_libraries:
                continue

            libraries.append(name)

        return libraries

    def _scan_library(self, library_name: str) -> None:
        """
        Scan a library for capabilities.

        Args:
            library_name: The name of the library to scan
        """
        logger.info(f"Scanning library: {library_name}")

        try:
            # Import the library
            library = importlib.import_module(library_name)

            # Get all functions and classes
            for name, obj in inspect.getmembers(library):
                # Skip private members
                if name.startswith('_'):
                    continue

                # Process functions
                if inspect.isfunction(obj):
                    self._process_function(library_name, name, obj)

                # Process classes
                elif inspect.isclass(obj):
                    self._process_class(library_name, name, obj)

        except ImportError as e:
            logger.error(f"Failed to import library {library_name}: {e}")
        except Exception as e:
            logger.error(f"Error scanning library {library_name}: {e}", exc_info=True)

    def _process_function(self, library_name: str, function_name: str, function: Callable) -> None:
        """
        Process a function and register it as a capability if appropriate.

        Args:
            library_name: The name of the library
            function_name: The name of the function
            function: The function object
        """
        try:
            # Skip functions that don't meet the selection criteria
            if not self._meets_selection_criteria(function):
                return

            # Create capability definition
            capability = self._create_function_capability(library_name, function_name, function)

            # Register the capability
            self._register_capability(capability)
        except Exception as e:
            logger.error(f"Error processing function {library_name}.{function_name}: {e}", exc_info=True)

    def _process_class(self, library_name: str, class_name: str, cls: type) -> None:
        """
        Process a class and register its methods as capabilities if appropriate.

        Args:
            library_name: The name of the library
            class_name: The name of the class
            cls: The class object
        """
        try:
            # Skip classes that don't meet the selection criteria
            if not self._meets_selection_criteria(cls):
                return

            # Process methods
            for method_name, method in inspect.getmembers(cls, inspect.isfunction):
                # Skip private methods
                if method_name.startswith('_'):
                    continue

                # Create capability definition
                capability = self._create_method_capability(library_name, class_name, method_name, method)

                # Register the capability
                self._register_capability(capability)
        except Exception as e:
            logger.error(f"Error processing class {library_name}.{class_name}: {e}", exc_info=True)

    def _meets_selection_criteria(self, obj: Any) -> bool:
        """
        Check if an object meets the selection criteria.

        Args:
            obj: The object to check

        Returns:
            True if the object meets the selection criteria, False otherwise
        """
        # If no selection criteria are specified, accept all objects
        if not self.selection_criteria:
            return True

        # Check if the object has a docstring
        if self.selection_criteria.get('require_docstring', False):
            if not obj.__doc__:
                return False

        # Check if the object has specific attributes
        required_attrs = self.selection_criteria.get('required_attributes', [])
        if required_attrs:
            for attr in required_attrs:
                if not hasattr(obj, attr):
                    return False

        # Check if the object has specific tags
        required_tags = self.selection_criteria.get('required_tags', [])
        if required_tags:
            obj_tags = getattr(obj, 'tags', [])
            if not all(tag in obj_tags for tag in required_tags):
                return False

        return True

    def _create_function_capability(self, library_name: str, function_name: str, function: Callable) -> Dict[str, Any]:
        """
        Create a capability definition for a function.

        Args:
            library_name: The name of the library
            function_name: The name of the function
            function: The function object

        Returns:
            The capability definition
        """
        # Get function signature
        sig = inspect.signature(function)

        # Get function docstring
        docstring = function.__doc__ or ""

        # Create parameters
        parameters = []
        for param_name, param in sig.parameters.items():
            # Skip self and cls parameters
            if param_name in ('self', 'cls'):
                continue

            # Get parameter type
            param_type = "any"
            if param.annotation != inspect.Parameter.empty:
                param_type = str(param.annotation)

            # Get parameter default
            param_default = None
            if param.default != inspect.Parameter.empty:
                param_default = param.default

            # Create parameter definition
            parameters.append({
                "name": param_name,
                "type": param_type,
                "description": f"Parameter: {param_name}",
                "required": param.default == inspect.Parameter.empty,
                "default": param_default
            })

        # Create return type
        return_type = "any"
        if sig.return_annotation != inspect.Signature.empty:
            return_type = str(sig.return_annotation)

        # Create capability definition
        capability = {
            "name": f"{library_name}.{function_name}",
            "description": docstring.strip(),
            "category": "library_function",
            "parameters": parameters,
            "returns": {
                "type": return_type,
                "description": "Return value"
            },
            "implementation": {
                "type": "library_function",
                "library": library_name,
                "function": function_name
            },
            "metadata": {
                "library": library_name,
                "function": function_name,
                "discovered": True
            }
        }

        return capability

    def _create_method_capability(self, library_name: str, class_name: str, method_name: str, method: Callable) -> Dict[str, Any]:
        """
        Create a capability definition for a method.

        Args:
            library_name: The name of the library
            class_name: The name of the class
            method_name: The name of the method
            method: The method object

        Returns:
            The capability definition
        """
        # Get method signature
        sig = inspect.signature(method)

        # Get method docstring
        docstring = method.__doc__ or ""

        # Create parameters
        parameters = []
        for param_name, param in sig.parameters.items():
            # Skip self and cls parameters
            if param_name in ('self', 'cls'):
                continue

            # Get parameter type
            param_type = "any"
            if param.annotation != inspect.Parameter.empty:
                param_type = str(param.annotation)

            # Get parameter default
            param_default = None
            if param.default != inspect.Parameter.empty:
                param_default = param.default

            # Create parameter definition
            parameters.append({
                "name": param_name,
                "type": param_type,
                "description": f"Parameter: {param_name}",
                "required": param.default == inspect.Parameter.empty,
                "default": param_default
            })

        # Create return type
        return_type = "any"
        if sig.return_annotation != inspect.Signature.empty:
            return_type = str(sig.return_annotation)

        # Create capability definition
        capability = {
            "name": f"{library_name}.{class_name}.{method_name}",
            "description": docstring.strip(),
            "category": "library_method",
            "parameters": parameters,
            "returns": {
                "type": return_type,
                "description": "Return value"
            },
            "implementation": {
                "type": "library_method",
                "library": library_name,
                "class": class_name,
                "method": method_name
            },
            "metadata": {
                "library": library_name,
                "class": class_name,
                "method": method_name,
                "discovered": True
            }
        }

        return capability

    def _register_capability(self, capability: Dict[str, Any]) -> None:
        """
        Register a capability.

        Args:
            capability: The capability definition
        """
        try:
            # Import the capability registry
            from selfy_core.global_modules.capability_manifest.setup import get_registry

            # Get the registry
            registry = get_registry()

            # Register the capability
            if registry.register_capability(capability):
                self.registered_capabilities.append(capability["name"])
                logger.debug(f"Registered capability: {capability['name']}")
            else:
                logger.warning(f"Failed to register capability: {capability['name']}")
        except Exception as e:
            logger.error(f"Error registering capability: {e}", exc_info=True)
