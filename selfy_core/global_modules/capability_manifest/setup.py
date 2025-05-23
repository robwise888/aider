"""
Setup functions for the capability manifest module.

This module provides functions for setting up the capability manifest system.
"""

import os
import json
import logging
import threading
from typing import Dict, Any, Optional

from selfy_core.global_modules.config import get as config_get
from selfy_core.global_modules.logging import get_logger
from .exceptions import ConfigurationError, PersistenceError, CapabilityError
from .data_structures import Capability

# Set up logger
logger = get_logger(__name__)

# Global state
_registry_initialized = False
_registry_instance = None


def setup_capability_manifest() -> Any:
    """
    Set up the capability manifest system.

    This function initializes the capability manifest system by creating a registry
    instance, loading capabilities from the manifest file, and optionally scanning
    libraries for capabilities.

    Returns:
        The registry instance

    Raises:
        ConfigurationError: If there's an issue with the configuration
        PersistenceError: If there's an issue with file I/O
        CapabilityError: If there's an issue with the capability manifest
    """
    # Import here to avoid circular imports
    from .capability_registry import CapabilityRegistry

    # Get configuration
    manifest_path = config_get('capability.manifest_path', 'data/capability_manifest.json')
    scan_on_startup = config_get('capability.scan_libraries_on_startup', True)
    convert_to_standard = config_get('capability.convert_to_standard_format', True)

    # Initialize the registry
    global _registry_initialized, _registry_instance

    if _registry_initialized:
        logger.warning("Capability manifest already initialized")
        return _registry_instance

    try:
        logger.info("Setting up capability manifest...")

        # Create registry instance
        _registry_instance = CapabilityRegistry(manifest_path)

        # Load capabilities from manifest file
        load_manifest(manifest_path, _registry_instance)

        # Scan libraries for capabilities if enabled
        # NOTE: This functionality is intended for the self-development pipeline
        # and is disabled in the production environment
        if scan_on_startup:
            logger.info("Library scanning is disabled in production environment")

        # Convert capabilities to standard format if enabled
        # NOTE: This functionality is intended for the self-development pipeline
        # and is disabled in the production environment
        if convert_to_standard:
            logger.info("Capability conversion is disabled in production environment")

        _registry_initialized = True
        logger.info(f"Capability manifest initialized with {len(_registry_instance.capabilities)} capabilities")
        return _registry_instance
    except Exception as e:
        logger.error(f"Failed to set up capability manifest: {e}", exc_info=True)
        raise ConfigurationError(f"Failed to set up capability manifest: {e}") from e


def is_setup_complete() -> bool:
    """
    Check if the capability manifest system is set up.

    Returns:
        True if the capability manifest system is set up, False otherwise
    """
    return _registry_initialized


def get_registry() -> Any:
    """
    Get the capability registry instance.

    Returns:
        The registry instance

    Raises:
        ConfigurationError: If the capability manifest system is not initialized
    """
    if not _registry_initialized:
        logger.warning("Capability manifest not initialized. Initializing now...")
        return setup_capability_manifest()

    return _registry_instance


def save_manifest(manifest_path: Optional[str] = None, registry: Any = None) -> None:
    """
    Save the capability manifest to a file.

    Args:
        manifest_path: The path to save the manifest to (optional)
        registry: The registry instance to save (optional)

    Raises:
        PersistenceError: If there's an issue with file I/O
    """
    # Get registry if not provided
    if registry is None:
        registry = get_registry()

    # Get manifest path if not provided
    if manifest_path is None:
        manifest_path = config_get('capability.manifest_path', 'data/capability_manifest.json')

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)

    # Write to a temporary file
    temp_path = f"{manifest_path}.tmp"
    try:
        with open(temp_path, 'w', encoding='utf-8') as f:
            # Convert capabilities to dictionary format
            capabilities_dict = {
                "capabilities": [
                    capability.to_dict() if isinstance(capability, Capability) else capability
                    for capability in registry.capabilities.values()
                ]
            }
            json.dump(capabilities_dict, f, indent=2)

        # Rename the temporary file to the target file (atomic operation)
        os.replace(temp_path, manifest_path)
        logger.debug(f"Saved capability manifest to {manifest_path}")
    except Exception as e:
        logger.error(f"Failed to save capability manifest: {e}")
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
        raise PersistenceError(f"Failed to save capability manifest: {e}")


def load_manifest(manifest_path: Optional[str] = None, registry: Any = None) -> None:
    """
    Load the capability manifest from a file.

    Args:
        manifest_path: The path to load the manifest from (optional)
        registry: The registry instance to load into (optional)

    Raises:
        PersistenceError: If there's an issue with file I/O
    """
    # Get registry if not provided
    if registry is None:
        registry = get_registry()

    # Get manifest path if not provided
    if manifest_path is None:
        manifest_path = config_get('capability.manifest_path', 'data/capability_manifest.json')

    # Check if the manifest file exists
    if not os.path.exists(manifest_path):
        logger.warning(f"Capability manifest not found at {manifest_path}")
        registry.capabilities = {}
        return

    try:
        # Load the manifest
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)

        # Process the manifest
        registry.capabilities = {}

        # Handle different formats
        if "capabilities" in manifest and isinstance(manifest["capabilities"], list):
            # List format
            for capability_dict in manifest["capabilities"]:
                try:
                    # Convert to Capability object
                    capability = Capability.from_dict(capability_dict)
                    registry.capabilities[capability.name] = capability
                except Exception as e:
                    logger.warning(f"Failed to load capability {capability_dict.get('name', 'unknown')}: {e}")
        else:
            # Dictionary format (old format)
            for section_name, section in manifest.items():
                if isinstance(section, dict):
                    for cap_name, cap_data in section.items():
                        if isinstance(cap_data, dict) and "name" in cap_data:
                            try:
                                # Convert to Capability object
                                capability = Capability.from_dict(cap_data)
                                registry.capabilities[capability.name] = capability
                            except Exception as e:
                                logger.warning(f"Failed to load capability {cap_name}: {e}")

        logger.info(f"Loaded {len(registry.capabilities)} capabilities from {manifest_path}")
    except Exception as e:
        logger.error(f"Failed to load capability manifest: {e}")
        registry.capabilities = {}
        raise PersistenceError(f"Failed to load capability manifest: {e}")
