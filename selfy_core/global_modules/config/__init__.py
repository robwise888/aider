"""
Configuration module for the Selfy agent.

This module provides a centralized configuration system for the Selfy agent,
with support for hierarchical configuration, environment variables, and validation.
It also supports different environments (development, testing, production) with
environment-specific configuration files and defaults.
"""

import os
from .config_manager import ConfigManager
from .exceptions import ConfigurationError

# Import schema if available
try:
    from .schema import get_schema
    HAS_SCHEMA = True
except ImportError:
    HAS_SCHEMA = False
    get_schema = None

# Determine the environment
environment = os.environ.get('SELFY_ENV', 'development')

# Create a singleton instance of the ConfigManager
config_manager = ConfigManager(environment=environment)

# Export the functions for easy access
get = config_manager.get
set = config_manager.set
load = config_manager.load

# Export environment information
get_environment = lambda: config_manager.environment

# Setup function
def setup(env=None):
    """
    Set up the configuration module.

    Args:
        env: The environment to use (default: None, which uses the environment from SELFY_ENV)

    Returns:
        True if the configuration was set up successfully, False otherwise
    """
    global config_manager, get, set, load, get_environment

    try:
        # Use the provided environment or the default
        actual_env = env or environment

        # Create a new ConfigManager instance
        config_manager = ConfigManager(environment=actual_env)

        # Update the exported functions
        get = config_manager.get
        set = config_manager.set
        load = config_manager.load
        get_environment = lambda: config_manager.environment

        # Export additional utility functions
        global get_all, get_section, load_config_file
        get_all = config_manager.get_all
        get_section = config_manager.get_section
        load_config_file = config_manager._load_config_file

        return True
    except Exception as e:
        print(f"Failed to set up configuration: {e}")
        return False

# Export additional utility functions
get_all = config_manager.get_all
get_section = config_manager.get_section
load_config_file = config_manager._load_config_file

# Export the ConfigurationError exception and schema
__all__ = ['get', 'set', 'load', 'get_environment', 'ConfigurationError', 'setup',
           'get_all', 'get_section', 'load_config_file', 'get_schema', 'HAS_SCHEMA']