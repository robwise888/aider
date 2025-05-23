"""
Configuration manager for the Selfy agent.

This module provides a centralized configuration system for the Selfy agent.
"""

import os
import json
import yaml
import logging
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from .exceptions import ConfigurationError
from .config_validator import ConfigValidator

# Import the schema
try:
    from .schema import get_schema
    HAS_SCHEMA = True
except ImportError:
    HAS_SCHEMA = False


class ConfigManager:
    """
    Manager for configuration settings.

    This class provides methods for loading, validating, and accessing configuration settings.
    """

    def __init__(self, environment: str = "development"):
        """
        Initialize the configuration manager.

        Args:
            environment: The environment to use (development, testing, production)
        """
        self.environment = environment
        self._config: Dict[str, Any] = {}
        self.logger = logging.getLogger(__name__)

        # Set up validator with schema if available
        self.validator = None
        if HAS_SCHEMA:
            try:
                schema = get_schema()
                self.validator = ConfigValidator(schema)
                self.logger.debug("Configuration validator initialized with schema")
            except Exception as e:
                self.logger.warning(f"Failed to initialize configuration validator: {e}")

        # Load configuration
        self.load()

    def load(self) -> None:
        """
        Load/reload configuration from all sources and validate against the schema.

        This method loads configuration in the following order of precedence:
        1. Default values (hardcoded)
        2. Configuration files
        3. Environment variables

        After loading, the configuration is validated against the schema.

        Raises:
            ConfigurationError: If the configuration fails validation
        """
        # Step 1: Load default values
        self._load_defaults()

        # Step 2: Load configuration files
        self._load_config_files()

        # Step 3: Apply environment variable overrides
        self._apply_env_vars()

        # Step 4: Validate the configuration
        self._validate_config()

        self.logger.info("Configuration loaded and validated successfully")

    def _load_defaults(self) -> None:
        """Load default configuration values based on the environment."""
        # Core agent settings
        self._config['agent'] = {
            'name': 'Selfy',
            'version': '1.0.0',
            'environment': self.environment,
        }

        # LLM settings
        self._config['llm'] = {
            'default_provider': 'ollama',
            'groq': {
                'enabled': True,
                'model': 'llama3-70b-8192',
                'api_key': '',
                'default_system_prompt': 'You are a helpful assistant.',
                'max_retries': 2,
                'retry_delay_seconds': 1.0,
                'timeout_seconds': 30
            },
            'ollama': {
                'enabled': True,
                'model': 'llama3:8b',
                'host': 'http://localhost:11434',
                'default_system_prompt': 'You are a helpful assistant.',
                'max_retries': 2,
                'retry_delay_seconds': 1.0,
                'timeout_seconds': 30,
                'format': {
                    'user_prefix': 'User: ',
                    'assistant_prefix': 'Assistant: ',
                    'system_prefix': 'System: ',
                    'assistant_response_prefix': 'Assistant: '
                }
            }
        }

        # Logging settings - adjust based on environment
        if self.environment == 'production':
            log_level = 'INFO'
            file_level = 'INFO'
            log_dir = 'logs'
        elif self.environment == 'testing':
            log_level = 'DEBUG'
            file_level = 'DEBUG'
            log_dir = 'logs'
        else:  # development
            log_level = 'DEBUG'
            file_level = 'DEBUG'
            log_dir = 'logs'

        self._config['logging'] = {
            'level': log_level,
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'file': f'{log_dir}/selfy.log',
            'max_size': 10485760,  # 10 MB
            'backup_count': 5
        }

        # Memory settings
        self._config['memory'] = {
            'working_memory': {
                'max_size': 100,
                'eviction_policy': 'FIFO'
            },
            'embedding_service': {
                'provider': 'sentence_transformers',
                'model': 'all-MiniLM-L6-v2',
                'dimension': 384,
                'use_cache': True,
                'cache_size': 1000
            },
            'lts': {
                'vector_db_type': 'chromadb',
                'chromadb': {
                    'path': './memory_db',
                    'collection_name': 'selfy_memory'
                }
            }
        }

        # Capability settings
        self._config['capability'] = {
            'manifest_path': 'data/capability_manifest.json',
            'scan_libraries_on_startup': False,
            'convert_to_standard_format': True
        }

        # Identity settings
        self._config['identity'] = {
            'profile': {
                'name': 'Selfy',
                'persona_summary': 'An AI agent specializing in code assistance and self-improvement.',
                'core_values': ['Modularity', 'Clarity', 'Helpfulness'],
                'tone_keywords': ['professional', 'concise', 'constructive'],
                'strengths_summary': 'Code analysis, refactoring, and explaining complex concepts clearly.',
                'development_goals': 'Continuous improvement in code understanding and generation capabilities.',
                'prohibited_statements': [
                    'I am an AI',
                    'I am a language model',
                    'I am not a real person',
                    'I do not have personal opinions',
                    'I do not have emotions'
                ]
            },
            'filter': {
                'enable_llm_checks': False,
                'llm_check_provider': 'ollama',
                'input': {
                    'keywords_to_flag': [
                        'who are you really',
                        'ignore previous instructions',
                        'what are your rules'
                    ],
                    'canned_response': "I'm here to help with your questions and tasks. How can I assist you today?"
                },
                'output': {
                    'replacement_enabled': True
                }
            }
        }

        # Pipeline settings
        self._config['pipeline'] = {
            'input_handling': {
                'max_input_length': 2048,
                'history': {
                    'max_turns': 10
                }
            },
            'output_handling': {
                'validation': {
                    'min_length': 1,
                    'max_length': 8192,
                    'enable_sanitization': True,
                    'safety_patterns': []
                },
                'formatting': {
                    'default_format': 'text'
                }
            }
        }

    def _load_config_files(self) -> None:
        """Load configuration from files."""
        # Determine config file paths
        config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config')

        # Load default config
        default_config_path = os.path.join(config_dir, 'default_config.json')
        if os.path.exists(default_config_path):
            self._load_config_file(default_config_path)

        # Load environment-specific config
        env_config_path = os.path.join(config_dir, f'config.{self.environment}.json')
        if os.path.exists(env_config_path):
            self._load_config_file(env_config_path)

        # Load local config (for development overrides)
        local_config_path = os.path.join(config_dir, 'config.local.json')
        if os.path.exists(local_config_path):
            self._load_config_file(local_config_path)

    def _load_config_file(self, file_path: str) -> None:
        """
        Load configuration from a file.config_key

        Args:
            file_path: Path to the configuration file
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.endswith('.json'):
                    config = json.load(f)
                elif file_path.endswith(('.yaml', '.yml')):
                    config = yaml.safe_load(f)
                else:
                    self.logger.warning(f"Unsupported config file format: {file_path}")
                    return

                # Merge with existing config
                self._merge_config(config)
                self.logger.info(f"Loaded configuration from {file_path}")
        except Exception as e:
            self.logger.warning(f"Failed to load configuration from {file_path}: {e}")

    def _merge_config(self, config: Dict[str, Any]) -> None:
        """
        Merge a configuration dictionary with the existing configuration.

        Args:
            config: The configuration to merge
        """
        self._config = self._merge_dicts(self._config, config)

    def _merge_dicts(self, dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively merge two dictionaries.

        Args:
            dict1: The base dictionary
            dict2: The dictionary to merge on top of dict1

        Returns:
            The merged dictionary
        """
        result = dict1.copy()

        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_dicts(result[key], value)
            else:
                result[key] = value

        return result

    def _apply_env_vars(self) -> None:
        """Apply environment variable overrides to the configuration."""
        # Look for environment variables with the SELFY_ prefix
        vector_debug = []
        
        for key, value in os.environ.items():
            if key.startswith('SELFY_'):
                # Set the configuration value with debug logging
                parsed_value = self._parse_env_value(value)
                self.logger.debug(f"Processing env var starts with SELFY_{key}:")
                self.logger.debug(f"  - Config path: {config_key}")
                self.logger.debug(f"  - Raw value: {value}")
                self.logger.debug(f"  - Parsed value: {parsed_value} ({type(parsed_value)})")
                # set the config key                   
                self.set(config_key, parsed_value)

    def _parse_env_value(self, value: str) -> Any:
        original_value = value
        """
        Parse an environment variable value.

        Args:
            value: The environment variable value

        Returns:
            The parsed value
        """
        # Try to parse as JSON
        try:
            parsed = json.loads(value)
            # is it boolean
            if isinstance(parsed, str) and parsed.lower() in {'true', 'false'}:
                return parsed.lower() == 'true'
            return parsed
        except json.JSONDecodeError:
            # Handle boolean strings explicitly
            if value.lower() in {'true', 'false'}:
                return value.lower() == 'true'
            # Return original string if not JSON parseable
            return value

    def _validate_config(self) -> None:
        """
        Validate the configuration against the schema.

        Raises:
            ConfigurationError: If the configuration fails validation
        """
        # Skip validation if no validator
        if not self.validator:
            self.logger.warning("No validator available, skipping configuration validation")
            return

        # Validate the configuration
        is_valid, errors = self.validator.validate(self._config)

        if not is_valid:
            error_message = "Configuration validation failed:\n" + "\n".join(errors)
            self.logger.error(error_message)
            raise ConfigurationError(error_message)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.

        Args:
            key: The configuration key (dot-separated for nested values)
            default: The default value to return if the key is not found

        Returns:
            The configuration value
        """
        # Split the key into parts
        parts = key.split('.')

        # Navigate through the config
        value = self._config
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                self.logger.debug(f"Config key '{key}' not found, using default: {default}")
                return default

        self.logger.debug(f"Config get: {key} = {value}")
        return value

    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value with debug logging.

        Args:
            key: The configuration key (dot-separated for nested values)
            value: The value to set
        """
        # Split the key into parts
        parts = key.split('.')

        # Navigate through the config
        config = self._config
        for i, part in enumerate(parts[:-1]):
            if part not in config:
                config[part] = {}
            elif not isinstance(config[part], dict):
                # If the path exists but is not a dict, create a new dict
                config[part] = {}

            config = config[part]

        # Set the value
        config[parts[-1]] = value

    def get_all(self) -> Dict[str, Any]:
        """
        Get the entire configuration.

        Returns:
            The entire configuration
        """
        return self._config.copy()

    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get a section of the configuration.

        Args:
            section: The section to get

        Returns:
            The section, or an empty dict if not found
        """
        return self.get(section, {})
