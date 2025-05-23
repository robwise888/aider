"""
Configuration validator for the Selfy agent.

This module provides validation for configuration values against a schema.
"""

import re
import os
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union

from .exceptions import ConfigurationError


class ConfigValidator:
    """
    Validator for configuration values.

    This class provides methods for validating configuration values against a schema.
    """

    def __init__(self, schema: Dict[str, Any]):
        """
        Initialize the validator with a schema.

        Args:
            schema: The schema to validate against
        """
        self.schema = schema

    def validate(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate a configuration against the schema.

        Args:
            config: The configuration to validate

        Returns:
            A tuple of (is_valid, error_messages)
        """
        errors = []
        self._validate_section(config, self.schema, "", errors)
        return len(errors) == 0, errors

    def _validate_section(self, config: Dict[str, Any], schema: Dict[str, Any],
                         path: str, errors: List[str]) -> None:
        """
        Validate a section of the configuration against the schema.

        Args:
            config: The configuration section to validate
            schema: The schema section to validate against
            path: The path to the current section (for error messages)
            errors: List to collect error messages
        """
        # Check for required fields
        for key, field_schema in schema.items():
            if isinstance(field_schema, dict) and field_schema.get('required', False):
                if key not in config:
                    errors.append(f"Missing required field: {self._get_full_path(path, key)}")

        # Validate each field in the config
        for key, value in config.items():
            field_path = self._get_full_path(path, key)
            
            # Check if the field is in the schema
            if key not in schema:
                errors.append(f"Unknown field: {field_path}")
                continue
            
            field_schema = schema[key]
            
            # If the field schema is a dictionary with a 'type' key, it's a field definition
            if isinstance(field_schema, dict) and 'type' in field_schema:
                self._validate_field(value, field_schema, field_path, errors)
            # If the field schema is a dictionary without a 'type' key, it's a nested section
            elif isinstance(field_schema, dict) and isinstance(value, dict):
               # Allow flat validation for previously nested structures
               if 'schema' in field_schema:
                   self._validate_section(value, field_schema['schema'], field_path, errors)
               else:
                   self._validate_field(value, field_schema, field_path, errors)
            # Otherwise, it's an error
            elif isinstance(value, dict):
                errors.append(f"Invalid field: {field_path} (expected a value, got a section)")

    def _validate_field(self, value: Any, field_schema: Dict[str, Any],
                       path: str, errors: List[str]) -> None:
        """
        Validate a field against its schema.

        Args:
            value: The field value to validate
            field_schema: The schema for the field
            path: The path to the field (for error messages)
            errors: List to collect error messages
        """
        # Check type
        expected_type = field_schema.get('type')
        if expected_type and not self._check_type(value, expected_type):
            errors.append(f"Invalid type for {path}: expected {expected_type.__name__}, got {type(value).__name__}")
            return
        
        # Check enum
        if 'enum' in field_schema and value not in field_schema['enum']:
            errors.append(f"Invalid value for {path}: {value} (must be one of {field_schema['enum']})")
        
        # Check pattern
        if 'pattern' in field_schema and isinstance(value, str):
            pattern = field_schema['pattern']
            if not re.match(pattern, value):
                errors.append(f"Invalid value for {path}: {value} (must match pattern {pattern})")
        
        # Check min/max for numbers
        if isinstance(value, (int, float)):
            if 'min' in field_schema and value < field_schema['min']:
                errors.append(f"Invalid value for {path}: {value} (must be >= {field_schema['min']})")
            if 'max' in field_schema and value > field_schema['max']:
                errors.append(f"Invalid value for {path}: {value} (must be <= {field_schema['max']})")
        
        # Check min_length/max_length for strings
        if isinstance(value, str):
            if 'min_length' in field_schema and len(value) < field_schema['min_length']:
                errors.append(f"Invalid value for {path}: {value} (length must be >= {field_schema['min_length']})")
            if 'max_length' in field_schema and len(value) > field_schema['max_length']:
                errors.append(f"Invalid value for {path}: {value} (length must be <= {field_schema['max_length']})")
        
        # Check min_items/max_items for lists
        if isinstance(value, list):
            if 'min_items' in field_schema and len(value) < field_schema['min_items']:
                errors.append(f"Invalid value for {path}: {value} (must have >= {field_schema['min_items']} items)")
            if 'max_items' in field_schema and len(value) > field_schema['max_items']:
                errors.append(f"Invalid value for {path}: {value} (must have <= {field_schema['max_items']} items)")
            
            # Validate items in the list
            if 'items' in field_schema and isinstance(field_schema['items'], dict):
                for i, item in enumerate(value):
                    item_path = f"{path}[{i}]"
                    self._validate_field(item, field_schema['items'], item_path, errors)
        
        # Check schema for dictionaries
        if isinstance(value, dict) and 'schema' in field_schema:
            self._validate_section(value, field_schema['schema'], path, errors)

    def _check_type(self, value: Any, expected_type: Union[Type, List[Type]]) -> bool:
        """
        Check if a value is of the expected type.

        Args:
            value: The value to check
            expected_type: The expected type or list of types

        Returns:
            True if the value is of the expected type, False otherwise
        """
        if expected_type == Any:
            return True
        
        if isinstance(expected_type, list):
            return any(self._check_type(value, t) for t in expected_type)
        
        if expected_type == bool:
            return isinstance(value, bool)
        elif expected_type == int:
            return isinstance(value, int) and not isinstance(value, bool)
        elif expected_type == float:
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        elif expected_type == str:
            return isinstance(value, str)
        elif expected_type == list:
            return isinstance(value, list)
        elif expected_type == dict:
            return isinstance(value, dict)
        
        return isinstance(value, expected_type)

    def _get_full_path(self, path: str, key: str) -> str:
        """
        Get the full path to a field.

        Args:
            path: The path to the current section
            key: The key of the field

        Returns:
            The full path to the field
        """
        if path:
            return f"{path}.{key}"
        return key
