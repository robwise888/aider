"""
Data structures for the capability manifest module.

This module defines the core data structures used by the capability manifest system.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


@dataclass
class Capability:
    """Standard structure representing a capability in the manifest."""
    name: str
    type: str  # 'library' or 'builtin'
    description: str
    source: str
    parameters: List[Dict[str, Any]]
    returns: Dict[str, str]
    category: Optional[str] = None
    examples: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Capability':
        """
        Create a capability from a dictionary.

        This method handles different capability formats and converts them to
        the standard format.

        Args:
            data: Dictionary representation of the capability

        Returns:
            Capability object

        Raises:
            ValueError: If the capability format is invalid
        """
        # Check if this is the standard format
        if all(key in data for key in ['name', 'type', 'description', 'source', 'parameters', 'returns']):
            # Extract required fields
            name = data['name']
            type_value = data['type']
            description = data['description']
            source = data['source']
            parameters = data['parameters']
            returns = data['returns']

            # Extract optional fields
            category = data.get('category')
            examples = data.get('examples', [])
            metadata = data.get('metadata', {})

            return cls(
                name=name,
                type=type_value,
                description=description,
                source=source,
                parameters=parameters,
                returns=returns,
                category=category,
                examples=examples,
                metadata=metadata
            )

        # Check if this is the old format
        elif all(key in data for key in ['name', 'description', 'functions']):
            # Extract required fields
            name = data['name']
            description = data['description']
            functions = data['functions']

            # Determine type and source
            if 'source' in data:
                type_value = 'library'
                source = data['source']
            else:
                type_value = 'builtin'
                source = name

            # Extract parameters and returns
            if functions and isinstance(functions, list) and len(functions) > 0:
                # Use the first function for parameters and returns
                function = functions[0]
                parameters = function.get('parameters', [])
                returns = {'type': function.get('returns', {}).get('type', 'any')}
            else:
                parameters = []
                returns = {'type': 'any'}

            # Extract category
            category = data.get('category')

            # Extract examples
            examples = data.get('examples', [])

            # Build metadata
            metadata = {}

            # Add version, author, enabled if present
            if 'version' in data:
                metadata['version'] = data['version']
            if 'author' in data:
                metadata['author'] = data['author']
            if 'enabled' in data:
                metadata['enabled'] = data.get('enabled', True)

            # Add functions to metadata for backward compatibility
            if 'functions' in data:
                metadata['functions'] = data['functions']

            return cls(
                name=name,
                type=type_value,
                description=description,
                source=source,
                parameters=parameters,
                returns=returns,
                category=category,
                examples=examples,
                metadata=metadata
            )

        # If we can't determine the format, raise an error
        else:
            raise ValueError(f"Invalid capability format: {data}")

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary format.

        Returns:
            Dictionary representation of the capability
        """
        result = {
            'name': self.name,
            'type': self.type,
            'description': self.description,
            'source': self.source,
            'parameters': self.parameters,
            'returns': self.returns
        }

        # Add optional fields if present
        if self.category:
            result['category'] = self.category

        if self.examples:
            result['examples'] = self.examples

        if self.metadata:
            result['metadata'] = self.metadata

        return result


@dataclass
class CapabilityResult:
    """Standardized result object for capability execution."""
    capability: str
    success: bool
    result: Any = None
    error_message: Optional[str] = None
    error_type: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def success(cls, capability: str, result: Any, execution_time: float = 0.0, metadata: Dict[str, Any] = None) -> 'CapabilityResult':
        """
        Create a success result.

        Args:
            capability: The name of the capability
            result: The result of the execution
            execution_time: The execution time in seconds
            metadata: Additional metadata

        Returns:
            CapabilityResult object
        """
        return cls(
            capability=capability,
            success=True,
            result=result,
            execution_time=execution_time,
            metadata=metadata or {}
        )

    @classmethod
    def error(cls, capability: str, error_message: str, error_type: str = 'Error', execution_time: float = 0.0, metadata: Dict[str, Any] = None) -> 'CapabilityResult':
        """
        Create an error result.

        Args:
            capability: The name of the capability
            error_message: The error message
            error_type: The error type
            execution_time: The execution time in seconds
            metadata: Additional metadata

        Returns:
            CapabilityResult object
        """
        return cls(
            capability=capability,
            success=False,
            error_message=error_message,
            error_type=error_type,
            execution_time=execution_time,
            metadata=metadata or {}
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary format.

        Returns:
            Dictionary representation of the result
        """
        result = {
            'capability': self.capability,
            'success': self.success,
            'execution_time': self.execution_time
        }

        if self.success:
            result['result'] = self.result
        else:
            result['error'] = self.error_message
            result['error_type'] = self.error_type

        if self.metadata:
            result['metadata'] = self.metadata

        return result
