"""
JSON utilities for the capability manifest module.

This module provides utilities for JSON serialization of capability-related objects.
"""

import json
from typing import Any, Dict, List, Optional, Union
from .data_structures import Capability, CapabilityResult

class CapabilityJSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for capability-related objects.
    
    This encoder handles serialization of Capability and CapabilityResult objects.
    """
    
    def default(self, obj: Any) -> Any:
        """
        Convert objects to JSON-serializable types.
        
        Args:
            obj: The object to convert
            
        Returns:
            A JSON-serializable representation of the object
        """
        # Handle Capability objects
        if isinstance(obj, Capability):
            # Use the to_dict method if available
            if hasattr(obj, 'to_dict') and callable(obj.to_dict):
                return obj.to_dict()
            
            # Otherwise, convert to dictionary manually
            result = {
                'name': obj.name,
                'type': getattr(obj, 'type', 'unknown'),
                'description': getattr(obj, 'description', ''),
                'source': getattr(obj, 'source', ''),
                'parameters': getattr(obj, 'parameters', []),
                'returns': getattr(obj, 'returns', {})
            }
            
            # Add optional fields if present
            if hasattr(obj, 'category') and obj.category:
                result['category'] = obj.category
                
            if hasattr(obj, 'examples') and obj.examples:
                result['examples'] = obj.examples
                
            if hasattr(obj, 'metadata') and obj.metadata:
                result['metadata'] = obj.metadata
                
            return result
            
        # Handle CapabilityResult objects
        elif isinstance(obj, CapabilityResult):
            # Use the to_dict method if available
            if hasattr(obj, 'to_dict') and callable(obj.to_dict):
                return obj.to_dict()
                
            # Otherwise, convert to dictionary manually
            result = {
                'capability': getattr(obj, 'capability', ''),
                'success': getattr(obj, 'success', False),
                'result': getattr(obj, 'result', None),
                'error_message': getattr(obj, 'error_message', None),
                'error_type': getattr(obj, 'error_type', None),
                'execution_time': getattr(obj, 'execution_time', 0.0)
            }
            
            # Add metadata if present
            if hasattr(obj, 'metadata') and obj.metadata:
                result['metadata'] = obj.metadata
                
            return result
            
        # Let the parent class handle other types
        return super().default(obj)


def dumps(obj: Any, **kwargs) -> str:
    """
    Serialize object to a JSON-formatted string.
    
    This function uses the CapabilityJSONEncoder to handle capability-related objects.
    
    Args:
        obj: The object to serialize
        **kwargs: Additional arguments to pass to json.dumps
        
    Returns:
        A JSON-formatted string
    """
    return json.dumps(obj, cls=CapabilityJSONEncoder, **kwargs)


def dump(obj: Any, fp, **kwargs) -> None:
    """
    Serialize object to a JSON-formatted file.
    
    This function uses the CapabilityJSONEncoder to handle capability-related objects.
    
    Args:
        obj: The object to serialize
        fp: A file-like object
        **kwargs: Additional arguments to pass to json.dump
    """
    return json.dump(obj, fp, cls=CapabilityJSONEncoder, **kwargs)
