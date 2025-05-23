"""
LLM Response for the Selfy agent.

This module defines the standardized response object for LLM interactions.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional


@dataclass
class LLMResponse:
    """
    Standardized response object for LLM interactions.

    This class represents the result of an LLM generation, including the content,
    token usage statistics, and metadata.
    """
    content: str
    provider_name: str
    model_name: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    finish_reason: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the response to a dictionary.

        Returns:
            Dictionary representation of the response
        """
        result = {
            'content': self.content,
            'provider_name': self.provider_name,
            'model_name': self.model_name,
            'prompt_tokens': self.prompt_tokens,
            'completion_tokens': self.completion_tokens,
            'total_tokens': self.total_tokens,
            'finish_reason': self.finish_reason
        }

        if self.tool_calls:
            result['tool_calls'] = self.tool_calls

        if self.metadata:
            result['metadata'] = self.metadata

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LLMResponse':
        """
        Create a response from a dictionary.

        Args:
            data: Dictionary representation of the response

        Returns:
            LLMResponse object
        """
        return cls(
            content=data.get('content', ''),
            provider_name=data.get('provider_name', 'unknown'),
            model_name=data.get('model_name', 'unknown'),
            prompt_tokens=data.get('prompt_tokens', 0),
            completion_tokens=data.get('completion_tokens', 0),
            total_tokens=data.get('total_tokens', 0),
            finish_reason=data.get('finish_reason'),
            tool_calls=data.get('tool_calls'),
            metadata=data.get('metadata', {})
        )
