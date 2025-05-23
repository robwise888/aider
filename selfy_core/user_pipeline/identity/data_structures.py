"""
Data structures for the Identity System.

This module defines the data structures used by the Identity System, including
IdentityProfile and FilterResult.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict


@dataclass
class IdentityProfile:
    """
    Structured definition of the agent's identity.
    
    Attributes:
        name: The agent's name
        persona_summary: A brief summary of the agent's persona
        core_values: List of core values that define the agent
        tone_keywords: List of keywords that describe the agent's tone
        strengths_summary: A summary of the agent's strengths
        development_goals: The agent's development goals
        prohibited_statements: List of statements the agent should not make
    """
    name: str
    persona_summary: str
    core_values: List[str]
    tone_keywords: List[str]
    strengths_summary: Optional[str] = None
    development_goals: Optional[str] = None
    prohibited_statements: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the profile to a dictionary."""
        return {
            'name': self.name,
            'persona_summary': self.persona_summary,
            'core_values': self.core_values,
            'tone_keywords': self.tone_keywords,
            'strengths_summary': self.strengths_summary,
            'development_goals': self.development_goals,
            'prohibited_statements': self.prohibited_statements
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'IdentityProfile':
        """Create a profile from a dictionary."""
        return cls(
            name=data.get('name', ''),
            persona_summary=data.get('persona_summary', ''),
            core_values=data.get('core_values', []),
            tone_keywords=data.get('tone_keywords', []),
            strengths_summary=data.get('strengths_summary'),
            development_goals=data.get('development_goals'),
            prohibited_statements=data.get('prohibited_statements', [])
        )


@dataclass
class FilterResult:
    """
    Standardized return object for filter operations.
    
    Attributes:
        status: The status of the filter operation ('allowed', 'blocked', 'modified')
        output_text: The resulting text (original, modified, or canned response)
        context_snippet: Optional identity context for prompt injection
        reason: Optional reason for blockage or modification
    """
    status: str  # 'allowed', 'blocked', 'modified'
    output_text: Optional[str]
    context_snippet: Optional[str] = None
    reason: Optional[str] = None
