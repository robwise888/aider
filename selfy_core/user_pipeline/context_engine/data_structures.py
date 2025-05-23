"""
Data structures for the Context Engine.

This module defines the data structures used by the Context Engine.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

@dataclass
class ProcessedInput:
    """
    Standardized input structure for the Context Engine.

    This class represents the processed input from the Input Handler.

    Attributes:
        validated_input: The user input text after validation and preprocessing
        conversation_history: The recent conversation history fetched from memory
        input_metadata: Dictionary containing key context identifiers
    """
    validated_input: str
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    input_metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ContextEngineResult:
    """
    Standardized output structure for the context engine.

    Attributes:
        response_text: The text response to the user
        capabilities_used: List of capabilities used to fulfill the request
        execution_steps: List of steps taken to fulfill the request
        metadata: Additional metadata about the request and response
    """
    response_text: str
    capabilities_used: List[str] = field(default_factory=list)
    execution_steps: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ExecutionStep:
    """
    Represents a single step in an execution plan.

    Attributes:
        step_id: The ID of the step
        description: A description of the step
        tool: The tool used for the step
        parameters: The parameters passed to the tool
        expected_output: A description of the expected output
        result: The result of executing the step
    """
    step_id: Any
    description: str
    tool: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    expected_output: str = ""
    result: Any = None

@dataclass
class ExecutionPlan:
    """
    Represents an execution plan.

    Attributes:
        steps: The steps in the plan
        fallback_steps: Fallback steps to use if primary steps fail
        final_response_template: Template for the final response
    """
    steps: List[ExecutionStep] = field(default_factory=list)
    fallback_steps: List[ExecutionStep] = field(default_factory=list)
    final_response_template: str = "{{result}}"
