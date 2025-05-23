"""
Context Engine for the Selfy agent.

This package provides the context engine for the Selfy agent, which is responsible
for processing user requests and generating appropriate responses.
"""

from selfy_core.user_pipeline.context_engine.core import (
    ContextEngine, setup_context_engine, get_context_engine
)
from selfy_core.user_pipeline.context_engine.data_structures import (
    ProcessedInput, ContextEngineResult, ExecutionPlan, ExecutionStep
)

__all__ = [
    'ContextEngine',
    'setup_context_engine',
    'get_context_engine',
    'ProcessedInput',
    'ContextEngineResult',
    'ExecutionPlan',
    'ExecutionStep'
]