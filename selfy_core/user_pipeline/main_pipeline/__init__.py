"""
Main Pipeline module for the User Chat Pipeline.

This module orchestrates the entire User Chat Pipeline, coordinating the flow of
information through the Input Handling, Context Engine, Identity, and Output Handling
modules. It manages conversation state and provides a unified interface for processing
user requests.
"""

from selfy_core.user_pipeline.main_pipeline.setup import (
    setup_user_pipeline, get_user_pipeline, UserPipeline, PipelineResult
)

__all__ = [
    'setup_user_pipeline',
    'get_user_pipeline',
    'UserPipeline',
    'PipelineResult'
]