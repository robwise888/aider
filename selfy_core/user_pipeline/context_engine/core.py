"""
Context Engine for the Selfy agent.

This module provides the ContextEngine class, which is responsible for
processing user requests and generating appropriate responses.
"""

import logging
import time
from typing import Dict, List, Any, Optional

from selfy_core.global_modules.config import get as config_get
from selfy_core.user_pipeline.context_engine.request_analyzer import RequestAnalyzer
from selfy_core.user_pipeline.context_engine.context_builder import ContextBuilder
from selfy_core.user_pipeline.context_engine.execution_planner import ExecutionPlanner
from selfy_core.user_pipeline.context_engine.data_structures import ProcessedInput, ContextEngineResult

logger = logging.getLogger(__name__)

class ContextEngine:
    """
    Processes user requests and generates appropriate responses.

    The ContextEngine is responsible for:
    1. Analyzing user requests
    2. Building appropriate contexts for LLM interactions
    3. Planning and executing capability calls
    4. Generating final responses
    """

    def __init__(self, llm_wrapper=None, capability_manifest=None, identity_filter=None, memory_system=None):
        """
        Initialize the context engine.

        Args:
            llm_wrapper: The LLM wrapper to use for processing
            capability_manifest: The capability manifest to use for capability execution
            identity_filter: The identity filter to use for filtering prompts and responses
            memory_system: The memory system for storing execution history
        """
        logger.info("Initializing ContextEngine")

        self.llm_wrapper = llm_wrapper
        self.capability_manifest = capability_manifest
        self.identity_filter = identity_filter
        self.memory_system = memory_system

        # Initialize components
        self.request_analyzer = RequestAnalyzer(
            llm_wrapper=llm_wrapper,
            capability_manifest=capability_manifest,
            identity_filter=identity_filter
        )

        self.context_builder = ContextBuilder(
            capability_manifest=capability_manifest,
            identity_filter=identity_filter,
            llm_wrapper=llm_wrapper
        )

        self.execution_planner = ExecutionPlanner(
            llm_wrapper=llm_wrapper,
            capability_manifest=capability_manifest,
            identity_filter=identity_filter,
            memory_system=memory_system
        )

        logger.info("ContextEngine initialized successfully")

    def process_request(self, input_data: ProcessedInput) -> ContextEngineResult:
        """
        Process a user request.

        Args:
            input_data: The processed input data

        Returns:
            The context engine result
        """
        start_time = time.time()
        logger.info(f"Processing request: {input_data.validated_input[:50]}...")

        # Extract input data
        request = input_data.validated_input
        conversation_history = input_data.conversation_history
        metadata = input_data.input_metadata

        # Step 1: Analyze the request
        analysis = self.request_analyzer.analyze_request(
            request=request,
            conversation_history=conversation_history,
            metadata=metadata
        )

        # Step 2: Build the context
        context_type = self._determine_context_type(analysis)
        context = self.context_builder.build_context(
            request=request,
            analysis=analysis,
            context_type=context_type,
            additional_context={"conversation_history": conversation_history}
        )

        # Step 3: Create an execution plan
        execution_plan = self.execution_planner.create_execution_plan(
            request=request,
            analysis=analysis,
            context=context
        )

        # Step 4: Execute the plan
        execution_results = self.execution_planner.execute_plan(
            plan=execution_plan,
            request=request,
            analysis=analysis,
            context=context
        )

        # Step 5: Create the result
        result = ContextEngineResult(
            response_text=execution_results.get("response", "I'm sorry, I couldn't process your request."),
            capabilities_used=execution_results.get("capabilities_used", []),
            execution_steps=execution_results.get("execution_steps", []),
            metadata={
                "request_type": analysis.get("request_type", "unknown"),
                "confidence": analysis.get("confidence", 0.0),
                "processing_time": time.time() - start_time
            }
        )

        logger.info(f"Request processed in {result.metadata['processing_time']:.2f}s")
        return result

    def _determine_context_type(self, analysis: Dict[str, Any]) -> str:
        """
        Determine the context type based on the request analysis.

        Args:
            analysis: The analysis of the request

        Returns:
            The context type
        """
        request_type = analysis.get("request_type", "unknown")

        if request_type == "general_query":
            return "default"
        elif request_type == "capability_query":
            return "capability_query"
        elif request_type == "code_generation":
            return "code_generation"
        elif request_type == "action":
            return "action"
        elif request_type == "error_recovery":
            return "error_recovery"
        else:
            return "default"


# Global instance
_context_engine_instance = None


def setup_context_engine(llm_wrapper=None, capability_manifest=None, identity_filter=None, memory_system=None) -> bool:
    """
    Set up the context engine.

    Args:
        llm_wrapper: The LLM wrapper to use for processing
        capability_manifest: The capability manifest to use for capability execution
        identity_filter: The identity filter to use for filtering prompts and responses
        memory_system: The memory system for storing execution history

    Returns:
        True if setup was successful, False otherwise
    """
    global _context_engine_instance

    try:
        _context_engine_instance = ContextEngine(
            llm_wrapper=llm_wrapper,
            capability_manifest=capability_manifest,
            identity_filter=identity_filter,
            memory_system=memory_system
        )
        logger.info("Context engine set up successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to set up context engine: {e}")
        return False


def get_context_engine() -> Optional[ContextEngine]:
    """
    Get the context engine instance.

    Returns:
        The context engine instance, or None if not set up
    """
    global _context_engine_instance

    if _context_engine_instance is None:
        logger.warning("Context engine not initialized")

    return _context_engine_instance
