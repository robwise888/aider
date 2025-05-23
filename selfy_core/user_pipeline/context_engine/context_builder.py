"""
Context Builder for the Context Engine.

This module provides the ContextBuilder class, which is responsible for
building appropriate contexts for LLM interactions.
"""

import logging
import json
from typing import Dict, List, Any, Optional

from selfy_core.global_modules.config import get as config_get
from selfy_core.user_pipeline.context_engine.utils.capability_utils import format_capabilities
from selfy_core.user_pipeline.context_engine.utils.prompt_utils import validate_and_deduplicate_prompt

class CustomJSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder that can handle complex objects.
    """
    def default(self, obj):
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        elif hasattr(obj, 'to_dict'):
            return obj.to_dict()
        # Handle MemoryItem objects specifically
        elif hasattr(obj, 'id') and hasattr(obj, 'timestamp') and hasattr(obj, 'type') and hasattr(obj, 'content'):
            return {
                'id': obj.id,
                'timestamp': obj.timestamp,
                'type': str(obj.type),
                'content': obj.content,
                'metadata': obj.metadata if hasattr(obj, 'metadata') else {}
            }
        return str(obj)

def safe_json_dumps(obj, indent=2):
    """
    Safely convert an object to a JSON string.

    Args:
        obj: The object to convert
        indent: The indentation level

    Returns:
        A JSON string representation of the object
    """
    return json.dumps(obj, indent=indent, cls=CustomJSONEncoder)

logger = logging.getLogger(__name__)

class ContextBuilder:
    """
    Builds appropriate contexts for LLM interactions.

    The ContextBuilder is responsible for:
    1. Building contexts for different types of requests
    2. Including relevant capabilities in contexts
    3. Including user preferences in contexts (placeholder for now)
    """

    def __init__(self, capability_manifest=None, identity_filter=None, llm_wrapper=None):
        """
        Initialize the context builder.

        Args:
            capability_manifest: The capability manifest to use for context building
            identity_filter: The identity filter to use for filtering prompts and responses
            llm_wrapper: The LLM wrapper to use for LLM-assisted context construction
        """
        logger.info("Initializing ContextBuilder")

        self.capability_manifest = capability_manifest
        self.identity_filter = identity_filter
        self.llm_wrapper = llm_wrapper

        # Configuration
        from selfy_core.global_modules.config import get as config_get
        self.use_llm_assisted_context = config_get('context_engine.use_llm_assisted_context', False)
        self.llm_assist_threshold = config_get('context_engine.llm_assist_threshold', 0.8)

        logger.info("ContextBuilder initialized successfully")

    def build_context(self, request: str, analysis: Dict[str, Any], context_type: str = "default",
                     additional_context: Optional[Dict[str, Any]] = None) -> str:
        """
        Build a context for an LLM interaction.

        Args:
            request: The user's request
            analysis: The analysis of the request
            context_type: The type of context to build
            additional_context: Additional context to include

        Returns:
            The constructed context as a string
        """
        logger.info(f"Building {context_type} context for request: {request[:50]}...")

        # Log detailed flow information
        logger.debug(f"CONTEXT ENGINE FLOW - CONTEXT BUILDER INPUT: Request: {request}")
        logger.debug(f"CONTEXT ENGINE FLOW - CONTEXT BUILDER ANALYSIS: {safe_json_dumps(analysis)}")
        logger.debug(f"CONTEXT ENGINE FLOW - CONTEXT TYPE: {context_type}")

        # Log the full analysis for debugging
        logger.debug(f"Full analysis received: {safe_json_dumps(analysis)}")

        # Check for potential_answer in analysis
        if "potential_answer" in analysis:
            logger.info(f"Analysis contains potential_answer: {analysis['potential_answer'][:100]}...")
            logger.info(f"Analysis confidence: {analysis.get('confidence', 0.0):.2f}")
        else:
            logger.info("Analysis does not contain a potential_answer")

        # Check if there's a potential answer with high confidence
        if "potential_answer" in analysis and analysis.get("confidence", 0.0) >= 0.9:
            logger.info(f"Using potential answer from analysis (confidence: {analysis.get('confidence', 0.0):.2f})")
            logger.info(f"Potential answer: {analysis['potential_answer']}")
            # Create a special context that includes the potential answer
            context = self._build_potential_answer_context(request, analysis, additional_context)
            logger.debug(f"Built potential answer context: {context[:200]}...")
        # Route to the appropriate context-specific building method based on context_type
        elif context_type == "capability_query":
            logger.info("Building capability query context")
            context = self._build_capability_query_context(request, analysis, additional_context)
        elif context_type == "action":
            logger.info("Building action context")
            context = self._build_action_context(request, analysis, additional_context)
        elif context_type == "code_generation":
            logger.info("Building code generation context")
            context = self._build_code_generation_context(request, analysis, additional_context)
        elif context_type == "error_recovery":
            logger.info("Building error recovery context")
            context = self._build_error_recovery_context(request, analysis, additional_context)
        elif context_type == "alternative_plan":
            logger.info("Building alternative plan context")
            context = self._build_alternative_plan_context(request, analysis, additional_context)
        else:  # default
            logger.info("Building default context")
            context = self._build_default_context(request, analysis, additional_context)

        # Check if we should use LLM-assisted context construction
        if self.use_llm_assisted_context and self.llm_wrapper:
            # If confidence is zero, always use LLM assistance regardless of complexity
            if analysis.get("confidence", 1.0) == 0.0:
                logger.info("Using LLM-assisted context construction due to zero confidence")
                # Get the preferred LLM from the analysis (should be cloud provider due to zero confidence)
                preferred_llm = analysis.get("preferred_llm", "ollama")
                logger.info(f"Using preferred LLM for context optimization: {preferred_llm}")
                # Use LLM to optimize the context
                context = self._optimize_context_with_llm(context, request, analysis, preferred_llm)
            else:
                # Determine if this is a complex query that would benefit from LLM assistance
                complexity = analysis.get("complexity", 0.0)
                if complexity >= self.llm_assist_threshold:
                    logger.info(f"Using LLM-assisted context construction (complexity: {complexity:.2f})")
                    # Get the preferred LLM from the analysis
                    preferred_llm = analysis.get("preferred_llm", "ollama")
                    # Use LLM to optimize the context
                    context = self._optimize_context_with_llm(context, request, analysis, preferred_llm)
                else:
                    logger.info(f"Skipping LLM-assisted context construction (complexity: {complexity:.2f} below threshold)")
        else:
            logger.info("LLM-assisted context construction is disabled or LLM wrapper is not available")

        # Apply identity filter if available
        if self.identity_filter:
            logger.debug("Applying identity filter to context")
            context = self.identity_filter.filter_prompt(context)

        # Validate and deduplicate the context to remove any duplicated content
        logger.debug("Validating and deduplicating context")
        validated_context, validation_stats = validate_and_deduplicate_prompt(context)

        # Log validation results
        if validation_stats["duplications_found"] > 0:
            logger.info(f"Context validation removed {validation_stats['duplications_found']} duplications "
                       f"({validation_stats['characters_removed']} characters)")
            logger.debug(f"Context validation stats: {validation_stats}")
        else:
            logger.debug("No duplications found in context")

        logger.debug(f"Final context built (first 200 chars): {validated_context[:200]}...")

        # Log the full final context for detailed flow tracking
        logger.debug(f"CONTEXT ENGINE FLOW - CONTEXT BUILDER OUTPUT: {validated_context}")

        return validated_context

    def _build_default_context(self,
                              request: str,
                              analysis: Dict[str, Any],
                              additional_context: Optional[Dict[str, Any]] = None) -> str:
        """
        Build a default context for an LLM interaction.

        Args:
            request: The user's request
            analysis: The analysis of the request
            additional_context: Additional context to include

        Returns:
            The constructed context as a string
        """
        # Determine if capabilities are needed based on request type
        request_type = analysis.get("request_type", "unknown")

        # For general queries, don't include capabilities
        if request_type == "general_query":
            logger.info("General query detected, skipping capability inclusion")
            capabilities_section = ""
        else:
            # Get matched capabilities from analysis
            matched_capabilities = analysis.get("matched_capabilities", [])

            # Rank capabilities by relevance if there are multiple
            if len(matched_capabilities) > 1:
                logger.info(f"Ranking {len(matched_capabilities)} capabilities by relevance")
                ranked_capabilities = self._rank_capabilities_by_relevance(request, matched_capabilities)
            else:
                ranked_capabilities = matched_capabilities

            # Format capabilities for inclusion in context
            capabilities_section = f"""
            Available capabilities:
            {format_capabilities(ranked_capabilities)}
            """
            logger.info(f"Including {len(ranked_capabilities)} capabilities in context")

        # Get parameters
        parameters = analysis.get("parameters", {})
        parameters_str = safe_json_dumps(parameters) if parameters else "No specific parameters identified."

        # Get knowledge domains if available
        knowledge_domains = analysis.get("required_knowledge_domains", [])
        knowledge_domains_section = ""
        if knowledge_domains:
            knowledge_domains_str = ", ".join(knowledge_domains)
            knowledge_domains_section = f"""
            Required knowledge domains:
            {knowledge_domains_str}
            """
            logger.info(f"Including {len(knowledge_domains)} knowledge domains in context")

        # Get user preferences if available
        user_preferences_str = self._get_user_preferences()

        # Build the context
        context = f"""
        User request: {request}

        Request analysis:
        - Description: {analysis.get('request_description', 'No description available.')}
        - Type: {request_type}
        """

        # Add capabilities section if needed
        if capabilities_section:
            context += capabilities_section

        # Add knowledge domains section if needed
        if knowledge_domains_section:
            context += knowledge_domains_section

        # Add parameters section
        context += f"""
        Parameters:
        {parameters_str}
        """

        # Add user preferences if available
        if user_preferences_str:
            context += f"""
            User preferences:
            {user_preferences_str}
            """

        # Add additional context if available
        if additional_context:
            additional_context_str = safe_json_dumps(additional_context)
            context += f"""
            Additional context:
            {additional_context_str}
            """

        return context

    def _build_capability_query_context(self,
                                       request: str,
                                       analysis: Dict[str, Any],
                                       additional_context: Optional[Dict[str, Any]] = None) -> str:
        """
        Build a context for a capability query.

        Args:
            request: The user's request
            analysis: The analysis of the request
            additional_context: Additional context to include

        Returns:
            The constructed context as a string
        """
        # Get matched capabilities from analysis
        matched_capabilities = analysis.get("matched_capabilities", [])

        # Rank capabilities by relevance if there are multiple
        if len(matched_capabilities) > 1:
            logger.info(f"Ranking {len(matched_capabilities)} capabilities by relevance")
            ranked_capabilities = self._rank_capabilities_by_relevance(request, matched_capabilities)
        else:
            ranked_capabilities = matched_capabilities

        # Format capabilities for inclusion in context
        capabilities_str = format_capabilities(ranked_capabilities)

        # Get parameters
        parameters = analysis.get("parameters", {})
        parameters_str = safe_json_dumps(parameters) if parameters else "No specific parameters identified."

        # Get user preferences if available
        user_preferences_str = self._get_user_preferences()

        # Build the context
        context = f"""
        User request: {request}

        The user is asking about available capabilities. This is a capability query.

        Available capabilities:
        {capabilities_str}

        Parameters:
        {parameters_str}
        """

        # Add user preferences if available
        if user_preferences_str:
            context += f"""
            User preferences:
            {user_preferences_str}
            """

        # Add additional context if available
        if additional_context:
            additional_context_str = safe_json_dumps(additional_context)
            context += f"""
            Additional context:
            {additional_context_str}
            """

        return context

    def _build_action_context(self,
                             request: str,
                             analysis: Dict[str, Any],
                             additional_context: Optional[Dict[str, Any]] = None) -> str:
        """
        Build a context for an action request.

        Args:
            request: The user's request
            analysis: The analysis of the request
            additional_context: Additional context to include

        Returns:
            The constructed context as a string
        """
        # Get matched capabilities from analysis
        matched_capabilities = analysis.get("matched_capabilities", [])

        # Rank capabilities by relevance if there are multiple
        if len(matched_capabilities) > 1:
            logger.info(f"Ranking {len(matched_capabilities)} capabilities by relevance")
            ranked_capabilities = self._rank_capabilities_by_relevance(request, matched_capabilities)
        else:
            ranked_capabilities = matched_capabilities

        # Format capabilities for inclusion in context
        capabilities_str = format_capabilities(ranked_capabilities)

        # Get parameters
        parameters = analysis.get("parameters", {})
        parameters_str = safe_json_dumps(parameters) if parameters else "No specific parameters identified."

        # Get user preferences if available
        user_preferences_str = self._get_user_preferences()

        # Build the context
        context = f"""
        User request: {request}

        The user is asking you to perform an action. This is an action request.

        Available capabilities:
        {capabilities_str}

        Parameters:
        {parameters_str}
        """

        # Add user preferences if available
        if user_preferences_str:
            context += f"""
            User preferences:
            {user_preferences_str}
            """

        # Add additional context if available
        if additional_context:
            additional_context_str = safe_json_dumps(additional_context)
            context += f"""
            Additional context:
            {additional_context_str}
            """

        return context

    def _build_code_generation_context(self,
                                      request: str,
                                      analysis: Dict[str, Any],
                                      additional_context: Optional[Dict[str, Any]] = None) -> str:
        """
        Build a context for a code generation request.

        Args:
            request: The user's request
            analysis: The analysis of the request
            additional_context: Additional context to include

        Returns:
            The constructed context as a string
        """
        # Get matched capabilities from analysis
        matched_capabilities = analysis.get("matched_capabilities", [])

        # Rank capabilities by relevance if there are multiple
        if len(matched_capabilities) > 1:
            logger.info(f"Ranking {len(matched_capabilities)} capabilities by relevance")
            ranked_capabilities = self._rank_capabilities_by_relevance(request, matched_capabilities)
        else:
            ranked_capabilities = matched_capabilities

        # Format capabilities for inclusion in context
        capabilities_str = format_capabilities(ranked_capabilities)

        # Get parameters
        parameters = analysis.get("parameters", {})
        parameters_str = safe_json_dumps(parameters) if parameters else "No specific parameters identified."

        # Get user preferences if available
        user_preferences_str = self._get_user_preferences()

        # Build the context
        context = f"""
        User request: {request}

        The user is asking you to generate code. This is a code generation request.

        Available capabilities:
        {capabilities_str}

        Parameters:
        {parameters_str}
        """

        # Add user preferences if available
        if user_preferences_str:
            context += f"""
            User preferences:
            {user_preferences_str}
            """

        # Add additional context if available
        if additional_context:
            additional_context_str = safe_json_dumps(additional_context)
            context += f"""
            Additional context:
            {additional_context_str}
            """

        return context

    def _build_error_recovery_context(self,
                                     request: str,
                                     analysis: Dict[str, Any],
                                     additional_context: Optional[Dict[str, Any]] = None) -> str:
        """
        Build a context for error recovery.

        Args:
            request: The user's request
            analysis: The analysis of the request
            additional_context: Additional context to include

        Returns:
            The constructed context as a string
        """
        # Get error information
        error = additional_context.get("error", "Unknown error") if additional_context else "Unknown error"
        previous_attempts = additional_context.get("previous_attempts", []) if additional_context else []
        previous_attempts_str = safe_json_dumps(previous_attempts) if previous_attempts else "No previous attempts recorded."

        # Build the context
        context = f"""
        User request: {request}

        I encountered an error while trying to fulfill this request:
        {error}

        Previous attempts:
        {previous_attempts_str}
        """

        # Add additional context if provided
        if additional_context:
            # Filter out error and previous_attempts
            filtered_context = {k: v for k, v in additional_context.items() if k not in ["error", "previous_attempts"]}
            if filtered_context:
                additional_context_str = safe_json_dumps(filtered_context)
                context += f"""
                Additional context:
                {additional_context_str}
                """

        return context

    def _build_alternative_plan_context(self,
                                     request: str,
                                     analysis: Dict[str, Any],
                                     additional_context: Optional[Dict[str, Any]] = None) -> str:
        """
        Build a context for generating an alternative plan.

        Args:
            request: The user's request
            analysis: The analysis of the request
            additional_context: Additional context to include

        Returns:
            The constructed context as a string
        """
        # Get matched capabilities from analysis
        matched_capabilities = analysis.get("matched_capabilities", [])

        # Rank capabilities by relevance if there are multiple
        if len(matched_capabilities) > 1:
            logger.info(f"Ranking {len(matched_capabilities)} capabilities by relevance")
            ranked_capabilities = self._rank_capabilities_by_relevance(request, matched_capabilities)
        else:
            ranked_capabilities = matched_capabilities

        # Format capabilities for inclusion in context
        capabilities_str = format_capabilities(ranked_capabilities)

        # Get failed approaches
        failed_approaches = additional_context.get("failed_approaches", []) if additional_context else []
        failed_approaches_str = safe_json_dumps(failed_approaches) if failed_approaches else "No previous approaches recorded."

        # Get execution history
        execution_history = additional_context.get("execution_history", []) if additional_context else []
        execution_history_str = safe_json_dumps(execution_history) if execution_history else "No execution history recorded."

        # Build the context
        context = f"""
        User request: {request}

        I need to generate an alternative plan to fulfill this request.
        Previous approaches have failed.

        Available capabilities:
        {capabilities_str}

        Failed approaches:
        {failed_approaches_str}

        Execution history:
        {execution_history_str}
        """

        # Add additional context if provided
        if additional_context:
            # Filter out failed_approaches and execution_history
            filtered_context = {k: v for k, v in additional_context.items() if k not in ["failed_approaches", "execution_history"]}
            if filtered_context:
                additional_context_str = safe_json_dumps(filtered_context)
                context += f"""
                Additional context:
                {additional_context_str}
                """

        return context

    def _build_potential_answer_context(self,
                                   request: str,
                                   analysis: Dict[str, Any],
                                   additional_context: Optional[Dict[str, Any]] = None) -> str:
        """
        Build a context for a request with a high-confidence potential answer.

        Args:
            request: The user's request
            analysis: The analysis of the request, including the potential answer
            additional_context: Additional context to include

        Returns:
            The constructed context as a string
        """
        # Get the potential answer
        potential_answer = analysis.get("potential_answer", "")
        confidence = analysis.get("confidence", 0.0)

        # Build the context
        context = f"""
        User request: {request}

        This is a simple factual query with a high-confidence answer.

        Potential answer (confidence: {confidence:.2f}):
        {potential_answer}

        Please use this answer directly if it fully addresses the user's request.
        """

        # Add additional context if available
        if additional_context:
            additional_context_str = safe_json_dumps(additional_context)
            context += f"""
            Additional context:
            {additional_context_str}
            """

        return context

    def _get_user_preferences(self) -> str:
        """
        Get user preferences for inclusion in a context.

        This is a placeholder implementation that returns an empty string.
        In the future, this will be replaced with actual user preferences.

        Returns:
            A formatted string representation of user preferences
        """
        # This is a placeholder implementation
        # In the future, this will be replaced with actual user preferences
        return ""

    def _optimize_context_with_llm(self, context: str, request: str, analysis: Dict[str, Any], preferred_llm: str = "ollama") -> str:
        """
        Use LLM to optimize the context for better results.

        Args:
            context: The original context
            request: The user's request
            analysis: The analysis of the request
            preferred_llm: The preferred LLM to use ("ollama" or "cloud")

        Returns:
            The optimized context
        """
        logger.info(f"Optimizing context with LLM (preferred: {preferred_llm})")

        # Create a prompt for context optimization
        prompt = f"""
        You are an expert at creating optimal contexts for LLM interactions. Your task is to optimize the following context
        to make it more effective for answering the user's request.

        User request: {request}

        Original context:
        {context}

        Please optimize this context by:
        1. Removing any redundant or unnecessary information
        2. Highlighting the most important information
        3. Structuring the context in a way that makes it easier for an LLM to understand
        4. Ensuring all critical information is preserved

        Return ONLY the optimized context, without any explanations or additional text.
        """

        try:
            # Use the appropriate LLM based on preference
            if preferred_llm == "cloud" and hasattr(self.llm_wrapper, 'generate_chat_completion'):
                # Use cloud LLM for optimization
                logger.info("Using cloud LLM for context optimization")
                messages = [
                    {"role": "system", "content": "You are an expert at optimizing contexts for LLM interactions."},
                    {"role": "user", "content": prompt}
                ]
                response = self.llm_wrapper.generate_chat_completion(messages, temperature=0.3)
                optimized_context = response.content
            else:
                # Use local LLM for optimization
                logger.info("Using local LLM for context optimization")
                if hasattr(self.llm_wrapper, 'generate_text'):
                    response = self.llm_wrapper.generate_text(prompt, temperature=0.3)
                    optimized_context = response.content
                else:
                    # Fallback to cloud if local is not available
                    logger.warning("Local LLM not available, falling back to cloud LLM")
                    messages = [
                        {"role": "system", "content": "You are an expert at optimizing contexts for LLM interactions."},
                        {"role": "user", "content": prompt}
                    ]
                    response = self.llm_wrapper.generate_chat_completion(messages, temperature=0.3)
                    optimized_context = response.content

            # Check if optimization was successful
            if optimized_context and len(optimized_context.strip()) > 0:
                logger.info(f"Context optimization successful: {len(context)} chars -> {len(optimized_context)} chars")
                return optimized_context
            else:
                logger.warning("Context optimization returned empty result, using original context")
                return context
        except Exception as e:
            logger.error(f"Error optimizing context with LLM: {e}")
            return context  # Return original context on error

    def _rank_capabilities_by_relevance(self, request: str, capabilities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Use LLM to rank capabilities by relevance to the request.

        Args:
            request: The user's request
            capabilities: List of capabilities to rank

        Returns:
            Ranked list of capabilities
        """
        if not capabilities or len(capabilities) <= 1 or not self.llm_wrapper:
            return capabilities

        logger.info(f"Ranking {len(capabilities)} capabilities by relevance")

        # Create a prompt for capability ranking
        capabilities_str = safe_json_dumps(capabilities)
        prompt = f"""
        You are an expert at matching user requests to capabilities. Your task is to rank the following capabilities
        by their relevance to the user's request.

        User request: {request}

        Available capabilities:
        {capabilities_str}

        Please return a JSON array of capability names, ordered from most relevant to least relevant.
        Format: ["capability_name1", "capability_name2", ...]
        """

        try:
            # Use local LLM for ranking (this is a simple task)
            if hasattr(self.llm_wrapper, 'generate_text'):
                response = self.llm_wrapper.generate_text(prompt, temperature=0.3)
                response_text = response.content
            else:
                # Fallback to cloud LLM
                messages = [
                    {"role": "system", "content": "You are an expert at matching user requests to capabilities."},
                    {"role": "user", "content": prompt}
                ]
                response = self.llm_wrapper.generate_chat_completion(messages, temperature=0.3)
                response_text = response.content

            # Extract JSON array from response
            import re
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if json_match:
                ranked_names = json.loads(json_match.group(0))

                # Create a dictionary mapping capability names to capabilities
                capability_map = {cap.get("name", ""): cap for cap in capabilities}

                # Create the ranked list
                ranked_capabilities = []
                for name in ranked_names:
                    if name in capability_map:
                        ranked_capabilities.append(capability_map[name])

                # Add any capabilities that weren't ranked
                for cap in capabilities:
                    name = cap.get("name", "")
                    if name not in ranked_names:
                        ranked_capabilities.append(cap)

                logger.info(f"Successfully ranked capabilities: {', '.join(ranked_names[:3])}...")
                return ranked_capabilities
            else:
                logger.warning("Failed to extract ranked capabilities from LLM response")
                return capabilities
        except Exception as e:
            logger.error(f"Error ranking capabilities: {e}")
            return capabilities  # Return original capabilities on error