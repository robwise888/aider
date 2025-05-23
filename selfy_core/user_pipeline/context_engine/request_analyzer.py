"""
Request Analyzer for the Context Engine.

This module provides the RequestAnalyzer class, which is responsible for
analyzing user requests, matching them to capabilities, and identifying required knowledge domains.
"""

import logging
import time
import json
from typing import Dict, List, Any, Optional

# Import config_get at the module level for use in the class
from selfy_core.global_modules.config import get as config_get
from selfy_core.user_pipeline.context_engine.utils.llm_utils import extract_json_from_response, log_llm_call
from selfy_core.global_modules.capability_manifest import json_dumps  # Import custom JSON encoder

logger = logging.getLogger(__name__)

class RequestAnalyzer:
    """
    Analyzes user requests and matches them to capabilities.

    The RequestAnalyzer is responsible for:
    1. Analyzing user requests to determine intent and parameters
    2. Matching requests to available capabilities
    3. Identifying required knowledge domains
    4. Providing analysis results for context building and execution planning
    """

    def __init__(self, llm_wrapper=None, capability_manifest=None, identity_filter=None):
        """
        Initialize the request analyzer.

        Args:
            llm_wrapper: The LLM wrapper to use for analysis
            capability_manifest: The capability manifest to use for capability matching
            identity_filter: The identity filter to use for filtering prompts and responses
        """
        logger.info("Initializing RequestAnalyzer")

        self.llm_wrapper = llm_wrapper
        self.capability_manifest = capability_manifest
        self.identity_filter = identity_filter

        # Configuration from config file
        from selfy_core.global_modules.config import get as config_get
        self.confidence_threshold = config_get('context_engine.request_analyzer.confidence_threshold', 0.95)
        self.skip_final_analysis_threshold = config_get('context_engine.request_analyzer.skip_final_analysis_threshold', 0.98)
        self.max_retries = config_get('context_engine.request_analyzer.max_retries', 2)
        self.retry_delay = config_get('context_engine.request_analyzer.retry_delay_seconds', 1.0)

        logger.info("RequestAnalyzer initialized successfully")

    def analyze_request(self, request: str, conversation_history: Optional[List[Dict[str, Any]]] = None,
                       metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze a user request.

        Args:
            request: The user's request
            conversation_history: The conversation history
            metadata: Additional metadata

        Returns:
            A dictionary containing the analysis results including required capabilities and knowledge domains
        """
        start_time = time.time()
        logger.info(f"Analyzing request: {request[:50]}...")
        logger.debug(f"CONTEXT ENGINE FLOW - REQUEST ANALYZER INPUT: {request}")

        # Initialize analysis
        analysis = {
            "request": request,
            "request_type": "unknown",
            "request_description": "",
            "parameters": {},
            "matched_capabilities": [],
            "available_capabilities": [],
            "requires_capabilities": False,  # Flag to indicate if capabilities are needed
            "required_knowledge_domains": [],  # New field for knowledge domains
            "confidence": 0.0,
            "analysis_time": 0.0
        }

        logger.debug(f"CONTEXT ENGINE FLOW - REQUEST ANALYZER INITIALIZED: {json_dumps(analysis, indent=2)}")

        # Apply identity filter to the request if available
        filtered_request = request
        if self.identity_filter:
            filter_result = self.identity_filter.filter_input(request)
            if filter_result.status == 'allowed':
                filtered_request = filter_result.output_text
            else:
                logger.warning(f"Request blocked by identity filter: {filter_result.reason}")
                analysis["request_type"] = "blocked"
                analysis["request_description"] = filter_result.reason
                analysis["analysis_time"] = time.time() - start_time
                return analysis

        # Perform two-stage analysis
        try:
            # Stage 1: Use local LLM for preprocessing
            preprocessing_result = self._preprocess_request(filtered_request)

            # Log preprocessing result
            if preprocessing_result:
                logger.debug(f"CONTEXT ENGINE FLOW - PREPROCESSING RESULT: {json_dumps(preprocessing_result, indent=2)}")
            else:
                logger.debug("CONTEXT ENGINE FLOW - PREPROCESSING RESULT: None")

            # Check if there was an error in preprocessing
            preprocessing_error = "error" in preprocessing_result if preprocessing_result else False

            # Check if preprocessing encountered an error
            if preprocessing_error:
                logger.info(f"Preprocessing encountered an error, using cloud LLM: {preprocessing_result.get('error', 'Unknown error')}")
                analysis_result = self._analyze_with_cloud_llm(filtered_request, conversation_history)
            # Check if preprocessing result has high confidence to skip final analysis
            elif preprocessing_result and preprocessing_result.get("confidence", 0.0) >= self.skip_final_analysis_threshold:
                # Skip final analysis for high confidence results
                logger.info(f"Skipping final analysis (confidence {preprocessing_result.get('confidence', 0.0):.2f} above threshold {self.skip_final_analysis_threshold:.2f})")
                analysis_result = preprocessing_result

                # Add potential answer if available in preprocessing result
                if "potential_answer" in preprocessing_result:
                    analysis_result["potential_answer"] = preprocessing_result["potential_answer"]
            # Stage 2: Determine whether to use local or cloud LLM for final analysis
            elif preprocessing_result and preprocessing_result.get("confidence", 0.0) >= self.confidence_threshold:
                # Use local LLM for final analysis
                logger.info("Using local LLM for final analysis (confidence above threshold)")
                analysis_result = self._analyze_with_local_llm(filtered_request, preprocessing_result, conversation_history)
            else:
                # Use cloud LLM for final analysis
                logger.info("Using cloud LLM for final analysis (confidence below threshold or preprocessing failed)")
                analysis_result = self._analyze_with_cloud_llm(filtered_request, conversation_history)

            # Update analysis with results
            if analysis_result:
                analysis.update(analysis_result)

                # Extract knowledge domains if present in the analysis result
                if "required_knowledge_domains" in analysis_result:
                    analysis["required_knowledge_domains"] = analysis_result["required_knowledge_domains"]
                    logger.info(f"Knowledge domains identified: {analysis['required_knowledge_domains']}")

            # Add preferred LLM based on confidence and analysis
            # Get provider names from config
            from selfy_core.global_modules.config import get as config_get
            cloud_provider = config_get('llm.cloud_provider', "groq")
            local_provider = config_get('llm.local_provider', "ollama")

            if analysis.get("confidence", 0.0) < self.confidence_threshold:
                # If confidence is below threshold, prefer cloud LLM
                analysis["preferred_llm"] = cloud_provider
                logger.info(f"Setting preferred LLM to '{cloud_provider}' due to low confidence: {analysis.get('confidence', 0.0):.2f}")
            else:
                # Otherwise, prefer local LLM
                analysis["preferred_llm"] = local_provider
                logger.info(f"Setting preferred LLM to '{local_provider}' due to high confidence: {analysis.get('confidence', 0.0):.2f}")

            # Match capabilities only if needed
            if self.capability_manifest:
                matched_capabilities = self._match_capabilities(analysis)
                analysis["matched_capabilities"] = matched_capabilities

                # Only populate available_capabilities if capabilities are required
                if analysis.get("requires_capabilities", False):
                    analysis["available_capabilities"] = self._get_available_capabilities(matched_capabilities)
                else:
                    # For general queries, don't waste memory with capabilities list
                    analysis["available_capabilities"] = []
                    logger.info("Skipping capability retrieval for general query")
        except Exception as e:
            logger.error(f"Error analyzing request: {e}")
            analysis["error"] = str(e)

        # Record analysis time
        analysis["analysis_time"] = time.time() - start_time

        logger.info(f"Request analysis completed in {analysis['analysis_time']:.2f}s")

        # Log the final analysis result
        logger.debug(f"CONTEXT ENGINE FLOW - REQUEST ANALYZER OUTPUT: {json_dumps(analysis, indent=2)}")

        return analysis

    def _preprocess_request(self, request: str) -> Optional[Dict[str, Any]]:
        """
        Preprocess a request using the local LLM.

        Args:
            request: The user's request

        Returns:
            A dictionary containing the preprocessing results, or None if preprocessing failed
        """
        logger.info(f"Preprocessing request: {request[:50]}...")

        # Create a prompt for preprocessing
        prompt = self._create_preprocessing_prompt(request)

        # Apply identity filter if available
        if self.identity_filter:
            prompt = self.identity_filter.filter_prompt(prompt)

        # Get response from local LLM
        try:
            if hasattr(self.llm_wrapper, 'generate_text'):
                # Local provider
                response = self.llm_wrapper.generate_text(prompt, temperature=0.3)
                response_text = response.content
            else:
                # Fallback to cloud provider
                messages = [
                    {"role": "system", "content": "You are a request analyzer."},
                    {"role": "user", "content": prompt}
                ]
                response = self.llm_wrapper.generate_chat_completion(messages, temperature=0.3)
                response_text = response.content

            # Extract JSON from response
            result = extract_json_from_response(response_text)

            # Ensure required fields
            if not result:
                result = {}

            if "request_type" not in result:
                result["request_type"] = "unknown"

            if "confidence" not in result:
                result["confidence"] = 0.5

            return result
        except Exception as e:
            logger.error(f"Error preprocessing request: {e}")
            return None

    def _create_preprocessing_prompt(self, request: str) -> str:
        """
        Create a prompt for preprocessing a request.

        Args:
            request: The user's request

        Returns:
            A prompt for preprocessing
        """
        prompt = f"""
        You are a request analyzer. Your task is to analyze a user request and determine its type and parameters.

        User request: "{request}"

        Please analyze the request and return a JSON object with the following fields:
        - request_type: The type of request (general_query, capability_query, code_generation, error_recovery, clarification, unknown)
        - confidence: A number between 0 and 1 indicating your confidence in the analysis
        - request_description: A brief description of the request
        - parameters: Any parameters extracted from the request
        - capability_categories: A list of capability categories that might be relevant to this request (e.g., "mathematical", "statistical", "text_processing", "file_operations")

        Example:
        {{
            "request_type": "general_query",
            "confidence": 0.8,
            "request_description": "User is asking about the weather",
            "parameters": {{
                "location": "New York"
            }},
            "capability_categories": ["weather_information"]
        }}
        """
        return prompt

    def _analyze_with_local_llm(self, request: str, preprocessing_result: Dict[str, Any],
                              conversation_history: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Analyze a request using the local LLM.

        Args:
            request: The user's request
            preprocessing_result: The preprocessing results
            conversation_history: The conversation history

        Returns:
            A dictionary containing the analysis results
        """
        logger.info(f"Analyzing request with local LLM: {request[:50]}...")

        # Create a prompt for analysis
        prompt = self._create_analysis_prompt(request, preprocessing_result, conversation_history)

        # Apply identity filter if available
        if self.identity_filter:
            prompt = self.identity_filter.filter_prompt(prompt)

        # Get response from local LLM
        try:
            if hasattr(self.llm_wrapper, 'generate_text'):
                # Local provider
                response = self.llm_wrapper.generate_text(prompt, temperature=0.3)
                response_text = response.content
            else:
                # Fallback to cloud provider
                messages = [
                    {"role": "system", "content": "You are a request analyzer."},
                    {"role": "user", "content": prompt}
                ]
                response = self.llm_wrapper.generate_chat_completion(messages, temperature=0.3)
                response_text = response.content

            # Extract JSON from response
            result = extract_json_from_response(response_text)

            # Ensure required fields
            if not result:
                result = preprocessing_result

            if "request_type" not in result:
                result["request_type"] = preprocessing_result.get("request_type", "unknown")

            if "confidence" not in result:
                result["confidence"] = preprocessing_result.get("confidence", 0.5)

            return result
        except Exception as e:
            logger.error(f"Error analyzing request with local LLM: {e}")
            return preprocessing_result

    def _analyze_with_cloud_llm(self, request: str,
                              conversation_history: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Analyze a request using the cloud LLM.

        Args:
            request: The user's request
            conversation_history: The conversation history

        Returns:
            A dictionary containing the analysis results
        """
        logger.info(f"Analyzing request with cloud LLM: {request[:50]}...")

        # Create a prompt for analysis
        prompt = self._create_detailed_analysis_prompt(request, conversation_history)

        # Apply identity filter if available
        if self.identity_filter:
            prompt = self.identity_filter.filter_prompt(prompt)

        # Get response from cloud LLM
        try:
            if hasattr(self.llm_wrapper, 'generate_chat_completion'):
                # Cloud provider
                messages = [
                    {"role": "system", "content": "You are a request analyzer."},
                    {"role": "user", "content": prompt}
                ]
                response = self.llm_wrapper.generate_chat_completion(messages, temperature=0.3)
                response_text = response.content
            else:
                # Fallback to local provider
                response = self.llm_wrapper.generate_text(prompt, temperature=0.3)
                response_text = response.content

            # Extract JSON from response
            result = extract_json_from_response(response_text)

            # Ensure required fields
            if not result:
                result = {}

            if "request_type" not in result:
                result["request_type"] = "unknown"

            if "confidence" not in result:
                result["confidence"] = 0.7

            return result
        except Exception as e:
            logger.error(f"Error analyzing request with cloud LLM: {e}")
            return {
                "request_type": "unknown",
                "confidence": 0.0,
                "error": str(e)
            }

    def _create_analysis_prompt(self, request: str, preprocessing_result: Dict[str, Any],
                              conversation_history: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Create a prompt for analyzing a request.

        Args:
            request: The user's request
            preprocessing_result: The preprocessing results
            conversation_history: The conversation history

        Returns:
            A prompt for analysis
        """
        # Format conversation history
        history_str = ""
        if conversation_history:
            history_str = "Conversation history:\n"
            for i, turn in enumerate(conversation_history[-5:]):  # Last 5 turns
                content = turn.get("content", "")
                role = turn.get("role", "unknown")
                history_str += f"{role.capitalize()}: {content}\n"

        # Format preprocessing results
        preprocessing_str = json_dumps(preprocessing_result, indent=2)

        prompt = f"""
        You are a request analyzer. Your task is to analyze a user request and determine its type, parameters, and required capability categories.

        {history_str}

        User request: "{request}"

        Preprocessing results:
        {preprocessing_str}

        Please analyze the request and return a JSON object with the following fields:
        - request_type: The type of request (general_query, capability_query, code_generation, error_recovery, clarification, unknown)
        - confidence: A number between 0 and 1 indicating your confidence in the analysis
        - request_description: A brief description of the request
        - parameters: Any parameters extracted from the request
        - capability_categories: A list of capability categories that might be relevant to this request
        - required_knowledge_domains: A list of knowledge domains required to fulfill the request

        Return your response as a JSON object.
        """
        return prompt

    def _create_detailed_analysis_prompt(self, request: str,
                                       conversation_history: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Create a detailed prompt for analyzing a request.

        Args:
            request: The user's request
            conversation_history: The conversation history

        Returns:
            A prompt for analysis
        """
        # Format conversation history
        history_str = ""
        if conversation_history:
            history_str = "Conversation history:\n"
            for i, turn in enumerate(conversation_history[-5:]):  # Last 5 turns
                content = turn.get("content", "")
                role = turn.get("role", "unknown")
                history_str += f"{role.capitalize()}: {content}\n"

        # Get available capability categories
        categories_str = "Available capability categories:\n"
        if self.capability_manifest:
            categories = self.capability_manifest.get_all_categories()
            for category in categories:
                categories_str += f"- {category}\n"
        else:
            categories_str += "No capability categories available."

        prompt = f"""
        You are a request analyzer. Your task is to analyze a user request and determine its type, parameters, and required capability categories.

        {history_str}

        {categories_str}

        User request: "{request}"

        Please analyze the request and return a JSON object with the following fields:
        - request_type: The type of request (general_query, capability_query, code_generation, error_recovery, clarification, unknown)
        - confidence: A number between 0 and 1 indicating your confidence in the analysis
        - request_description: A brief description of the request
        - parameters: Any parameters extracted from the request
        - capability_categories: A list of capability categories that might be relevant to this request
        - required_knowledge_domains: A list of knowledge domains required to fulfill the request

        Return your response as a JSON object.
        """
        return prompt

    def _match_capabilities(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Match capability categories to available capabilities.

        Args:
            analysis: The analysis results

        Returns:
            A list of matched capabilities
        """
        if not self.capability_manifest:
            return []

        # Get capability categories from analysis
        capability_categories = analysis.get("capability_categories", [])

        # If no categories specified, try to infer from request type
        if not capability_categories:
            request_type = analysis.get("request_type", "unknown")

            if request_type == "general_query":
                capability_categories = ["general_knowledge"]
            elif request_type == "code_generation":
                capability_categories = ["code_generation"]
            elif request_type == "capability_query":
                capability_categories = ["capability_info"]

        # Get capabilities for each category
        matched_capabilities = []
        for category in capability_categories:
            # Get capabilities for this category
            category_capabilities = self.capability_manifest.find_capabilities_by_category(category)
            matched_capabilities.extend(category_capabilities)

            # If no capabilities found for this category, try to find similar capabilities
            if not category_capabilities:
                similar_capabilities = self.capability_manifest.find_capabilities(category)
                matched_capabilities.extend(similar_capabilities)

        # Remove duplicates while preserving order
        seen = set()
        unique_capabilities = []
        for cap in matched_capabilities:
            if cap.name not in seen:
                seen.add(cap.name)
                unique_capabilities.append(cap)

        return unique_capabilities

    def _get_available_capabilities(self, matched_capabilities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Get available capabilities based on matched capabilities.

        Args:
            matched_capabilities: The matched capabilities

        Returns:
            A list of available capabilities
        """
        if not self.capability_manifest:
            return []

        # If no matched capabilities, return all capabilities
        if not matched_capabilities:
            return self.capability_manifest.get_all_capabilities()

        # Otherwise, return matched capabilities
        return matched_capabilities