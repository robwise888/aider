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

        logger.debug(f"CONTEXT ENGINE FLOW - REQUEST ANALYZER INITIALIZED: {json.dumps(analysis, indent=2)}")

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
                logger.debug(f"CONTEXT ENGINE FLOW - PREPROCESSING RESULT: {json.dumps(preprocessing_result, indent=2)}")
            else:
                logger.debug("CONTEXT ENGINE FLOW - PREPROCESSING RESULT: None")

            # Check if there was an error in preprocessing
            preprocessing_error = "error" in preprocessing_result if preprocessing_result else False

            # Check for specific analysis keywords that might benefit from cloud LLM
            if "analyze" in filtered_request.lower():
                logger.info("Request contains 'analyze', using cloud LLM for analysis")
                analysis_result = self._analyze_with_cloud_llm(filtered_request, conversation_history)
            elif preprocessing_error:
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

            if "analyze" in filtered_request.lower():
                # If request contains "analyze", prefer cloud LLM
                analysis["preferred_llm"] = cloud_provider
                logger.info(f"Setting preferred LLM to '{cloud_provider}' due to 'analyze' keyword")
            elif analysis.get("confidence", 0.0) < self.confidence_threshold:
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
        logger.debug(f"CONTEXT ENGINE FLOW - REQUEST ANALYZER OUTPUT: {json.dumps(analysis, indent=2)}")

        return analysis
