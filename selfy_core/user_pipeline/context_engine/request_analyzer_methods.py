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
                # Handle both dictionary and object formats safely
                if isinstance(turn, dict):
                    content = turn.get("content", "")
                    role = turn.get("role", "unknown")
                else:
                    # Assume it's a MemoryItem or similar object
                    try:
                        content = getattr(turn, "content", "")
                        role = getattr(turn, "role", "unknown")
                    except Exception as e:
                        logger.warning(f"Error accessing conversation history item attributes: {e}")
                        content = str(turn)
                        role = "unknown"

                history_str += f"{role.capitalize()}: {content}\n"

        # Format preprocessing results
        preprocessing_str = json.dumps(preprocessing_result, indent=2)

        prompt = f"""
        You are a request analyzer. Your task is to analyze a user request and determine its type, parameters, required capabilities, and knowledge domains.

        {history_str}

        User request: "{request}"

        Preprocessing results:
        {preprocessing_str}

        IMPORTANT DEFINITIONS:
        
        1. Capabilities are FUNCTIONAL TOOLS or ACTIONS that the system can perform:
        Examples of valid capabilities:
        - "file_operations" (a tool that can read/write files)
        - "web_search" (a tool that can search the web)
        - "code_execution" (a tool that can run code)
        - "memory_management" (a tool that can store/retrieve memories)
        
        2. Knowledge Domains are SUBJECT AREAS or EXPERTISE that help answer the query:
        Examples of knowledge domains:
        - "domain_expertise_quantum_computing" (knowledge about quantum computing)
        - "domain_expertise_AI_ethics" (knowledge about AI ethics)
        - "strategic_planning" (knowledge about strategic planning)
        - "cross_disciplinary_synthesis" (ability to connect multiple fields)
        
        You must identify BOTH:
        - The functional capabilities (tools/actions) needed to fulfill the request
        - The knowledge domains required to properly answer the query

        Please analyze the request and return a JSON object with the following fields:
        - request_type: The type of request (general_query, capability_query, code_generation, error_recovery, clarification, unknown)
        - confidence: A number between 0 and 1 indicating your confidence in the analysis
        - request_description: A brief description of the request
        - parameters: Any parameters extracted from the request
        - required_capabilities: A list of FUNCTIONAL capabilities required to fulfill the request (tools/actions, not knowledge domains)
        - required_knowledge_domains: A list of knowledge domains or subject expertise needed to properly answer the query

        IMPORTANT: Wrap your JSON response ONLY in ```json and ``` delimiters.
        Do not include any text before or after these delimiters.

        Example of correct response format:
        ```json
        {{
            "request_type": "code_generation",
            "confidence": 0.8,
            "request_description": "User is asking for help with Python code",
            "parameters": {{
                "language": "Python",
                "task": "file handling"
            }},
            "required_capabilities": ["code_generation", "python_knowledge"],
            "required_knowledge_domains": ["domain_expertise_python", "domain_expertise_file_systems"]
        }}
        ```
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
                # Handle both dictionary and object formats safely
                if isinstance(turn, dict):
                    content = turn.get("content", "")
                    role = turn.get("role", "unknown")
                else:
                    # Assume it's a MemoryItem or similar object
                    try:
                        content = getattr(turn, "content", "")
                        role = getattr(turn, "role", "unknown")
                    except Exception as e:
                        logger.warning(f"Error accessing conversation history item attributes: {e}")
                        content = str(turn)
                        role = "unknown"

                history_str += f"{role.capitalize()}: {content}\n"

        # Get available capabilities
        capabilities_str = "Available capabilities:\n"
        if self.capability_manifest:
            capabilities = self.capability_manifest.get_all_capabilities()
            for capability in capabilities[:10]:  # Limit to 10 capabilities
                name = capability.name if hasattr(capability, 'name') else "Unknown"
                description = capability.description if hasattr(capability, 'description') else "No description"
                capabilities_str += f"- {name}: {description}\n"
        else:
            capabilities_str += "No capabilities available."

        prompt = f"""
        You are a request analyzer. Your task is to analyze a user request and determine its type, parameters, required capabilities, and knowledge domains.

        {history_str}

        {capabilities_str}

        User request: "{request}"

        IMPORTANT DEFINITIONS:
        
        1. Capabilities are FUNCTIONAL TOOLS or ACTIONS that the system can perform:
        Examples of valid capabilities:
        - "file_operations" (a tool that can read/write files)
        - "web_search" (a tool that can search the web)
        - "code_execution" (a tool that can run code)
        - "memory_management" (a tool that can store/retrieve memories)
        
        2. Knowledge Domains are SUBJECT AREAS or EXPERTISE that help answer the query:
        Examples of knowledge domains:
        - "domain_expertise_quantum_computing" (knowledge about quantum computing)
        - "domain_expertise_AI_ethics" (knowledge about AI ethics)
        - "strategic_planning" (knowledge about strategic planning)
        - "cross_disciplinary_synthesis" (ability to connect multiple fields)
        
        You must identify BOTH:
        - The functional capabilities (tools/actions) needed to fulfill the request
        - The knowledge domains required to properly answer the query

        Please analyze the request and return a JSON object with the following fields:
        - request_type: The type of request (general_query, capability_query, code_generation, error_recovery, clarification, unknown)
        - confidence: A number between 0 and 1 indicating your confidence in the analysis
        - request_description: A brief description of the request
        - parameters: Any parameters extracted from the request
        - required_capabilities: A list of FUNCTIONAL capabilities required to fulfill the request (tools/actions, not knowledge domains)
        - required_knowledge_domains: A list of knowledge domains or subject expertise needed to properly answer the query

        IMPORTANT: Wrap your JSON response ONLY in ```json and ``` delimiters.
        Do not include any text before or after these delimiters.

        Example of correct response format:
        ```json
        {{
            "request_type": "general_query",
            "confidence": 0.9,
            "request_description": "User is asking about quantum computing and AI",
            "parameters": {{
                "topic": "quantum computing and AI convergence"
            }},
            "required_capabilities": ["web_search", "knowledge_aggregation"],
            "required_knowledge_domains": ["domain_expertise_quantum_computing", "domain_expertise_AI", "cross_disciplinary_synthesis"]
        }}
        ```
        """
        return prompt
