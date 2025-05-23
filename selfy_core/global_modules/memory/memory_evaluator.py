"""
Memory Evaluator for the Selfy agent.

This module provides the MemoryEvaluator class, which is responsible for
evaluating the value of memories for long-term storage.
"""

import logging
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from selfy_core.global_modules.config import get as config_get
from selfy_core.global_modules.memory.interfaces import MemoryItem, MemoryItemType
from selfy_core.global_modules.llm_wrapper.base_llm_provider import BaseLLMProvider

logger = logging.getLogger(__name__)

class MemoryEvaluator:
    """
    Evaluates the value of memories for long-term storage.

    The MemoryEvaluator is responsible for:
    1. Assessing the value of conversation turns for long-term storage
    2. Summarizing valuable information for efficient storage
    3. Determining the appropriate memory type for storage

    This class uses an LLM (Ollama by default) to evaluate conversation content
    against four key parameters:
    - Future tasks: Information that could be useful for future tasks or goals
    - Reasoning: Information that demonstrates important reasoning patterns
    - Context retrieval: Information that provides important context for future interactions
    - Self-improvement: Information that could help the agent improve its capabilities

    Only information that scores above the configured threshold (default: 0.6)
    will be stored in long-term memory, reducing storage requirements and
    improving retrieval relevance.
    """

    def __init__(self, llm_provider: Optional[BaseLLMProvider] = None):
        """
        Initialize the memory evaluator.

        Args:
            llm_provider: The LLM provider to use for evaluation
        """
        logger.info("Initializing MemoryEvaluator")

        self.llm_provider = llm_provider

        # Load configuration
        self.min_score_threshold = config_get('memory.evaluator.min_score_threshold', 0.6)
        self.enable_evaluation = config_get('memory.evaluator.enable', True)

        logger.info(f"MemoryEvaluator initialized with min_score_threshold={self.min_score_threshold}")

    def evaluate_conversation(self, conversation_turns: List[MemoryItem]) -> List[MemoryItem]:
        """
        Evaluate a list of conversation turns for long-term storage.

        This method analyzes a conversation to identify valuable information worth
        storing in long-term memory. It uses the configured LLM (Ollama by default)
        to evaluate the conversation against four key parameters:

        1. Future tasks: Information that could be useful for future tasks or goals
        2. Reasoning: Information that demonstrates important reasoning patterns
        3. Context retrieval: Information that provides important context for future interactions
        4. Self-improvement: Information that could help the agent improve its capabilities

        The LLM assigns a score to each piece of valuable information, and only
        information scoring above the configured threshold (default: 0.6) will be
        included in the returned list of memory items.

        Each returned memory item contains:
        - A concise summary of the valuable information (1-2 sentences)
        - The appropriate memory type (LEARNED_FACT, USER_PREFERENCE, etc.)
        - Metadata including the score and which criteria it satisfied

        Args:
            conversation_turns: List of conversation turns to evaluate

        Returns:
            List of memory items to store long-term, or an empty list if no valuable
            information was found or if evaluation is disabled/failed
        """
        if not self.enable_evaluation:
            logger.info("Memory evaluation is disabled, storing all conversation turns")
            return conversation_turns

        if not self.llm_provider:
            logger.warning("No LLM provider available for memory evaluation, storing all conversation turns")
            return conversation_turns

        if not conversation_turns:
            logger.debug("No conversation turns to evaluate")
            return []

        logger.info(f"Evaluating {len(conversation_turns)} conversation turns for long-term storage")

        # Format conversation for evaluation
        conversation_text = self._format_conversation(conversation_turns)

        # Evaluate the conversation
        evaluation_result = self._evaluate_with_llm(conversation_text)

        if not evaluation_result:
            logger.warning("Failed to evaluate conversation, storing all turns")
            return conversation_turns

        # Create memory items from the evaluation result
        memory_items = self._create_memory_items(evaluation_result, conversation_turns)

        logger.info(f"Created {len(memory_items)} memory items from evaluation")
        return memory_items

    def _format_conversation(self, conversation_turns: List[MemoryItem]) -> str:
        """
        Format a list of conversation turns for evaluation.

        Args:
            conversation_turns: List of conversation turns

        Returns:
            Formatted conversation text
        """
        formatted_turns = []

        for turn in conversation_turns:
            role = turn.metadata.get('role', 'unknown')
            content = turn.content

            if role == 'user':
                formatted_turns.append(f"User: {content}")
            elif role == 'assistant':
                formatted_turns.append(f"Assistant: {content}")
            elif role == 'system':
                formatted_turns.append(f"System: {content}")
            else:
                formatted_turns.append(f"{role.capitalize()}: {content}")

        return "\n\n".join(formatted_turns)

    def _evaluate_with_llm(self, conversation_text: str) -> Optional[Dict[str, Any]]:
        """
        Evaluate a conversation using the LLM provider.

        Args:
            conversation_text: Formatted conversation text

        Returns:
            Evaluation result as a dictionary, or None if evaluation failed
        """
        prompt = f"""
        You are an expert at evaluating conversations for long-term memory storage.

        Please analyze the following conversation and identify information that would be valuable to store in long-term memory.

        Conversation:
        {conversation_text}

        Evaluate the conversation based on these criteria:
        1. Future Tasks: Information that could be useful for future tasks or goals
        2. Reasoning: Information that demonstrates important reasoning patterns or problem-solving approaches
        3. Context Retrieval: Information that provides important context for future interactions
        4. Self-Improvement: Information that could help the agent improve its capabilities or performance

        For each valuable piece of information you identify, provide:
        1. A concise summary (1-2 sentences)
        2. The type of information (LEARNED_FACT, USER_PREFERENCE, etc.)
        3. A score from 0.0 to 1.0 indicating how valuable this information is
        4. Which criteria it satisfies (can be multiple)

        Return your analysis as a JSON object with this structure:
        {{
            "valuable_information": [
                {{
                    "summary": "Concise summary of the information",
                    "type": "LEARNED_FACT",
                    "score": 0.85,
                    "criteria": ["Future Tasks", "Context Retrieval"]
                }},
                ...
            ]
        }}

        Only include information with a score of {self.min_score_threshold} or higher.
        If there is no valuable information, return an empty list.
        """

        try:
            # Use the LLM provider to evaluate the conversation
            response = self.llm_provider.generate_text(prompt, temperature=0.3)

            # Extract JSON from the response
            result_text = response.content

            # Try to parse the JSON
            try:
                # Find JSON in the response
                json_start = result_text.find('{')
                json_end = result_text.rfind('}') + 1

                if json_start >= 0 and json_end > json_start:
                    json_str = result_text[json_start:json_end]
                    result = json.loads(json_str)

                    # Validate the result structure
                    if 'valuable_information' not in result:
                        logger.warning("Invalid evaluation result: missing 'valuable_information' key")
                        return None

                    return result
                else:
                    logger.warning("No JSON found in evaluation response")
                    return None
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse evaluation result as JSON: {e}")
                return None

        except Exception as e:
            logger.error(f"Error evaluating conversation with LLM: {e}")
            return None

    def _create_memory_items(self, evaluation_result: Dict[str, Any],
                           original_turns: List[MemoryItem]) -> List[MemoryItem]:
        """
        Create memory items from the evaluation result.

        Args:
            evaluation_result: Evaluation result from the LLM
            original_turns: Original conversation turns

        Returns:
            List of memory items to store long-term
        """
        memory_items = []

        # Get session and user IDs from original turns
        session_id = None
        user_id = None

        for turn in original_turns:
            if 'session_id' in turn.metadata and not session_id:
                session_id = turn.metadata['session_id']
            if 'user_id' in turn.metadata and not user_id:
                user_id = turn.metadata['user_id']

        # Process valuable information
        for info in evaluation_result.get('valuable_information', []):
            summary = info.get('summary', '')
            memory_type_str = info.get('type', 'LEARNED_FACT')
            score = info.get('score', 0.0)
            criteria = info.get('criteria', [])

            # Skip if below threshold
            if score < self.min_score_threshold:
                continue

            # Convert string type to enum
            try:
                memory_type = getattr(MemoryItemType, memory_type_str)
            except AttributeError:
                logger.warning(f"Invalid memory type: {memory_type_str}, using LEARNED_FACT")
                memory_type = MemoryItemType.LEARNED_FACT

            # Create a new memory item
            item = MemoryItem(
                id=None,  # Will be assigned when added to memory
                timestamp=datetime.now(),
                type=memory_type,
                content=summary,
                metadata={
                    'session_id': session_id,
                    'user_id': user_id,
                    'score': score,
                    'criteria': criteria,
                    'source': 'memory_evaluator'
                }
            )

            memory_items.append(item)

        return memory_items
