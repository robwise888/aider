"""
Memory Consolidation Service for the Selfy agent.

This module provides the MemoryConsolidationService class, which is responsible for
consolidating short-term memory into long-term memory.
"""

import logging
import time
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

from selfy_core.global_modules.config import get as config_get
from selfy_core.global_modules.memory.interfaces import MemoryItem, MemoryItemType
from selfy_core.global_modules.memory.memory_evaluator import MemoryEvaluator
from selfy_core.global_modules.llm_wrapper.llm_factory import get_llm_provider

# Forward declaration to avoid circular imports
get_memory_system = None

logger = logging.getLogger(__name__)

class MemoryConsolidationService:
    """
    Consolidates short-term memory into long-term memory.

    The MemoryConsolidationService is responsible for:
    1. Periodically checking for sessions that need consolidation
    2. Evaluating conversation turns for long-term storage
    3. Storing valuable information in long-term memory
    4. Clearing short-term memory that has been consolidated

    This service implements intelligent filtering of conversation content to reduce
    long-term storage requirements while preserving valuable information. It works by:

    - Periodically checking for sessions that have been inactive for a configurable
      period of time (default: 30 minutes)
    - Using the MemoryEvaluator to analyze conversation content against four parameters:
      future tasks, reasoning, context retrieval, and self-improvement
    - Creating concise summaries of valuable information that score above the threshold
    - Storing only these summaries in long-term memory with appropriate metadata
    - Optionally clearing working memory after consolidation

    Configuration options:
    - memory.consolidation.enable: Enable/disable consolidation (default: true)
    - memory.consolidation.interval_seconds: Time between checks (default: 3600s)
    - memory.consolidation.session_inactive_threshold_seconds: Time before a session
      is considered inactive (default: 1800s)
    - memory.consolidation.check_after_each_turn: Whether to check after each new
      conversation turn (default: false)
    - memory.evaluator.min_score_threshold: Minimum score for storing (default: 0.6)
    """

    def __init__(self):
        """Initialize the memory consolidation service."""
        logger.info("Initializing MemoryConsolidationService")

        # Load configuration
        self.consolidation_interval = config_get('memory.consolidation.interval_seconds', 3600)  # 1 hour
        self.session_inactive_threshold = config_get('memory.consolidation.session_inactive_threshold_seconds', 1800)  # 30 minutes
        self.enable_consolidation = config_get('memory.consolidation.enable', True)

        # Initialize state
        self.last_consolidation_time = time.time()
        self.consolidated_sessions = set()

        # Initialize memory evaluator with Ollama LLM
        llm_provider = get_llm_provider('ollama')
        self.memory_evaluator = MemoryEvaluator(llm_provider)

        logger.info(f"MemoryConsolidationService initialized with interval={self.consolidation_interval}s")

    def check_for_consolidation(self) -> None:
        """
        Check if it's time to consolidate memory and perform consolidation if needed.

        This method is called periodically to check if enough time has passed since
        the last consolidation. If so, it identifies inactive sessions and triggers
        the consolidation process for each one.

        The consolidation interval is controlled by the configuration setting
        'memory.consolidation.interval_seconds' (default: 3600 seconds or 1 hour).

        This method can also be called explicitly to force a consolidation check,
        for example after each conversation turn if 'check_after_each_turn' is enabled.
        """
        current_time = time.time()

        # Check if consolidation is enabled
        if not self.enable_consolidation:
            logger.debug("Memory consolidation is disabled")
            return

        # Check if it's time to consolidate
        if current_time - self.last_consolidation_time < self.consolidation_interval:
            return

        logger.info("Starting memory consolidation")
        self.last_consolidation_time = current_time

        # Get memory system
        memory_system = get_memory_system()
        if not memory_system:
            logger.error("Memory system not initialized")
            return

        # Find inactive sessions to consolidate
        inactive_sessions = self._find_inactive_sessions()

        if not inactive_sessions:
            logger.info("No inactive sessions found for consolidation")
            return

        logger.info(f"Found {len(inactive_sessions)} inactive sessions for consolidation")

        # Consolidate each inactive session
        for session_id in inactive_sessions:
            self._consolidate_session(session_id)

    def _find_inactive_sessions(self) -> List[str]:
        """
        Find inactive sessions that need consolidation.

        Returns:
            List of session IDs for inactive sessions
        """
        # Get memory system
        memory_system = get_memory_system()
        if not memory_system:
            logger.error("Memory system not initialized")
            return []

        # This is a placeholder - in a real implementation, we would query
        # the memory system for all sessions and check their last activity time
        # For now, we'll just return an empty list

        # TODO: Implement session tracking and activity time tracking
        return []

    def consolidate_session(self, session_id: str) -> bool:
        """
        Consolidate a specific session's memory.

        This method evaluates all conversation turns for a specific session,
        identifies valuable information based on the four parameters (future tasks,
        reasoning, context retrieval, self-improvement), and stores concise summaries
        of that information in long-term memory.

        The evaluation is performed by the MemoryEvaluator using Ollama to analyze
        the conversation content. Only information that scores above the configured
        threshold (default: 0.6) will be stored in long-term memory.

        This method can be called explicitly to consolidate a specific session,
        or it will be called automatically by the check_for_consolidation method
        for inactive sessions.

        Args:
            session_id: The session ID to consolidate

        Returns:
            True if consolidation was successful, False otherwise
        """
        return self._consolidate_session(session_id)

    def _consolidate_session(self, session_id: str) -> bool:
        """
        Consolidate a session's memory.

        Args:
            session_id: The session ID to consolidate

        Returns:
            True if consolidation was successful, False otherwise
        """
        logger.info(f"Consolidating memory for session {session_id}")

        # Get memory system
        memory_system = get_memory_system()
        if not memory_system:
            logger.error("Memory system not initialized")
            return False

        try:
            # Get all conversation turns for the session
            conversation_turns = memory_system.get_recent_conversation(session_id, 100)  # Get up to 100 turns

            if not conversation_turns:
                logger.info(f"No conversation turns found for session {session_id}")
                return True  # Nothing to consolidate

            logger.info(f"Found {len(conversation_turns)} conversation turns for session {session_id}")

            # Evaluate conversation turns for long-term storage
            valuable_items = self.memory_evaluator.evaluate_conversation(conversation_turns)

            if not valuable_items:
                logger.info(f"No valuable information found in session {session_id}")
                return True  # Nothing to store

            logger.info(f"Found {len(valuable_items)} valuable items to store from session {session_id}")

            # Store valuable items in long-term memory
            for item in valuable_items:
                memory_system.add_memory(item)

                # Detailed debug logging for the stored memory
                logger.debug(f"MEMORY CONSOLIDATION - STORED SUMMARY: {item.content}")
                logger.debug(f"MEMORY CONSOLIDATION - MEMORY DETAILS: Type={item.type}, Score={item.metadata.get('score', 'N/A')}, Criteria={item.metadata.get('criteria', [])}")

                # Log the full memory item for debugging
                memory_json = {
                    "id": item.id,
                    "type": str(item.type),
                    "content": item.content,
                    "timestamp": item.timestamp.isoformat() if hasattr(item.timestamp, 'isoformat') else str(item.timestamp),
                    "metadata": item.metadata
                }
                logger.debug(f"MEMORY CONSOLIDATION - FULL MEMORY: {json.dumps(memory_json, indent=2)}")

            # Mark session as consolidated
            self.consolidated_sessions.add(session_id)

            logger.info(f"Successfully consolidated memory for session {session_id}")
            return True

        except Exception as e:
            logger.error(f"Error consolidating memory for session {session_id}: {e}")
            return False

    def clear_consolidated_sessions(self) -> None:
        """
        Clear working memory for sessions that have been consolidated.
        """
        # Get memory system
        memory_system = get_memory_system()
        if not memory_system:
            logger.error("Memory system not initialized")
            return

        # Clear working memory
        memory_system.clear_working_memory()

        logger.info("Cleared working memory for consolidated sessions")

        # Reset consolidated sessions
        self.consolidated_sessions.clear()


# Global instance
_memory_consolidation_service = None


def setup_memory_consolidation_service() -> bool:
    """
    Set up the memory consolidation service.

    Returns:
        True if setup was successful, False otherwise
    """
    global _memory_consolidation_service

    try:
        if _memory_consolidation_service is None:
            _memory_consolidation_service = MemoryConsolidationService()

        logger.info("Memory consolidation service set up successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to set up memory consolidation service: {e}")
        return False


def get_memory_consolidation_service() -> Optional[MemoryConsolidationService]:
    """
    Get the memory consolidation service instance.

    Returns:
        The memory consolidation service instance, or None if not set up
    """
    global _memory_consolidation_service

    if _memory_consolidation_service is None:
        logger.warning("Memory consolidation service not initialized")

    return _memory_consolidation_service
