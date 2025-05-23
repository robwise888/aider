"""
Memory integration for the Input Handling module.

This module provides functions for interacting with the memory system.
"""

import uuid
from datetime import datetime
from typing import List, Any, Dict, Optional

from selfy_core.global_modules.config import get as config_get
from selfy_core.global_modules.logging import get_logger
from selfy_core.global_modules.memory import (
    MemoryItem, MemoryItemType, get_memory_system
)

# Set up logger
logger = get_logger(__name__)


def get_recent_conversation(session_id: str, last_n: int) -> List[Any]:
    """
    Get recent conversation history from memory.

    Args:
        session_id: Identifier for the current conversation session
        last_n: Number of recent conversation turns to fetch

    Returns:
        List of conversation turns from memory
    """
    try:
        # Get memory system
        memory_system = get_memory_system()
        if memory_system is None:
            logger.warning("Memory system not initialized. Returning empty conversation history.")
            return []

        # Get recent conversation
        conversation_history = memory_system.get_recent_conversation(session_id, last_n)

        logger.debug(f"Retrieved {len(conversation_history)} conversation turns for session_id={session_id}")
        return conversation_history
    except Exception as e:
        logger.error(f"Error retrieving conversation history: {e}", exc_info=True)
        return []


def add_memory_item(content: str, user_id: str, session_id: str) -> None:
    """
    Add a memory item to the memory system.

    Args:
        content: The content of the memory item
        user_id: Identifier for the user
        session_id: Identifier for the current conversation session
    """
    try:
        # Get memory system
        memory_system = get_memory_system()
        if memory_system is None:
            logger.warning("Memory system not initialized. Memory item not stored.")
            return

        # Get memory type from config or use default
        memory_type = config_get('memory.types.conversation_turn', MemoryItemType.CONVERSATION_TURN)

        # Get user role from config or use default
        user_role = config_get('memory.roles.user', 'user')

        # Create memory item
        memory_item = MemoryItem(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            type=memory_type,
            content=content,
            metadata={
                'user_id': user_id,
                'session_id': session_id,
                'role': user_role
            }
        )

        # Add to memory
        memory_system.add_memory(memory_item)
        logger.debug(f"Added memory item for user_id={user_id}, session_id={session_id}")
    except Exception as e:
        logger.error(f"Error adding memory item: {e}", exc_info=True)
