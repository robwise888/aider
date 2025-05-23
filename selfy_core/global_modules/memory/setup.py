"""
Setup functions for the memory system.

This module provides functions for setting up and initializing the memory system.
"""

import logging
from selfy_core.global_modules.logging import get_logger
from selfy_core.global_modules.memory.core import setup_memory_core, get_memory_system
from selfy_core.global_modules.memory.memory_consolidation import (
    setup_memory_consolidation_service, get_memory_consolidation_service
)
from selfy_core.global_modules.config import get as config_get

# Set up logger
logger = get_logger(__name__)


def setup_memory_system() -> bool:
    """
    Set up the memory system.
    
    This function initializes the memory system and all its components.
    
    Returns:
        True if setup was successful, False otherwise
    """
    logger.info("Setting up memory system")
    
    # Load memory consolidation configuration
    try:
        # Check if intelligent filtering is enabled
        use_intelligent_filtering = config_get('memory.use_intelligent_filtering', True)
        logger.info(f"Intelligent memory filtering is {'enabled' if use_intelligent_filtering else 'disabled'}")
        
        # Log consolidation settings
        consolidation_enabled = config_get('memory.consolidation.enable', True)
        consolidation_interval = config_get('memory.consolidation.interval_seconds', 3600)
        logger.info(f"Memory consolidation is {'enabled' if consolidation_enabled else 'disabled'} "
                   f"with interval {consolidation_interval}s")
    except Exception as e:
        logger.warning(f"Failed to load memory consolidation configuration: {e}. Using defaults.")
    
    # Set up memory core
    if not setup_memory_core():
        logger.error("Failed to set up memory core")
        return False
    
    logger.info("Memory system set up successfully")
    return True


def consolidate_session_memory(session_id: str) -> bool:
    """
    Consolidate memory for a specific session.
    
    This function triggers the memory consolidation process for a specific session.
    
    Args:
        session_id: The session ID to consolidate
        
    Returns:
        True if consolidation was successful, False otherwise
    """
    logger.info(f"Consolidating memory for session {session_id}")
    
    # Get memory consolidation service
    consolidation_service = get_memory_consolidation_service()
    if not consolidation_service:
        logger.error("Memory consolidation service not initialized")
        return False
    
    # Consolidate session
    return consolidation_service.consolidate_session(session_id)


def clear_session_memory(session_id: str) -> bool:
    """
    Clear memory for a specific session.
    
    This function clears all memory items for a specific session.
    
    Args:
        session_id: The session ID to clear
        
    Returns:
        True if clearing was successful, False otherwise
    """
    logger.info(f"Clearing memory for session {session_id}")
    
    # Get memory system
    memory_system = get_memory_system()
    if not memory_system:
        logger.error("Memory system not initialized")
        return False
    
    # Clear working memory
    memory_system.clear_working_memory()
    
    # TODO: Implement selective clearing of session-specific memory
    
    return True
