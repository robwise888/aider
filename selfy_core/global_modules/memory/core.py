"""
Memory core for the Selfy agent.

This module implements the UnifiedMemorySystem interface and orchestrates the
working memory, embedding service, and long-term storage components.
"""

import time
import uuid
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Union

from selfy_core.global_modules.config import get as config_get
from selfy_core.global_modules.logging import get_logger
from selfy_core.global_modules.memory.interfaces import (
    MemoryItem, MemoryQuery, MemoryResult, UnifiedMemorySystem, MemoryItemType
)
from selfy_core.global_modules.memory.working_memory import (
    setup_working_memory, get_working_memory
)
from selfy_core.global_modules.memory.embedding_service import (
    setup_embedding_service, get_embedding_service
)
from selfy_core.global_modules.memory.lts_vector_store import (
    setup_lts_vector_store, get_lts_vector_store
)

# Import only the setup function to avoid circular imports
from selfy_core.global_modules.memory.memory_consolidation import setup_memory_consolidation_service

# Forward declaration to avoid circular imports
get_memory_consolidation_service = None

# Set up logger
logger = get_logger(__name__)


class MemoryCore(UnifiedMemorySystem):
    """
    Memory core for the Selfy agent.

    This class implements the UnifiedMemorySystem interface and orchestrates the
    working memory, embedding service, and long-term storage components.
    """

    def __init__(self):
        """Initialize the memory core."""
        logger.info("Initializing MemoryCore")

    def setup(self) -> bool:
        """
        Set up the memory core.

        Returns:
            True if setup was successful, False otherwise
        """
        try:
            # Set up working memory
            if not setup_working_memory():
                logger.error("Failed to set up working memory")
                return False

            # Set up embedding service
            if not setup_embedding_service():
                logger.error("Failed to set up embedding service")
                return False

            # Set up LTS vector store
            if not setup_lts_vector_store():
                logger.error("Failed to set up LTS vector store")
                return False

            # Set up memory consolidation service
            if not setup_memory_consolidation_service():
                logger.warning("Failed to set up memory consolidation service - continuing without intelligent memory filtering")
                # Don't return False here, as this is an optional component

            logger.info("Set up MemoryCore successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to set up MemoryCore: {e}", exc_info=True)
            return False

    def add_memory(self, item: MemoryItem) -> None:
        """
        Add or update a memory item.

        This method handles embedding generation and storage in working memory
        and long-term storage.

        Args:
            item: The memory item to add or update
        """
        try:
            # Get components
            working_memory = get_working_memory()
            embedding_service = get_embedding_service()
            lts_vector_store = get_lts_vector_store()
            memory_consolidation = get_memory_consolidation_service()

            if not working_memory or not embedding_service or not lts_vector_store:
                logger.error("Memory components not initialized")
                return

            # Generate embedding if needed
            if item.content and item.embedding is None:
                try:
                    item.embedding = embedding_service.generate_embedding(item.content)
                    logger.debug(f"Generated embedding for item {item.id}")
                except Exception as e:
                    logger.error(f"Failed to generate embedding for item {item.id}: {e}", exc_info=True)

            # Add to working memory
            working_memory.add_or_update(item)
            logger.debug(f"Added item {item.id} to working memory")

            # Check if this is a conversation turn and if we should use intelligent filtering
            use_intelligent_filtering = (
                memory_consolidation is not None and
                item.type == MemoryItemType.CONVERSATION_TURN and
                config_get('memory.use_intelligent_filtering', True)
            )

            # For non-conversation items or when intelligent filtering is disabled,
            # add directly to LTS vector store
            if not use_intelligent_filtering:
                if item.embedding:
                    success = lts_vector_store.add_or_update(item)
                    if success:
                        logger.debug(f"Added item {item.id} to LTS vector store")
                    else:
                        logger.error(f"Failed to add item {item.id} to LTS vector store")
            else:
                # For conversation turns with intelligent filtering enabled,
                # we'll let the consolidation service handle it later
                logger.debug(f"Item {item.id} will be evaluated for long-term storage during consolidation")

                # Check if we should trigger immediate consolidation for this session
                if 'session_id' in item.metadata and memory_consolidation is not None:
                    # Check if enough time has passed since last consolidation
                    should_consolidate = config_get('memory.consolidation.check_after_each_turn', False)
                    if should_consolidate:
                        memory_consolidation.check_for_consolidation()

        except Exception as e:
            logger.error(f"Failed to add memory item {item.id}: {e}", exc_info=True)

    def query_memory(self, query: MemoryQuery) -> MemoryResult:
        """
        Perform a query against the memory system.

        This method uses semantic search and filtering to find relevant memory items.

        Args:
            query: The query to execute

        Returns:
            The query results
        """
        start_time = time.time()

        # Log detailed information about the query
        query_text_preview = query.query_text[:50] + "..." if query.query_text and len(query.query_text) > 50 else query.query_text
        logger.info(f"Querying memory: Text='{query_text_preview}', Filters={query.filters}, TopK={query.top_k}")

        try:
            # Get components
            embedding_service = get_embedding_service()
            lts_vector_store = get_lts_vector_store()

            if not embedding_service or not lts_vector_store:
                logger.error("Memory components not initialized")
                return MemoryResult(
                    retrieved_items=[],
                    similarity_scores=[],
                    total_found=0,
                    query_time_ms=(time.time() - start_time) * 1000
                )

            # Generate embedding if needed
            embedding_start_time = time.time()
            if query.query_text and query.query_embedding is None:
                try:
                    query.query_embedding = embedding_service.generate_embedding(query.query_text)
                    embedding_time_ms = (time.time() - embedding_start_time) * 1000
                    logger.info(f"Generated embedding for query in {embedding_time_ms:.2f}ms")
                    if query.query_embedding:
                        logger.debug(f"Query embedding dimensions: {len(query.query_embedding)}")
                except Exception as e:
                    logger.error(f"Failed to generate embedding for query: {e}", exc_info=True)

            # Search in LTS vector store
            search_start_time = time.time()
            logger.info(f"Executing vector store search")
            result = lts_vector_store.search(query)
            search_time_ms = (time.time() - search_start_time) * 1000

            # Log detailed information about the results
            query_time_ms = (time.time() - start_time) * 1000
            logger.info(f"Memory query completed in {query_time_ms:.2f}ms (Embedding: {embedding_time_ms if 'embedding_time_ms' in locals() else 0:.2f}ms, Search: {search_time_ms:.2f}ms)")
            logger.info(f"Found {result.total_found} items matching the query")

            # Log the top 3 results with their similarity scores
            if result.retrieved_items:
                logger.info("Top memory query results:")
                for i in range(min(3, len(result.retrieved_items))):
                    item = result.retrieved_items[i]
                    similarity = result.similarity_scores[i]
                    content_preview = str(item.content)[:50] + "..." if len(str(item.content)) > 50 else str(item.content)
                    logger.info(f"  {i+1}. ID={item.id}, Type={item.type}, Similarity={similarity:.4f}, Content='{content_preview}'")

            return result
        except Exception as e:
            logger.error(f"Failed to query memory: {e}", exc_info=True)
            query_time_ms = (time.time() - start_time) * 1000

            return MemoryResult(
                retrieved_items=[],
                similarity_scores=[],
                total_found=0,
                query_time_ms=query_time_ms
            )

    def get_memory_by_id(self, item_id: str) -> Optional[MemoryItem]:
        """
        Retrieve a specific memory item by its unique ID.

        Args:
            item_id: The ID of the memory item to retrieve

        Returns:
            The memory item, or None if not found
        """
        try:
            # Get components
            working_memory = get_working_memory()
            lts_vector_store = get_lts_vector_store()

            if not working_memory or not lts_vector_store:
                logger.error("Memory components not initialized")
                return None

            # Check working memory first
            item = working_memory.get_by_id(item_id)
            if item:
                logger.debug(f"Found item {item_id} in working memory")
                return item

            # Check LTS vector store
            item = lts_vector_store.get_by_id(item_id)
            if item:
                logger.debug(f"Found item {item_id} in LTS vector store")
                return item

            logger.debug(f"Item {item_id} not found")
            return None
        except Exception as e:
            logger.error(f"Failed to get memory item {item_id}: {e}", exc_info=True)
            return None

    def delete_memory(self, item_id: str) -> bool:
        """
        Delete a memory item by its unique ID.

        Args:
            item_id: The ID of the memory item to delete

        Returns:
            True if the item was deleted, False otherwise
        """
        try:
            # Get components
            working_memory = get_working_memory()
            lts_vector_store = get_lts_vector_store()

            if not working_memory or not lts_vector_store:
                logger.error("Memory components not initialized")
                return False

            # Delete from working memory
            working_memory.delete(item_id)

            # Delete from LTS vector store
            success = lts_vector_store.delete(item_id)

            logger.debug(f"Deleted item {item_id} from memory")
            return success
        except Exception as e:
            logger.error(f"Failed to delete memory item {item_id}: {e}", exc_info=True)
            return False

    def get_recent_conversation(self, session_id: str, last_n: int) -> List[MemoryItem]:
        """
        Get the most recent conversation turns for a session.

        Args:
            session_id: The session ID to get conversation turns for
            last_n: The maximum number of turns to retrieve

        Returns:
            List of conversation turns, ordered from oldest to newest
        """
        try:
            # Get components
            lts_vector_store = get_lts_vector_store()

            if not lts_vector_store:
                logger.error("Memory components not initialized")
                return []

            # Get conversation turns from LTS vector store
            items = lts_vector_store.get_by_session(session_id, limit=last_n * 2)

            # Filter by type
            conversation_turns = [item for item in items if item.type == 'conversation_turn']

            # Sort by timestamp
            conversation_turns.sort(key=lambda item: item.timestamp)

            # Limit to last_n
            if len(conversation_turns) > last_n:
                conversation_turns = conversation_turns[-last_n:]

            logger.debug(f"Got {len(conversation_turns)} conversation turns for session {session_id}")
            return conversation_turns
        except Exception as e:
            logger.error(f"Failed to get recent conversation: {e}", exc_info=True)
            return []

    def clear_working_memory(self) -> None:
        """
        Clear all items from working memory.

        This does not affect items in long-term storage.
        """
        try:
            # Get components
            working_memory = get_working_memory()

            if not working_memory:
                logger.error("Working memory not initialized")
                return

            # Clear working memory
            working_memory.clear()
            logger.info("Cleared working memory")
        except Exception as e:
            logger.error(f"Failed to clear working memory: {e}", exc_info=True)

    def store_capability_info(self, capability: Any) -> None:
        """
        Store capability information in memory.

        Creates a MemoryItem of type CAPABILITY_INFO containing the capability
        details and stores it in the memory system.

        Args:
            capability: The capability object or dictionary to store
        """
        try:
            # Convert capability to dictionary if needed
            if not isinstance(capability, dict):
                try:
                    capability_dict = capability.__dict__
                except AttributeError:
                    capability_dict = {"capability": str(capability)}
            else:
                capability_dict = capability

            # Create a content string
            if 'name' in capability_dict and 'description' in capability_dict:
                content = f"{capability_dict['name']}: {capability_dict['description']}"
            elif 'name' in capability_dict:
                content = f"{capability_dict['name']}"
            else:
                content = str(capability_dict)

            # Create memory item
            item = MemoryItem(
                id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                type=MemoryItemType.CAPABILITY_INFO,
                content=content,
                metadata=capability_dict
            )

            # Add to memory
            self.add_memory(item)
            logger.debug(f"Stored capability info: {content[:100]}")
        except Exception as e:
            logger.error(f"Failed to store capability info: {e}", exc_info=True)

    def log_capability_usage(self, capability_name: str, parameters: Dict[str, Any],
                           result: Any, success: bool, session_id: str,
                           user_id: str) -> None:
        """
        Log capability usage in memory.

        Creates a MemoryItem of type EXECUTION_LOG containing the usage details
        and stores it in the memory system.

        Args:
            capability_name: The name of the capability
            parameters: The parameters passed to the capability
            result: The result of the capability execution
            success: Whether the execution was successful
            session_id: The session ID
            user_id: The user ID
        """
        try:
            # Create a content string
            timestamp = datetime.now()
            content = f"Capability: {capability_name}\n"
            content += f"Parameters: {json.dumps(parameters, default=str)}\n"
            content += f"Result: {str(result)}\n"
            content += f"Success: {success}\n"
            content += f"Timestamp: {timestamp.isoformat()}"

            # Create memory item
            item = MemoryItem(
                id=str(uuid.uuid4()),
                timestamp=timestamp,
                type=MemoryItemType.EXECUTION_LOG,
                content=content,
                metadata={
                    'capability_id': capability_name,
                    'parameters': json.dumps(parameters, default=str),
                    'result': str(result),
                    'success': success,
                    'execution_timestamp': timestamp.isoformat(),
                    'session_id': session_id,
                    'user_id': user_id
                }
            )

            # Add to memory
            self.add_memory(item)
            logger.debug(f"Logged capability usage for {capability_name}")
        except Exception as e:
            logger.error(f"Failed to log capability usage: {e}", exc_info=True)

    def store_error(self, error: Any) -> None:
        """
        Store error information in memory.

        Creates a MemoryItem of type ERROR_LOG containing the error details
        and stores it in the memory system.

        Args:
            error: The error object to store
        """
        try:
            # Convert error to dictionary if needed
            if hasattr(error, 'to_dict'):
                error_dict = error.to_dict()
            elif hasattr(error, '__dict__'):
                error_dict = error.__dict__
            else:
                error_dict = {"error": str(error)}

            # Create a content string
            if 'message' in error_dict:
                content = f"Error: {error_dict['message']}"
            elif 'error' in error_dict:
                content = f"Error: {error_dict['error']}"
            else:
                content = f"Error: {str(error)}"

            if 'category' in error_dict:
                content += f"\nCategory: {error_dict['category']}"
            if 'severity' in error_dict:
                content += f"\nSeverity: {error_dict['severity']}"
            if 'recovery_strategy' in error_dict:
                content += f"\nRecovery Strategy: {error_dict['recovery_strategy']}"

            # Create memory item
            item = MemoryItem(
                id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                type=MemoryItemType.ERROR_LOG,
                content=content,
                metadata=error_dict
            )

            # Add to memory
            self.add_memory(item)
            logger.debug(f"Stored error info: {content[:100]}")
        except Exception as e:
            logger.error(f"Failed to store error info: {e}", exc_info=True)

    def find_similar_errors(self, error_message: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Find similar errors in memory.

        Args:
            error_message: The error message to search for
            top_k: Maximum number of errors to return

        Returns:
            List of similar errors
        """
        try:
            # Create a query
            query = MemoryQuery(
                query_text=error_message,
                filters={"type": MemoryItemType.ERROR_LOG},
                top_k=top_k
            )

            # Execute the query
            result = self.query_memory(query)

            # Extract error info from the results
            errors = []
            for item in result.retrieved_items:
                if item.metadata:
                    errors.append(item.metadata)
                else:
                    errors.append({"error": item.content})

            logger.debug(f"Found {len(errors)} similar errors for query: {error_message[:100]}")
            return errors
        except Exception as e:
            logger.error(f"Failed to find similar errors: {e}", exc_info=True)
            return []

    def store_memory(self, memory_type: str, content: Any, metadata: Optional[Dict[str, Any]] = None) -> MemoryItem:
        """
        Store a memory in the memory system.

        Args:
            memory_type: The type of memory to store
            content: The content of the memory
            metadata: Additional metadata for the memory

        Returns:
            The created memory item
        """
        try:
            # Create a unique ID for the memory item
            memory_id = str(uuid.uuid4())
            timestamp = datetime.now()

            logger.info(f"Storing new memory: ID={memory_id}, Type={memory_type}, Timestamp={timestamp.isoformat()}")

            # Log content preview
            content_str = str(content)
            content_preview = content_str[:100] + "..." if len(content_str) > 100 else content_str
            logger.info(f"Memory content preview: '{content_preview}'")

            # Log metadata if available
            if metadata:
                metadata_preview = str(metadata)[:200] + "..." if len(str(metadata)) > 200 else str(metadata)
                logger.debug(f"Memory metadata: {metadata_preview}")

            # Create memory item
            item = MemoryItem(
                id=memory_id,
                timestamp=timestamp,
                type=memory_type,
                content=content,
                metadata=metadata
            )

            # Add to memory
            start_time = time.time()
            self.add_memory(item)
            store_time_ms = (time.time() - start_time) * 1000

            logger.info(f"Memory stored successfully: ID={memory_id}, Type={memory_type}, Processing time={store_time_ms:.2f}ms")

            return item
        except Exception as e:
            logger.error(f"Failed to store memory: {e}", exc_info=True)
            # Return a placeholder item
            return MemoryItem(
                id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                type=memory_type,
                content=content,
                metadata=metadata
            )

    def find_relevant_capabilities(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Find capabilities relevant to a query.

        Args:
            query_text: The query text
            top_k: Maximum number of capabilities to return

        Returns:
            List of relevant capabilities
        """
        try:
            # Create a query
            query = MemoryQuery(
                query_text=query_text,
                filters={"type": MemoryItemType.CAPABILITY_INFO},
                top_k=top_k
            )

            # Execute the query
            result = self.query_memory(query)

            # Extract capability info from the results
            capabilities = []
            for item in result.retrieved_items:
                if item.metadata:
                    capabilities.append(item.metadata)
                else:
                    capabilities.append({"name": "Unknown", "description": item.content})

            logger.debug(f"Found {len(capabilities)} relevant capabilities for query: {query_text[:100]}")
            return capabilities
        except Exception as e:
            logger.error(f"Failed to find relevant capabilities: {e}", exc_info=True)
            return []


# Global instance
_memory_core_instance = None


def setup_memory_core() -> bool:
    """
    Set up the memory core.

    Returns:
        True if setup was successful, False otherwise
    """
    global _memory_core_instance

    try:
        if _memory_core_instance is None:
            _memory_core_instance = MemoryCore()

        # Set up the memory core
        success = _memory_core_instance.setup()

        # Set the get_memory_system function in memory_consolidation
        # This breaks the circular import
        from selfy_core.global_modules.memory import memory_consolidation
        memory_consolidation.get_memory_system = get_memory_system

        # Set the get_memory_consolidation_service function in this module
        # This breaks the circular import
        global get_memory_consolidation_service
        get_memory_consolidation_service = memory_consolidation.get_memory_consolidation_service

        return success
    except Exception as e:
        logger.error(f"Failed to set up memory core: {e}", exc_info=True)
        return False


def get_memory_system() -> Optional[UnifiedMemorySystem]:
    """
    Get the memory system instance.

    Returns:
        The memory system instance, or None if not set up
    """
    global _memory_core_instance

    if _memory_core_instance is None:
        logger.warning("Memory system not initialized")

    return _memory_core_instance
