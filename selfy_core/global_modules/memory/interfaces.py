"""
Memory interfaces for the Selfy agent.

This module defines the standard data types and interfaces for memory operations,
including MemoryItem, MemoryQuery, MemoryResult, and UnifiedMemorySystem.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional, Protocol, Union


class MemoryItemType(str, Enum):
    """
    Standardized types for memory items.

    These types help categorize different kinds of information stored in memory.
    """
    CONVERSATION_TURN = "conversation_turn"
    CAPABILITY_INFO = "capability_info"
    EXECUTION_LOG = "execution_log"
    LEARNED_FACT = "learned_fact"
    USER_PREFERENCE = "user_preference"
    ERROR_LOG = "error_log"
    # CODE_* types will be added in Phase 3


@dataclass
class MemoryItem:
    """
    Represents a single unit of information stored in memory.

    Attributes:
        id: Unique identifier (e.g., generated UUID)
        timestamp: Timestamp of creation or last significant update
        type: Category of memory (use MemoryItemType value or string)
        content: Textual content of the memory item
        embedding: Vector embedding of the content (optional)
        metadata: Dictionary for storing additional context
    """
    id: str
    timestamp: datetime
    type: Union[MemoryItemType, str]
    content: str
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryQuery:
    """
    Represents a query to the memory system.

    Attributes:
        query_text: Text to search for (semantic search)
        query_embedding: Pre-computed embedding for the query text (optional)
        filters: Dictionary of filters to apply to the search results
        top_k: Maximum number of results to return
        min_similarity: Minimum similarity score for results (0-1)
    """
    query_text: Optional[str] = None
    query_embedding: Optional[List[float]] = None
    filters: Dict[str, Any] = field(default_factory=dict)
    top_k: int = 10
    min_similarity: float = 0.0


@dataclass
class MemoryResult:
    """
    Represents the result of a memory query.

    Attributes:
        retrieved_items: List of memory items that match the query
        similarity_scores: List of similarity scores for each retrieved item
        total_found: Total number of items found before limiting to top_k
        query_time_ms: Time taken to execute the query in milliseconds
    """
    retrieved_items: List[MemoryItem]
    similarity_scores: List[float]
    total_found: int
    query_time_ms: float


class UnifiedMemorySystem(Protocol):
    """
    Defines the primary interface for interacting with the agent's memory system.

    This interface is implemented by the memory core, which orchestrates the
    working memory and long-term storage components.
    """

    def add_memory(self, item: MemoryItem) -> None:
        """
        Add or update a memory item.

        This method handles embedding generation and storage in working memory
        and long-term storage.

        Args:
            item: The memory item to add or update
        """
        ...

    def query_memory(self, query: MemoryQuery) -> MemoryResult:
        """
        Perform a query against the memory system.

        This method uses semantic search and filtering to find relevant memory items.

        Args:
            query: The query to execute

        Returns:
            The query results
        """
        ...

    def get_memory_by_id(self, item_id: str) -> Optional[MemoryItem]:
        """
        Retrieve a specific memory item by its unique ID.

        Args:
            item_id: The ID of the memory item to retrieve

        Returns:
            The memory item, or None if not found
        """
        ...

    def delete_memory(self, item_id: str) -> bool:
        """
        Delete a memory item by its unique ID.

        Args:
            item_id: The ID of the memory item to delete

        Returns:
            True if the item was deleted, False otherwise
        """
        ...

    def get_recent_conversation(self, session_id: str, last_n: int) -> List[MemoryItem]:
        """
        Get the most recent conversation turns for a session.

        Args:
            session_id: The session ID to get conversation turns for
            last_n: The maximum number of turns to retrieve

        Returns:
            List of conversation turns, ordered from oldest to newest
        """
        ...

    def clear_working_memory(self) -> None:
        """
        Clear all items from working memory.

        This does not affect items in long-term storage.
        """
        ...

    def store_capability_info(self, capability: Any) -> None:
        """
        Store capability information in memory.

        Creates a MemoryItem of type CAPABILITY_INFO containing the capability
        details and stores it in the memory system.

        Args:
            capability: The capability object or dictionary to store
        """
        ...

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
        ...

    def store_error(self, error: Any) -> None:
        """
        Store error information in memory.

        Creates a MemoryItem of type ERROR_LOG containing the error details
        and stores it in the memory system.

        Args:
            error: The error object to store
        """
        ...

    def find_similar_errors(self, error_message: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Find similar errors in memory.

        Args:
            error_message: The error message to search for
            top_k: Maximum number of errors to return

        Returns:
            List of similar errors
        """
        ...

    def find_relevant_capabilities(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Find capabilities relevant to a query.

        Args:
            query_text: The query text
            top_k: Maximum number of capabilities to return

        Returns:
            List of relevant capabilities
        """
        ...
