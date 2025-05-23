"""
Long-term storage vector store for the Selfy agent.

This module provides a vector database for storing and retrieving memory items.
It supports semantic search and filtering based on metadata.
"""

import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union

from selfy_core.global_modules.config import get as config_get
from selfy_core.global_modules.logging import get_logger
from selfy_core.global_modules.memory.interfaces import MemoryItem, MemoryQuery, MemoryResult

# Set up logger
logger = get_logger(__name__)


class LTSVectorStore:
    """
    Long-term storage vector store for memory items.

    This class provides a vector database for storing and retrieving memory items.
    It supports semantic search and filtering based on metadata.
    """

    def __init__(self):
        """Initialize the vector store."""
        # Get configuration
        self.vector_db_type = config_get('memory.lts.vector_db_type', 'chromadb')
        self.db_path = config_get(f'memory.lts.{self.vector_db_type}.path', './memory_db')
        self.collection_name = config_get(f'memory.lts.{self.vector_db_type}.collection_name', 'selfy_memory')

        # Initialize database client and collection
        self.client = None
        self.collection = None

        logger.info(f"Initialized LTSVectorStore with vector_db_type={self.vector_db_type}, "
                   f"db_path={self.db_path}, collection_name={self.collection_name}")

    def setup(self) -> bool:
        """
        Set up the vector store.

        Returns:
            True if setup was successful, False otherwise
        """
        try:
            # Reload configuration in case it changed
            self.vector_db_type = config_get('memory.lts.vector_db_type', 'chromadb')
            self.db_path = config_get(f'memory.lts.{self.vector_db_type}.path', './memory_db')
            self.collection_name = config_get(f'memory.lts.{self.vector_db_type}.collection_name', 'selfy_memory')

            # Check for recreate flag file
            import os
            flag_file = os.path.join(os.path.dirname(self.db_path), ".recreate_db")
            recreate_db = False

            if os.path.exists(flag_file):
                logger.info(f"Found database recreation flag file: {flag_file}")
                recreate_db = True
                try:
                    os.remove(flag_file)
                    logger.info(f"Removed flag file: {flag_file}")
                except Exception as e:
                    logger.warning(f"Failed to remove flag file: {e}")

            # Initialize the vector database based on the type
            if self.vector_db_type == 'chromadb':
                try:
                    import chromadb
                    from chromadb.config import Settings

                    # Create client
                    self.client = chromadb.PersistentClient(
                        path=self.db_path,
                        settings=Settings(anonymized_telemetry=False)
                    )

                    # Get or create collection
                    if recreate_db:
                        # Try to delete the collection if it exists
                        try:
                            self.client.delete_collection(name=self.collection_name)
                            logger.info(f"Deleted existing ChromaDB collection: {self.collection_name}")
                        except Exception as e:
                            logger.warning(f"Failed to delete collection (it may not exist): {e}")

                        # Create a new collection
                        self.collection = self.client.create_collection(name=self.collection_name)
                        logger.info(f"Created new ChromaDB collection: {self.collection_name}")
                    else:
                        try:
                            self.collection = self.client.get_collection(name=self.collection_name)
                            logger.info(f"Using existing ChromaDB collection: {self.collection_name}")
                        except Exception:
                            self.collection = self.client.create_collection(name=self.collection_name)
                            logger.info(f"Created new ChromaDB collection: {self.collection_name}")
                except ImportError:
                    logger.error("ChromaDB not available. Please install it with 'pip install chromadb'.")
                    return False
            elif self.vector_db_type == 'faiss':
                logger.error("FAISS support not implemented yet")
                return False
            else:
                logger.error(f"Unknown vector database type: {self.vector_db_type}")
                return False

            logger.info(f"Set up LTSVectorStore with vector_db_type={self.vector_db_type}, "
                       f"db_path={self.db_path}, collection_name={self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to set up LTSVectorStore: {e}", exc_info=True)
            return False

    def add_or_update(self, item: MemoryItem) -> bool:
        """
        Add or update a memory item.

        Args:
            item: The memory item to add or update

        Returns:
            True if the operation was successful, False otherwise
        """
        if not self.collection:
            logger.error("Vector store not initialized")
            return False

        try:
            # Ensure the item has an embedding
            if item.embedding is None:
                logger.error(f"Item {item.id} has no embedding")
                return False

            # Convert metadata to JSON-serializable format
            metadata = self._prepare_metadata(item)

            # Log detailed information about the memory item
            logger.info(f"Processing memory item: ID={item.id}, Type={item.type}, Content length={len(str(item.content))}")
            logger.debug(f"Memory item metadata: {metadata}")
            logger.debug(f"Memory item embedding shape: {len(item.embedding)} dimensions")

            # Check if the item already exists
            try:
                existing = self.collection.get(ids=[item.id])
                if len(existing['ids']) > 0:
                    # Update existing item
                    logger.info(f"Updating existing memory item: ID={item.id}, Type={item.type}")
                    start_time = time.time()

                    self.collection.update(
                        ids=[item.id],
                        embeddings=[item.embedding],
                        metadatas=[metadata],
                        documents=[item.content]
                    )

                    update_time_ms = (time.time() - start_time) * 1000
                    logger.info(f"Updated item {item.id} in vector store in {update_time_ms:.2f}ms")
                else:
                    # Add new item
                    logger.info(f"Adding new memory item: ID={item.id}, Type={item.type}")
                    start_time = time.time()

                    self.collection.add(
                        ids=[item.id],
                        embeddings=[item.embedding],
                        metadatas=[metadata],
                        documents=[item.content]
                    )

                    add_time_ms = (time.time() - start_time) * 1000
                    logger.info(f"Added item {item.id} to vector store in {add_time_ms:.2f}ms")
            except Exception as e:
                # Add new item
                logger.info(f"Exception during check, adding as new item: ID={item.id}, Type={item.type}, Error: {str(e)}")
                start_time = time.time()

                self.collection.add(
                    ids=[item.id],
                    embeddings=[item.embedding],
                    metadatas=[metadata],
                    documents=[item.content]
                )

                add_time_ms = (time.time() - start_time) * 1000
                logger.info(f"Added item {item.id} to vector store in {add_time_ms:.2f}ms")

            return True
        except Exception as e:
            logger.error(f"Failed to add or update item {item.id}: {e}", exc_info=True)

            # Check if this is a known ChromaDB error
            if detect_and_handle_chromadb_error(e):
                logger.warning("ChromaDB error detected and handled. The database will be recreated on next startup.")

            return False

    def get_by_id(self, item_id: str) -> Optional[MemoryItem]:
        """
        Get a memory item by its ID.

        Args:
            item_id: The ID of the memory item to get

        Returns:
            The memory item, or None if not found
        """
        if not self.collection:
            logger.error("Vector store not initialized")
            return None

        try:
            # Get the item from the collection
            result = self.collection.get(ids=[item_id])

            if len(result['ids']) == 0:
                logger.debug(f"Item {item_id} not found in vector store")
                return None

            # Convert to MemoryItem
            return self._convert_to_memory_item(
                id=result['ids'][0],
                content=result['documents'][0],
                metadata=result['metadatas'][0],
                embedding=result['embeddings'][0] if 'embeddings' in result else None
            )
        except Exception as e:
            logger.error(f"Failed to get item {item_id}: {e}", exc_info=True)

            # Check if this is a known ChromaDB error
            if detect_and_handle_chromadb_error(e):
                logger.warning("ChromaDB error detected and handled. The database will be recreated on next startup.")

            return None

    def delete(self, item_id: str) -> bool:
        """
        Delete a memory item by its ID.

        Args:
            item_id: The ID of the memory item to delete

        Returns:
            True if the item was deleted, False otherwise
        """
        if not self.collection:
            logger.error("Vector store not initialized")
            return False

        try:
            # Delete the item from the collection
            self.collection.delete(ids=[item_id])
            logger.debug(f"Deleted item {item_id} from vector store")
            return True
        except Exception as e:
            logger.error(f"Failed to delete item {item_id}: {e}", exc_info=True)

            # Check if this is a known ChromaDB error
            if detect_and_handle_chromadb_error(e):
                logger.warning("ChromaDB error detected and handled. The database will be recreated on next startup.")

            return False

    def search(self, query: MemoryQuery) -> MemoryResult:
        """
        Search for memory items.

        Args:
            query: The query to execute

        Returns:
            The search results
        """
        if not self.collection:
            logger.error("Vector store not initialized")
            return MemoryResult(
                retrieved_items=[],
                similarity_scores=[],
                total_found=0,
                query_time_ms=0.0
            )

        start_time = time.time()

        # Log detailed information about the query
        logger.info(f"Executing memory search: Text='{query.query_text[:50]}...' if query.query_text else 'None'")
        logger.info(f"Search parameters: Filters={query.filters}, TopK={query.top_k}")
        if query.query_embedding:
            logger.debug(f"Query embedding dimensions: {len(query.query_embedding)}")

        try:
            # Prepare query parameters
            query_embedding = query.query_embedding
            query_text = query.query_text
            filters = self._prepare_filters(query.filters)
            top_k = query.top_k

            logger.debug(f"Prepared filters: {filters}")

            # Execute the query
            query_start_time = time.time()
            if query_embedding:
                # Search by embedding
                logger.info(f"Searching by embedding vector")
                result = self.collection.query(
                    query_embeddings=[query_embedding],
                    where=filters,
                    n_results=top_k,
                    include=['metadatas', 'documents', 'distances', 'embeddings']
                )
            elif query_text:
                # Search by text
                logger.info(f"Searching by text: '{query_text[:50]}...'")
                result = self.collection.query(
                    query_texts=[query_text],
                    where=filters,
                    n_results=top_k,
                    include=['metadatas', 'documents', 'distances', 'embeddings']
                )
            else:
                # Get items by filter only
                logger.info(f"Searching by filters only: {filters}")
                result = self.collection.get(
                    where=filters,
                    limit=top_k,
                    include=['metadatas', 'documents', 'embeddings']
                )
                # Add distances for consistency
                result['distances'] = [[0.0] * len(result['ids'])]

            db_query_time_ms = (time.time() - query_start_time) * 1000
            logger.info(f"Database query completed in {db_query_time_ms:.2f}ms")

            # Convert to MemoryResult
            retrieved_items = []
            similarity_scores = []

            processing_start_time = time.time()
            if 'ids' in result and len(result['ids']) > 0:
                ids = result['ids'][0]
                documents = result['documents'][0]
                metadatas = result['metadatas'][0]
                distances = result['distances'][0] if 'distances' in result else [0.0] * len(ids)
                embeddings = result['embeddings'][0] if 'embeddings' in result else [None] * len(ids)

                logger.info(f"Processing {len(ids)} search results")

                for i in range(len(ids)):
                    item = self._convert_to_memory_item(
                        id=ids[i],
                        content=documents[i],
                        metadata=metadatas[i],
                        embedding=embeddings[i]
                    )
                    retrieved_items.append(item)

                    # Convert distance to similarity score (1.0 - normalized distance)
                    # ChromaDB uses cosine distance, which is 1 - cosine similarity
                    similarity = 1.0 - distances[i]
                    similarity_scores.append(similarity)

                    logger.debug(f"Result {i+1}: ID={ids[i]}, Type={metadatas[i].get('type', 'unknown')}, Similarity={similarity:.4f}")

            processing_time_ms = (time.time() - processing_start_time) * 1000
            query_time_ms = (time.time() - start_time) * 1000

            logger.info(f"Search completed in {query_time_ms:.2f}ms (DB: {db_query_time_ms:.2f}ms, Processing: {processing_time_ms:.2f}ms)")
            logger.info(f"Found {len(retrieved_items)} items matching the query")

            # Log the top 3 results with their similarity scores
            if retrieved_items:
                logger.info("Top search results:")
                for i in range(min(3, len(retrieved_items))):
                    item = retrieved_items[i]
                    similarity = similarity_scores[i]
                    content_preview = str(item.content)[:50] + "..." if len(str(item.content)) > 50 else str(item.content)
                    logger.info(f"  {i+1}. ID={item.id}, Type={item.type}, Similarity={similarity:.4f}, Content='{content_preview}'")

            return MemoryResult(
                retrieved_items=retrieved_items,
                similarity_scores=similarity_scores,
                total_found=len(retrieved_items),
                query_time_ms=query_time_ms
            )
        except Exception as e:
            logger.error(f"Failed to search: {e}", exc_info=True)

            # Check if this is a known ChromaDB error
            if detect_and_handle_chromadb_error(e):
                logger.warning("ChromaDB error detected and handled. The database will be recreated on next startup.")

            return MemoryResult(
                retrieved_items=[],
                similarity_scores=[],
                total_found=0,
                query_time_ms=(time.time() - start_time) * 1000
            )

    def get_by_type(self, item_type: str, limit: int = 100) -> List[MemoryItem]:
        """
        Get memory items by type.

        Args:
            item_type: The type of memory items to get
            limit: Maximum number of items to return

        Returns:
            List of memory items
        """
        if not self.collection:
            logger.error("Vector store not initialized")
            return []

        try:
            # Get items by type
            result = self.collection.get(
                where={"type": item_type},
                limit=limit,
                include=['metadatas', 'documents', 'embeddings']
            )

            # Convert to MemoryItem objects
            items = []

            if 'ids' in result and len(result['ids']) > 0:
                for i in range(len(result['ids'])):
                    item = self._convert_to_memory_item(
                        id=result['ids'][i],
                        content=result['documents'][i],
                        metadata=result['metadatas'][i],
                        embedding=result['embeddings'][i] if 'embeddings' in result else None
                    )
                    items.append(item)

            logger.debug(f"Got {len(items)} items of type {item_type}")
            return items
        except Exception as e:
            logger.error(f"Failed to get items by type {item_type}: {e}", exc_info=True)

            # Check if this is a known ChromaDB error
            if detect_and_handle_chromadb_error(e):
                logger.warning("ChromaDB error detected and handled. The database will be recreated on next startup.")

            return []

    def get_by_session(self, session_id: str, limit: int = 100) -> List[MemoryItem]:
        """
        Get memory items by session ID.

        Args:
            session_id: The session ID to get items for
            limit: Maximum number of items to return

        Returns:
            List of memory items
        """
        if not self.collection:
            logger.error("Vector store not initialized")
            return []

        try:
            # Get items by session ID
            # The session_id is stored directly in the metadata, not under metadata.session_id
            result = self.collection.get(
                where={"session_id": session_id},
                limit=limit,
                include=['metadatas', 'documents', 'embeddings']
            )

            # Convert to MemoryItem objects
            items = []

            if 'ids' in result and len(result['ids']) > 0:
                for i in range(len(result['ids'])):
                    item = self._convert_to_memory_item(
                        id=result['ids'][i],
                        content=result['documents'][i],
                        metadata=result['metadatas'][i],
                        embedding=result['embeddings'][i] if 'embeddings' in result else None
                    )
                    items.append(item)

            logger.debug(f"Got {len(items)} items for session {session_id}")
            return items
        except Exception as e:
            logger.error(f"Failed to get items by session {session_id}: {e}", exc_info=True)

            # Check if this is a known ChromaDB error
            if detect_and_handle_chromadb_error(e):
                logger.warning("ChromaDB error detected and handled. The database will be recreated on next startup.")

            return []

    def _prepare_metadata(self, item: MemoryItem) -> Dict[str, Any]:
        """
        Prepare metadata for storage.

        Args:
            item: The memory item

        Returns:
            The prepared metadata
        """
        # Create a copy of the metadata
        metadata = dict(item.metadata) if item.metadata else {}

        # Add type and timestamp
        metadata['type'] = item.type
        # Handle timestamp (could be datetime object or float)
        if hasattr(item.timestamp, 'isoformat'):
            metadata['timestamp'] = item.timestamp.isoformat()
        else:
            metadata['timestamp'] = str(item.timestamp)

        # Convert non-primitive types to strings
        for key, value in metadata.items():
            if not isinstance(value, (str, int, float, bool, type(None))):
                metadata[key] = str(value)

        return metadata

    def _prepare_filters(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare filters for search.

        Args:
            filters: The filters to prepare

        Returns:
            The prepared filters
        """
        if not filters:
            return {}

        # Create a copy of the filters
        prepared_filters = dict(filters)

        # Handle special filter keys
        if 'type' in prepared_filters:
            # Type is stored directly in the metadata
            prepared_filters['type'] = prepared_filters['type']

        # Convert non-primitive types to strings
        for key, value in prepared_filters.items():
            if not isinstance(value, (str, int, float, bool, type(None))):
                prepared_filters[key] = str(value)

        return prepared_filters

    def _convert_to_memory_item(self, id: str, content: str, metadata: Dict[str, Any],
                               embedding: Optional[List[float]]) -> MemoryItem:
        """
        Convert database result to a MemoryItem.

        Args:
            id: The item ID
            content: The item content
            metadata: The item metadata
            embedding: The item embedding

        Returns:
            The memory item
        """
        # Extract type from metadata
        item_type = metadata.pop('type', 'unknown')

        # Extract timestamp from metadata
        timestamp_str = metadata.pop('timestamp', None)
        if timestamp_str:
            try:
                timestamp = datetime.fromisoformat(timestamp_str)
            except ValueError:
                timestamp = datetime.now()
        else:
            timestamp = datetime.now()

        # Create the memory item
        return MemoryItem(
            id=id,
            timestamp=timestamp,
            type=item_type,
            content=content,
            embedding=embedding,
            metadata=metadata
        )


# Global instance
_lts_vector_store_instance = None


def setup_lts_vector_store() -> bool:
    """
    Set up the LTS vector store.

    Returns:
        True if setup was successful, False otherwise
    """
    global _lts_vector_store_instance

    try:
        if _lts_vector_store_instance is None:
            _lts_vector_store_instance = LTSVectorStore()

        return _lts_vector_store_instance.setup()
    except Exception as e:
        logger.error(f"Failed to set up LTS vector store: {e}", exc_info=True)

        # Check if this is a known ChromaDB error
        if detect_and_handle_chromadb_error(e):
            logger.warning("ChromaDB error detected and handled. The database will be recreated on next startup.")

        return False


def detect_and_handle_chromadb_error(error: Exception) -> bool:
    """
    Detect and handle ChromaDB errors.

    This function checks if an error is a known ChromaDB schema error and
    creates a flag file to trigger database recreation on next startup.

    Args:
        error: The exception to check

    Returns:
        True if the error was handled, False otherwise
    """
    error_str = str(error)

    # Check for the specific type mismatch error
    if "mismatched types; Rust type `u64` (as SQL type `INTEGER`) is not compatible with SQL type `BLOB`" in error_str:
        logger.warning("Detected ChromaDB schema mismatch error")

        try:
            # Get the database path
            db_path = config_get('memory.lts.chromadb.path', './memory_db')

            # Create a flag file to trigger database recreation
            import os
            flag_file = os.path.join(os.path.dirname(db_path), ".recreate_db")

            with open(flag_file, 'w') as f:
                f.write(f"ChromaDB schema error detected at {datetime.now().isoformat()}\n")
                f.write(f"Error: {error_str}\n")

            logger.warning(f"Created database recreation flag file: {flag_file}")
            logger.warning("The database will be recreated on next startup")

            return True
        except Exception as e:
            logger.error(f"Failed to create database recreation flag file: {e}")
            return False

    return False


def get_lts_vector_store() -> Optional[LTSVectorStore]:
    """
    Get the LTS vector store instance.

    Returns:
        The LTS vector store instance, or None if not set up
    """
    global _lts_vector_store_instance

    if _lts_vector_store_instance is None:
        logger.warning("LTS vector store not initialized")

    return _lts_vector_store_instance
