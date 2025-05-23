"""
Self-indexing manager for Aider.

This module provides functionality for Aider to index its own codebase
and make it available for RAG operations.
"""

import os
import time
import logging
import glob
from pathlib import Path
from datetime import datetime

from .self_indexing_config import DEFAULT_CONFIG, AIDER_ROOT

logger = logging.getLogger(__name__)

class SelfIndexingManager:
    """Manager for Aider's self-indexing capability."""

    def __init__(
        self,
        vector_db_manager=None,
        config=None,
        verbose=False,
    ):
        """
        Initialize the self-indexing manager.

        Args:
            vector_db_manager: Vector database manager
            config: Configuration dictionary
            verbose: Whether to enable verbose logging
        """
        print("SELF-INDEXING: Initializing self-indexing manager")
        self.config = config or DEFAULT_CONFIG
        self.verbose = verbose
        self.vector_db_manager = vector_db_manager
        self.collection_name = self.config.get("collection_name", "aider_self_reference")
        self.enabled = self.config.get("enabled", True)
        self.aider_root = AIDER_ROOT

        print(f"SELF-INDEXING: Configuration: {self.config}")
        print(f"SELF-INDEXING: Enabled: {self.enabled}")
        print(f"SELF-INDEXING: Collection name: {self.collection_name}")
        print(f"SELF-INDEXING: Aider root: {self.aider_root}")
        print(f"SELF-INDEXING: Vector DB manager available: {self.vector_db_manager is not None}")

        # Last indexing timestamp file
        if self.vector_db_manager and hasattr(self.vector_db_manager, 'db_path'):
            self.timestamp_file = Path(self.vector_db_manager.db_path) / ".aider_self_index_timestamp"
            print(f"SELF-INDEXING: Using timestamp file in vector DB path: {self.timestamp_file}")
        else:
            # Fallback to a location in the aider root
            self.timestamp_file = self.aider_root / ".aider_self_index_timestamp"
            print(f"SELF-INDEXING: Using fallback timestamp file in Aider root: {self.timestamp_file}")

        if self.verbose:
            logger.info(f"Self-indexing manager initialized with root: {self.aider_root}")

        print("SELF-INDEXING: Initialization complete")

    def should_refresh_index(self):
        """Check if the index should be refreshed."""
        print(f"SELF-INDEXING: Checking if index should be refreshed (enabled={self.enabled})")

        if not self.enabled:
            print("SELF-INDEXING: Self-indexing is disabled, no refresh needed")
            return False

        if not self.timestamp_file.exists():
            print(f"SELF-INDEXING: Timestamp file {self.timestamp_file} does not exist, refresh needed")
            return True

        # Read the timestamp
        try:
            with open(self.timestamp_file, "r") as f:
                timestamp_str = f.read().strip()
                last_indexed = datetime.fromisoformat(timestamp_str)

                # Check if refresh interval has passed
                refresh_hours = self.config.get("refresh_interval_hours", 1)
                hours_since_last_index = (datetime.now() - last_indexed).total_seconds() / 3600
                refresh_needed = hours_since_last_index > refresh_hours

                print(f"SELF-INDEXING: Last indexed on {last_indexed.isoformat()}, {hours_since_last_index:.2f} hours ago")
                print(f"SELF-INDEXING: Refresh interval is {refresh_hours} hours")
                print(f"SELF-INDEXING: Refresh needed: {refresh_needed}")

                return refresh_needed
        except (ValueError, IOError) as e:
            print(f"SELF-INDEXING: Error reading timestamp file: {e}, refresh needed")
            logger.warning(f"Error reading timestamp file: {e}")
            return True

    def update_timestamp(self):
        """Update the timestamp file."""
        try:
            # Ensure parent directory exists
            self.timestamp_file.parent.mkdir(parents=True, exist_ok=True)

            with open(self.timestamp_file, "w") as f:
                f.write(datetime.now().isoformat())
        except IOError as e:
            logger.warning(f"Error writing timestamp file: {e}")

    def index_aider_codebase(self, force=False):
        """
        Index Aider's own codebase.

        Args:
            force: Whether to force reindexing even if not needed

        Returns:
            Number of documents indexed
        """
        print(f"SELF-INDEXING: Checking if self-indexing is needed (enabled={self.enabled}, force={force})")

        if not self.enabled:
            print("SELF-INDEXING: Self-indexing is disabled")
            if self.verbose:
                logger.info("Self-indexing is disabled")
            return 0

        if not force and not self.should_refresh_index():
            print("SELF-INDEXING: Self-index is up to date, skipping")
            if self.verbose:
                logger.info("Self-index is up to date, skipping")
            return 0

        print(f"SELF-INDEXING: Starting to index Aider codebase at {self.aider_root}")
        if self.verbose:
            logger.info(f"Indexing Aider codebase at {self.aider_root}")
            start_time = time.time()

        # Prepare the vector database
        if self.vector_db_manager:
            # Use create_collection if available, otherwise try create_or_get_collection, or just use the collection directly
            if hasattr(self.vector_db_manager, 'create_collection'):
                print(f"SELF-INDEXING: Creating collection: {self.collection_name}")
                self.vector_db_manager.create_collection(self.collection_name)
            elif hasattr(self.vector_db_manager, 'create_or_get_collection'):
                print(f"SELF-INDEXING: Creating or getting collection: {self.collection_name}")
                self.vector_db_manager.create_or_get_collection(self.collection_name)
            else:
                # The collection might already exist or be created automatically when adding documents
                print(f"SELF-INDEXING: Using existing collection or will create on first document: {self.collection_name}")

            # Try to update collection metadata using a file-based approach instead of directly setting it
            try:
                if hasattr(self.vector_db_manager, 'update_collection_metadata'):
                    print(f"SELF-INDEXING: Updating collection metadata for: {self.collection_name}")
                    try:
                        self.vector_db_manager.update_collection_metadata(
                            collection_name=self.collection_name,
                            metadata={"description": "Aider self-reference collection", "last_updated": datetime.now().isoformat()}
                        )
                    except Exception as metadata_error:
                        # Log the error but continue with initialization
                        print(f"SELF-INDEXING: Could not update collection metadata: {metadata_error}")
                        logger.warning(f"Could not update collection metadata: {metadata_error}")
                        # This is a non-critical error, so we can continue
            except Exception as e:
                print(f"SELF-INDEXING: Error during metadata handling: {e}")
                logger.warning(f"Error during metadata handling: {e}")
        else:
            logger.warning("No vector database manager available, skipping indexing")
            return 0

        # Get files to index
        files_to_index = self._get_files_to_index()

        if self.verbose:
            logger.info(f"Found {len(files_to_index)} files to index")

        # Index the files
        num_indexed = self._index_files(files_to_index)

        # Update timestamp
        self.update_timestamp()

        if self.verbose:
            duration = time.time() - start_time
            logger.info(f"Indexed {num_indexed} documents in {duration:.2f}s")

        return num_indexed

    def _get_files_to_index(self):
        """Get the list of files to index."""
        print("SELF-INDEXING: Getting files to index")
        files_to_index = []
        exclude_patterns = self.config.get("exclude_paths", [])
        include_patterns = self.config.get("include_paths", [])

        print(f"SELF-INDEXING: Exclude patterns: {exclude_patterns}")
        print(f"SELF-INDEXING: Include patterns: {include_patterns}")
        print(f"SELF-INDEXING: Scanning directory: {self.aider_root}")

        total_files = 0
        excluded_files = 0
        included_files = 0

        # Walk through the directory
        for root, dirs, files in os.walk(self.aider_root):
            # Skip excluded directories
            original_dirs = list(dirs)
            dirs[:] = [d for d in dirs if not any(
                glob.fnmatch.fnmatch(d, pattern) for pattern in exclude_patterns
            )]
            skipped_dirs = set(original_dirs) - set(dirs)
            if skipped_dirs and self.verbose:
                print(f"SELF-INDEXING: Skipping directories in {root}: {skipped_dirs}")

            for file in files:
                total_files += 1
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, self.aider_root)

                # Skip excluded files
                if any(glob.fnmatch.fnmatch(rel_path, pattern) for pattern in exclude_patterns):
                    # But include if it matches an include pattern
                    if not any(glob.fnmatch.fnmatch(rel_path, pattern) for pattern in include_patterns):
                        excluded_files += 1
                        continue

                files_to_index.append(file_path)
                included_files += 1
                if self.verbose:
                    print(f"SELF-INDEXING: Including file: {rel_path}")

        print(f"SELF-INDEXING: Found {total_files} total files")
        print(f"SELF-INDEXING: Excluded {excluded_files} files")
        print(f"SELF-INDEXING: Including {included_files} files for indexing")

        return files_to_index

    def _index_files(self, files):
        """Index the files."""
        print("SELF-INDEXING: Starting to index files")
        num_indexed = 0
        chunk_size = self.config.get("chunk_size", 1000)
        chunk_overlap = self.config.get("chunk_overlap", 200)

        print(f"SELF-INDEXING: Using chunk size: {chunk_size}, overlap: {chunk_overlap}")
        print(f"SELF-INDEXING: Total files to process: {len(files)}")

        files_processed = 0
        files_skipped = 0
        total_chunks = 0

        for file_path in files:
            files_processed += 1
            if files_processed % 10 == 0:
                print(f"SELF-INDEXING: Processed {files_processed}/{len(files)} files...")

            try:
                # Read the file
                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()

                # Skip empty files
                if not content.strip():
                    files_skipped += 1
                    if self.verbose:
                        print(f"SELF-INDEXING: Skipping empty file: {file_path}")
                    continue

                # Get relative path for metadata
                rel_path = os.path.relpath(file_path, self.aider_root)

                # Chunk the file
                chunks = self._chunk_file(
                    content,
                    file_path,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )

                total_chunks += len(chunks)
                if self.verbose:
                    print(f"SELF-INDEXING: File {rel_path} split into {len(chunks)} chunks")

                # Add chunks to vector database
                for i, chunk in enumerate(chunks):
                    # Check which method is available and use the appropriate one
                    if hasattr(self.vector_db_manager, 'add_document'):
                        self.vector_db_manager.add_document(
                            collection_name=self.collection_name,
                            document_id=f"aider:{rel_path}:{i}",
                            content=chunk["content"],
                            metadata={
                                "file_path": rel_path,
                                "chunk_index": i,
                                "total_chunks": len(chunks),
                                "file_type": os.path.splitext(file_path)[1],
                                "document_type": "code",
                            }
                        )
                    else:
                        # Fallback to using add_file if add_document is not available
                        print(f"SELF-INDEXING: Using add_file fallback for {rel_path}")
                        # Create a temporary file with the chunk content
                        temp_file_path = f"{rel_path}.chunk{i}"
                        with open(temp_file_path, "w", encoding="utf-8") as temp_f:
                            temp_f.write(chunk["content"])

                        try:
                            # Add the file to the vector database
                            self.vector_db_manager.add_file(temp_file_path, chunk["content"])
                        finally:
                            # Clean up the temporary file
                            if os.path.exists(temp_file_path):
                                os.remove(temp_file_path)

                    num_indexed += 1

            except Exception as e:
                print(f"SELF-INDEXING: Error indexing file {file_path}: {e}")
                logger.warning(f"Error indexing file {file_path}: {e}")

        print(f"SELF-INDEXING: Indexing complete")
        print(f"SELF-INDEXING: Files processed: {files_processed}")
        print(f"SELF-INDEXING: Files skipped: {files_skipped}")
        print(f"SELF-INDEXING: Total chunks created: {total_chunks}")
        print(f"SELF-INDEXING: Total documents indexed: {num_indexed}")

        return num_indexed

    def _chunk_file(self, content, file_path, chunk_size=1000, chunk_overlap=200):
        """
        Chunk a file into smaller pieces for indexing.

        This is a simple implementation that splits by lines.
        A more sophisticated implementation would use a proper tokenizer.
        """
        # Simple chunking by lines
        lines = content.splitlines()
        chunks = []

        # Estimate average tokens per line (very rough approximation)
        avg_tokens_per_line = 10
        lines_per_chunk = max(1, chunk_size // avg_tokens_per_line)
        overlap_lines = max(1, chunk_overlap // avg_tokens_per_line)

        i = 0
        while i < len(lines):
            end = min(i + lines_per_chunk, len(lines))
            chunk_content = "\n".join(lines[i:end])

            chunks.append({
                "content": chunk_content,
                "start_line": i,
                "end_line": end - 1,
                "file_path": file_path,
            })

            # Move to next chunk with overlap
            i = end - overlap_lines if end < len(lines) else end

        return chunks

    def query_aider_knowledge(self, query, limit=5):
        """
        Query Aider's self-knowledge.

        Args:
            query: The query string
            limit: Maximum number of results to return

        Returns:
            List of relevant documents
        """
        print(f"SELF-INDEXING: Querying Aider knowledge with: '{query}'")

        if not self.enabled or not self.vector_db_manager:
            print(f"SELF-INDEXING: Cannot query - enabled={self.enabled}, vector_db_available={self.vector_db_manager is not None}")
            return []

        # Ensure the index exists
        if self.should_refresh_index():
            print("SELF-INDEXING: Index needs refresh before querying")
            self.index_aider_codebase()

        # Query the vector database
        print(f"SELF-INDEXING: Querying collection '{self.collection_name}' with limit={limit}")
        try:
            # Check which query method is available
            if hasattr(self.vector_db_manager, 'query') and callable(getattr(self.vector_db_manager, 'query')):
                # Check the signature of the query method
                import inspect
                sig = inspect.signature(self.vector_db_manager.query)
                params = list(sig.parameters.keys())

                if 'collection_name' in params:
                    # New API with collection_name parameter
                    results = self.vector_db_manager.query(
                        query=query,
                        collection_name=self.collection_name,
                        limit=limit
                    )
                elif 'collection' in params:
                    # Old API with collection parameter
                    results = self.vector_db_manager.query(
                        collection=self.collection_name,
                        query=query,
                        limit=limit
                    )
                else:
                    # Default to search method if query method doesn't have collection parameter
                    print(f"SELF-INDEXING: Using search method instead of query")
                    results = self.vector_db_manager.search(query, limit=limit)
            else:
                # Fallback to search method
                print(f"SELF-INDEXING: Using search method as fallback")
                results = self.vector_db_manager.search(query, limit=limit)

            print(f"SELF-INDEXING: Query returned {len(results)} results")
            if self.verbose and results:
                for i, result in enumerate(results):
                    print(f"SELF-INDEXING: Result {i+1}: score={result.get('score', 0)}, file={result.get('metadata', {}).get('file_path', 'unknown')}")

            return results
        except Exception as e:
            print(f"SELF-INDEXING: Error querying vector database: {e}")
            logger.error(f"Error querying vector database: {e}")
            return []
