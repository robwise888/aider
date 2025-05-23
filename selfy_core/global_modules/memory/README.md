# Memory Module

This module provides memory capabilities for the Selfy agent. It implements a unified memory system that combines working memory (for fast access to recent items) and long-term storage (for persistent storage and semantic search).

## Components

- **Memory Interface**: Defines the standard data types and interfaces for memory operations
- **Memory Core**: Implements the UnifiedMemorySystem interface and orchestrates the other components
- **Working Memory**: Provides a simple in-memory cache for recently accessed memory items
- **Embedding Service**: Generates vector embeddings from text for semantic search
- **LTS Vector Store**: Provides persistent storage and semantic search capabilities

## Usage

### Setup

```python
from selfy_core.global_modules.memory import setup_memory_core, get_memory_system

# Set up the memory system
setup_memory_core()

# Get the memory system instance
memory_system = get_memory_system()
```

### Adding Memory Items

```python
from selfy_core.global_modules.memory import MemoryItem, MemoryItemType
from datetime import datetime
import uuid

# Create a memory item
item = MemoryItem(
    id=str(uuid.uuid4()),
    timestamp=datetime.now(),
    type=MemoryItemType.CONVERSATION_TURN,
    content="User: What is the capital of France?",
    metadata={
        "session_id": "session123",
        "user_id": "user456",
        "role": "user"
    }
)

# Add the item to memory
memory_system.add_memory(item)
```

### Querying Memory

```python
from selfy_core.global_modules.memory import MemoryQuery

# Create a query
query = MemoryQuery(
    query_text="Why is the sky blue?",
    filters={"type": MemoryItemType.LEARNED_FACT},
    top_k=3
)

# Execute the query
results = memory_system.query_memory(query)

# Process the results
for i, item in enumerate(results.retrieved_items):
    print(f"{i+1}. {item.content} (Score: {results.similarity_scores[i]:.2f})")
```

### Retrieving Conversation History

```python
# Get recent conversation turns for a session
conversation_history = memory_system.get_recent_conversation(
    session_id="session123",
    last_n=10
)

# Process the conversation history
for turn in conversation_history:
    print(f"{turn.timestamp}: {turn.content}")
```

### Storing Capability Information

```python
# Store capability information
capability = {
    "name": "weather_lookup",
    "description": "Look up the weather for a location",
    "parameters": {
        "location": {
            "type": "string",
            "description": "The location to look up weather for"
        }
    }
}

memory_system.store_capability_info(capability)
```

### Logging Capability Usage

```python
# Log capability usage
memory_system.log_capability_usage(
    capability_name="weather_lookup",
    parameters={"location": "Paris"},
    result={"temperature": 22, "conditions": "Sunny"},
    success=True,
    session_id="session123",
    user_id="user456"
)
```

### Finding Relevant Capabilities

```python
# Find capabilities relevant to a query
capabilities = memory_system.find_relevant_capabilities(
    query_text="What's the weather like in Paris?",
    top_k=3
)

# Process the capabilities
for capability in capabilities:
    print(f"{capability['name']}: {capability['description']}")
```

## Configuration

The memory module can be configured through the configuration system. The following configuration options are available:

- `memory.working_memory.max_size`: Maximum number of items in working memory (default: 100)
- `memory.working_memory.eviction_policy`: Eviction policy for working memory (default: "FIFO")
- `memory.lts.vector_db_type`: Type of vector database to use (default: "chromadb")
- `memory.lts.chromadb.path`: Path to the ChromaDB database (default: "./memory_db")
- `memory.lts.chromadb.collection_name`: Name of the ChromaDB collection (default: "selfy_memory")
- `memory.embedding_service.model`: Embedding model to use (default: "all-MiniLM-L6-v2")
- `memory.embedding_service.provider`: Embedding provider to use (default: "sentence_transformers")
- `memory.embedding_service.dimension`: Dimension of the embeddings (default: 384)
- `memory.embedding_service.use_cache`: Whether to cache embeddings (default: true)

## Dependencies

- **ChromaDB**: For vector storage and semantic search
- **SentenceTransformers**: For generating embeddings (optional, falls back to LLM if not available)
- **LLM Wrapper**: For generating embeddings when SentenceTransformers is not available
