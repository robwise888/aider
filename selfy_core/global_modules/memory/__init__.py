"""
Memory module for the Selfy agent.

This package contains the memory-related components for the Selfy agent,
including the UnifiedMemorySystem interface, working memory, embedding service,
and long-term storage.
"""

# Import key components for easy access
from selfy_core.global_modules.memory.interfaces import (
    MemoryItem, MemoryQuery, MemoryResult, UnifiedMemorySystem, MemoryItemType
)
from selfy_core.global_modules.memory.core import (
    setup_memory_core, get_memory_system
)

__all__ = [
    'MemoryItem',
    'MemoryQuery',
    'MemoryResult',
    'UnifiedMemorySystem',
    'MemoryItemType',
    'setup_memory_core',
    'get_memory_system'
]