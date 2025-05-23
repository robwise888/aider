"""
Working memory for the Selfy agent.

This module implements a simple in-memory cache for recently accessed memory items.
It provides fast access to frequently used items and serves as a buffer before
items are stored in long-term storage.
"""

import time
from collections import OrderedDict
from typing import Dict, List, Optional, Any

from selfy_core.global_modules.config import get as config_get
from selfy_core.global_modules.logging import get_logger
from selfy_core.global_modules.memory.interfaces import MemoryItem

# Set up logger
logger = get_logger(__name__)


class WorkingMemory:
    """
    A simple in-memory cache for recently accessed memory items.
    
    This class implements a cache with configurable size and eviction policy.
    It supports FIFO (First In, First Out) and LRU (Least Recently Used) policies.
    """
    
    def __init__(self):
        """Initialize the working memory."""
        self.max_size = config_get('memory.working_memory.max_size', 100)
        self.eviction_policy = config_get('memory.working_memory.eviction_policy', 'FIFO')
        
        # Use OrderedDict to maintain insertion order for FIFO
        # and access order for LRU
        self.items: OrderedDict[str, MemoryItem] = OrderedDict()
        
        # Track access times for LRU
        self.access_times: Dict[str, float] = {}
        
        logger.info(f"Initialized WorkingMemory with max_size={self.max_size}, "
                   f"eviction_policy={self.eviction_policy}")
    
    def setup(self) -> bool:
        """
        Set up the working memory.
        
        Returns:
            True if setup was successful, False otherwise
        """
        try:
            # Reload configuration in case it changed
            self.max_size = config_get('memory.working_memory.max_size', 100)
            self.eviction_policy = config_get('memory.working_memory.eviction_policy', 'FIFO')
            
            logger.info(f"Set up WorkingMemory with max_size={self.max_size}, "
                       f"eviction_policy={self.eviction_policy}")
            return True
        except Exception as e:
            logger.error(f"Failed to set up WorkingMemory: {e}", exc_info=True)
            return False
    
    def add_or_update(self, item: MemoryItem) -> None:
        """
        Add or update a memory item.
        
        If the item already exists, it will be updated. If the cache is full,
        an item will be evicted according to the eviction policy.
        
        Args:
            item: The memory item to add or update
        """
        # Check if we need to evict an item
        if len(self.items) >= self.max_size and item.id not in self.items:
            self._evict_item()
        
        # Add or update the item
        self.items[item.id] = item
        self.access_times[item.id] = time.time()
        
        # If using LRU, move the item to the end of the OrderedDict
        if self.eviction_policy == 'LRU':
            self.items.move_to_end(item.id)
    
    def get_by_id(self, item_id: str) -> Optional[MemoryItem]:
        """
        Get a memory item by its ID.
        
        Args:
            item_id: The ID of the memory item to get
            
        Returns:
            The memory item, or None if not found
        """
        item = self.items.get(item_id)
        
        if item:
            # Update access time
            self.access_times[item_id] = time.time()
            
            # If using LRU, move the item to the end of the OrderedDict
            if self.eviction_policy == 'LRU':
                self.items.move_to_end(item_id)
        
        return item
    
    def delete(self, item_id: str) -> bool:
        """
        Delete a memory item by its ID.
        
        Args:
            item_id: The ID of the memory item to delete
            
        Returns:
            True if the item was deleted, False otherwise
        """
        if item_id in self.items:
            del self.items[item_id]
            
            if item_id in self.access_times:
                del self.access_times[item_id]
            
            return True
        
        return False
    
    def clear(self) -> None:
        """
        Clear all items from the working memory.
        """
        self.items.clear()
        self.access_times.clear()
        logger.info("Cleared working memory")
    
    def get_all_items(self) -> List[MemoryItem]:
        """
        Get all items in the working memory.
        
        Returns:
            List of all memory items
        """
        return list(self.items.values())
    
    def get_item_count(self) -> int:
        """
        Get the number of items in the working memory.
        
        Returns:
            Number of items
        """
        return len(self.items)
    
    def _evict_item(self) -> None:
        """
        Evict an item from the working memory.
        
        The item to evict is determined by the eviction policy.
        """
        if not self.items:
            return
        
        if self.eviction_policy == 'FIFO':
            # Evict the first item (oldest)
            item_id, _ = next(iter(self.items.items()))
            self.delete(item_id)
            logger.debug(f"Evicted item {item_id} from working memory (FIFO)")
        elif self.eviction_policy == 'LRU':
            # Evict the least recently used item
            item_id, _ = next(iter(self.items.items()))
            self.delete(item_id)
            logger.debug(f"Evicted item {item_id} from working memory (LRU)")
        else:
            # Unknown policy, default to FIFO
            item_id, _ = next(iter(self.items.items()))
            self.delete(item_id)
            logger.debug(f"Evicted item {item_id} from working memory (unknown policy)")


# Global instance
_working_memory_instance = None


def setup_working_memory() -> bool:
    """
    Set up the working memory.
    
    Returns:
        True if setup was successful, False otherwise
    """
    global _working_memory_instance
    
    try:
        if _working_memory_instance is None:
            _working_memory_instance = WorkingMemory()
        
        return _working_memory_instance.setup()
    except Exception as e:
        logger.error(f"Failed to set up working memory: {e}", exc_info=True)
        return False


def get_working_memory() -> Optional[WorkingMemory]:
    """
    Get the working memory instance.
    
    Returns:
        The working memory instance, or None if not set up
    """
    global _working_memory_instance
    
    if _working_memory_instance is None:
        logger.warning("Working memory not initialized")
    
    return _working_memory_instance
