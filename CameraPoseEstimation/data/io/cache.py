from collections import OrderedDict
from typing import Callable, Optional, Any, Dict


class LRUCache:
    """
    Simple LRU (Least Recently Used) cache.
    
    Usage:
        cache = LRUCache(max_size=100)
        
        # Store
        cache.put('key', value)
        
        # Retrieve
        value = cache.get('key')
        
        # Or use as decorator
        @cache.cached
        def expensive_function(arg):
            return compute(arg)
    """
    
    def __init__(self, max_size: int = 100):
        """
        Initialize cache.
        
        Args:
            max_size: Maximum number of items to cache
        """
        self.max_size = max_size
        self.cache = OrderedDict()
        self.hits = 0
        self.misses = 0
    
    def get(self, key: Any) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
        
        Returns:
            Cached value or None if not found
        """
        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            self.hits += 1
            return self.cache[key]
        
        self.misses += 1
        return None
    
    def put(self, key: Any, value: Any):
        """
        Put value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        if key in self.cache:
            # Update existing
            self.cache.move_to_end(key)
        else:
            # Add new
            if len(self.cache) >= self.max_size:
                # Remove least recently used
                self.cache.popitem(last=False)
        
        self.cache[key] = value
    
    def clear(self):
        """Clear cache"""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate
        }
    
    def cached(self, func: Callable) -> Callable:
        """
        Decorator to cache function results.
        
        Usage:
            cache = LRUCache(100)
            
            @cache.cached
            def expensive_function(arg):
                return compute(arg)
        """
        def wrapper(*args, **kwargs):
            # Create cache key from args
            key = (args, tuple(sorted(kwargs.items())))
            
            # Try cache
            result = self.get(key)
            if result is not None:
                return result
            
            # Compute and cache
            result = func(*args, **kwargs)
            self.put(key, result)
            return result
        
        return wrapper
