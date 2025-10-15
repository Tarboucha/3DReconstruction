"""
I/O utilities for data access.

Components:
    - PickleIO: Safe pickle operations
    - BatchReader: Efficient batch file reading
    - LRUCache: Least Recently Used cache

Usage:
    from data.io import PickleIO, BatchReader, LRUCache
    
    # Read pickle safely
    data = PickleIO.read('batch_0.pkl')
    
    # Read batches efficiently
    reader = BatchReader('./results/')
    for batch_file, batch_data in reader.iter_batches():
        process(batch_data)
    
    # Cache expensive operations
    cache = LRUCache(max_size=100)
    value = cache.get('key')
    if value is None:
        value = expensive_computation()
        cache.put('key', value)
"""

from .pickle_io import PickleIO
from .batch_reader import BatchReader
from .cache import LRUCache

__all__ = [
    'PickleIO',
    'BatchReader',
    'LRUCache',
]
