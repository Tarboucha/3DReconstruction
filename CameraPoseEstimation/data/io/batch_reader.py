from typing import List, Dict, Iterator, Tuple , Optional
import os
from pathlib import Path
from .pickle_io import PickleIO

class BatchReader:
    """
    Efficient reader for batch files.
    
    Features:
    - Lazy loading (only load when needed)
    - Memory efficient (don't load everything at once)
    - Caching support
    """
    
    def __init__(self, batch_dir: str):
        """
        Initialize batch reader.
        
        Args:
            batch_dir: Directory containing batch files
        """
        self.batch_dir = Path(batch_dir)
        self._batch_files = None
    
    def list_batch_files(self) -> List[str]:
        """
        List all batch files in directory.
        
        Returns:
            List of batch file paths
        """
        if self._batch_files is None:
            self._batch_files = sorted([
                str(f) for f in self.batch_dir.glob('batch_*.pkl')
            ])
        return self._batch_files
    
    def read_batch(self, batch_file: str) -> Dict:
        """
        Read a single batch file.
        
        Args:
            batch_file: Path to batch file
        
        Returns:
            Batch data dictionary
        """
        return PickleIO.read(batch_file)
    
    def iter_batches(self) -> Iterator[Tuple[str, Dict]]:
        """
        Iterate over all batch files lazily.
        
        Yields:
            Tuple[str, Dict]: (batch_file_path, batch_data)
        """
        for batch_file in self.list_batch_files():
            yield batch_file, self.read_batch(batch_file)
    
    def get_batch_for_pair(self, pair: Tuple[str, str]) -> Optional[str]:
        """
        Find which batch file contains a specific pair.
        
        Args:
            pair: Image pair tuple
        
        Returns:
            Batch file path or None if not found
        """
        # This is a simple linear search
        # Could be optimized with an index
        for batch_file, batch_data in self.iter_batches():
            if pair in batch_data.get('matches_data', {}):
                return batch_file
        return None