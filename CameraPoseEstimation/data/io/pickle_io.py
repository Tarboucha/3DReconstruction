import pickle
from pathlib import Path
from typing import Any, Optional


class PickleIO:
    """Safe pickle file operations with error handling"""
    
    @staticmethod
    def read(filepath: str) -> Any:
        """
        Safely read pickle file.
        
        Args:
            filepath: Path to pickle file
        
        Returns:
            Unpickled data
        
        Raises:
            FileNotFoundError: If file doesn't exist
            pickle.UnpicklingError: If file is corrupted
        """
        path = Path(filepath)
        
        if not path.exists():
            raise FileNotFoundError(f"Pickle file not found: {filepath}")
        
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except pickle.UnpicklingError as e:
            raise pickle.UnpicklingError(f"Corrupted pickle file: {filepath}") from e
    
    @staticmethod
    def write(data: Any, filepath: str):
        """
        Safely write pickle file.
        
        Args:
            data: Data to pickle
            filepath: Output path
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    @staticmethod
    def exists(filepath: str) -> bool:
        """Check if pickle file exists"""
        return Path(filepath).exists()