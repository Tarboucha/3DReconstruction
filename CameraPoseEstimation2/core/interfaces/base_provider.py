"""
Base interface for match data providers.

This defines the contract that all data providers must implement,
enabling dependency injection and easy swapping of data sources.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass, field


@dataclass
class ValidationResult:
    """
    Result of provider validation.
    
    Attributes:
        is_valid: Whether validation passed
        errors: List of error messages
        warnings: List of warning messages
        stats: Dictionary of validation statistics
    """
    is_valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stats: Dict[str, any] = field(default_factory=dict)
    
    def __bool__(self) -> bool:
        """Allow truthiness check"""
        return self.is_valid and len(self.errors) == 0
    
    def add_error(self, message: str):
        """Add an error message"""
        self.errors.append(message)
        self.is_valid = False
    
    def add_warning(self, message: str):
        """Add a warning message"""
        self.warnings.append(message)
    
    def print_report(self):
        """Print validation report"""
        if self.is_valid and not self.warnings:
            print("✅ Validation passed")
            return
        
        if self.errors:
            print("\n❌ ERRORS:")
            for error in self.errors:
                print(f"  - {error}")
        
        if self.warnings:
            print("\n⚠️  WARNINGS:")
            for warning in self.warnings:
                print(f"  - {warning}")
        
        if self.stats:
            print("\n📊 STATISTICS:")
            for key, value in self.stats.items():
                print(f"  {key}: {value}")


class IMatchDataProvider(ABC):
    """
    Abstract interface for match data providers.
    
    This interface defines the contract for accessing match data,
    allowing different implementations (file-based, database, mock, etc.)
    while maintaining a consistent API.
    
    Design Principles:
    - Dependency Injection: Components receive providers, don't create them
    - Interface Segregation: Only essential methods in the interface
    - Open/Closed: Open for extension (new providers), closed for modification
    """
    
    # ========================================================================
    # CORE DATA ACCESS METHODS (Required)
    # ========================================================================
    
    @abstractmethod
    def get_match_data(self, pair: Tuple[str, str]) -> 'StructuredMatchData':
        """
        Get match data for an image pair.
        
        Args:
            pair: Tuple of (image1, image2) identifiers
        
        Returns:
            StructuredMatchData: Match information between the images
        
        Raises:
            KeyError: If pair not found
            ValueError: If pair is invalid
        """
        pass
    
    @abstractmethod
    def get_image_info(self, image_id: str) -> dict:
        """
        Get metadata for a specific image.
        
        Args:
            image_id: Image identifier
        
        Returns:
            dict: Image metadata containing at minimum:
                - 'size': (width, height) tuple
                - 'path': str (optional)
                - Other metadata as available
        
        Raises:
            KeyError: If image not found
        """
        pass
    
    @abstractmethod
    def has_pair(self, pair: Tuple[str, str]) -> bool:
        """
        Check if a pair exists in the dataset.
        
        Args:
            pair: Tuple of (image1, image2) identifiers
        
        Returns:
            bool: True if pair exists
        """
        pass
    
    @abstractmethod
    def get_all_images(self) -> List[str]:
        """
        Get list of all image identifiers.
        
        Returns:
            List[str]: List of image IDs
        """
        pass
    
    @abstractmethod
    def get_all_pairs(self) -> List[Tuple[str, str]]:
        """
        Get list of all image pairs.
        
        Returns:
            List[Tuple[str, str]]: List of image pair tuples
        """
        pass
    
    # ========================================================================
    # QUERY METHODS (Required)
    # ========================================================================
    
    @abstractmethod
    def get_image_count(self) -> int:
        """
        Get total number of images.
        
        Returns:
            int: Number of images
        """
        pass
    
    @abstractmethod
    def get_pair_count(self) -> int:
        """
        Get total number of image pairs.
        
        Returns:
            int: Number of pairs
        """
        pass
    
    @abstractmethod
    def get_pairs_for_image(self, image_id: str) -> List[Tuple[str, str]]:
        """
        Get all pairs that include a specific image.
        
        Args:
            image_id: Image identifier
        
        Returns:
            List[Tuple[str, str]]: List of pairs containing this image
        """
        pass
    
    # ========================================================================
    # FILTERING & SELECTION METHODS (Required)
    # ========================================================================
    
    @abstractmethod
    def filter_pairs(self, 
                    min_matches: Optional[int] = None,
                    min_quality: Optional[float] = None,
                    max_quality: Optional[float] = None,
                    images: Optional[Set[str]] = None) -> Dict[Tuple[str, str], 'StructuredMatchData']:
        """
        Filter pairs based on criteria.
        
        Args:
            min_matches: Minimum number of matches required
            min_quality: Minimum quality score (0-1)
            max_quality: Maximum quality score (0-1)
            images: Only include pairs with these images
        
        Returns:
            Dict: Filtered pairs mapping (img1, img2) -> StructuredMatchData
        """
        pass
    
    @abstractmethod
    def get_best_pairs(self, 
                      k: int = 10,
                      criterion: str = 'quality') -> List[Tuple[str, str]]:
        """
        Get top k pairs by specified criterion.
        
        Args:
            k: Number of pairs to return
            criterion: Sorting criterion ('quality', 'matches', 'both')
        
        Returns:
            List[Tuple[str, str]]: Top k pairs
        """
        pass
    
    # ========================================================================
    # VALIDATION & METADATA (Required)
    # ========================================================================
    
    @abstractmethod
    def validate(self) -> ValidationResult:
        """
        Validate the provider's data integrity.
        
        Returns:
            ValidationResult: Validation result with errors/warnings
        """
        pass
    
    @abstractmethod
    def get_statistics(self) -> dict:
        """
        Get dataset statistics.
        
        Returns:
            dict: Statistics including:
                - 'num_images': int
                - 'num_pairs': int
                - 'avg_matches_per_pair': float
                - 'quality_distribution': dict
                - Other relevant stats
        """
        pass
    
    # ========================================================================
    # UTILITY METHODS (Optional but recommended)
    # ========================================================================
    
    def print_summary(self):
        """
        Print a human-readable summary of the dataset.
        
        This is a convenience method with a default implementation.
        Providers can override for custom formatting.
        """
        stats = self.get_statistics()
        print("="*60)
        print("MATCH DATA PROVIDER SUMMARY")
        print("="*60)
        print(f"Images: {stats.get('num_images', 'N/A')}")
        print(f"Pairs: {stats.get('num_pairs', 'N/A')}")
        print(f"Avg matches/pair: {stats.get('avg_matches_per_pair', 'N/A'):.1f}")
        
        if 'quality_distribution' in stats:
            print("\nQuality Distribution:")
            for key, value in stats['quality_distribution'].items():
                print(f"  {key}: {value}")
        print("="*60)
    
    def get_connected_component(self, start_image: str, min_quality: float = 0.5) -> Set[str]:
        """
        Get connected component of images starting from a seed image.
        
        Useful for finding images that can be reconstructed together.
        
        Args:
            start_image: Starting image ID
            min_quality: Minimum quality threshold for connections
        
        Returns:
            Set[str]: Set of connected image IDs
        """
        visited = set()
        to_visit = [start_image]
        
        while to_visit:
            current = to_visit.pop()
            if current in visited:
                continue
            
            visited.add(current)
            
            # Get pairs for this image
            pairs = self.get_pairs_for_image(current)
            
            for pair in pairs:
                match_data = self.get_match_data(pair)
                
                # Only follow high-quality connections
                if match_data.standardized_pair_quality >= min_quality:
                    other_image = pair[1] if pair[0] == current else pair[0]
                    if other_image not in visited:
                        to_visit.append(other_image)
        
        return visited
    
    # ========================================================================
    # CONTEXT MANAGER SUPPORT (Optional)
    # ========================================================================
    
    def __enter__(self):
        """Support context manager protocol"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Support context manager protocol"""
        # Default: do nothing
        # Override in subclasses if cleanup needed
        pass