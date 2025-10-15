"""
Mock provider for testing and prototyping.

File: CameraPoseEstimation/data_providers/mock_provider.py
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Callable
from itertools import combinations

from core.interfaces import IMatchDataProvider, ValidationResult
from core.structures import StructuredMatchData, EnhancedDMatch, ScoreType


class MockProvider(IMatchDataProvider):
    """
    Mock data provider for testing and prototyping.
    
    Generates synthetic but valid match data without requiring actual images.
    Useful for:
    - Unit testing
    - Algorithm development
    - Performance benchmarking
    - Edge case testing
    """
    
    def __init__(self,
                 num_images: int = 10,
                 pairs_per_image: int = 5,
                 matches_per_pair: int = 150,
                 quality_range: Tuple[float, float] = (0.5, 0.95),
                 image_size: Tuple[int, int] = (640, 480),
                 seed: Optional[int] = None):
        """
        Initialize mock provider.
        
        Args:
            num_images: Number of images to simulate
            pairs_per_image: Average pairs per image
            matches_per_pair: Average matches per pair
            quality_range: (min, max) quality scores
            image_size: Simulated image dimensions
            seed: Random seed for reproducibility
        """
        self.num_images = num_images
        self.pairs_per_image = pairs_per_image
        self.matches_per_pair = matches_per_pair
        self.quality_range = quality_range
        self.image_size = image_size
        
        # Set random seed
        if seed is not None:
            np.random.seed(seed)
        
        # Generate synthetic data
        self._generate_data()
        
        print(f"✓ MockProvider initialized")
        print(f"  Images: {len(self._image_names)}")
        print(f"  Pairs: {len(self._pairs)}")
        print(f"  Avg matches/pair: {self.matches_per_pair}")
    
    def _generate_data(self):
        """Generate synthetic match data"""
        # Generate image names
        self._image_names = [f"image_{i:04d}.jpg" for i in range(self.num_images)]
        
        # Generate image info
        self._image_info = {}
        for name in self._image_names:
            self._image_info[name] = {
                'name': name,
                'width': self.image_size[0],
                'height': self.image_size[1],
                'size': (self.image_size[1], self.image_size[0], 3),
                'channels': 3
            }
        
        # Generate pairs (ensure connected graph)
        self._pairs = []
        self._match_data = {}
        
        # First ensure all images are connected (spanning tree)
        for i in range(self.num_images - 1):
            pair = (self._image_names[i], self._image_names[i + 1])
            self._pairs.append(pair)
        
        # Add random additional pairs
        total_pairs = min(
            self.num_images * self.pairs_per_image // 2,
            self.num_images * (self.num_images - 1) // 2
        )
        
        all_possible_pairs = list(combinations(self._image_names, 2))
        np.random.shuffle(all_possible_pairs)
        
        for pair in all_possible_pairs:
            if len(self._pairs) >= total_pairs:
                break
            if pair not in self._pairs:
                self._pairs.append(pair)
        
        # Generate match data for each pair
        for pair in self._pairs:
            self._match_data[pair] = self._generate_match_data_for_pair()
        
        # Build image->pairs mapping
        self._image_to_pairs = {}
        for pair in self._pairs:
            for img in pair:
                if img not in self._image_to_pairs:
                    self._image_to_pairs[img] = []
                self._image_to_pairs[img].append(pair)
    
    def _generate_match_data_for_pair(self) -> StructuredMatchData:
        """Generate synthetic match data for a pair"""
        # Randomize match count (with some variation)
        num_matches = max(10, int(np.random.normal(
            self.matches_per_pair, 
            self.matches_per_pair * 0.3
        )))
        
        # Generate random keypoints
        kp1 = [
            cv2.KeyPoint(
                x=float(np.random.uniform(50, self.image_size[0] - 50)),
                y=float(np.random.uniform(50, self.image_size[1] - 50)),
                size=float(np.random.uniform(5, 20)),
                angle=float(np.random.uniform(0, 360)),
                response=float(np.random.uniform(0.5, 1.0)),
                octave=int(np.random.randint(0, 4)),
                class_id=-1
            )
            for _ in range(num_matches)
        ]
        
        # Generate kp2 with some spatial correlation to kp1 (realistic matching)
        kp2 = []
        for kp in kp1:
            # Add small displacement + rotation
            dx = np.random.normal(0, 20)
            dy = np.random.normal(0, 20)
            x2 = np.clip(kp.pt[0] + dx, 0, self.image_size[0])
            y2 = np.clip(kp.pt[1] + dy, 0, self.image_size[1])
            
            kp2.append(cv2.KeyPoint(
                x=float(x2),
                y=float(y2),
                size=float(kp.size * np.random.uniform(0.8, 1.2)),
                angle=float((kp.angle + np.random.normal(0, 10)) % 360),
                response=float(kp.response * np.random.uniform(0.8, 1.0)),
                octave=kp.octave,
                class_id=-1
            ))
        
        # Generate match quality scores
        base_quality = np.random.uniform(*self.quality_range)
        match_qualities = np.clip(
            np.random.normal(base_quality, 0.1, num_matches),
            0.0, 1.0
        )
        
        # Create EnhancedDMatch objects
        matches = []
        for i in range(num_matches):
            match = EnhancedDMatch(
                queryIdx=i,
                trainIdx=i,
                score=float(match_qualities[i]),
                score_type=ScoreType.CONFIDENCE,
                confidence=float(match_qualities[i]),
                standardized_quality=float(match_qualities[i]),
                source_method='mock'
            )
            matches.append(match)
        
        # Compute pair quality
        pair_quality = float(np.mean(match_qualities))
        
        # Quality statistics
        quality_stats = {
            'mean': float(np.mean(match_qualities)),
            'std': float(np.std(match_qualities)),
            'min': float(np.min(match_qualities)),
            'max': float(np.max(match_qualities)),
            'median': float(np.median(match_qualities))
        }
        
        return StructuredMatchData(
            matches=matches,
            keypoints1=kp1,
            keypoints2=kp2,
            method='mock',
            score_type=ScoreType.CONFIDENCE,
            num_matches=num_matches,
            standardized_pair_quality=pair_quality,
            match_quality_stats=quality_stats,
            matching_time=0.0
        )
    
    # =========================================================================
    # IMatchDataProvider Implementation
    # =========================================================================
    
    def get_match_data(self, pair: Tuple[str, str]) -> StructuredMatchData:
        """Get match data for a pair"""
        if pair not in self._match_data:
            # Try reverse
            pair_rev = (pair[1], pair[0])
            if pair_rev in self._match_data:
                return self._match_data[pair_rev]
            raise KeyError(f"Pair {pair} not found")
        return self._match_data[pair]
    
    def get_all_pairs(self) -> List[Tuple[str, str]]:
        """Get all pairs"""
        return self._pairs.copy()
    
    def get_image_info(self, image_name: str) -> Dict:
        """Get image metadata"""
        if image_name not in self._image_info:
            raise KeyError(f"Image {image_name} not found")
        return self._image_info[image_name].copy()
    
    def get_all_images(self) -> List[str]:
        """Get all image names"""
        return self._image_names.copy()
    
    def get_pairs_with_image(self, image_name: str) -> List[Tuple[str, str]]:
        """Get pairs containing an image"""
        return self._image_to_pairs.get(image_name, []).copy()
    
    def get_best_pairs(self, k: int = 10, criterion: str = 'quality') -> List[Tuple[str, str]]:
        """Get best pairs"""
        if criterion == 'quality':
            sorted_pairs = sorted(
                self._pairs,
                key=lambda p: self._match_data[p].standardized_pair_quality,
                reverse=True
            )
        elif criterion == 'match_count':
            sorted_pairs = sorted(
                self._pairs,
                key=lambda p: self._match_data[p].num_matches,
                reverse=True
            )
        else:
            sorted_pairs = self._pairs
        
        return sorted_pairs[:k]
    
    def filter_pairs(self,
                    min_quality: Optional[float] = None,
                    min_matches: Optional[int] = None,
                    predicate: Optional[Callable] = None) -> Dict[Tuple[str, str], StructuredMatchData]:
        """Filter pairs"""
        result = {}
        
        for pair in self._pairs:
            match_data = self._match_data[pair]
            
            if min_quality and match_data.standardized_pair_quality < min_quality:
                continue
            if min_matches and match_data.num_matches < min_matches:
                continue
            if predicate and not predicate(pair, match_data):
                continue
            
            result[pair] = match_data
        
        return result
    
    def get_statistics(self) -> Dict:
        """Get statistics"""
        all_qualities = [data.standardized_pair_quality for data in self._match_data.values()]
        all_matches = [data.num_matches for data in self._match_data.values()]
        
        return {
            'total_pairs': len(self._pairs),
            'total_images': len(self._image_names),
            'total_matches': sum(all_matches),
            'avg_matches_per_pair': float(np.mean(all_matches)),
            'quality_distribution': {
                'mean': float(np.mean(all_qualities)),
                'std': float(np.std(all_qualities)),
                'min': float(np.min(all_qualities)),
                'max': float(np.max(all_qualities)),
                'median': float(np.median(all_qualities))
            },
            'match_count_distribution': {
                'mean': float(np.mean(all_matches)),
                'std': float(np.std(all_matches)),
                'min': int(np.min(all_matches)),
                'max': int(np.max(all_matches)),
                'median': float(np.median(all_matches))
            },
            'provider_type': 'mock'
        }
    
    def validate(self) -> ValidationResult:
        """Validate data"""
        errors = []
        warnings = []
        
        # Check we have data
        if not self._pairs:
            errors.append("No pairs generated")
        if not self._image_names:
            errors.append("No images generated")
        
        # Check graph connectivity (all images should have at least one pair)
        for img in self._image_names:
            if img not in self._image_to_pairs or not self._image_to_pairs[img]:
                warnings.append(f"Image {img} has no pairs")
        
        stats = self.get_statistics()
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            statistics=stats
        )


# =============================================================================
# Example usage
# =============================================================================

if __name__ == "__main__":
    # Create mock provider
    mock = MockProvider(
        num_images=20,
        pairs_per_image=6,
        matches_per_pair=200,
        quality_range=(0.6, 0.95),
        seed=42  # Reproducible
    )
    
    # Print summary
    mock.print_summary()
    
    # Validate
    validation = mock.validate()
    validation.print_report()
    
    # Get best pairs
    best = mock.get_best_pairs(k=5)
    print("\nTop 5 pairs:")
    for pair in best:
        data = mock.get_match_data(pair)
        print(f"  {pair}: quality={data.standardized_pair_quality:.3f}, matches={data.num_matches}")
    
    # Use in testing
    print("\nExample: Testing with mock data")
    for pair in best[:2]:
        match_data = mock.get_match_data(pair)
        print(f"\nPair: {pair}")
        print(f"  Matches: {len(match_data.matches)}")
        print(f"  First match: query={match_data.matches[0].queryIdx}, "
              f"train={match_data.matches[0].trainIdx}, "
              f"quality={match_data.matches[0].standardized_quality:.3f}")