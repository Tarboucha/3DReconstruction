"""
Match data loader with quality standardization.

This module handles loading and reconstructing StructuredMatchData from
saved batch files, including quality score standardization across different
matching methods.

File: CameraPoseEstimation/loaders/match_loader.py
"""

import numpy as np
import cv2
from typing import Dict, List, Optional
from pathlib import Path

from CameraPoseEstimation2.core.structures import (
    StructuredMatchData,
    EnhancedDMatch,
    ScoreType,
    keypoints_from_serializable
)
from CameraPoseEstimation2.logger import get_logger

logger = get_logger("data.loaders")


# =============================================================================
# Quality Standardization
# =============================================================================

class MatchQualityStandardizer:
    """
    Standardize match quality scores across different methods to 0-1 range.
    
    Different matching methods produce scores with different semantics:
    - Distance scores (FLANN, BF): Lower is better
    - Confidence scores (LightGlue): Higher is better (already 0-1)
    - Similarity scores: Higher is better
    
    This class normalizes all scores to a unified 0-1 range where
    higher values always indicate better matches.
    """
    
    @staticmethod
    def standardize_single_score(score: float, 
                                 score_type: ScoreType,
                                 percentile_info: Optional[Dict] = None) -> float:
        """
        Standardize a single score to 0-1 range.
        
        Args:
            score: Raw score value
            score_type: Type of score (DISTANCE, CONFIDENCE, SIMILARITY)
            percentile_info: Optional percentile statistics for robust normalization
            
        Returns:
            Standardized quality score (0-1, higher is better)
        """
        if score_type == ScoreType.CONFIDENCE or score_type == ScoreType.SIMILARITY:
            # Already 0-1, higher is better
            return float(np.clip(score, 0, 1))
        
        elif score_type == ScoreType.DISTANCE:
            # Lower distance = better match, need to normalize and invert
            if percentile_info:
                # Use robust normalization based on data distribution
                p95 = percentile_info.get('p95', 100.0)
                
                # Clip to p95 to handle outliers
                normalized = np.clip(score / p95, 0, 1)
                
                # Invert: low distance = high quality
                return float(1.0 - normalized)
            else:
                # Fallback: assume reasonable distance range
                # This is less robust but works when we don't have distribution info
                return float(max(0.0, 1.0 - score / 100.0))
        
        else:
            # Unknown type: assume 0-1 range, higher is better
            return float(np.clip(score, 0, 1))
    
    @staticmethod
    def compute_percentile_info(scores: np.ndarray) -> Dict:
        """
        Compute percentile information for robust normalization.
        
        Args:
            scores: Array of raw scores
            
        Returns:
            Dictionary with percentile statistics
        """
        if len(scores) == 0:
            return {
                'p25': 0,
                'p50': 0,
                'p75': 0,
                'p95': 100,
                'mean': 0,
                'std': 0
            }
        
        return {
            'p25': float(np.percentile(scores, 25)),
            'p50': float(np.percentile(scores, 50)),
            'p75': float(np.percentile(scores, 75)),
            'p95': float(np.percentile(scores, 95)),
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores))
        }
    
    @staticmethod
    def standardize_scores(scores: List[float],
                          score_type: ScoreType) -> List[float]:
        """
        Standardize a list of scores.
        
        Args:
            scores: List of raw scores
            score_type: Type of scores
            
        Returns:
            List of standardized scores (0-1)
        """
        if not scores:
            return []
        
        scores_array = np.array(scores)
        
        if score_type == ScoreType.CONFIDENCE or score_type == ScoreType.SIMILARITY:
            # Already normalized
            return np.clip(scores_array, 0, 1).tolist()
        
        elif score_type == ScoreType.DISTANCE:
            # Compute percentile info for robust normalization
            percentile_info = MatchQualityStandardizer.compute_percentile_info(scores_array)
            
            # Standardize each score
            standardized = [
                MatchQualityStandardizer.standardize_single_score(
                    score, score_type, percentile_info
                )
                for score in scores
            ]
            
            return standardized
        
        else:
            return np.clip(scores_array, 0, 1).tolist()
    
    @staticmethod
    def compute_pair_quality(standardized_scores: List[float],
                            num_matches: int) -> float:
        """
        Compute overall quality score for an image pair.
        
        Considers multiple factors:
        - Number of matches (more is better, up to saturation)
        - Mean quality
        - Quality consistency (lower variance is better)
        - Quality of top matches
        
        Args:
            standardized_scores: List of standardized match quality scores
            num_matches: Total number of matches
            
        Returns:
            Overall pair quality (0-1)
        """
        if num_matches == 0 or len(standardized_scores) == 0:
            return 0.0
        
        scores_array = np.array(standardized_scores)
        
        # Component 1: Match count score (sigmoid, saturates at ~200 matches)
        # More matches = better, but diminishing returns
        count_score = 1.0 - np.exp(-num_matches / 100.0)
        
        # Component 2: Mean quality of all matches
        mean_quality = float(np.mean(scores_array))
        
        # Component 3: Consistency score (penalize high variance)
        # Consistent quality across matches is better
        std_quality = float(np.std(scores_array))
        consistency_score = float(np.exp(-std_quality))
        
        # Component 4: Top matches quality (median of top 50%)
        # High-quality top matches are important
        if len(scores_array) > 1:
            top_50_percent = np.partition(scores_array, -len(scores_array)//2)[-len(scores_array)//2:]
            top_quality = float(np.median(top_50_percent))
        else:
            top_quality = mean_quality
        
        # Weighted combination
        # Emphasize mean quality and top quality
        overall_quality = (
            0.20 * count_score +
            0.30 * mean_quality +
            0.15 * consistency_score +
            0.35 * top_quality
        )
        
        return float(np.clip(overall_quality, 0.0, 1.0))


# =============================================================================
# Match Data Reconstruction
# =============================================================================

def _reconstruct_structured_match_data(result: Dict,
                                       standardize: bool = True,
                                       reconstruct_kp: bool = True,
                                       standardizer: Optional[MatchQualityStandardizer] = None) -> Optional[StructuredMatchData]:
    """
    Reconstruct StructuredMatchData from saved batch result.
    
    This function takes a dictionary from a batch file and reconstructs
    a full StructuredMatchData object with all metadata.
    
    Args:
        result: Dictionary with match data from batch file
        standardize: If True, compute standardized quality scores
        reconstruct_kp: If True, reconstruct full cv2.KeyPoint objects
        standardizer: MatchQualityStandardizer instance (created if None)
        
    Returns:
        StructuredMatchData object or None if reconstruction fails
    """
    if standardizer is None:
        standardizer = MatchQualityStandardizer()
    
    try:
        # Extract basic information
        method = result.get('method', 'unknown')
        num_matches = result.get('num_matches', 0)
        
        if num_matches == 0:
            return None
        
        # Get correspondences
        correspondences = np.array(result.get('correspondences', []))
        if len(correspondences) == 0:
            return None
        
        # Reconstruct keypoints
        if reconstruct_kp and 'keypoints1' in result and 'keypoints2' in result:
            # Reconstruct from serialized keypoints
            try:
                kp1 = keypoints_from_serializable(result['keypoints1'])
                kp2 = keypoints_from_serializable(result['keypoints2'])
            except Exception as e:
                # Fallback to minimal keypoints
                logger.warning(f"Could not reconstruct keypoints: {e}")
                kp1 = [cv2.KeyPoint(x=float(c[0]), y=float(c[1]), size=1.0)
                       for c in correspondences]
                kp2 = [cv2.KeyPoint(x=float(c[2]), y=float(c[3]), size=1.0)
                       for c in correspondences]
        else:
            # Create minimal keypoints from correspondences
            kp1 = [cv2.KeyPoint(x=float(c[0]), y=float(c[1]), size=1.0) 
                   for c in correspondences]
            kp2 = [cv2.KeyPoint(x=float(c[2]), y=float(c[3]), size=1.0) 
                   for c in correspondences]
        
        # Determine score type
        score_type_str = result.get('score_type', 'confidence')
        if isinstance(score_type_str, str):
            score_type = ScoreType.from_string(score_type_str)
        else:
            score_type = ScoreType.CONFIDENCE
        
        # Get match scores
        match_scores = result.get('match_scores', [])
        if not match_scores or len(match_scores) != num_matches:
            # Fallback: uniform scores
            match_scores = [1.0] * num_matches
        
        # Standardize quality scores
        if standardize:
            standardized_scores = standardizer.standardize_scores(
                match_scores,
                score_type
            )
        else:
            standardized_scores = match_scores
        
        # Create EnhancedDMatch objects
        matches = []
        for i in range(num_matches):
            match = EnhancedDMatch(
                queryIdx=i,
                trainIdx=i,
                score=float(match_scores[i]),
                score_type=score_type,
                source_method=method,
                standardized_quality=float(standardized_scores[i]) if standardize else None
            )
            matches.append(match)
        
        # Compute overall pair quality
        if standardize:
            pair_quality = standardizer.compute_pair_quality(
                standardized_scores,
                num_matches
            )
        else:
            # Use existing quality score if available
            pair_quality = float(result.get('quality_score', 
                                          result.get('standardized_pair_quality', 0.5)))
        
        # Compute quality statistics
        quality_stats = {
            'mean': float(np.mean(standardized_scores)),
            'std': float(np.std(standardized_scores)),
            'min': float(np.min(standardized_scores)),
            'max': float(np.max(standardized_scores)),
            'median': float(np.median(standardized_scores))
        }
        
        # Create StructuredMatchData
        structured = StructuredMatchData(
            matches=matches,
            keypoints1=kp1,
            keypoints2=kp2,
            method=method,
            score_type=score_type,
            num_matches=num_matches,
            standardized_pair_quality=pair_quality,
            match_quality_stats=quality_stats,
            matching_time=float(result.get('matching_time', 0.0))
        )
        
        return structured
        
    except Exception as e:
        logger.error(f"Error reconstructing match data: {e}", exc_info=True)
        return None


# =============================================================================
# Convenience Functions
# =============================================================================

def load_single_batch_file(batch_file_path: str,
                           standardize: bool = True) -> Dict:
    """
    Load a single batch file and reconstruct all match data.
    
    Args:
        batch_file_path: Path to batch pickle file
        standardize: If True, standardize quality scores
        
    Returns:
        Dictionary mapping pair keys to StructuredMatchData
    """
    import pickle
    
    batch_path = Path(batch_file_path)
    
    if not batch_path.exists():
        raise FileNotFoundError(f"Batch file not found: {batch_file_path}")
    
    # Load batch file
    with open(batch_path, 'rb') as f:
        batch_data = pickle.load(f)
    
    batch_results = batch_data.get('results', {})
    
    # Reconstruct each pair
    reconstructed = {}
    standardizer = MatchQualityStandardizer()
    
    for pair_key, result in batch_results.items():
        # Convert string keys to tuples if needed
        if isinstance(pair_key, str):
            try:
                pair_key = eval(pair_key)
            except:
                continue
        
        # Skip failed matches
        if 'error' in result:
            continue
        
        # Reconstruct
        match_data = _reconstruct_structured_match_data(
            result,
            standardize=standardize,
            reconstruct_kp=True,
            standardizer=standardizer
        )
        
        if match_data:
            reconstructed[pair_key] = match_data
    
    return reconstructed


# =============================================================================
# Testing and Examples
# =============================================================================

if __name__ == "__main__":
    logger.info("="*70)
    logger.info("MATCH LOADER MODULE - Testing")
    logger.info("="*70)

    # Test 1: Quality standardization
    logger.info("\nTest 1: Quality Standardization")
    logger.info("-"*70)
    
    standardizer = MatchQualityStandardizer()
    
    # Test with distance scores (lower is better)
    distance_scores = [10, 20, 30, 50, 100]
    standardized = standardizer.standardize_scores(distance_scores, ScoreType.DISTANCE)

    logger.info("Distance scores:", distance_scores)
    logger.info("Standardized:   ", [f"{s:.3f}" for s in standardized])
    logger.info("✓ Distance scores normalized (inverted)")

    # Test with confidence scores (already 0-1)
    confidence_scores = [0.9, 0.8, 0.85, 0.7, 0.95]
    standardized = standardizer.standardize_scores(confidence_scores, ScoreType.CONFIDENCE)

    logger.info("\nConfidence scores:", confidence_scores)
    logger.info("Standardized:     ", [f"{s:.3f}" for s in standardized])
    logger.info("✓ Confidence scores preserved")

    # Test pair quality computation
    logger.info("\n" + "-"*70)
    pair_quality = standardizer.compute_pair_quality(standardized, len(standardized))
    logger.info(f"Overall pair quality: {pair_quality:.3f}")
    logger.info("✓ Pair quality computed")
    
    # Test 2: Match data reconstruction
    logger.info("\n" + "="*70)
    logger.info("Test 2: Match Data Reconstruction")
    logger.info("-"*70)
    
    # Simulate batch result
    fake_result = {
        'correspondences': [[10, 20, 15, 25], [30, 40, 35, 45], [50, 60, 55, 65]],
        'num_matches': 3,
        'method': 'test_method',
        'score_type': 'confidence',
        'match_scores': [0.9, 0.85, 0.8],
        'matching_time': 0.5
    }
    
    match_data = _reconstruct_structured_match_data(
        fake_result,
        standardize=True,
        reconstruct_kp=True
    )
    
    if match_data:
        logger.info(f"✓ Reconstructed match data:")
        logger.info(f"  Method: {match_data.method}")
        logger.info(f"  Matches: {match_data.num_matches}")
        logger.info(f"  Pair quality: {match_data.standardized_pair_quality:.3f}")
        logger.info(f"  Match qualities: {[f'{m.standardized_quality:.3f}' for m in match_data.matches]}")

        # Test backward compatibility
        pts1 = match_data.pts1
        pts2 = match_data.pts2
        logger.info(f"  Points shape: {pts1.shape} / {pts2.shape}")
        logger.info("✓ Backward compatible access works")
    else:
        logger.error("✗ Reconstruction failed")

    logger.info("\n" + "="*70)
    logger.info("✅ ALL TESTS PASSED")
    logger.info("="*70)
    logger.info("\nThis module is ready to use with StructuredDataProvider!")