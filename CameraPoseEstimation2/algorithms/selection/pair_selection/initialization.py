import pickle
import numpy as np
import cv2
from typing import Dict, Tuple, List, Optional, Set
from dataclasses import dataclass
from pathlib import Path
from enum import Enum
from CameraPoseEstimation2.core.interfaces import IMatchDataProvider
from CameraPoseEstimation2.core.structures import StructuredMatchData

class ScoreType(Enum):
    """Enumeration of score types"""
    DISTANCE = "distance"      # Lower is better (traditional matchers)
    CONFIDENCE = "confidence"  # Higher is better (deep learning matchers)
    SIMILARITY = "similarity"  # Higher is better

@dataclass
class ScoringConfig:
    """Configuration for scoring image pairs"""
    # Scoring weights (must sum to 1.0)
    num_matches_weight: float = 0.25
    match_distribution_weight: float = 0.20
    geometric_consistency_weight: float = 0.25
    baseline_adequacy_weight: float = 0.15
    match_confidence_weight: float = 0.15
    
    # Quality thresholds
    min_matches: int = 50
    min_inlier_ratio: float = 0.3
    min_coverage_ratio: float = 0.02
    optimal_displacement_range: Tuple[float, float] = (10.0, 100.0)
    
    # Multi-view specific thresholds
    min_matches_incremental: int = 30  # Lower threshold for incremental views
    min_overlap_with_existing: float = 0.4  # Minimum overlap with existing cameras
    max_cameras_per_iteration: int = 1  # How many cameras to add per iteration
    
    # Image assumptions (if not available in data)
    default_image_width: int = 1920
    default_image_height: int = 1080


class InitializationPairSelector:

    def __init__(self, config: Optional[ScoringConfig] = None):
        """
        Initialize pair selector with configuration
        
        Args:
            config: Scoring configuration for monument-specific requirements
        """
        self.provider = None
        self.config = config if config is not None else ScoringConfig()
        self.matches_data = None
        self.pair_scores = []
        
    def _validate_matches_data(self):
        """Validate the structure of loaded matches data"""
        for pair_key, pair_data in self.matches_data.items():
            # Check pair key format
            if not isinstance(pair_key, tuple) or len(pair_key) != 2:
                raise ValueError(f"Invalid pair key format: {pair_key}")
            
            # Check required data
            if 'correspondences' not in pair_data:
                raise ValueError(f"Missing 'correspondences' for pair {pair_key}")
            
            correspondences = pair_data['correspondences']
            if len(correspondences[0]) > 0 and len(correspondences[0][0]) != 4:
                raise ValueError(f"Correspondences must be Nx4 format for pair {pair_key}")


    def _calculate_connectivity_score(self, pair_key: Tuple[str, str], 
                                    existing_cameras: Set[str]) -> float:
        """
        Calculate how well this pair connects to existing reconstruction
        
        Args:
            pair_key: The image pair being evaluated
            existing_cameras: Set of camera IDs already in reconstruction
            
        Returns:
            Connectivity score between 0 and 1
        """
        img1, img2 = pair_key
        
        # Count connections to existing cameras
        connections = 0
        total_possible_connections = 0
        
        for existing_cam in existing_cameras:
            total_possible_connections += 2  # Could connect to both images in pair
            
            # Check if we have matches between existing camera and either image in pair
            if (existing_cam, img1) in self.matches_data or (img1, existing_cam) in self.matches_data:
                pair_data = (self.matches_data.get((existing_cam, img1)) or 
                           self.matches_data.get((img1, existing_cam)))
                if pair_data and len(pair_data['correspondences']) >= self.config.min_matches_incremental:
                    connections += 1
            
            if (existing_cam, img2) in self.matches_data or (img2, existing_cam) in self.matches_data:
                pair_data = (self.matches_data.get((existing_cam, img2)) or 
                           self.matches_data.get((img2, existing_cam)))
                if pair_data and len(pair_data['correspondences']) >= self.config.min_matches_incremental:
                    connections += 1
        
        if total_possible_connections == 0:
            return 0.0
        
        connectivity_ratio = connections / total_possible_connections
        
        # Boost score if at least one image has good connectivity
        min_required_connections = max(1, len(existing_cameras) * self.config.min_overlap_with_existing)
        
        if connections >= min_required_connections:
            return min(connectivity_ratio * 2, 1.0)  # Bonus for good connectivity
        else:
            return connectivity_ratio * 0.5  # Penalty for poor connectivity

    def _calculate_geometric_consistency(self, pts1: np.ndarray, pts2: np.ndarray) -> float:
        """Calculate geometric consistency using fundamental matrix RANSAC"""
        try:
            F, mask = cv2.findFundamentalMat(
                pts1, pts2,
                method=cv2.FM_RANSAC,
                ransacReprojThreshold=2.0,
                confidence=0.99,
                maxIters=2000
            )
            
            if F is not None and mask is not None:
                return np.sum(mask) / len(pts1)
            else:
                return 0.0
                
        except Exception:
            return 0.0

    def _calculate_baseline_score(self, pts1: np.ndarray, pts2: np.ndarray) -> float:
        """Calculate baseline adequacy score based on point displacements"""
        displacements = np.linalg.norm(pts2 - pts1, axis=1)
        mean_displacement = np.mean(displacements)
        
        opt_min, opt_max = self.config.optimal_displacement_range
        
        if mean_displacement < opt_min:
            return mean_displacement / opt_min
        elif mean_displacement <= opt_max:
            return 1.0
        else:
            return max(0.0, 1.0 - (mean_displacement - opt_max) / (opt_max * 2))

    def find_best_pair(self, matches_data: Optional[Dict] = None) -> Dict:
        """Find the best image pair for initialization (2-view case)"""
        return self._find_best_pairs(matches_data, existing_cameras=None, num_pairs=1)

    def find_next_best_pairs(self, existing_cameras: Set[str], 
                           matches_data: Optional[Dict] = None,
                           num_pairs: int = 1) -> Dict:
        """
        Find the best image pair(s) to add to existing reconstruction
        
        Args:
            existing_cameras: Set of camera IDs already in reconstruction
            matches_data: Optional matches data (uses loaded data if None)
            num_pairs: Number of pairs to return (for adding multiple cameras)
            
        Returns:
            Dictionary with best pair(s) information for incremental reconstruction
        """
        if len(existing_cameras) < 2:
            raise ValueError("Need at least 2 existing cameras for incremental reconstruction")
        
        print(f"Finding next best pair(s) to add to {len(existing_cameras)} existing cameras...")
        return self._find_best_pairs(matches_data, existing_cameras, num_pairs)

    def _find_best_pairs(self, matches_data: Optional[Dict] = None,
                        existing_cameras: Optional[Set[str]] = None,
                        num_pairs: int = 1) -> Dict:
        """
        Internal method to find best pairs for either initialization or incremental reconstruction
        
        Args:
            matches_data: Optional matches data
            existing_cameras: Set of existing camera IDs (None for initialization)
            num_pairs: Number of pairs to return
            
        Returns:
            Dictionary with best pair(s) information
        """
        if matches_data is not None:
            self.matches_data = matches_data
        
        if self.matches_data is None:
            raise ValueError("No matches data available. Call load_matches_from_pickle first.")
        
        is_incremental = existing_cameras is not None and len(existing_cameras) >= 2
        mode = "incremental" if is_incremental else "initialization"
        
        print(f"Evaluating {len(self.matches_data)} image pairs for {mode}...")
        
        self.pair_scores = []
        
        for pair_key, pair_data in self.matches_data.items():
            img1, img2 = pair_key
            
            # For incremental: skip pairs where both images are already in reconstruction
            if is_incremental and img1 in existing_cameras and img2 in existing_cameras:
                continue
            
            # For incremental: require at least one new camera
            if is_incremental:
                has_new_camera = img1 not in existing_cameras or img2 not in existing_cameras
                if not has_new_camera:
                    continue
            
            # Basic filtering
            correspondences = pair_data['correspondences']
            min_matches = (self.config.min_matches_incremental if is_incremental 
                         else self.config.min_matches)
            
            if len(correspondences) < min_matches:
                continue
            
            # Score this pair
            score_result = self.score_image_pair(pair_key, pair_data, existing_cameras)
            
            # Quality filtering
            min_inlier_ratio = (self.config.min_inlier_ratio * 0.8 if is_incremental 
                              else self.config.min_inlier_ratio)
            
            if (score_result.get('inlier_ratio', 0) >= min_inlier_ratio and
                score_result.get('mean_coverage', 0) >= self.config.min_coverage_ratio):
                
                # For incremental: additional connectivity check
                if is_incremental:
                    connectivity_score = score_result['component_scores'].get('connectivity', 0)
                    if connectivity_score < 0.3:  # Minimum connectivity threshold
                        continue
                
                self.pair_scores.append({
                    'pair': pair_key,
                    'score_result': score_result,
                    'pair_data': pair_data
                })
        
        if not self.pair_scores:
            criteria_msg = (f"min_matches={min_matches}, min_inlier_ratio={min_inlier_ratio:.2f}"
                          + (", min_connectivity=0.3" if is_incremental else ""))
            raise ValueError(
                f"No suitable {mode} pairs found! Criteria: {criteria_msg}"
            )
        
        # Sort by total score (descending)
        self.pair_scores.sort(key=lambda x: x['score_result']['total_score'], reverse=True)
        
        # Print top candidates
        self._print_top_candidates(mode=mode)
        
        # Return requested number of pairs
        best_pairs = []
        selected_cameras = set(existing_cameras) if existing_cameras else set()
        
        for i in range(min(num_pairs, len(self.pair_scores))):
            candidate = self.pair_scores[i]
            pair_key = candidate['pair']
            
            # For multiple pairs: avoid selecting pairs with overlapping cameras
            if num_pairs > 1 and i > 0:
                img1, img2 = pair_key
                if img1 in selected_cameras and img2 in selected_cameras:
                    continue  # Skip this pair to avoid camera overlap
            
            best_pairs.append(candidate)
            
            # Update selected cameras
            selected_cameras.update(pair_key)
        
        if len(best_pairs) == 1:
            # Return single pair (backward compatibility)
            best_pair_info = best_pairs[0]
            return {
                'best_pair': best_pair_info['pair'],
                'best_pair_data': best_pair_info['pair_data'],
                'best_score_result': best_pair_info['score_result'],
                'all_candidates': self.pair_scores,
                'num_candidates': len(self.pair_scores),
                'mode': mode
            }
        else:
            # Return multiple pairs
            return {
                'best_pairs': [(p['pair'], p['pair_data'], p['score_result']) for p in best_pairs],
                'all_candidates': self.pair_scores,
                'num_candidates': len(self.pair_scores),
                'mode': mode
            }

    def _print_top_candidates(self, num_candidates: int = 5, mode: str = "initialization"):
        """Print information about top candidate pairs"""
        print(f"\nTop {min(num_candidates, len(self.pair_scores))} {mode} candidates:")
        print("-" * 80)
        
        for i, candidate in enumerate(self.pair_scores[:num_candidates]):
            pair = candidate['pair']
            score_result = candidate['score_result']
            
            print(f"{i+1:2d}. {pair[0]} <-> {pair[1]}")
            print(f"    Total Score: {score_result['total_score']:.3f}")
            print(f"    Matches: {score_result['num_matches']:3d}, "
                  f"Inliers: {score_result['inlier_ratio']:.1%}, "
                  f"Coverage: {score_result['mean_coverage']:.3f}")
            
            # Show component scores
            components = score_result['component_scores']
            comp_str = (f"matches={components['num_matches']:.2f}, "
                       f"distrib={components['match_distribution']:.2f}, "
                       f"geom={components['geometric_consistency']:.2f}")
            
            if 'connectivity' in components:
                comp_str += f", connect={components['connectivity']:.2f}"
            
            print(f"    Components: {comp_str}")
            print()

    def get_next_cameras_to_add(self, existing_cameras: Set[str],
                               max_new_cameras: int = 3) -> List[Dict]:
        """
        Get prioritized list of individual cameras to add to reconstruction
        
        Args:
            existing_cameras: Set of camera IDs already in reconstruction
            max_new_cameras: Maximum number of new cameras to consider
            
        Returns:
            List of camera candidates with their best connection pairs
        """
        if self.matches_data is None:
            raise ValueError("No matches data available. Call load_matches_from_pickle first.")
        
        # Find all cameras that could be added
        all_cameras = set()
        for pair_key in self.matches_data.keys():
            all_cameras.update(pair_key)
        
        candidate_cameras = all_cameras - existing_cameras
        
        if not candidate_cameras:
            return []
        
        print(f"Evaluating {len(candidate_cameras)} candidate cameras to add...")
        
        camera_scores = []
        
        for new_camera in candidate_cameras:
            # Find all pairs involving this new camera and existing cameras
            connection_pairs = []
            
            for existing_camera in existing_cameras:
                # Check both directions
                pair_key1 = (existing_camera, new_camera)
                pair_key2 = (new_camera, existing_camera)
                
                pair_data = None
                final_pair_key = None
                
                if pair_key1 in self.matches_data:
                    pair_data = self.matches_data[pair_key1]
                    final_pair_key = pair_key1
                elif pair_key2 in self.matches_data:
                    pair_data = self.matches_data[pair_key2]
                    final_pair_key = pair_key2
                
                if pair_data and len(pair_data['correspondences']) >= self.config.min_matches_incremental:
                    score_result = self.score_image_pair(final_pair_key, pair_data, existing_cameras)
                    connection_pairs.append({
                        'pair_key': final_pair_key,
                        'existing_camera': existing_camera,
                        'pair_data': pair_data,
                        'score_result': score_result
                    })
            
            if not connection_pairs:
                continue  # No valid connections
            
            # Calculate overall score for this camera
            connection_scores = [cp['score_result']['total_score'] for cp in connection_pairs]
            best_connection_score = max(connection_scores)
            mean_connection_score = np.mean(connection_scores)
            num_connections = len(connection_pairs)
            
            # Combined score favoring cameras with multiple good connections
            overall_score = (best_connection_score * 0.6 + 
                           mean_connection_score * 0.3 + 
                           min(num_connections / len(existing_cameras), 1.0) * 0.1)
            
            camera_scores.append({
                'camera_id': new_camera,
                'overall_score': overall_score,
                'num_connections': num_connections,
                'best_connection_score': best_connection_score,
                'mean_connection_score': mean_connection_score,
                'connection_pairs': connection_pairs,
                'best_pair': max(connection_pairs, key=lambda x: x['score_result']['total_score'])
            })
        
        # Sort by overall score
        camera_scores.sort(key=lambda x: x['overall_score'], reverse=True)
        
        # Print top candidates
        print(f"\nTop candidate cameras to add:")
        print("-" * 60)
        for i, candidate in enumerate(camera_scores[:min(5, len(camera_scores))]):
            print(f"{i+1}. {candidate['camera_id']}")
            print(f"   Score: {candidate['overall_score']:.3f}, "
                  f"Connections: {candidate['num_connections']}")
            best_pair = candidate['best_pair']
            print(f"   Best pair: {best_pair['pair_key']} "
                  f"(score: {best_pair['score_result']['total_score']:.3f})")
            print()
        
        return camera_scores[:max_new_cameras]

    def get_best_pair_for_pipeline(self, pickle_file: str, validate: bool = True) -> Dict:
        """
        Complete pipeline to get best initialization pair from pickle file
        
        Args:
            pickle_file: Path to feature matches pickle file
            validate: Whether to perform monument-specific validation
            
        Returns:
            Complete information about selected pair ready for pose estimation
        """
        # Load and find best pair
        self.matches_data=pickle_file
        self._validate_matches_data()

        result = self.find_best_pair()

        
        best_pair = result['best_pair']
        best_data = result['best_pair_data']
        best_score = result['best_score_result']
        
        # Validation (existing code...)
        validation_result = None
        if validate:
            validation_result = self.validate_pair_for_monument(
                best_pair, best_data, best_score
            )
        
        # Prepare correspondences for matrix estimation
        correspondences = np.array(best_data['correspondences'])
        pts1 = correspondences[:, :2].astype(np.float32)
        pts2 = correspondences[:, 2:].astype(np.float32)
        
        
        return {
            'image_pair': best_pair,
            'correspondences': correspondences,
            'pts1': pts1,
            'pts2': pts2,
            'num_matches': len(correspondences),
            'quality_metrics': best_score,
            'validation': validation_result,
            'pair_data': best_data,
            'selection_metadata': {
                'total_pairs_evaluated': len(self.matches_data),
                'qualified_pairs': result['num_candidates'],
                'selection_config': self.config
            }
        }

    def get_selected_pair_for_pipeline(self, pickle_file: str, selected_pair: tuple, image1_size: tuple, image2_size: tuple, validate: bool = True) -> Dict:
        """
        Complete pipeline to get selected initialization pair from pickle file
        
        Args:
            pickle_file: Path to feature matches pickle file
            selected_pair: Tuple of (image1, image2) specifying the desired pair
            validate: Whether to perform monument-specific validation
            
        Returns:
            Complete information about selected pair ready for pose estimation
            
        Raises:
            KeyError: If the selected pair is not found in the matches data
            ValueError: If the selected pair data is invalid
        """
        # Load matches data
        self.matches_data = pickle_file
        self._validate_matches_data()
        
        # Check if selected pair exists in matches data
        if selected_pair not in self.matches_data:
            available_pairs = list(self.matches_data.keys())
            raise KeyError(f"Selected pair {selected_pair} not found in matches data. "
                        f"Available pairs: {available_pairs}")
        
        # Get data for the selected pair
        selected_data = self.matches_data[selected_pair]
        
        # Validate that the selected pair has required data
        if 'correspondences' not in selected_data or len(selected_data['correspondences']) == 0:
            raise ValueError(f"Selected pair {selected_pair} has no valid correspondences")
        
        # Calculate quality metrics for the selected pair (similar to find_best_pair scoring)
        selected_score = self.score_image_pair(selected_pair, selected_data, image1_size=image1_size, image2_size=image2_size)
        
        # Validation (existing code...)
        validation_result = None
        if validate:
            validation_result = self.validate_pair_for_monument(
                selected_pair, selected_data, selected_score
            )
        
        # Prepare correspondences for matrix estimation
        correspondences = np.array(selected_data['correspondences'][0])
        pts1 = correspondences[:, :2].astype(np.float32)
        pts2 = correspondences[:, 2:].astype(np.float32)
        
        return {
            'image_pair': selected_pair,
            'correspondences': correspondences,
            'pts1': pts1,
            'pts2': pts2,
            'num_matches': len(correspondences),
            'quality_metrics': selected_score,
            'validation': validation_result,
            'pair_data': selected_data,
            'selection_metadata': {
                'total_pairs_available': len(self.matches_data),
                'pair_selection_method': 'manual',
                'selected_pair': selected_pair,
                'selection_config': self.config
            }
        }   

    def validate_pair_for_monument(self, 
                                pair_key: Tuple[str, str], 
                                pair_data: Dict, 
                                score_result: Dict) -> Dict:
        """
        Perform monument-specific validation of selected pair
        
        Args:
            pair_key: Tuple of (image1, image2) filenames
            pair_data: Dictionary containing match data
            score_result: Result from score_image_pair function
            
        Returns:
            Validation dictionary with suitability assessment
        """
        validation = {
            'suitable': True,
            'warnings': [],
            'recommendations': [],
            'critical_issues': [],
            'quality_metrics': {}
        }
        
        # Extract key metrics
        component_scores = score_result.get('component_scores', {})
        score_type = score_result.get('score_type', 'unknown')
        method = score_result.get('method', 'unknown')
        
        # 1. Check baseline adequacy using normalized displacement
        normalized_displacement = score_result.get('normalized_displacement', 0)
        mean_displacement = score_result.get('mean_displacement', 0)
        
        if normalized_displacement < 0.02:
            validation['warnings'].append(
                f"Very small baseline (normalized: {normalized_displacement:.3f}) - may have poor depth resolution"
            )
            validation['recommendations'].append(
                "Consider using image pairs with larger baseline for better 3D reconstruction"
            )
        elif normalized_displacement > 0.5:
            validation['warnings'].append(
                f"Very large baseline (normalized: {normalized_displacement:.3f}) - may have matching difficulties"
            )
        
        # Store baseline quality
        validation['quality_metrics']['baseline_quality'] = component_scores.get('baseline_adequacy', 0)
        
        # 2. Check spatial coverage (using new coverage score)
        coverage_score = component_scores.get('coverage', 0)
        distribution_score = component_scores.get('match_distribution', 0)
        
        if coverage_score < 0.3:
            validation['critical_issues'].append(
                f"Poor spatial coverage ({coverage_score:.2f}) - matches don't cover monument adequately"
            )
            validation['suitable'] = False
            validation['recommendations'].append(
                "Need better distributed matches across the monument surface"
            )
        elif coverage_score < 0.5:
            validation['warnings'].append(
                f"Limited spatial coverage ({coverage_score:.2f}) - monument features may be incomplete"
            )
        
        validation['quality_metrics']['coverage_quality'] = coverage_score
        validation['quality_metrics']['distribution_quality'] = distribution_score
        
        # 3. Check geometric consistency
        geometric_consistency = component_scores.get('geometric_consistency', 0)
        
        if geometric_consistency < 0.3:
            validation['critical_issues'].append(
                f"Very low geometric consistency ({geometric_consistency:.1%}) - likely incorrect matches"
            )
            validation['suitable'] = False
        elif geometric_consistency < 0.5:
            validation['warnings'].append(
                f"Low geometric consistency ({geometric_consistency:.1%}) - challenging scene geometry"
            )
            validation['recommendations'].append(
                "Manual verification of matches recommended"
            )
        
        validation['quality_metrics']['geometric_quality'] = geometric_consistency
        
        # 4. Check resolution compatibility (important for monument reconstruction)
        resolution_compatibility = component_scores.get('resolution_compatibility', 1.0)
        
        if resolution_compatibility < 0.5:
            validation['warnings'].append(
                f"Poor resolution compatibility ({resolution_compatibility:.2f}) - images have very different resolutions"
            )
            validation['recommendations'].append(
                "Consider using images with more similar resolutions for better reconstruction"
            )
        
        # 5. Method-specific validation
        match_quality = component_scores.get('match_quality', 0)
        method_confidence = component_scores.get('method_confidence', 0)
        
        if score_type == 'confidence':
            # Deep learning methods
            if match_quality < 0.6:
                validation['warnings'].append(
                    f"Low confidence scores from {method} (avg: {match_quality:.2f})"
                )
                if 'loftr' in method.lower():
                    validation['recommendations'].append(
                        "LoFTR struggling - consider using images with more texture"
                    )
                elif 'superglue' in method.lower():
                    validation['recommendations'].append(
                        "SuperGlue confidence low - check image quality and overlap"
                    )
        elif score_type == 'distance':
            # Traditional methods
            if match_quality < 0.4:
                validation['warnings'].append(
                    f"Poor match distances from {method} (quality: {match_quality:.2f})"
                )
                if 'orb' in method.lower():
                    validation['recommendations'].append(
                        "ORB matches unreliable - consider using SIFT for better accuracy"
                    )
        
        validation['quality_metrics']['match_quality'] = match_quality
        validation['quality_metrics']['method_confidence'] = method_confidence
        
        # 6. Check score consistency (if available)
        score_consistency = component_scores.get('score_consistency', 0)
        if score_consistency < 0.5:
            validation['warnings'].append(
                f"Inconsistent match scores ({score_consistency:.2f}) - varying match quality"
            )
            validation['recommendations'].append(
                "Consider filtering matches by score threshold"
            )
        
        # 7. Number of matches validation
        num_matches = score_result.get('num_matches', 0)
        num_matches_score = component_scores.get('num_matches', 0)
        
        # Adjust thresholds based on image resolution
        image1_size = score_result.get('image1_size', (1920, 1080))
        image2_size = score_result.get('image2_size', (1920, 1080))
        avg_megapixels = ((image1_size[0] * image1_size[1] + image2_size[0] * image2_size[1]) / 2) / 1e6
        
        # Scale minimum matches with resolution
        adjusted_min_matches = int(self.config.min_matches * np.sqrt(avg_megapixels / 2.0))
        
        if num_matches < adjusted_min_matches:
            validation['critical_issues'].append(
                f"Insufficient matches ({num_matches} < {adjusted_min_matches} required for {avg_megapixels:.1f}MP images)"
            )
            validation['suitable'] = False
        elif num_matches < adjusted_min_matches * 1.5:
            validation['warnings'].append(
                f"Low number of matches ({num_matches}) for monument reconstruction"
            )
            validation['recommendations'].append(
                "More matches would improve reconstruction quality"
            )
        
        validation['quality_metrics']['match_count_quality'] = num_matches_score
        
        # 8. Overall score validation
        total_score = score_result.get('total_score', 0)
        
        # Dynamic threshold based on method and context
        is_incremental = score_result.get('is_incremental', False)
        
        if is_incremental:
            min_acceptable_score = 0.25  # Lower threshold for incremental
            good_score = 0.4
        else:
            min_acceptable_score = 0.35  # Higher threshold for initialization
            good_score = 0.5
        
        if total_score < min_acceptable_score:
            validation['critical_issues'].append(
                f"Overall quality score too low ({total_score:.2f} < {min_acceptable_score:.2f})"
            )
            validation['suitable'] = False
        elif total_score < good_score:
            validation['warnings'].append(
                f"Marginal overall quality ({total_score:.2f}) - reconstruction may have issues"
            )
        
        # 9. Monument-specific checks
        # Monuments often have repetitive patterns - check for this
        if geometric_consistency > 0.7 and num_matches > 200:
            # High consistency with many matches might indicate repetitive patterns
            validation['warnings'].append(
                "High match count with high consistency - check for repetitive architectural patterns"
            )
            validation['recommendations'].append(
                "Verify matches aren't concentrated on repetitive elements (columns, windows, etc.)"
            )
        
        # Check if this is a wide-baseline stereo pair (good for monuments)
        if 0.1 <= normalized_displacement <= 0.25 and geometric_consistency > 0.6:
            validation['recommendations'].append(
                "Good stereo baseline for monument reconstruction ✓"
            )
        
        # 10. Generate final recommendation
        if validation['suitable']:
            if len(validation['warnings']) == 0:
                validation['recommendations'].insert(0, 
                    f"Excellent pair for reconstruction (score: {total_score:.2f})"
                )
            elif len(validation['warnings']) <= 2:
                validation['recommendations'].insert(0,
                    f"Good pair with minor issues (score: {total_score:.2f})"
                )
            else:
                validation['recommendations'].insert(0,
                    f"Acceptable pair but with concerns (score: {total_score:.2f})"
                )
        else:
            validation['recommendations'].insert(0,
                "Pair not suitable for monument reconstruction - critical issues found"
            )
        
        # Add method information
        validation['method_info'] = {
            'method': method,
            'score_type': score_type,
            'is_incremental': is_incremental
        }
        
        return validation

    def score_image_pair(self, 
                        pair_key: Tuple[str, str], 
                        pair_data: Dict,
                        image1_size: Tuple[int, int],
                        image2_size: Tuple[int, int],
                        existing_cameras: Optional[Set[str]] = None) -> Dict:
        """
        Calculate comprehensive quality score for an image pair
        
        Args:
            pair_key: Tuple of (image1, image2) filenames
            pair_data: Dictionary containing match data for this pair
            image1_size: (width, height) of first image
            image2_size: (width, height) of second image
            existing_cameras: Set of camera IDs already in reconstruction (for incremental)
            
        Returns:
            Dictionary with total score and component scores
        """
        correspondences_data = pair_data['correspondences']
        
        # Extract correspondences and scores from the two-array structure
        if isinstance(correspondences_data, (list, tuple)) and len(correspondences_data) == 2:
            correspondences = np.array(correspondences_data[0])  # First array: actual correspondences
            raw_scores = np.array(correspondences_data[1])       # Second array: scores for each correspondence
        else:
            # Fallback for old format
            correspondences = np.array(correspondences_data)
            raw_scores = pair_data.get('match_scores', pair_data.get('confidence_scores', []))
        
        if len(correspondences) == 0:
            return {
                'total_score': 0.0,
                'component_scores': {},
                'error': 'No correspondences available'
            }
        
        scores = {}
        
        # Extract points
        pts1 = correspondences[:, :2].astype(np.float32)
        pts2 = correspondences[:, 2:].astype(np.float32)
        
        # Calculate image areas
        image1_area = image1_size[0] * image1_size[1]
        image2_area = image2_size[0] * image2_size[1]
        
        # Get score type and raw scores
        score_type_str = pair_data.get('score_type', 'distance')
        try:
            score_type = ScoreType(score_type_str)
        except ValueError:
            print(f"Warning: Unknown score type '{score_type_str}', defaulting to 'distance'")
            score_type = ScoreType.DISTANCE
        
        # Adjust thresholds for incremental reconstruction
        is_incremental = existing_cameras is not None and len(existing_cameras) >= 2
        min_matches_threshold = (self.config.min_matches_incremental if is_incremental 
                            else self.config.min_matches)
        
        # 1. Number of matches score (normalized by average image area)
        num_matches = len(correspondences)
        avg_image_area = (image1_area + image2_area) / 2
        area_factor = np.sqrt(avg_image_area / (1920 * 1080))  # Normalize to FHD baseline
        match_saturation = (200 if not is_incremental else 100) * area_factor
        scores['num_matches'] = min(num_matches / match_saturation, 1.0)
        
        # 2. Spatial distribution score (calculated separately for each image)
        scores['match_distribution'] = self._calculate_spatial_distribution(
            pts1, pts2, image1_size, image2_size
        )
        
        # 3. Coverage score (how well matches cover both images)
        scores['coverage'] = self._calculate_coverage_score(
            pts1, pts2, image1_size, image2_size
        )
        
        # 4. Geometric consistency score
        scores['geometric_consistency'] = self._calculate_geometric_consistency(pts1, pts2)
        
        # 5. Baseline adequacy score (considering image sizes)
        scores['baseline_adequacy'] = self._calculate_baseline_score_with_sizes(
            pts1, pts2, image1_size, image2_size
        )
        
        score_distribution = self._analyze_score_distribution(raw_scores, score_type)
        scores['score_consistency'] = score_distribution['consistency']
        
        # 6. Match quality score (normalized based on score type)
        scores['match_quality'] = self._normalize_match_scores(
            raw_scores, 
            score_type, 
            pair_data.get('method', 'unknown')
        )

        score_stats = {
            'score_mean': score_distribution['mean'],
            'score_median': score_distribution['median'],
            'score_std': score_distribution['std'],
            'score_iqr': score_distribution['percentile_75'] - score_distribution['percentile_25']
        }


        # 7. Score type specific adjustments
        scores['method_confidence'] = self._get_method_confidence(
            score_type,
            pair_data.get('method', 'unknown'),
            num_matches,
            scores['match_quality']
        )
        
        # 8. Resolution compatibility score
        scores['resolution_compatibility'] = self._calculate_resolution_compatibility(
            image1_size, image2_size
        )
        
        # 9. Incremental-specific: connectivity score
        if is_incremental:
            scores['connectivity'] = self._calculate_connectivity_score(pair_key, existing_cameras)
            
            # Adjust weights for incremental reconstruction
            weights = self.config
            total_score = (
                scores['num_matches'] * weights.num_matches_weight * 0.6 +
                scores['match_distribution'] * weights.match_distribution_weight * 0.9 +
                scores['coverage'] * 0.15 +  # Coverage is important
                scores['geometric_consistency'] * weights.geometric_consistency_weight +
                scores['baseline_adequacy'] * weights.baseline_adequacy_weight +
                scores['match_quality'] * weights.match_confidence_weight +
                scores['resolution_compatibility'] * 0.05 +
                scores['method_confidence'] * 0.1 +
                scores['connectivity'] * 0.3
            )
        else:
            # Original scoring for initialization
            total_score = (
                scores['num_matches'] * self.config.num_matches_weight +
                scores['match_distribution'] * self.config.match_distribution_weight +
                scores['coverage'] * 0.15 +  # Add coverage importance
                scores['geometric_consistency'] * self.config.geometric_consistency_weight +
                scores['baseline_adequacy'] * self.config.baseline_adequacy_weight +
                scores['match_quality'] * self.config.match_confidence_weight +
                scores['resolution_compatibility'] * 0.05 +
                scores['method_confidence'] * 0.1
            )
        
        # Apply score type-specific scaling
        total_score = self._apply_score_type_scaling(total_score, score_type, pair_data.get('method'))
        
        # Calculate additional statistics
        mean_displacement = np.mean(np.linalg.norm(pts2 - pts1, axis=1))
        
        # Normalize displacement by image diagonal average
        diag1 = np.sqrt(image1_size[0]**2 + image1_size[1]**2)
        diag2 = np.sqrt(image2_size[0]**2 + image2_size[1]**2)
        avg_diagonal = (diag1 + diag2) / 2
        normalized_displacement = mean_displacement / avg_diagonal if avg_diagonal > 0 else 0
        
        return {
            'total_score': total_score,
            'component_scores': scores,
            'num_matches': num_matches,
            'inlier_ratio': scores['geometric_consistency'],
            'mean_displacement': mean_displacement,
            'normalized_displacement': normalized_displacement,
            'score_type': score_type.value,
            'method': pair_data.get('method', 'unknown'),
            'image1_size': image1_size,
            'image2_size': image2_size,
            'is_incremental': is_incremental,
            'score_statistics': score_stats  # ADD: Include score distribution stats
        }


    def _calculate_spatial_distribution(self, pts1: np.ndarray, pts2: np.ndarray,
                                    image1_size: Tuple[int, int], 
                                    image2_size: Tuple[int, int]) -> float:
        """
        Calculate how well distributed the matches are across both images
        
        Args:
            pts1: Points in first image
            pts2: Points in second image
            image1_size: (width, height) of first image
            image2_size: (width, height) of second image
            
        Returns:
            Distribution score in [0, 1]
        """
        # Calculate standard deviation for each image normalized by image dimensions
        pts1_std_norm = np.std(pts1 / np.array(image1_size[:2]), axis=0)
        pts2_std_norm = np.std(pts2 / np.array(image2_size[:2]), axis=0)
        
        # Calculate normalized spread (product of normalized std devs)
        spread1 = pts1_std_norm[0] * pts1_std_norm[1]
        spread2 = pts2_std_norm[0] * pts2_std_norm[1]
        
        # Good distribution has spread around 0.08-0.15 (normalized)
        optimal_spread = 0.12
        
        # Score based on how close to optimal spread
        dist_score1 = 1.0 - abs(spread1 - optimal_spread) / optimal_spread
        dist_score2 = 1.0 - abs(spread2 - optimal_spread) / optimal_spread
        
        # Combine scores (both images should have good distribution)
        combined_score = (dist_score1 + dist_score2) / 2
        
        return max(0, min(1, combined_score))


    def _calculate_coverage_score(self, pts1: np.ndarray, pts2: np.ndarray,
                                image1_size: Tuple[int, int], 
                                image2_size: Tuple[int, int]) -> float:
        """
        Calculate how well the matches cover both images
        
        Args:
            pts1: Points in first image
            pts2: Points in second image
            image1_size: (width, height) of first image
            image2_size: (width, height) of second image
            
        Returns:
            Coverage score in [0, 1]
        """
        # Create bounding boxes for points
        bbox1_min = np.min(pts1, axis=0)
        bbox1_max = np.max(pts1, axis=0)
        bbox2_min = np.min(pts2, axis=0)
        bbox2_max = np.max(pts2, axis=0)
        
        # Calculate coverage areas
        bbox1_area = (bbox1_max[0] - bbox1_min[0]) * (bbox1_max[1] - bbox1_min[1])
        bbox2_area = (bbox2_max[0] - bbox2_min[0]) * (bbox2_max[1] - bbox2_min[1])
        
        # Calculate coverage ratios
        coverage1 = bbox1_area / (image1_size[0] * image1_size[1])
        coverage2 = bbox2_area / (image2_size[0] * image2_size[1])
        
        # Good coverage is typically 0.3-0.8 of image area
        min_good_coverage = 0.3
        max_good_coverage = 0.8
        
        def score_coverage(cov):
            if cov < min_good_coverage:
                return cov / min_good_coverage
            elif cov > max_good_coverage:
                return 1.0 - (cov - max_good_coverage) / (1.0 - max_good_coverage) * 0.2
            else:
                return 1.0
        
        # Score both coverages
        score1 = score_coverage(coverage1)
        score2 = score_coverage(coverage2)
        
        # Both images should have good coverage
        return (score1 + score2) / 2


    def _calculate_baseline_score_with_sizes(self, pts1: np.ndarray, pts2: np.ndarray,
                                            image1_size: Tuple[int, int], 
                                            image2_size: Tuple[int, int]) -> float:
        """
        Calculate baseline adequacy considering different image sizes
        
        Args:
            pts1: Points in first image
            pts2: Points in second image
            image1_size: (width, height) of first image
            image2_size: (width, height) of second image
            
        Returns:
            Baseline score in [0, 1]
        """
        # Calculate average displacement normalized by image dimensions
        displacements = pts2 - pts1
        
        # Normalize displacements by respective image dimensions
        # Use average of dimensions for normalization
        avg_width = (image1_size[0] + image2_size[0]) / 2
        avg_height = (image1_size[1] + image2_size[1]) / 2
        
        norm_displacements = displacements / np.array([avg_width, avg_height])
        norm_displacement_magnitudes = np.linalg.norm(norm_displacements, axis=1)
        
        mean_norm_displacement = np.mean(norm_displacement_magnitudes)
        
        # Good baseline typically has normalized displacement of 0.05-0.3
        if mean_norm_displacement < 0.02:
            # Too small baseline
            return mean_norm_displacement / 0.02 * 0.5
        elif mean_norm_displacement > 0.4:
            # Very large baseline (might be good for wide baseline)
            return max(0.3, 1.0 - (mean_norm_displacement - 0.4) / 0.6)
        else:
            # Optimal range
            optimal = 0.15
            distance_from_optimal = abs(mean_norm_displacement - optimal)
            return 1.0 - distance_from_optimal / optimal * 0.3


    def _calculate_resolution_compatibility(self, image1_size: Tuple[int, int], 
                                        image2_size: Tuple[int, int]) -> float:
        """
        Calculate how compatible the image resolutions are
        
        Args:
            image1_size: (width, height) of first image
            image2_size: (width, height) of second image
            
        Returns:
            Compatibility score in [0, 1]
        """
        # Calculate resolution ratios
        width_ratio = min(image1_size[0], image2_size[0]) / max(image1_size[0], image2_size[0])
        height_ratio = min(image1_size[1], image2_size[1]) / max(image1_size[1], image2_size[1])
        
        # Calculate area ratio
        area1 = image1_size[0] * image1_size[1]
        area2 = image2_size[0] * image2_size[1]
        area_ratio = min(area1, area2) / max(area1, area2)
        
        # Calculate aspect ratio difference
        aspect1 = image1_size[0] / image1_size[1]
        aspect2 = image2_size[0] / image2_size[1]
        aspect_similarity = min(aspect1, aspect2) / max(aspect1, aspect2)
        
        # Combine scores
        # Area ratio is most important, then aspect ratio, then individual dimensions
        compatibility = (
            area_ratio * 0.4 +
            aspect_similarity * 0.3 +
            width_ratio * 0.15 +
            height_ratio * 0.15
        )
        
        # Apply non-linear scaling (small differences are ok, large differences are penalized)
        return compatibility ** 0.5


    def _normalize_match_scores(self, raw_scores: list, score_type: ScoreType, 
                            method: str) -> float:
        """
        Normalize match scores based on their type
        
        Args:
            raw_scores: List of raw match scores
            score_type: Type of scores (distance, confidence, similarity)
            method: Matching method used
            
        Returns:
            Normalized score in [0, 1] where 1 is best
        """
        if raw_scores is None or len(raw_scores) == 0:
            # Return neutral score if no scores available
            return 0.5
        
        # Ensure it's a numpy array
        scores_array = np.array(raw_scores) if not isinstance(raw_scores, np.ndarray) else raw_scores
        
        if score_type == ScoreType.DISTANCE:
            # Lower is better - need to invert
            # Common distance ranges by method
            if 'sift' in method.lower() or 'surf' in method.lower():
                # SIFT/SURF distances typically 0-500
                max_distance = 500.0
                normalized = 1.0 - np.clip(scores_array / max_distance, 0, 1)
            elif 'orb' in method.lower():
                # ORB Hamming distances typically 0-64
                max_distance = 64.0
                normalized = 1.0 - np.clip(scores_array / max_distance, 0, 1)
            elif 'akaze' in method.lower() or 'brisk' in method.lower():
                # Binary descriptors, Hamming distance
                max_distance = 256.0
                normalized = 1.0 - np.clip(scores_array / max_distance, 0, 1)
            else:
                # Generic normalization using percentiles
                p95 = np.percentile(scores_array, 95)
                normalized = 1.0 - np.clip(scores_array / (p95 + 1e-6), 0, 1)
                
        elif score_type == ScoreType.CONFIDENCE:
            # Higher is better - already in correct direction
            if 'superpoint' in method.lower() or 'superglue' in method.lower():
                # SuperPoint/SuperGlue confidence typically 0-1
                normalized = scores_array
            elif 'loftr' in method.lower() or 'lightglue' in method.lower():
                # LoFTR/LightGlue confidence scores
                normalized = scores_array  # Usually already normalized
            elif 'd2net' in method.lower():
                # D2-Net scores
                normalized = np.clip(scores_array, 0, 1)
            else:
                # Generic normalization to [0, 1]
                min_score = np.min(scores_array)
                max_score = np.max(scores_array)
                if max_score > min_score:
                    normalized = (scores_array - min_score) / (max_score - min_score)
                else:
                    normalized = np.ones_like(scores_array) * 0.5
                    
        elif score_type == ScoreType.SIMILARITY:
            # Higher is better
            if 'ncc' in method.lower():
                # Normalized Cross-Correlation: -1 to 1, convert to 0-1
                normalized = (scores_array + 1.0) / 2.0
            elif 'zncc' in method.lower():
                # Zero-mean NCC: -1 to 1, convert to 0-1
                normalized = (scores_array + 1.0) / 2.0
            else:
                # Assume similarity is in [0, 1] or needs normalization
                if np.min(scores_array) < 0:
                    # Has negative values, shift and scale
                    min_score = np.min(scores_array)
                    max_score = np.max(scores_array)
                    normalized = (scores_array - min_score) / (max_score - min_score + 1e-6)
                else:
                    # Assume already in reasonable range
                    normalized = np.clip(scores_array, 0, 1)
        else:
            # Unknown type, use neutral scoring
            normalized = np.ones_like(scores_array) * 0.5
        
        # Return mean of normalized scores
        return float(np.mean(normalized))


    def _analyze_score_distribution(self, raw_scores: np.ndarray, score_type: ScoreType) -> Dict:
        """
        Analyze the distribution of correspondence scores
        
        Args:
            raw_scores: Array of scores (one per correspondence)
            score_type: Type of scores
            
        Returns:
            Dictionary with distribution statistics
        """
        if raw_scores is None or len(raw_scores) == 0:
            return {
                'mean': 0.5,
                'std': 0.0,
                'median': 0.5,
                'percentile_25': 0.5,
                'percentile_75': 0.5,
                'consistency': 0.0
            }
        
        scores = np.array(raw_scores)
        
        # For distance scores, lower is better, so invert for statistics
        if score_type == ScoreType.DISTANCE:
            # Normalize and invert
            max_val = np.max(scores) if len(scores) > 0 else 1.0
            if max_val > 0:
                scores = 1.0 - (scores / max_val)
        
        stats = {
            'mean': np.mean(scores),
            'std': np.std(scores),
            'median': np.median(scores),
            'percentile_25': np.percentile(scores, 25),
            'percentile_75': np.percentile(scores, 75),
        }
        
        # Consistency score: how consistent are the scores (lower std is better)
        # Normalized by mean to handle different scales
        if stats['mean'] > 0:
            stats['consistency'] = 1.0 - min(stats['std'] / stats['mean'], 1.0)
        else:
            stats['consistency'] = 0.0
        
        return stats

    def _get_method_confidence(self, score_type: ScoreType, method: str, 
                            num_matches: int, match_quality: float) -> float:
        """
        Get confidence factor based on the matching method and score type
        
        Args:
            score_type: Type of scores
            method: Matching method name
            num_matches: Number of matches
            match_quality: Normalized match quality score
            
        Returns:
            Method confidence factor in [0, 1]
        """
        method_lower = method.lower()
        
        # Base confidence by method type
        if score_type == ScoreType.CONFIDENCE:
            # Deep learning methods with confidence scores
            if 'superglue' in method_lower:
                base_confidence = 0.95
            elif 'lightglue' in method_lower:
                base_confidence = 0.92
            elif 'loftr' in method_lower:
                base_confidence = 0.90
            elif 'superpoint' in method_lower:
                base_confidence = 0.88
            elif 'd2net' in method_lower:
                base_confidence = 0.85
            else:
                base_confidence = 0.80
                
        elif score_type == ScoreType.DISTANCE:
            # Traditional feature matchers
            if 'sift' in method_lower:
                base_confidence = 0.85
            elif 'surf' in method_lower:
                base_confidence = 0.82
            elif 'orb' in method_lower:
                base_confidence = 0.75
            elif 'akaze' in method_lower:
                base_confidence = 0.78
            elif 'brisk' in method_lower:
                base_confidence = 0.76
            else:
                base_confidence = 0.70
                
        elif score_type == ScoreType.SIMILARITY:
            # Correlation-based methods
            if 'zncc' in method_lower:
                base_confidence = 0.80
            elif 'ncc' in method_lower:
                base_confidence = 0.78
            else:
                base_confidence = 0.75
        else:
            base_confidence = 0.60
        
        # Adjust based on match quality and quantity
        if num_matches < 20:
            confidence_penalty = 0.8
        elif num_matches < 50:
            confidence_penalty = 0.9
        else:
            confidence_penalty = 1.0
        
        # Combine base confidence with match quality
        final_confidence = base_confidence * confidence_penalty * (0.5 + 0.5 * match_quality)
        
        return min(final_confidence, 1.0)


    def _apply_score_type_scaling(self, base_score: float, score_type: ScoreType, 
                                method: Optional[str] = None) -> float:
        """
        Apply final scaling based on score type and method reliability
        
        Args:
            base_score: Base calculated score
            score_type: Type of scores
            method: Matching method name
            
        Returns:
            Scaled final score
        """
        if method:
            method_lower = method.lower()
        else:
            method_lower = ""
        
        # Reliability multipliers based on empirical performance
        if score_type == ScoreType.CONFIDENCE:
            # Deep learning methods tend to be more reliable
            if 'superglue' in method_lower or 'lightglue' in method_lower:
                scale = 1.1  # Slight boost for highly reliable methods
            elif 'loftr' in method_lower:
                scale = 1.05
            else:
                scale = 1.0
                
        elif score_type == ScoreType.DISTANCE:
            # Traditional methods may need adjustment
            if 'sift' in method_lower:
                scale = 1.0  # SIFT is gold standard
            elif 'orb' in method_lower:
                scale = 0.9  # ORB can be noisy
            else:
                scale = 0.95
                
        elif score_type == ScoreType.SIMILARITY:
            # Correlation methods
            scale = 0.95  # Slightly conservative
        else:
            scale = 0.85  # Unknown methods get penalized
        
        return min(base_score * scale, 1.0)  # Cap at 1.0



# Factory function for easy configuration
def create_monument_pair_selector(monument_type: str = 'general') -> InitializationPairSelector:
    """
    Create a pair selector configured for specific monument types
    
    Args:
        monument_type: 'general', 'building', 'statue', 'ruins'
        
    Returns:
        Configured InitializationPairSelector
    """
    if monument_type == 'building':
        config = ScoringConfig(
            min_matches=60,
            min_inlier_ratio=0.4,
            optimal_displacement_range=(15.0, 120.0),
            match_distribution_weight=0.3,
            min_matches_incremental=35
        )
    elif monument_type == 'statue':
        config = ScoringConfig(
            min_matches=40,
            min_inlier_ratio=0.35,
            optimal_displacement_range=(8.0, 80.0),
            geometric_consistency_weight=0.3,
            min_matches_incremental=25
        )
    elif monument_type == 'ruins':
        config = ScoringConfig(
            min_matches=35,
            min_inlier_ratio=0.25,
            optimal_displacement_range=(12.0, 150.0),
            baseline_adequacy_weight=0.2,
            min_matches_incremental=20
        )
    else:  # general
        config = ScoringConfig()
    
    return InitializationPairSelector(config)