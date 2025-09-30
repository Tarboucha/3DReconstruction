"""
Main processing pipeline for feature detection and matching.

This module contains the high-level pipeline classes that orchestrate
the entire feature detection and matching workflow.
"""

import cv2
import numpy as np
import time
import pickle
import json
import os
import glob
from itertools import combinations
from typing import List, Tuple, Set, Dict, Optional, Union, Any

from .core_data_structures import FeatureData, MatchData, EnhancedDMatch, MultiMethodMatchData, DetectorType, ScoreType
from .traditional_detectors import create_traditional_detector
from .deep_learning_detectors import create_deep_learning_detector
from .feature_matchers import (
    create_traditional_matcher, 
    create_deep_learning_matcher, 
    auto_select_matcher
)
from .utils import (
    enhanced_filter_matches_with_homography,
    adaptive_match_filtering,
    extract_correspondences,
    visualize_matches_with_scores,
    MatchQualityAnalyzer,
    save_enhanced_results,
    FolderImageSource,
    ImageInfo,
    ImageSource,
    load_images_from_folder,
    load_single_image
)


class MultiMethodFeatureDetector:
    """Combines multiple feature detection methods"""
    
    def __init__(self, 
                 methods: List[Union[str, DetectorType]], 
                 max_features_per_method: int = 2000,
                 combine_strategy: str = "independent",  # ✓ Changed default
                 detector_params: Optional[Dict[str, Dict]] = None):
        """
        Initialize multi-method detector
        
        Args:
            methods: List of detection methods to use
            max_features_per_method: Max features per individual method
            combine_strategy: 'independent' (for matching), 'best' (single method),
                            'spatial_merge' (keypoints only, for single-image analysis)
            detector_params: Optional parameters for each detector
        """
        self.methods = methods
        self.max_features_per_method = max_features_per_method
        self.combine_strategy = combine_strategy
        self.detector_params = detector_params or {}
        
        # Validate strategy
        valid_strategies = ['independent', 'best', 'spatial_merge']
        if combine_strategy not in valid_strategies:
            raise ValueError(f"combine_strategy must be one of {valid_strategies}")
        
        self.detectors = self._initialize_detectors()
        self.lightglue_matcher = self._initialize_lightglue()

    def _initialize_detectors(self) -> Dict[str, Any]:
        """Initialize all requested detectors"""
        detectors = {}
        
        # Traditional detectors
        traditional_methods = ['SIFT', 'ORB', 'AKAZE', 'BRISK', 'Harris', 'GoodFeatures']
        
        # Deep learning detectors  
        deep_learning_methods = ['SuperPoint', 'DISK', 'ALIKED']
        
        for method in self.methods:
            method_str = str(method)
            
            # Skip LightGlue as it's handled separately
            if method_str.lower() in ['lightglue']:
                continue
                
            try:
                params = self.detector_params.get(method_str, {})
                params['max_features'] = self.max_features_per_method
                
                if method_str in traditional_methods:
                    detector = create_traditional_detector(method_str, **params)
                    detectors[method_str] = detector
                    print(f"Initialized {method_str} detector")
                    
                elif method_str in deep_learning_methods:
                    detector = create_deep_learning_detector(method_str, **params)
                    detectors[method_str] = detector
                    print(f"Initialized {method_str} detector")
                    
                else:
                    print(f"Unknown detector type: {method_str}")
                    
            except Exception as e:
                print(f"Failed to initialize {method_str}: {e}")
        
        return detectors
    
    def _initialize_lightglue(self):
        """Initialize LightGlue matcher if requested"""
        lightglue_methods = ['lightglue', 'LightGlue']
        
        if any(str(method).lower() in [m.lower() for m in lightglue_methods] 
               for method in self.methods):
            try:
                params = self.detector_params.get('lightglue', {})
                lightglue = create_deep_learning_matcher('LightGlue', **params)
                print("Initialized LightGlue matcher")
                return lightglue
            except Exception as e:
                print(f"Failed to initialize LightGlue: {e}")
                return None
        
        return None
    
    def detect_all(self, image: np.ndarray) -> Dict[str, FeatureData]:
            """Detect features using all methods - ALWAYS returns separate results"""
            results = {}
            
            for name, detector in self.detectors.items():
                print(f"Detecting with {name}...")
                try:
                    features = detector.detect(image)
                    results[name] = features
                    print(f"  Found {len(features)} features in {features.detection_time:.3f}s")
                except Exception as e:
                    print(f"  Failed: {str(e)}")
                    results[name] = FeatureData([], None, name, raw_image=image)
            
            return results
    
    def detect_combined(self, image: np.ndarray) -> FeatureData:
        """
        Detect and combine features from all methods
        
        WARNING: This should ONLY be used for single-image analysis where you want
        diverse keypoint coverage. DO NOT use the result for matching against another
        image's combined features - descriptors from different methods are incompatible!
        """
        all_features = self.detect_all(image)
        
        if self.combine_strategy == "independent":
            # For matching: don't combine, return empty (matching should be done separately)
            print("Warning: detect_combined() called with 'independent' strategy")
            print("For matching, use detect_all() and match each method independently")
            return FeatureData([], None, "NotCombined", raw_image=image)
            
        elif self.combine_strategy == "best":
            return self._select_best_features(all_features, image)
            
        elif self.combine_strategy == "spatial_merge":
            return self._spatial_merge_keypoints(all_features, image)
            
        else:
            raise ValueError(f"Unknown combine strategy: {self.combine_strategy}")
    
    def _select_best_features(self, features_dict: Dict[str, FeatureData],
                             image: np.ndarray) -> FeatureData:
        """Select features from the best performing method - SAFE for matching"""
        best_method = None
        best_count = 0
        
        for method_name, features in features_dict.items():
            if len(features) > best_count:
                best_count = len(features)
                best_method = method_name
        
        if best_method:
            result = features_dict[best_method]
            result.raw_image = image
            return result
        
        return FeatureData([], None, "None", raw_image=image)
    
    def _spatial_merge_keypoints(self, features_dict: Dict[str, FeatureData], 
                                image: np.ndarray) -> FeatureData:
        """
        Merge keypoints spatially (for visualization/coverage analysis only)
        
        ⚠️ WARNING: Results CANNOT be used for matching!
        Descriptors are set to None since they're incompatible across methods.
        """
        all_keypoints = []
        methods_used = []
        total_time = 0
        
        for method_name, features in features_dict.items():
            if len(features) > 0:
                all_keypoints.extend(features.keypoints)
                methods_used.append(method_name)
                total_time += features.detection_time
        
        # Remove spatial duplicates
        unique_keypoints = self._remove_duplicate_keypoints(all_keypoints)
        
        return FeatureData(
            keypoints=unique_keypoints,
            descriptors=None,  # ✓ No descriptors - incompatible!
            method=f"SpatialMerge({','.join(methods_used)})",
            detection_time=total_time,
            raw_image=image
        )
    
    def _remove_duplicate_keypoints(self, keypoints: List[cv2.KeyPoint], 
                                   threshold: float = 5.0) -> List[cv2.KeyPoint]:
        """Remove duplicate keypoints within threshold distance"""
        if not keypoints:
            return []
        
        unique = []
        for kp in keypoints:
            is_duplicate = False
            for unique_kp in unique:
                dist = np.sqrt((kp.pt[0] - unique_kp.pt[0])**2 + 
                              (kp.pt[1] - unique_kp.pt[1])**2)
                if dist < threshold:
                    is_duplicate = True
                    # Keep the one with higher response
                    if kp.response > unique_kp.response:
                        unique.remove(unique_kp)
                        unique.append(kp)
                    break
            
            if not is_duplicate:
                unique.append(kp)
        
        return unique

    def _select_best_features(self, features_dict: Dict[str, FeatureData],
                             image: np.ndarray) -> FeatureData:
        """Select features from the best performing method"""
        best_method = None
        best_count = 0
        
        for method_name, features in features_dict.items():
            if len(features) > best_count:
                best_count = len(features)
                best_method = method_name
        
        if best_method:
            result = features_dict[best_method]
            result.raw_image = image
            return result
        
        return FeatureData([], None, "None", raw_image=image)
    

class FeatureProcessingPipeline:
    """Complete pipeline for feature detection and matching"""
    

    def __init__(self, config: Dict[str, Any]):
            """Initialize pipeline"""
            self.config = config
            
            # ✓ Force 'independent' strategy for multi-method pipelines
            combine_strategy = config.get('combine_strategy', 'independent')
            if len(config.get('methods', [])) > 1 and combine_strategy not in ['independent', 'best']:
                print(f"Warning: combine_strategy '{combine_strategy}' not suitable for multi-method matching")
                print("Forcing 'independent' strategy for correct matching")
                combine_strategy = 'independent'
            
            self.multi_detector = MultiMethodFeatureDetector(
                methods=config.get('methods', ['SIFT']),
                max_features_per_method=config.get('max_features', 2000),
                combine_strategy=combine_strategy,
                detector_params=config.get('detector_params', {})
            )
            self.analyzer = MatchQualityAnalyzer()
    
    def process_image_pair(self, img1: np.ndarray, img2: np.ndarray,
                        visualize: bool = True,
                        filter_matches: bool = True) -> Dict:
        """Process a single image pair with multi-method support"""
        print("Processing image pair...")
        
        # Check what methods are available
        has_lightglue = self.multi_detector.lightglue_matcher is not None
        has_traditional = len(self.multi_detector.detectors) > 0
        
        # Single method processing
        if has_lightglue and not has_traditional:
            return self._process_with_lightglue(img1, img2, visualize, filter_matches)
        elif has_traditional and not has_lightglue and len(self.multi_detector.detectors) == 1:
            return self._process_single_traditional(img1, img2, visualize, filter_matches)
        
        # Multi-method processing - ALWAYS use independent matching
        return self._process_merged_methods(img1, img2, visualize, filter_matches)
    
    def _process_single_traditional(self, img1: np.ndarray, img2: np.ndarray,
                                   visualize: bool, filter_matches: bool) -> Dict:
        """Process using a SINGLE traditional method"""
        print("Using single traditional method...")
        
        # Get the single method
        method_name = list(self.multi_detector.detectors.keys())[0]
        detector = self.multi_detector.detectors[method_name]
        
        # Detect features
        features1 = detector.detect(img1)
        features2 = detector.detect(img2)
        
        print(f"Image 1: {len(features1)} features ({features1.method})")
        print(f"Image 2: {len(features2)} features ({features2.method})")
        
        if len(features1) == 0 or len(features2) == 0:
            print("No features detected!")
            return {
                'features1': features1,
                'features2': features2,
                'match_data': MatchData([]),
                'correspondences': np.array([]).reshape(0, 4),
                'method_used': method_name
            }
        
        # Match features
        matcher = auto_select_matcher(features1, features2)
        match_data = matcher.match(features1, features2)
        
        print(f"Found {len(match_data.matches)} matches")
        
        if filter_matches and len(match_data.matches) >= 4:
            filtered_data = adaptive_match_filtering(match_data, top_k=200)
            final_matches, homography = enhanced_filter_matches_with_homography(
                features1.keypoints, features2.keypoints,
                filtered_data.get_best_matches(), filtered_data
            )
            filtered_data.filtered_matches = final_matches
            filtered_data.homography = homography
            print(f"Filtered to {len(final_matches)} good matches")
            match_data = filtered_data
        
        if visualize and len(match_data.get_best_matches()) > 0:
            visualize_matches_with_scores(
                img1, img2, features1.keypoints, features2.keypoints,
                match_data, f"{features1.method} Matches"
            )
        
        return {
            'features1': features1,
            'features2': features2,
            'match_data': match_data,
            'correspondences': extract_correspondences(
                features1.keypoints, features2.keypoints,
                match_data.get_best_matches()
            ),
            'method_used': features1.method
        }

    def _process_with_lightglue(self, img1: np.ndarray, img2: np.ndarray,
                               visualize: bool, filter_matches: bool) -> Dict:
        """Process using LightGlue matcher"""
        print("Using LightGlue processing...")
        
        try:
            features1, features2, match_data = self.multi_detector.lightglue_matcher.process_pair(img1, img2)
            
            print(f"LightGlue found {len(match_data.matches)} matches")
            
            if filter_matches and len(match_data.matches) >= 4:
                # Apply adaptive filtering
                filtered_data = adaptive_match_filtering(match_data, top_k=500)
                
                # Apply homography filtering
                final_matches, homography = enhanced_filter_matches_with_homography(
                    features1.keypoints, features2.keypoints,
                    filtered_data.get_best_matches(), filtered_data
                )
                filtered_data.filtered_matches = final_matches
                filtered_data.homography = homography
                
                print(f"Filtered to {len(final_matches)} good matches")
                match_data = filtered_data
            
            if visualize and len(match_data.get_best_matches()) > 0:
                visualize_matches_with_scores(
                    img1, img2, features1.keypoints, features2.keypoints,
                    match_data, "LightGlue Matches"
                )
            
            return {
                'features1': features1,
                'features2': features2,
                'match_data': match_data,
                'correspondences': extract_correspondences(
                    features1.keypoints, features2.keypoints,
                    match_data.get_best_matches()
                ),
                'method_used': 'LightGlue'
            }
            
        except Exception as e:
            print(f"LightGlue processing failed: {e}")
            # Fallback to traditional methods
            return self._process_with_traditional(img1, img2, visualize, filter_matches)

    def process_image_pair_comprehensive(self, img1: np.ndarray, img2: np.ndarray,
                                       visualize: bool = True,
                                       analyze_quality: bool = True) -> Dict:
        """Comprehensive processing with multiple methods and analysis"""
        print("Comprehensive processing with multiple methods...")
        results = {}
        
        # 1. Try LightGlue if available
        if self.multi_detector.lightglue_matcher is not None:
            print("\n1. LightGlue processing...")
            try:
                lg_result = self._process_with_lightglue(img1, img2, False, True)
                results['lightglue'] = lg_result
                print(f"  LightGlue: {len(lg_result['match_data'].get_best_matches())} matches")
            except Exception as e:
                print(f"  LightGlue failed: {e}")
        
        # 2. Try traditional methods
        traditional_methods = [m for m in self.config.get('methods', []) 
                             if str(m).lower() not in ['lightglue']]
        
        if traditional_methods:
            print(f"\n2. Traditional methods: {traditional_methods}")
            try:
                trad_result = self._process_with_traditional(img1, img2, False, True)
                results['traditional'] = trad_result
                print(f"  {trad_result['method_used']}: {len(trad_result['match_data'].get_best_matches())} matches")
            except Exception as e:
                print(f"  Traditional methods failed: {e}")
        
        # 3. Quality analysis
        if analyze_quality and results:
            print("\n3. Quality analysis...")
            quality_comparison = {}
            
            for method_name, result in results.items():
                analysis = self.analyzer.analyze_match_data(result['match_data'])
                quality_comparison[method_name] = analysis
                print(f"  {method_name}: {analysis['num_matches']} matches, "
                      f"quality={analysis['quality_score']:.3f}")
            
            # Find best method
            if quality_comparison:
                best_method = max(quality_comparison.keys(), 
                                key=lambda x: quality_comparison[x]['quality_score'])
                print(f"  Best method: {best_method}")
        
        # 4. Select best result
        if not results:
            print("No successful matches!")
            empty_features = FeatureData([], None, "None")
            return {
                'features1': empty_features,
                'features2': empty_features,
                'match_data': MatchData([]),
                'correspondences': np.array([]).reshape(0, 4),
                'all_results': {},
                'quality_analysis': {}
            }
        
        # Find best result by quality score
        best_method = None
        best_quality = -1
        
        for method_name, result in results.items():
            analysis = self.analyzer.analyze_match_data(result['match_data'])
            if analysis['quality_score'] > best_quality:
                best_quality = analysis['quality_score']
                best_method = method_name
        
        best_result = results[best_method]
        
        # 5. Visualization
        if visualize and len(best_result['match_data'].get_best_matches()) > 0:
            visualize_matches_with_scores(
                img1, img2,
                best_result['features1'].keypoints,
                best_result['features2'].keypoints,
                best_result['match_data'],
                f"{best_method.title()} Matches (Best)"
            )
        
        return {
            'features1': best_result['features1'],
            'features2': best_result['features2'],
            'match_data': best_result['match_data'],
            'correspondences': best_result['correspondences'],
            'method_used': best_method,  
            'best_method': best_method,  
            'all_results': results,
            'quality_analysis': quality_comparison if analyze_quality else {}
        }
        
    def _process_merged_methods(self, img1: np.ndarray, img2: np.ndarray,
                                visualize: bool, filter_matches: bool) -> Dict:
        """
        Merge results from multiple methods
        
        CORRECTED: Each method maintains its own MatchData with correct score_type
        """
        print("Processing multiple methods independently...")
        
        all_features1 = []
        all_features2 = []
        multi_match_data = MultiMethodMatchData()  # ✓ NEW container
        methods_used = []
        total_time = 0
        
        # ==================================================================
        # PHASE 1: LIGHTGLUE
        # ==================================================================
        
        if self.multi_detector.lightglue_matcher is not None:
            try:
                print("\n  Processing LightGlue...")
                lg_features1, lg_features2, lg_match_data = self.multi_detector.lightglue_matcher.match_images_directly(img1, img2)
                
                print(f"    LightGlue: {len(lg_features1)} features, {len(lg_match_data.matches)} matches")
                
                offset1 = len(all_features1)
                offset2 = len(all_features2)
                
                # Store features
                all_features1.append({
                    'keypoints': lg_features1.keypoints,
                    'method': 'LightGlue',
                    'count': len(lg_features1.keypoints)
                })
                
                all_features2.append({
                    'keypoints': lg_features2.keypoints,
                    'method': 'LightGlue',
                    'count': len(lg_features2.keypoints)
                })
                
                # ✓ Store matches with method separation
                multi_match_data.add_method_matches('LightGlue', lg_match_data, offset1, offset2)
                
                methods_used.append('LightGlue')
                total_time += lg_features1.detection_time + lg_match_data.matching_time
                
            except Exception as e:
                print(f"    LightGlue failed: {e}")
        
        # ==================================================================
        # PHASE 2: TRADITIONAL METHODS
        # ==================================================================
        
        for method_name, detector in self.multi_detector.detectors.items():
            try:
                print(f"\n  Processing {method_name}...")
                
                features1 = detector.detect(img1)
                features2 = detector.detect(img2)
                
                if len(features1) == 0 or len(features2) == 0:
                    print(f"    {method_name}: No features detected")
                    continue
                
                if features1.descriptors is None or features2.descriptors is None:
                    print(f"    {method_name}: No descriptors available")
                    continue
                
                print(f"    {method_name}: Detected {len(features1)} and {len(features2)} features")
                
                offset1 = sum(f['count'] for f in all_features1)
                offset2 = sum(f['count'] for f in all_features2)
                
                # Match
                matcher = auto_select_matcher(features1, features2)
                match_data = matcher.match(features1, features2)
                
                print(f"    {method_name}: Found {len(match_data.matches)} matches")
                
                # Store features
                all_features1.append({
                    'keypoints': features1.keypoints,
                    'method': method_name,
                    'count': len(features1.keypoints)
                })
                
                all_features2.append({
                    'keypoints': features2.keypoints,
                    'method': method_name,
                    'count': len(features2.keypoints)
                })
                
                # ✓ Store matches with method separation
                if len(match_data.matches) > 0:
                    multi_match_data.add_method_matches(method_name, match_data, offset1, offset2)
                    methods_used.append(method_name)
                
                total_time += features1.detection_time + features2.detection_time + match_data.matching_time
                
            except Exception as e:
                print(f"    {method_name} failed: {e}")
        
        # ==================================================================
        # PHASE 3: COMBINE AND FILTER
        # ==================================================================
        
        if not all_features1 or not all_features2:
            print("\nNo features detected from any method!")
            return {
                'features1': FeatureData([], None, "None"),
                'features2': FeatureData([], None, "None"),
                'match_data': MultiMethodMatchData(),
                'correspondences': np.array([]).reshape(0, 4),
                'method_used': 'None'
            }
        
        # Combine keypoints
        combined_kp1 = []
        combined_kp2 = []
        
        for feat_set in all_features1:
            combined_kp1.extend(feat_set['keypoints'])
        
        for feat_set in all_features2:
            combined_kp2.extend(feat_set['keypoints'])
        
        # Store in multi_match_data for reference
        multi_match_data.all_keypoints1 = combined_kp1
        multi_match_data.all_keypoints2 = combined_kp2
        
        print(f"\nCombined features: img1={len(combined_kp1)}, img2={len(combined_kp2)}")
        print(f"Combined matches: {len(multi_match_data)}")
        
        # Apply filtering if requested
        if filter_matches and len(multi_match_data) >= 4:
            print("\nApplying filtering to merged matches...")
            
            all_matches = multi_match_data.get_all_matches()
            
            # ✓ Adaptive filtering that respects per-match score types
            filtered_matches = self._adaptive_multi_method_filtering(all_matches, top_k=500)
            
            # Geometric filtering with homography RANSAC
            final_matches, homography = enhanced_filter_matches_with_homography(
                combined_kp1, combined_kp2,
                filtered_matches, multi_match_data  # Pass the container
            )
            
            # Store filtering results
            multi_match_data.homography = homography
            multi_match_data.filtered_match_indices = [
                i for i, m in enumerate(all_matches) if m in final_matches
            ]
            
            print(f"Filtered to {len(final_matches)} good matches")
        
        # Create FeatureData objects
        merged_features1 = FeatureData(
            keypoints=combined_kp1,
            descriptors=None,
            method=f"Merged({','.join(methods_used)})",
            detection_time=total_time,
            raw_image=img1
        )
        
        merged_features2 = FeatureData(
            keypoints=combined_kp2,
            descriptors=None,
            method=f"Merged({','.join(methods_used)})",
            detection_time=0,
            raw_image=img2
        )
        
        # Visualization
        if visualize and len(multi_match_data.get_filtered_matches()) > 0:
            visualize_matches_with_scores(
                img1, img2, combined_kp1, combined_kp2,
                multi_match_data, f"Merged: {', '.join(methods_used)}"
            )
        
        return {
            'features1': merged_features1,
            'features2': merged_features2,
            'match_data': multi_match_data,  # ✓ Now properly structured
            'correspondences': extract_correspondences(
                combined_kp1, combined_kp2,
                multi_match_data.get_filtered_matches()
            ),
            'method_used': f"Merged({','.join(methods_used)})",
            'methods_breakdown': multi_match_data.get_stats()
        }


    def _adaptive_multi_method_filtering(self, matches: List[EnhancedDMatch], 
                                        top_k: int = 500) -> List[EnhancedDMatch]:
        """
        Filter matches respecting per-match score types
        
        ✅ FIXED: Correctly handles mixed score types with proper normalization
        """
        if not matches:
            return []
        
        # First pass: collect scores by type to determine normalization parameters
        distance_scores = []
        confidence_scores = []
        
        for match in matches:
            if isinstance(match, EnhancedDMatch):
                if match.score_type == ScoreType.CONFIDENCE:
                    confidence_scores.append(match.score)
                else:  # DISTANCE
                    distance_scores.append(match.score)
            else:
                # cv2.DMatch - assume distance
                distance_scores.append(match.distance)
        
        # Calculate normalization parameters
        # For distances: use percentile-based normalization to handle outliers
        if distance_scores:
            # Use 95th percentile as max to avoid outlier bias
            distance_max = np.percentile(distance_scores, 95)
            distance_min = min(distance_scores)
        
        # Normalize scores based on type for fair comparison
        normalized_matches = []
        
        for match in matches:
            if isinstance(match, EnhancedDMatch):
                if match.score_type == ScoreType.CONFIDENCE:
                    # Already 0-1, higher is better
                    norm_score = match.score
                else:  # DISTANCE
                    # ✅ FIXED: Normalize to 0-1 range, then invert
                    # Clamp to [distance_min, distance_max] to handle outliers
                    clamped_distance = np.clip(match.score, distance_min, distance_max)
                    
                    # Normalize to 0-1
                    if distance_max > distance_min:
                        normalized_distance = (clamped_distance - distance_min) / (distance_max - distance_min)
                    else:
                        normalized_distance = 0.0
                    
                    # Invert: lower distance = higher quality
                    norm_score = 1.0 - normalized_distance
                
                normalized_matches.append((match, norm_score))
            else:
                # cv2.DMatch - assume distance
                clamped_distance = np.clip(match.distance, distance_min, distance_max)
                if distance_max > distance_min:
                    normalized_distance = (clamped_distance - distance_min) / (distance_max - distance_min)
                else:
                    normalized_distance = 0.0
                norm_score = 1.0 - normalized_distance
                normalized_matches.append((match, norm_score))
        
        # Sort by normalized score (higher is better)
        normalized_matches.sort(key=lambda x: x[1], reverse=True)
        
        # Take top k
        top_matches = [m for m, score in normalized_matches[:top_k]]
        
        return top_matches

    def _create_per_method_stats(self, all_matches: List[Dict]) -> Dict[str, Dict]:
        """
        Create per-method statistics for breakdown analysis
        
        Args:
            all_matches: List of match dictionaries with method information
            
        Returns:
            Dictionary with per-method statistics
        """
        stats = {}
        
        for match_set in all_matches:
            method = match_set['method']
            matches = match_set['matches']
            
            stats[method] = {
                'num_matches': len(matches),
                'offset1': match_set['offset1'],
                'offset2': match_set['offset2']
            }
            
            # Add score statistics if available
            if match_set.get('match_data'):
                match_data = match_set['match_data']
                scores = match_data.get_match_scores()
                
                if len(scores) > 0:
                    stats[method]['score_mean'] = float(np.mean(scores))
                    stats[method]['score_std'] = float(np.std(scores))
                    stats[method]['score_min'] = float(np.min(scores))
                    stats[method]['score_max'] = float(np.max(scores))
                    stats[method]['score_type'] = match_data.score_type.value
        
        return stats
    # =============================================================================
    # New Methods for Folder Processing
    # =============================================================================
    
    def process_folder(self, folder_path: str,
                      max_images: Optional[int] = None,
                      resize_to: Optional[Tuple[int, int]] = None,
                      visualize: bool = False,
                      output_file: Optional[str] = None,
                      save_format: str = 'pickle') -> Dict[str, Any]:
        """
        Process all images in a folder for feature detection and matching
        
        Args:
            folder_path: Path to folder containing images
            max_images: Maximum number of images to process
            resize_to: Resize images to (width, height) if provided
            visualize: Whether to show visualizations
            output_file: Optional file to save results
            save_format: Format to save results ('json', 'pickle', 'both')
            
        Returns:
            Dictionary with processing results
        """
        print(f"Processing folder: {folder_path}")
        
        # Create image source
        image_source = FolderImageSource(
            folder_path=folder_path,
            max_images=max_images,
            resize_to=resize_to
        )
        
        # Get source information
        source_info = image_source.get_source_info()
        print(f"Found {image_source.get_image_count()} images")
        
        # Convert to old format for compatibility with existing process_dataset
        images = [(img_info.image, img_info.identifier) for img_info in image_source.get_images()]
        
        if not images:
            print("No valid images found in folder!")
            return {
                'source_info': source_info,
                'results': {},
                'error': 'No valid images found'
            }
        
        # Process using existing dataset method
        if output_file is None:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            output_file = f"folder_results_{timestamp}"
        
        results = self.process_dataset(
            images=images,
            output_file=output_file,
            save_format=save_format,
            batch_size=100
        )
        
        # Add source information
        results['source_info'] = source_info
        results['folder_path'] = folder_path
        
        print(f"\nFolder processing completed:")
        print(f"  Total images: {len(images)}")
        print(f"  Success rate: {results['overall_stats']['success_rate']:.1%}")
        print(f"  Average matches: {results['overall_stats']['avg_matches']:.1f}")
        
        return results
    
    def process_folder_pairs(self, folder_path: str,
                            max_images: Optional[int] = None,
                            resize_to: Optional[Tuple[int, int]] = None,
                            visualize: bool = False) -> List[Dict[str, Any]]:
        """
        Process consecutive image pairs from a folder
        
        Args:
            folder_path: Path to folder containing images
            max_images: Maximum number of images to load
            resize_to: Resize images to (width, height) if provided
            visualize: Whether to show match visualizations
            
        Returns:
            List of pair processing results
        """
        print(f"Processing image pairs from folder: {folder_path}")
        
        # Create image source
        image_source = FolderImageSource(
            folder_path=folder_path,
            max_images=max_images,
            resize_to=resize_to
        )
        
        results = []
        pair_count = 0
        
        # Process image pairs
        for img1_info, img2_info in image_source.get_image_pairs():
            pair_count += 1
            print(f"\nProcessing pair {pair_count}: {img1_info.identifier} ↔ {img2_info.identifier}")
            
            try:
                result = self.process_image_pair(
                    img1_info.image, img2_info.image,
                    visualize=visualize
                )
                
                # Add metadata
                result['pair_info'] = {
                    'image1': img1_info.identifier,
                    'image2': img2_info.identifier,
                    'image1_size': img1_info.size,
                    'image2_size': img2_info.size,
                    'pair_number': pair_count
                }
                
                results.append(result)
                
                num_matches = len(result['match_data'].get_best_matches())
                print(f"  Result: {num_matches} matches ({result['method_used']})")
                
            except Exception as e:
                print(f"  Error processing pair: {e}")
                error_result = {
                    'error': str(e),
                    'pair_info': {
                        'image1': img1_info.identifier,
                        'image2': img2_info.identifier,
                        'pair_number': pair_count
                    }
                }
                results.append(error_result)
        
        print(f"\nProcessed {len(results)} image pairs")
        successful_pairs = [r for r in results if 'error' not in r]
        print(f"Success rate: {len(successful_pairs)}/{len(results)} ({len(successful_pairs)/len(results):.1%})")
        
        if successful_pairs:
            avg_matches = np.mean([len(r['match_data'].get_best_matches()) for r in successful_pairs])
            print(f"Average matches per pair: {avg_matches:.1f}")
        
        return results
    
    def process_single_image_file(self, image_path: str,
                                 resize_to: Optional[Tuple[int, int]] = None,
                                 create_transformed_pair: bool = True,
                                 visualize: bool = True) -> Dict[str, Any]:
        """
        Process a single image file (creates transformed pair for matching)
        
        Args:
            image_path: Path to image file
            resize_to: Resize image to (width, height) if provided  
            create_transformed_pair: Whether to create a transformed version for matching
            visualize: Whether to show visualizations
            
        Returns:
            Processing results
        """
        print(f"Processing single image: {image_path}")
        
        # Load image
        image, filename = load_single_image(image_path, resize_to)
        
        if create_transformed_pair:
            # Create a simple transformation for demonstration
            h, w = image.shape[:2]
            M = cv2.getRotationMatrix2D((w/2, h/2), 15, 0.9)  # 15 degree rotation, slight scale
            transformed_image = cv2.warpAffine(image, M, (w, h))
            
            print(f"Created transformed pair for matching")
            result = self.process_image_pair(image, transformed_image, visualize=visualize)
            
            result['image_info'] = {
                'original_path': image_path,
                'filename': filename,
                'transformation': 'rotation_15deg_scale_0.9',
                'image_size': (w, h)
            }
            
        else:
            # Just extract features from single image
            features = self.multi_detector.detect_combined(image)
            print(f"Extracted {len(features)} features using {features.method}")
            
            if visualize:
                from .utils import visualize_keypoints
                visualize_keypoints(image, features, f"Features: {filename}")
            
            result = {
                'features': features,
                'image_info': {
                    'original_path': image_path,
                    'filename': filename,
                    'image_size': (w, h)
                },
                'method_used': features.method
            }
        
        return result

    # =============================================================================
    # Enhanced Dataset Processing (existing method with improvements)
    # =============================================================================

    def process_dataset(self, images: List[Tuple[np.ndarray, str]],
                        output_file: str,
                        max_pairs: Optional[int] = None,
                        save_format: str = 'pickle',
                        batch_size: int = 500,
                        resume: bool = True) -> Dict:
        """Process entire dataset of images with batched saving and resume capability"""
        print(f"Processing dataset with {len(images)} images")
        
        # Generate image pairs
        image_pairs = list(combinations(range(len(images)), 2))
        if max_pairs:
            image_pairs = image_pairs[:max_pairs]
        
        # Prepare output file paths
        output_dir = os.path.dirname(output_file) if os.path.dirname(output_file) else '.'
        output_name = os.path.splitext(os.path.basename(output_file))[0]
        output_ext = os.path.splitext(output_file)[1] or ('.pkl' if save_format == 'pickle' else '.json')
        
        # ===== ADDED: Save image metadata to independent pickle file =====
        image_metadata_file = os.path.join(output_dir, f"{output_name}_image_metadata.pkl")
        
        # Extract and store image metadata
        image_metadata = {
            'total_images': len(images),
            'images': []
        }
        
        for img_array, img_name in images:
            img_info = {
                'name': img_name,
                'size': img_array.shape,  # (height, width, channels) or (height, width)
                'height': img_array.shape[0],
                'width': img_array.shape[1],
                'channels': img_array.shape[2] if len(img_array.shape) > 2 else 1,
                'dtype': str(img_array.dtype)
            }
            image_metadata['images'].append(img_info)
        
        # Save image metadata to pickle file
        try:
            with open(image_metadata_file, 'wb') as f:
                pickle.dump(image_metadata, f)
            print(f"Image metadata saved to: {image_metadata_file}")
            print(f"  Total images: {len(images)}")
            print(f"  First image: {image_metadata['images'][0]['name']} - {image_metadata['images'][0]['size']}")
            if len(images) > 1:
                print(f"  Last image: {image_metadata['images'][-1]['name']} - {image_metadata['images'][-1]['size']}")
        except Exception as e:
            print(f"Warning: Failed to save image metadata: {e}")


        def load_existing_results() -> Tuple[Set[Tuple[str, str]], int, Dict]:
            """Load existing batch files and return processed pairs, next batch number, and stats"""
            processed_pairs = set()
            existing_stats = {
                'successful_pairs': 0,
                'total_matches': 0,
                'batch_files': []
            }
            next_batch_num = 1
            
            if not resume:
                print("Resume disabled - starting fresh")
                return processed_pairs, next_batch_num, existing_stats
            
            # Find existing batch files
            batch_pattern = os.path.join(output_dir, f"{output_name}_batch_*.{output_ext.lstrip('.')}")
            existing_batch_files = sorted(glob.glob(batch_pattern))
            
            if not existing_batch_files:
                print("No existing batch files found - starting fresh")
                return processed_pairs, next_batch_num, existing_stats
            
            print(f"Found {len(existing_batch_files)} existing batch files - checking for resume...")
            
            # Load each existing batch file
            valid_batches = []
            for batch_file in existing_batch_files:
                try:
                    if save_format == 'pickle':
                        with open(batch_file, 'rb') as f:
                            batch_data = pickle.load(f)
                    else:
                        with open(batch_file, 'r') as f:
                            batch_data = json.load(f)
                    
                    # Extract processed pairs from this batch
                    batch_results = batch_data.get('results', {})
                    
                    # Handle both string and tuple keys (from previous saves)
                    for key, result in batch_results.items():
                        if isinstance(key, str) and key.startswith('(') and key.endswith(')'):
                            # Convert string representation back to tuple
                            try:
                                # Parse string like "('img1.jpg', 'img2.jpg')" back to tuple
                                key = eval(key)
                            except:
                                continue
                        
                        if isinstance(key, tuple) and len(key) == 2:
                            processed_pairs.add(key)
                            
                            # Only count successful pairs for statistics
                            if 'error' not in result:
                                existing_stats['successful_pairs'] += 1
                                existing_stats['total_matches'] += result.get('num_matches', 0)
                    
                    valid_batches.append(batch_file)
                    existing_stats['batch_files'].append(batch_file)
                    
                    # Extract batch number to determine next batch
                    batch_num = batch_data.get('batch_stats', {}).get('batch_number', 0)
                    next_batch_num = max(next_batch_num, batch_num + 1)
                    
                    print(f"  Loaded: {os.path.basename(batch_file)} ({len(batch_results)} pairs)")
                    
                except Exception as e:
                    print(f"  Warning: Could not load {batch_file}: {e}")
                    continue
            
            if processed_pairs:
                print(f"Resume: Found {len(processed_pairs)} already processed pairs")
                print(f"Resume: Starting from batch {next_batch_num}")
            
            return processed_pairs, next_batch_num, existing_stats
        
        # Load existing results if resuming
        processed_pairs, current_batch, existing_stats = load_existing_results()
        
        # Initialize overall statistics
        overall_stats = {
            'total_pairs': len(image_pairs),
            'successful_pairs': existing_stats['successful_pairs'],
            'total_matches': existing_stats['total_matches'],
            'methods_used': self.config['methods'],
            'processing_time': 0,
            'batch_files': existing_stats['batch_files'].copy(),
            'batch_count': current_batch - 1,
            'resumed_from_batch': current_batch if processed_pairs else 0,
            'skipped_pairs': len(processed_pairs)
        }
        
        # Current batch data
        current_batch_results = {}
        current_batch_stats = {
            'successful_pairs': 0,
            'total_matches': 0,
            'batch_start_time': 0
        }
        
        def save_current_batch(batch_num: int, is_final: bool = False):
            """Save current batch to file"""
            if not current_batch_results and not is_final:
                return
                
            batch_filename = os.path.join(output_dir, f"{output_name}_batch_{batch_num:03d}{output_ext}")
            
            batch_data = {
                'results': current_batch_results.copy(),
                'batch_stats': {
                    'batch_number': batch_num,
                    'successful_pairs': current_batch_stats['successful_pairs'],
                    'total_matches': current_batch_stats['total_matches'],
                    'batch_processing_time': time.time() - current_batch_stats['batch_start_time'],
                    'pairs_in_batch': len(current_batch_results)
                },
                'config': self.config,
                'overall_progress': {
                    'completed_pairs': overall_stats['successful_pairs'],
                    'total_pairs': overall_stats['total_pairs'],
                    'progress_percent': (overall_stats['successful_pairs'] / overall_stats['total_pairs']) * 100,
                    'resumed_from': overall_stats['resumed_from_batch']
                }
            }
            
            try:
                save_enhanced_results(batch_data, batch_filename, save_format)
                overall_stats['batch_files'].append(batch_filename)
                print(f"  Batch {batch_num} saved to: {batch_filename}")
                
                # Clear current batch to free memory
                current_batch_results.clear()
                current_batch_stats['successful_pairs'] = 0
                current_batch_stats['total_matches'] = 0
                current_batch_stats['batch_start_time'] = time.time()
                
            except Exception as e:
                print(f"  Warning: Failed to save batch {batch_num}: {e}")
        
        start_time = time.time()
        current_batch_stats['batch_start_time'] = start_time
        pairs_processed_in_session = 0
        pairs_skipped_in_session = 0
        
        # Create name pairs for easier lookup
        pair_names = [(images[idx1][1], images[idx2][1]) for idx1, idx2 in image_pairs]
        
        for i, ((idx1, idx2), (name1, name2)) in enumerate(zip(image_pairs, pair_names)):
            # Check if this pair was already processed
            pair_key = (name1, name2)
            reverse_pair_key = (name2, name1)  # Check both directions
            
            if pair_key in processed_pairs or reverse_pair_key in processed_pairs:
                pairs_skipped_in_session += 1
                if pairs_skipped_in_session <= 10:  # Limit spam
                    print(f"Skipping pair {i+1}/{len(image_pairs)}: {name1} <-> {name2} (already processed)")
                elif pairs_skipped_in_session == 11:
                    print("  ... (further skip messages suppressed)")
                continue
            
            img1, _ = images[idx1]
            img2, _ = images[idx2]
            
            print(f"\nProcessing pair {i+1}/{len(image_pairs)}: {name1} <-> {name2}")
            pairs_processed_in_session += 1
            
            try:
                result = self.process_image_pair(img1, img2, visualize=False)
                
                # Store essential information
                pair_result = {
                    'correspondences': result['correspondences'].tolist() if isinstance(result['correspondences'], np.ndarray) else result['correspondences'],
                    'num_matches': len(result['match_data'].get_best_matches()),
                    'method': result['method_used'],
                    'score_type': result['match_data'].score_type.value,
                    'quality_score': self.analyzer.analyze_match_data(result['match_data'])['quality_score'],
                    'homography': result['match_data'].homography.tolist()
                                if result['match_data'].homography is not None else None,
                    'processing_time': result.get('processing_time', 0),
                    'pair_index': i + 1,
                    'session_pair_index': pairs_processed_in_session
                }
                
                current_batch_results[pair_key] = pair_result
                current_batch_stats['successful_pairs'] += 1
                current_batch_stats['total_matches'] += pair_result['num_matches']
                overall_stats['successful_pairs'] += 1
                overall_stats['total_matches'] += pair_result['num_matches']
                
                print(f"  Success: {pair_result['num_matches']} matches "
                    f"(quality: {pair_result['quality_score']:.3f})")
                
            except Exception as e:
                print(f"  Failed: {str(e)}")
                current_batch_results[pair_key] = {
                    'error': str(e),
                    'num_matches': 0,
                    'pair_index': i + 1,
                    'session_pair_index': pairs_processed_in_session
                }
            
            # Save batch if we've reached the batch size
            if len(current_batch_results) >= batch_size:
                save_current_batch(current_batch)
                current_batch += 1
                overall_stats['batch_count'] = current_batch - 1
                
                # Print progress
                total_processed = overall_stats['successful_pairs'] + overall_stats.get('skipped_pairs', 0)
                progress = (total_processed / len(image_pairs)) * 100
                print(f"\n--- Batch {current_batch-1} completed ({progress:.1f}% total progress) ---")
        
        # Save final batch if there are remaining results
        if current_batch_results:
            save_current_batch(current_batch)
            overall_stats['batch_count'] = current_batch
        
        # Calculate final statistics
        overall_stats['processing_time'] = time.time() - start_time
        overall_stats['avg_matches'] = overall_stats['total_matches'] / max(overall_stats['successful_pairs'], 1)
        overall_stats['success_rate'] = overall_stats['successful_pairs'] / overall_stats['total_pairs']
        overall_stats['pairs_processed_this_session'] = pairs_processed_in_session
        overall_stats['pairs_skipped_this_session'] = pairs_skipped_in_session
        
        # Save overall summary
        summary_filename = os.path.join(output_dir, f"{output_name}_summary{output_ext}")
        summary_data = {
            'overall_stats': overall_stats,
            'config': self.config,
            'batch_files': overall_stats['batch_files'],
            'dataset_info': {
                'total_images': len(images),
                'image_names': [name for _, name in images]
            },
            'resume_info': {
                'was_resumed': overall_stats['resumed_from_batch'] > 0,
                'resumed_from_batch': overall_stats['resumed_from_batch'],
                'total_existing_pairs': len(processed_pairs),
                'new_pairs_processed': pairs_processed_in_session
            }
        }
        
        try:
            save_enhanced_results(summary_data, summary_filename, save_format)
            print(f"\nSummary saved to: {summary_filename}")
        except Exception as e:
            print(f"Warning: Failed to save summary: {e}")
        
        # Print final summary
        print(f"\nDataset processing completed:")
        if overall_stats['resumed_from_batch'] > 0:
            print(f"  Resumed from batch: {overall_stats['resumed_from_batch']}")
            print(f"  Previously processed: {overall_stats['skipped_pairs']} pairs")
            print(f"  Newly processed: {pairs_processed_in_session} pairs")
        print(f"  Success rate: {overall_stats['successful_pairs']}/{overall_stats['total_pairs']} ({overall_stats['success_rate']:.1%})")
        print(f"  Average matches: {overall_stats['avg_matches']:.1f}")
        print(f"  Session time: {overall_stats['processing_time']:.1f}s")
        print(f"  Total batches: {overall_stats['batch_count']}")
        print(f"  Batch files: {len(overall_stats['batch_files'])}")
        
        return {
            'overall_stats': overall_stats,
            'batch_files': overall_stats['batch_files'],
            'summary_file': summary_filename,
            'resume_info': summary_data['resume_info'],
            'image_metadata_file': image_metadata_file 
        }


# =============================================================================
# Factory Function for Easy Pipeline Creation
# =============================================================================
def create_pipeline(preset: str = 'balanced', **custom_config) -> FeatureProcessingPipeline:
    """Create a pipeline with predefined presets"""
    presets = {
        'fast': {
            'methods': ['ORB'],
            'max_features': 1000,
            'combine_strategy': 'best',  # ✓ Single method
            'detector_params': {
                'ORB': {'scale_factor': 1.5, 'n_levels': 6}
            }
        },
        'balanced': {
            'methods': ['SIFT', 'ORB'],
            'max_features': 2000,
            'combine_strategy': 'independent',  # ✓ Changed from 'best'
            'detector_params': {
                'SIFT': {'contrast_threshold': 0.04},
                'ORB': {'scale_factor': 1.2, 'n_levels': 8}
            }
        },
        'accurate': {
            'methods': ['SIFT', 'AKAZE', 'BRISK'],
            'max_features': 3000,
            'combine_strategy': 'independent',  # ✓ Changed from 'weighted'
            'detector_params': {
                'SIFT': {'contrast_threshold': 0.03},
                'AKAZE': {'threshold': 0.0005},
                'BRISK': {'threshold': 20}
            }
        },
        'deep_learning': {
            'methods': ['lightglue', 'SuperPoint'],
            'max_features': 2048,
            'combine_strategy': 'independent',  # ✓ Changed from 'best'
            'detector_params': {
                'lightglue': {'confidence_threshold': 0.2},
                'SuperPoint': {'keypoint_threshold': 0.005}
            }
        }
    }
    
    if preset not in presets:
        available = ', '.join(presets.keys())
        raise ValueError(f"Unknown preset: {preset}. Available: {available}")
    
    config = presets[preset].copy()
    config.update(custom_config)
    
    return FeatureProcessingPipeline(config)
    """
    Create a pipeline with predefined presets
    
    Args:
        preset: Preset configuration ('fast', 'balanced', 'accurate', 'deep_learning')
        **custom_config: Custom configuration overrides
        
    Returns:
        Configured pipeline instance
    """
    presets = {
        'fast': {
            'methods': ['ORB'],
            'max_features': 1000,
            'combine_strategy': 'best',
            'detector_params': {
                'ORB': {'scale_factor': 1.5, 'n_levels': 6}
            }
        },
        'balanced': {
            'methods': ['SIFT', 'ORB'],
            'max_features': 2000,
            'combine_strategy': 'best',
            'detector_params': {
                'SIFT': {'contrast_threshold': 0.04},
                'ORB': {'scale_factor': 1.2, 'n_levels': 8}
            }
        },
        'accurate': {
            'methods': ['SIFT', 'AKAZE', 'BRISK'],
            'max_features': 3000,
            'combine_strategy': 'weighted',
            'detector_params': {
                'SIFT': {'contrast_threshold': 0.03},
                'AKAZE': {'threshold': 0.0005},
                'BRISK': {'threshold': 20}
            }
        },
        'deep_learning': {
            'methods': ['lightglue', 'SuperPoint'],
            'max_features': 2048,
            'combine_strategy': 'best',
            'detector_params': {
                'lightglue': {'confidence_threshold': 0.2},
                'SuperPoint': {'keypoint_threshold': 0.005}
            }
        }
    }
    
    if preset not in presets:
        available = ', '.join(presets.keys())
        raise ValueError(f"Unknown preset: {preset}. Available: {available}")
    
    # Start with preset config
    config = presets[preset].copy()
    
    # Apply custom overrides
    config.update(custom_config)
    
    return FeatureProcessingPipeline(config)