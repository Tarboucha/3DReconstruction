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

from .core_data_structures import FeatureData, MatchData, DetectorType, ScoreType
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
    save_enhanced_results
)


class MultiMethodFeatureDetector:
    """Combines multiple feature detection methods"""
    
    def __init__(self, 
                 methods: List[Union[str, DetectorType]], 
                 max_features_per_method: int = 2000,
                 combine_strategy: str = "merge",
                 detector_params: Optional[Dict[str, Dict]] = None):
        """
        Initialize multi-method detector
        
        Args:
            methods: List of detection methods to use
            max_features_per_method: Max features per individual method
            combine_strategy: 'merge', 'best', or 'weighted'
            detector_params: Optional parameters for each detector
        """
        self.methods = methods
        self.max_features_per_method = max_features_per_method
        self.combine_strategy = combine_strategy
        self.detector_params = detector_params or {}
        
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
        """Detect features using all methods"""
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
        """Detect and combine features from all methods"""
        all_features = self.detect_all(image)
        
        if self.combine_strategy == "merge":
            return self._merge_features(all_features, image)
        elif self.combine_strategy == "best":
            return self._select_best_features(all_features, image)
        elif self.combine_strategy == "weighted":
            return self._weighted_combination(all_features, image)
        else:
            raise ValueError(f"Unknown combine strategy: {self.combine_strategy}")
    
    def _merge_features(self, features_dict: Dict[str, FeatureData], 
                       image: np.ndarray) -> FeatureData:
        """Merge all features, removing duplicates"""
        all_keypoints = []
        all_descriptors = []
        methods_used = []
        total_time = 0
        
        for method_name, features in features_dict.items():
            if len(features) > 0:
                all_keypoints.extend(features.keypoints)
                if features.descriptors is not None:
                    all_descriptors.append(features.descriptors)
                methods_used.append(method_name)
                total_time += features.detection_time
        
        # Remove duplicate keypoints (within threshold distance)
        unique_keypoints = self._remove_duplicate_keypoints(all_keypoints)
        
        # For simplicity, use descriptors from first method with descriptors
        combined_descriptors = None
        if all_descriptors:
            combined_descriptors = all_descriptors[0]
        
        return FeatureData(
            keypoints=unique_keypoints,
            descriptors=combined_descriptors,
            method=f"Combined({','.join(methods_used)})",
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
    
    def _weighted_combination(self, features_dict: Dict[str, FeatureData],
                         image: np.ndarray) -> FeatureData:
        """Combine features with weights based on performance"""
        
        # Calculate weights based on number of features and detection time
        weights = {}
        for method_name, features in features_dict.items():
            if len(features) > 0 and features.detection_time >= 0:
                # Higher weight for more features and faster detection
                weight = len(features) / (1 + features.detection_time)
                weights[method_name] = weight
        
        # Early return if no valid features
        if not weights:
            return FeatureData(
                keypoints=[],
                descriptors=None,
                method="Weighted(empty)",
                detection_time=0.0,
                raw_image=image
            )
        
        # Normalize weights
        total_weight = sum(weights.values())
        weights = {k: v/total_weight for k, v in weights.items()}
        
        # Select features proportionally
        selected_keypoints = []
        selected_descriptors = []
        total_time = 0
        
        for method_name, weight in weights.items():
            features = features_dict[method_name]
            n_select = int(weight * self.max_features_per_method * len(self.methods))
            
            if n_select > 0 and len(features) > 0:
                # Create indexed keypoints and sort by response
                indexed_keypoints = list(enumerate(features.keypoints))
                sorted_indexed = sorted(indexed_keypoints, 
                                    key=lambda x: x[1].response, 
                                    reverse=True)
                
                # Select top N
                top_n = sorted_indexed[:n_select]
                top_indices = [idx for idx, kp in top_n]
                top_keypoints = [kp for idx, kp in top_n]
                
                selected_keypoints.extend(top_keypoints)
                
                # Handle descriptors efficiently
                if features.descriptors is not None and len(top_indices) > 0:
                    try:
                        selected_desc = features.descriptors[top_indices]
                        if selected_desc.size > 0:  # Ensure not empty
                            selected_descriptors.append(selected_desc)
                    except (IndexError, AttributeError) as e:
                        print(f"Warning: Could not extract descriptors for {method_name}: {e}")
                
                total_time += features.detection_time
        
        # Combine descriptors with shape validation
        combined_descriptors = None
        if selected_descriptors:
            try:
                # Check if all descriptors have the same feature dimension
                shapes = [desc.shape[1] if desc.ndim > 1 else desc.shape[0] 
                        for desc in selected_descriptors]
                
                if len(set(shapes)) == 1:
                    combined_descriptors = np.vstack(selected_descriptors)
                else:
                    print(f"Warning: Descriptor shape mismatch: {shapes}. Using first compatible set.")
                    # Use only descriptors with the most common shape
                    most_common_shape = max(set(shapes), key=shapes.count)
                    compatible_descriptors = [desc for desc in selected_descriptors 
                                            if (desc.shape[1] if desc.ndim > 1 else desc.shape[0]) == most_common_shape]
                    if compatible_descriptors:
                        combined_descriptors = np.vstack(compatible_descriptors)
                        
            except Exception as e:
                print(f"Warning: Could not combine descriptors: {e}")
                combined_descriptors = None
        
        method_names = list(weights.keys())
        return FeatureData(
            keypoints=selected_keypoints,
            descriptors=combined_descriptors,
            method=f"Weighted({','.join(method_names)})",
            detection_time=total_time,
            raw_image=image
        )


class FeatureProcessingPipeline:
    """Complete pipeline for feature detection and matching"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize pipeline
        
        Args:
            config: Configuration dictionary with pipeline settings
                {
                    'methods': ['SIFT', 'ORB', 'lightglue'],
                    'max_features': 2000,
                    'combine_strategy': 'best',
                    'detector_params': {...},
                    'matcher_params': {...}
                }
        """
        self.config = config
        self.multi_detector = MultiMethodFeatureDetector(
            methods=config.get('methods', ['SIFT']),
            max_features_per_method=config.get('max_features', 2000),
            combine_strategy=config.get('combine_strategy', 'merge'),
            detector_params=config.get('detector_params', {})
        )
        self.analyzer = MatchQualityAnalyzer()
        
    def process_image_pair(self, img1: np.ndarray, img2: np.ndarray,
                          visualize: bool = True,
                          filter_matches: bool = True) -> Dict:
        """Process a single image pair"""
        print("Processing image pair...")
        
        # Check if LightGlue is available and prioritize it
        if self.multi_detector.lightglue_matcher is not None:
            return self._process_with_lightglue(img1, img2, visualize, filter_matches)
        
        # Use traditional pipeline
        return self._process_with_traditional(img1, img2, visualize, filter_matches)
    
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
    
    def _process_with_traditional(self, img1: np.ndarray, img2: np.ndarray,
                                 visualize: bool, filter_matches: bool) -> Dict:
        """Process using traditional detectors and matchers"""
        print("Using traditional processing...")
        
        # Detect features
        features1 = self.multi_detector.detect_combined(img1)
        features2 = self.multi_detector.detect_combined(img2)
        
        print(f"Image 1: {len(features1)} features ({features1.method})")
        print(f"Image 2: {len(features2)} features ({features2.method})")
        
        if len(features1) == 0 or len(features2) == 0:
            print("No features detected!")
            return {
                'features1': features1,
                'features2': features2,
                'match_data': MatchData([]),
                'correspondences': np.array([]).reshape(0, 4),
                'method_used': 'None'
            }
        
        # Match features
        matcher = auto_select_matcher(features1, features2)
        match_data = matcher.match(features1, features2)
        
        print(f"Found {len(match_data.matches)} matches")
        
        if filter_matches and len(match_data.matches) >= 4:
            # Apply adaptive filtering
            filtered_data = adaptive_match_filtering(match_data, top_k=200)
            
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
            'all_results': results,
            'best_method': best_method,
            'quality_analysis': quality_comparison if analyze_quality else {}
        }



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