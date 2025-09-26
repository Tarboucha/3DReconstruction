"""
Feature matching algorithms and classes.

This module contains implementations of various feature matching algorithms,
from traditional approaches to deep learning-based methods.
"""

import cv2
import numpy as np
import time
from typing import List, Tuple, Optional, Dict, Union
from .base_classes import BaseFeatureMatcher, BasePairMatcher
from .core_data_structures import FeatureData, MatchData, EnhancedDMatch, ScoreType

# Try to import PyTorch for deep learning matchers
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class EnhancedFLANNMatcher(BaseFeatureMatcher):
    """Enhanced FLANN-based matcher with proper score handling"""
    
    def __init__(self, ratio_threshold: float = 0.7, algorithm: str = 'kdtree',
                 trees: int = 5, checks: int = 50):
        """
        Initialize FLANN matcher
        
        Args:
            ratio_threshold: Ratio test threshold for Lowe's ratio test
            algorithm: FLANN algorithm ('kdtree' or 'lsh')
            trees: Number of trees for kdtree algorithm
            checks: Number of checks for search
        """
        self.ratio_threshold = ratio_threshold
        
        if algorithm == 'kdtree':
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=trees)
        else:  # LSH for binary descriptors
            FLANN_INDEX_LSH = 6
            index_params = dict(
                algorithm=FLANN_INDEX_LSH,
                table_number=6,
                key_size=12,
                multi_probe_level=1
            )
        
        search_params = dict(checks=checks)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
    
    def match(self, features1: FeatureData, features2: FeatureData) -> MatchData:
        start_time = time.time()
        
        # Validate features
        if not self.validate_features(features1, features2):
            return MatchData([], method="FLANN", score_type=ScoreType.DISTANCE)
        
        try:
            desc1 = features1.descriptors
            desc2 = features2.descriptors


            if desc1.shape[0] == desc2.shape[0] and \
                desc1.shape[1] != desc2.shape[1] and \
                desc1.ndim == 2:

                desc1 = desc1.T  # Transpose: [256, N] -> [N, 256]
                desc2 = desc2.T

            # Detect descriptor type and create appropriate matcher
            is_binary = desc1.dtype == np.uint8
            
            if is_binary:
                # Binary descriptors (ORB, BRISK, AKAZE with MLDB)
                # Use LSH (Locality Sensitive Hashing) for Hamming distance
                FLANN_INDEX_LSH = 6
                index_params = dict(
                    algorithm=FLANN_INDEX_LSH,
                    table_number=12,      # 12 hash tables
                    key_size=31,          # 20-31 is good for ORB
                    multi_probe_level=2   # Higher = more accurate but slower
                )
                search_params = dict(checks=100)  # Higher for binary
                
                # Create new matcher for binary descriptors
                matcher = cv2.FlannBasedMatcher(index_params, search_params)
                
            else:
                # Float descriptors (SIFT, SURF, SuperPoint)
                # Convert to float32 if needed (FLANN requires float32)
                if desc1.dtype != np.float32:
                    desc1 = desc1.astype(np.float32)
                if desc2.dtype != np.float32:
                    desc2 = desc2.astype(np.float32)
                
                # Use KD-Tree for Euclidean distance
                FLANN_INDEX_KDTREE = 1
                index_params = dict(
                    algorithm=FLANN_INDEX_KDTREE,
                    trees=5  # More trees = more accurate but slower
                )
                search_params = dict(checks=50)
                
                # Create new matcher for float descriptors
                matcher = cv2.FlannBasedMatcher(index_params, search_params)
            
            # Perform kNN matching with k=2 for ratio test
            # Need at least 2 descriptors in train set for kNN with k=2
            if len(desc2) < 2:
                # Fallback to simple matching without ratio test
                matches_raw = matcher.match(desc1, desc2)
                enhanced_matches = []
                for m in matches_raw:
                    enhanced_match = EnhancedDMatch(
                        queryIdx=m.queryIdx,
                        trainIdx=m.trainIdx,
                        score=m.distance,
                        score_type=ScoreType.DISTANCE,
                        raw_distance=m.distance
                    )
                    enhanced_matches.append(enhanced_match)
            else:
                # Standard kNN matching with ratio test
                matches_knn = matcher.knnMatch(desc1, desc2, k=2)
                
                enhanced_matches = []
                for match_pair in matches_knn:
                    if len(match_pair) == 2:
                        # Apply Lowe's ratio test
                        best_match, second_match = match_pair
                        
                        # Ratio test: best match should be significantly better than second
                        if best_match.distance < self.ratio_threshold * second_match.distance:
                            enhanced_match = EnhancedDMatch(
                                queryIdx=best_match.queryIdx,
                                trainIdx=best_match.trainIdx,
                                score=best_match.distance,
                                score_type=ScoreType.DISTANCE,
                                raw_distance=best_match.distance,
                                # Store ratio as confidence (inverse of distance ratio)
                                confidence=1.0 - (best_match.distance / second_match.distance) if second_match.distance > 0 else 1.0
                            )
                            enhanced_matches.append(enhanced_match)
                    elif len(match_pair) == 1:
                        # Only one match found (shouldn't happen with k=2, but handle it)
                        m = match_pair[0]
                        enhanced_match = EnhancedDMatch(
                            queryIdx=m.queryIdx,
                            trainIdx=m.trainIdx,
                            score=m.distance,
                            score_type=ScoreType.DISTANCE,
                            raw_distance=m.distance,
                            confidence=0.5  # Unknown confidence
                        )
                        enhanced_matches.append(enhanced_match)
            
            # Sort matches by distance (best first)
            enhanced_matches.sort(key=lambda x: x.score)
            
            return MatchData(
                matches=enhanced_matches,
                method="FLANN",
                matching_time=time.time() - start_time,
                score_type=ScoreType.DISTANCE
            )
            
        except cv2.error as e:
            print(f"FLANN matching OpenCV error: {e}")
            print(f"Descriptor shapes: {features1.descriptors.shape}, {features2.descriptors.shape}")
            print(f"Descriptor types: {features1.descriptors.dtype}, {features2.descriptors.dtype}")
            return MatchData([], method="FLANN", score_type=ScoreType.DISTANCE)
        except Exception as e:
            print(f"FLANN matching failed: {e}")
            import traceback
            traceback.print_exc()
            return MatchData([], method="FLANN", score_type=ScoreType.DISTANCE)


class EnhancedBFMatcher(BaseFeatureMatcher):
    """Enhanced Brute-force matcher with proper score handling"""
    
    def __init__(self, norm_type: int = cv2.NORM_L2, cross_check: bool = False,
                 ratio_threshold: Optional[float] = None):
        """
        Initialize BF matcher
        
        Args:
            norm_type: Distance measurement type (cv2.NORM_L2, cv2.NORM_HAMMING, etc.)
            cross_check: If True, only return consistent matches
            ratio_threshold: If provided, apply ratio test
        """
        self.matcher = cv2.BFMatcher(norm_type, crossCheck=cross_check)
        self.ratio_threshold = ratio_threshold
        self.cross_check = cross_check
    
    def match(self, features1: FeatureData, features2: FeatureData) -> MatchData:
        start_time = time.time()
        
        if not self.validate_features(features1, features2):
            return MatchData([], method="BF", score_type=ScoreType.DISTANCE)
        
        try:
            if self.ratio_threshold is not None and not self.cross_check:
                # Use ratio test
                matches = self.matcher.knnMatch(features1.descriptors, features2.descriptors, k=2)
                
                enhanced_matches = []
                for match_pair in matches:
                    if len(match_pair) == 2:
                        m, n = match_pair
                        if m.distance < self.ratio_threshold * n.distance:
                            enhanced_match = EnhancedDMatch(
                                queryIdx=m.queryIdx,
                                trainIdx=m.trainIdx,
                                score=m.distance,
                                score_type=ScoreType.DISTANCE,
                                raw_distance=m.distance
                            )
                            enhanced_matches.append(enhanced_match)
            else:
                # Direct matching
                matches = self.matcher.match(features1.descriptors, features2.descriptors)
                
                enhanced_matches = []
                for m in matches:
                    enhanced_match = EnhancedDMatch(
                        queryIdx=m.queryIdx,
                        trainIdx=m.trainIdx,
                        score=m.distance,
                        score_type=ScoreType.DISTANCE,
                        raw_distance=m.distance
                    )
                    enhanced_matches.append(enhanced_match)
            
            # Sort by distance (lower is better)
            enhanced_matches.sort(key=lambda x: x.score)
            
            return MatchData(
                matches=enhanced_matches,
                method="BF",
                matching_time=time.time() - start_time,
                score_type=ScoreType.DISTANCE
            )
            
        except Exception as e:
            print(f"BF matching failed: {e}")
            return MatchData([], method="BF", score_type=ScoreType.DISTANCE)


class LightGlueMatcher(BasePairMatcher):
    """LightGlue deep learning-based matcher with proper score handling"""
    
    def __init__(self, features: str = 'superpoint', 
                 confidence_threshold: float = 0.2,
                 max_num_keypoints: int = 2048,
                 filter_threshold: float = 0.1):
        """
        Initialize LightGlue matcher
        
        Args:
            features: Type of features to match ('superpoint', 'disk', 'aliked')
            confidence_threshold: Minimum confidence for matches
            max_num_keypoints: Maximum number of keypoints to extract
            filter_threshold: Filtering threshold for matches
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for LightGlue. Install with: pip install torch torchvision lightglue")
        
        self.features = features
        self.confidence_threshold = confidence_threshold
        self.max_num_keypoints = max_num_keypoints
        self.filter_threshold = filter_threshold
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.matcher = None
        self.extractor = None
        self._load_model()
    
    def _load_model(self):
        """Load LightGlue model and feature extractor"""
        try:
            from .LightGlue.lightglue import LightGlue, SuperPoint, DISK, ALIKED
            
            if self.features.lower() == 'superpoint':
                self.extractor = SuperPoint(max_num_keypoints=self.max_num_keypoints).eval().to(self.device)
                self.matcher = LightGlue(features='superpoint', filter_threshold=self.filter_threshold).eval().to(self.device)
            elif self.features.lower() == 'disk':
                self.extractor = DISK(max_num_keypoints=self.max_num_keypoints).eval().to(self.device)
                self.matcher = LightGlue(features='disk', filter_threshold=self.filter_threshold).eval().to(self.device)
            elif self.features.lower() == 'aliked':
                self.extractor = ALIKED(max_num_keypoints=self.max_num_keypoints).eval().to(self.device)
                self.matcher = LightGlue(features='aliked', filter_threshold=self.filter_threshold).eval().to(self.device)
            else:
                print(f"Feature type '{self.features}' not supported. Using SuperPoint.")
                self.extractor = SuperPoint(max_num_keypoints=self.max_num_keypoints).eval().to(self.device)
                self.matcher = LightGlue(features='superpoint', filter_threshold=self.filter_threshold).eval().to(self.device)
            
            print(f"LightGlue matcher with {self.features} loaded successfully")
            
        except ImportError as e:
            print(f"LightGlue not available. Install with: pip install lightglue")
            print(f"Error: {e}")
            self.matcher = None
            self.extractor = None
        except Exception as e:
            print(f"Failed to load LightGlue: {e}")
            self.matcher = None
            self.extractor = None
    
    def _image_to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """Convert image to tensor format expected by LightGlue"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Normalize to [0, 1] and add batch dimension
        tensor = torch.from_numpy(gray).float() / 255.0
        tensor = tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        return tensor.to(self.device)
    
    def match_images_directly(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[FeatureData, FeatureData, MatchData]:
        """
        Match images directly with LightGlue and proper score handling
        """
        start_time = time.time()
        
        if self.matcher is None or self.extractor is None:
            print("LightGlue matcher not loaded")
            empty_features = FeatureData([], None, "LightGlue")
            return empty_features, empty_features, MatchData([], method="LightGlue", score_type=ScoreType.CONFIDENCE)
        
        try:
            # Convert images to tensors
            tensor0 = self._image_to_tensor(img1)
            tensor1 = self._image_to_tensor(img2)
            
            # Extract features
            with torch.no_grad():
                feats0 = self.extractor.extract(tensor0)
                feats1 = self.extractor.extract(tensor1)
                
                # Match features
                matches = self.matcher({'image0': feats0, 'image1': feats1})
            
            # Extract match information
            matches01 = matches['matches0'][0]  # [N] tensor
            match_confidence = matches['matching_scores0'][0]  # [N] confidence scores
            
            # Get valid matches
            valid = matches01 > -1
            confident = match_confidence >= self.confidence_threshold
            final_valid = valid & confident
            
            if not final_valid.any():
                empty_features = FeatureData([], None, "LightGlue")
                return empty_features, empty_features, MatchData([], method="LightGlue", score_type=ScoreType.CONFIDENCE)
            
            # Extract matched keypoints and scores
            mkpts0 = feats0['keypoints'][0][final_valid].cpu().numpy()
            mkpts1 = feats1['keypoints'][0][matches01[final_valid]].cpu().numpy()
            match_confs = match_confidence[final_valid].cpu().numpy()
            
            # Extract keypoint confidence scores
            kp_scores0 = feats0['keypoint_scores'][0][final_valid].cpu().numpy()
            kp_scores1 = feats1['keypoint_scores'][0][matches01[final_valid]].cpu().numpy()
            
            # Convert to cv2.KeyPoint format
            kp0_cv = [cv2.KeyPoint(x=float(pt[0]), y=float(pt[1]), size=8, response=float(score)) 
                     for pt, score in zip(mkpts0, kp_scores0)]
            kp1_cv = [cv2.KeyPoint(x=float(pt[0]), y=float(pt[1]), size=8, response=float(score)) 
                     for pt, score in zip(mkpts1, kp_scores1)]
            
            # Create EnhancedDMatch objects with confidence scores
            enhanced_matches = []
            for i, conf in enumerate(match_confs):
                match = EnhancedDMatch(
                    queryIdx=i,
                    trainIdx=i,  # Since we already filtered to matched pairs
                    score=float(conf),
                    score_type=ScoreType.CONFIDENCE,
                    confidence=float(conf)
                )
                enhanced_matches.append(match)
            
            # Extract descriptors if available
            desc0 = None
            desc1 = None
            if 'descriptors' in feats0 and 'descriptors' in feats1:
                desc0 = feats0['descriptors'][0][final_valid].cpu().numpy()
                desc1 = feats1['descriptors'][0][matches01[final_valid]].cpu().numpy()
            
            # Create FeatureData objects
            features1 = FeatureData(
                keypoints=kp0_cv,
                descriptors=desc0,
                method="LightGlue",
                confidence_scores=kp_scores0.tolist(),
                detection_time=time.time() - start_time,
                raw_image=img1
            )
            
            features2 = FeatureData(
                keypoints=kp1_cv,
                descriptors=desc1,
                method="LightGlue",
                confidence_scores=kp_scores1.tolist(),
                detection_time=0.0,
                raw_image=img2
            )
            
            match_data = MatchData(
                matches=enhanced_matches,
                method="LightGlue",
                matching_time=time.time() - start_time,
                score_type=ScoreType.CONFIDENCE,
                match_confidences=match_confs,
                keypoint_confidences=(kp_scores0, kp_scores1)
            )
            
            return features1, features2, match_data
            
        except Exception as e:
            print(f"LightGlue matching failed: {e}")
            import traceback
            traceback.print_exc()
            empty_features = FeatureData([], None, "LightGlue")
            return empty_features, empty_features, MatchData([], method="LightGlue", score_type=ScoreType.CONFIDENCE)
    
    def match(self, features1: FeatureData, features2: FeatureData) -> MatchData:
        """Match features using LightGlue (requires raw images)"""
        if not hasattr(features1, 'raw_image') or not hasattr(features2, 'raw_image'):
            print("LightGlue requires raw images for feature extraction")
            return MatchData([], method="LightGlue", score_type=ScoreType.CONFIDENCE)
        
        _, _, match_data = self.match_images_directly(features1.raw_image, features2.raw_image)
        return match_data


# Factory functions for easy matcher creation
def create_traditional_matcher(matcher_type: str, **kwargs) -> BaseFeatureMatcher:
    """
    Factory function to create traditional feature matchers
    
    Args:
        matcher_type: Type of matcher ('FLANN', 'BF')
        **kwargs: Additional parameters for the matcher
        
    Returns:
        Initialized matcher instance
        
    Raises:
        ValueError: If matcher_type is not supported
    """
    matcher_map = {
        'FLANN': EnhancedFLANNMatcher,
        'BF': EnhancedBFMatcher
    }
    
    if matcher_type not in matcher_map:
        available = ', '.join(matcher_map.keys())
        raise ValueError(f"Unknown matcher type: {matcher_type}. Available: {available}")
    
    return matcher_map[matcher_type](**kwargs)


def create_deep_learning_matcher(matcher_type: str, **kwargs) -> BasePairMatcher:
    """
    Factory function to create deep learning feature matchers
    
    Args:
        matcher_type: Type of matcher ('LightGlue')
        **kwargs: Additional parameters for the matcher
        
    Returns:
        Initialized matcher instance
        
    Raises:
        ValueError: If matcher_type is not supported
        ImportError: If required dependencies are not available
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for deep learning matchers. Install with: pip install torch torchvision")
    
    matcher_map = {
        'LightGlue': LightGlueMatcher
    }
    
    if matcher_type not in matcher_map:
        available = ', '.join(matcher_map.keys())
        raise ValueError(f"Unknown matcher type: {matcher_type}. Available: {available}")
    
    return matcher_map[matcher_type](**kwargs)


def auto_select_matcher(features1: FeatureData, features2: FeatureData, **kwargs) -> BaseFeatureMatcher:
    """
    Automatically select appropriate matcher based on feature types
    
    Args:
        features1: Features from first image
        features2: Features from second image
        **kwargs: Additional parameters for matcher creation
        
    Returns:
        Appropriate matcher instance
    """
    # Check feature types
    method1 = features1.method.upper()
    method2 = features2.method.upper()
    
    # For binary descriptors, use BF matcher with Hamming distance
    if any(method in ['ORB', 'BRISK', 'AKAZE'] for method in [method1, method2]):
        return EnhancedBFMatcher(norm_type=cv2.NORM_HAMMING, **kwargs)
    
    # For deep learning features, prefer specific matchers if available
    if any(method in ['SUPERPOINT', 'DISK', 'ALIKED', 'LIGHTGLUE'] for method in [method1, method2]):
        if TORCH_AVAILABLE:
            try:
                return LightGlueMatcher(**kwargs)
            except:
                pass
    
    # Default to FLANN for float descriptors
    return EnhancedFLANNMatcher(**kwargs)