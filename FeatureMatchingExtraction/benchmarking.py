"""
 Benchmarking and evaluation tools for feature detection and matching.

This module provides comprehensive benchmarking capabilities with improved
accuracy metrics, statistical analysis, and robust synthetic data generation.
"""

import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any, Union
import os
import tracemalloc
import psutil
from dataclasses import dataclass, field
from pathlib import Path
from scipy import stats
import json
from collections import defaultdict

from .core_data_structures import FeatureData, MatchData, DetectorType, ScoreType
from .traditional_detectors import create_traditional_detector
from .feature_matchers import create_traditional_matcher, auto_select_matcher
from .pipeline import FeatureProcessingPipeline, create_pipeline
from .utils import MatchQualityAnalyzer, extract_correspondences, save_benchmark_summary
from .config import create_config_from_preset


@dataclass
class BenchmarkConfig:
    """Centralized configuration for benchmarks"""
    num_runs: int = 3
    image_sizes: List[Tuple[int, int]] = field(default_factory=lambda: [(240, 320), (480, 640), (720, 1280)])
    transformation_params: Dict = field(default_factory=lambda: {
        'perspective_offset_range': (0.05, 0.15),
        'affine_angle_range': (-15, 15),
        'affine_scale_range': (0.8, 1.2),
        'affine_translation_range': 0.1,
        'rotation_angle_range': (-30, 30),
        'scale_factor_range': (0.7, 1.3)
    })
    quality_thresholds: Dict = field(default_factory=lambda: {
        'min_matches': 10,
        'max_reprojection_error': 3.0,
        'min_inlier_ratio': 0.3
    })
    statistical_confidence: float = 0.95
    memory_profiling: bool = True
    save_intermediate_results: bool = False


@dataclass
class BenchmarkResult:
    """ container for benchmark results with statistical data"""
    method_name: str
    detection_time: float
    matching_time: float
    num_features1: int
    num_features2: int
    num_matches: int
    num_filtered_matches: int
    quality_score: float
    score_type: str
    memory_usage: Optional[Dict[str, float]] = None
    error_message: Optional[str] = None
    reprojection_error: Optional[float] = None
    inlier_ratio: Optional[float] = None
    geometric_consistency: Optional[float] = None
    statistical_data: Optional[Dict] = None
    
    @property
    def total_time(self) -> float:
        return self.detection_time + self.matching_time
    
    @property
    def success(self) -> bool:
        return self.error_message is None
    
    @property
    def meets_quality_threshold(self) -> bool:
        """Check if result meets minimum quality standards"""
        if not self.success:
            return False
        return (self.num_matches >= 10 and 
                (self.reprojection_error is None or self.reprojection_error < 3.0) and
                (self.inlier_ratio is None or self.inlier_ratio > 0.3))


class SyntheticImageGenerator:
    """Advanced synthetic image generation for testing"""
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        self.seed = seed
    
    def create_realistic_test_image(self, height: int, width: int, 
                                   feature_density: str = 'medium',
                                   texture_complexity: str = 'medium') -> np.ndarray:
        """
        Create realistic test image with controlled features and textures
        
        Args:
            height: Image height in pixels (NumPy convention for generation)
            width: Image width in pixels
            feature_density: 'low', 'medium', 'high'
            texture_complexity: 'low', 'medium', 'high'
            
        Returns:
            Realistic synthetic test image with shape (height, width, 3)
            
        Note:
            This function uses (height, width) order internally for NumPy array creation.
            External APIs should convert from (width, height) before calling.
        """
        # ✅ Validate inputs
        if height <= 0 or width <= 0:
            raise ValueError(f"Image dimensions must be positive, got {width}x{height}")
        
        # Base image with gradient background
        image = self._create_gradient_background(height, width)
        
        # Add texture based on complexity
        if texture_complexity != 'low':
            image = self._add_procedural_texture(image, texture_complexity)
        
        # Add geometric features
        image = self._add_geometric_features(image, feature_density)
        
        # Add natural-looking patterns
        image = self._add_natural_patterns(image, feature_density)
        
        # Add noise
        image = self._add_realistic_noise(image)
        
        return image
    
    def _create_gradient_background(self, height: int, width: int) -> np.ndarray:
        """Create gradient background"""
        # Create smooth gradient
        x = np.linspace(0, 1, width)
        y = np.linspace(0, 1, height)
        X, Y = np.meshgrid(x, y)
        
        # Multi-directional gradient
        gradient = 0.3 * X + 0.3 * Y + 0.4 * np.sin(X * np.pi) * np.cos(Y * np.pi)
        gradient = (gradient - gradient.min()) / (gradient.max() - gradient.min())
        
        # Convert to 3-channel image
        base_color = np.random.randint(80, 180, 3)
        image = np.zeros((height, width, 3), dtype=np.uint8)
        
        for c in range(3):
            image[:, :, c] = (gradient * 100 + base_color[c]).astype(np.uint8)
        
        return image
    
    def _add_procedural_texture(self, image: np.ndarray, complexity: str) -> np.ndarray:
        """Add procedural texture using Perlin-like noise"""
        height, width = image.shape[:2]
        
        # Create multiple octaves of noise
        octaves = {'low': 2, 'medium': 4, 'high': 6}[complexity]
        texture = np.zeros((height, width))
        
        for octave in range(octaves):
            freq = 2 ** octave
            amplitude = 1.0 / (2 ** octave)
            
            # Simple noise generation
            noise_h = max(1, height // freq)
            noise_w = max(1, width // freq)
            noise = np.random.random((noise_h, noise_w))
            
            # Upsample noise
            upsampled = cv2.resize(noise, (width, height), interpolation=cv2.INTER_LINEAR)
            texture += amplitude * upsampled
        
        # Normalize and apply
        texture = (texture - texture.min()) / (texture.max() - texture.min())
        texture = (texture * 50).astype(np.uint8)
        
        # Add to all channels
        for c in range(3):
            image[:, :, c] = np.clip(image[:, :, c].astype(int) + texture, 0, 255).astype(np.uint8)
        
        return image
    
    def _add_geometric_features(self, image: np.ndarray, density: str) -> np.ndarray:
        """Add geometric features with better control"""
        height, width = image.shape[:2]
        
        # Feature counts based on density and image size
        base_counts = {'low': 5, 'medium': 15, 'high': 30}
        scale_factor = min(height, width) / 300
        num_features = int(base_counts[density] * scale_factor)
        
        for i in range(num_features):
            # Ensure features are not too close to edges
            margin = 30
            x = np.random.randint(margin, width - margin)
            y = np.random.randint(margin, height - margin)
            
            # Varied feature sizes
            min_size = max(8, int(10 * scale_factor))
            max_size = max(15, int(30 * scale_factor))
            size = np.random.randint(min_size, max_size)
            
            
            # More varied colors
            base_brightness = np.mean(image[y, x])
            if base_brightness < 128:
                color_range = (180, 255)
            else:
                color_range = (50, 120)
            
            color = tuple(np.random.randint(*color_range, 3).tolist())
            
            # Different shape types
            shape_type = np.random.choice(['rectangle', 'circle', 'line', 'cross'])
            
            if shape_type == 'rectangle':
                cv2.rectangle(image, (x-size//2, y-size//2), (x+size//2, y+size//2), color, -1)
                # Add border for better detectability
                cv2.rectangle(image, (x-size//2, y-size//2), (x+size//2, y+size//2), (255, 255, 255), 1)
                
            elif shape_type == 'circle':
                cv2.circle(image, (x, y), size//2, color, -1)
                cv2.circle(image, (x, y), size//2, (255, 255, 255), 1)
                
            elif shape_type == 'line':
                angle = np.random.uniform(0, 2*np.pi)
                x2 = int(x + size * np.cos(angle))
                y2 = int(y + size * np.sin(angle))
                cv2.line(image, (x, y), (x2, y2), color, 3)
                
            elif shape_type == 'cross':
                cv2.line(image, (x-size//2, y), (x+size//2, y), color, 3)
                cv2.line(image, (x, y-size//2), (x, y+size//2), color, 3)
        
        return image
    
    def _add_natural_patterns(self, image: np.ndarray, density: str) -> np.ndarray:
        """Add natural-looking patterns"""
        height, width = image.shape[:2]
        
        if density in ['medium', 'high']:
            # Add some curved patterns
            num_curves = {'medium': 3, 'high': 6}[density]
            
            for _ in range(num_curves):
                # Create smooth curves
                points = []
                num_points = np.random.randint(3, 8)
                
                for i in range(num_points):
                    x = np.random.randint(0, width)
                    y = np.random.randint(0, height)
                    points.append([x, y])
                
                points = np.array(points, dtype=np.int32)
                color = tuple(np.random.randint(100, 200, 3).tolist())
                
                # Draw smooth curve using polylines
                cv2.polylines(image, [points], False, color, 2)
        
        return image
    
    def _add_realistic_noise(self, image: np.ndarray) -> np.ndarray:
        """Add realistic noise patterns"""
        height, width = image.shape[:2]
        
        # Gaussian noise
        gaussian_noise = np.random.normal(0, 8, (height, width, 3))
        
        # Salt and pepper noise
        salt_pepper = np.random.random((height, width, 3))
        salt_mask = salt_pepper > 0.98
        pepper_mask = salt_pepper < 0.02
        
        # Apply noise
        noisy_image = image.astype(float) + gaussian_noise
        noisy_image[salt_mask] = 255
        noisy_image[pepper_mask] = 0
        
        return np.clip(noisy_image, 0, 255).astype(np.uint8)


class AdvancedQualityMetrics:
    """Advanced quality assessment for matching results"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
    
    def comprehensive_quality_assessment(self, matches: np.ndarray, 
                                       keypoints1: List, keypoints2: List,
                                       gt_transform: np.ndarray = None) -> Dict[str, float]:
        """
        Comprehensive quality assessment of matches
        
        Args:
            matches: Array of matches [(idx1, idx2), ...]
            keypoints1: Keypoints from first image
            keypoints2: Keypoints from second image  
            gt_transform: Ground truth transformation matrix
            
        Returns:
            Dictionary of quality metrics
        """
        if len(matches) < 4:
            return {'error': 'Insufficient matches for analysis'}
        
        # Extract matched points
        pts1 = np.float32([keypoints1[m[0]].pt for m in matches]).reshape(-1, 1, 2)
        pts2 = np.float32([keypoints2[m[1]].pt for m in matches]).reshape(-1, 1, 2)
        
        results = {}
        
        # 1. Geometric consistency via homography
        try:
            H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 3.0)
            if H is not None:
                inliers = np.sum(mask)
                results['inlier_ratio'] = inliers / len(matches)
                results['num_inliers'] = int(inliers)
                
                # Reprojection error for inliers
                if inliers > 0:
                    inlier_pts1 = pts1[mask.ravel() == 1]
                    projected_pts = cv2.perspectiveTransform(inlier_pts1.reshape(-1, 1, 2), H)
                    inlier_pts2 = pts2[mask.ravel() == 1]
                    
                    errors = np.sqrt(np.sum((projected_pts - inlier_pts2) ** 2, axis=2))
                    results['mean_reprojection_error'] = float(np.mean(errors))
                    results['std_reprojection_error'] = float(np.std(errors))
                    results['max_reprojection_error'] = float(np.max(errors))
                
                # Ground truth comparison if available
                if gt_transform is not None:
                    results.update(self._compare_with_ground_truth(H, gt_transform))
            
        except cv2.error:
            results['homography_error'] = 'Failed to compute homography'
        
        # 2. Spatial distribution analysis
        results.update(self._analyze_spatial_distribution(pts1, pts2))
        
        # 3. Distance consistency
        results.update(self._analyze_distance_consistency(pts1, pts2))
        
        # 4. Overall quality score
        results['overall_quality'] = self._compute_overall_quality(results)
        
        return results
    
    def _compare_with_ground_truth(self, estimated_H: np.ndarray, 
                                  gt_transform: np.ndarray) -> Dict[str, float]:
        """Compare estimated homography with ground truth"""
        results = {}
        
        try:
            # Handle different transformation types
            if gt_transform.shape == (2, 3):  # Affine
                # Convert affine to homography
                gt_H = np.vstack([gt_transform, [0, 0, 1]])
            elif gt_transform.shape == (3, 3):  # Homography
                gt_H = gt_transform
            else:
                return {'gt_comparison_error': 'Unsupported ground truth format'}
            
            # Normalize homographies
            estimated_H_norm = estimated_H / estimated_H[2, 2]
            gt_H_norm = gt_H / gt_H[2, 2]
            
            # Matrix difference metrics
            diff_matrix = np.abs(estimated_H_norm - gt_H_norm)
            results['matrix_frobenius_error'] = float(np.linalg.norm(diff_matrix, 'fro'))
            results['matrix_max_error'] = float(np.max(diff_matrix))
            
            # Corner reprojection test
            h, w = 300, 400  # Assume standard test image size
            corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
            
            gt_corners = cv2.perspectiveTransform(corners, gt_H_norm)
            est_corners = cv2.perspectiveTransform(corners, estimated_H_norm)
            
            corner_errors = np.sqrt(np.sum((gt_corners - est_corners) ** 2, axis=2))
            results['mean_corner_error'] = float(np.mean(corner_errors))
            results['max_corner_error'] = float(np.max(corner_errors))
            
        except Exception as e:
            results['gt_comparison_error'] = str(e)
        
        return results
    
    def _analyze_spatial_distribution(self, pts1: np.ndarray, pts2: np.ndarray) -> Dict[str, float]:
        """Analyze spatial distribution of matches"""
        results = {}
        
        # Points should be well distributed across the image
        pts1_flat = pts1.reshape(-1, 2)
        pts2_flat = pts2.reshape(-1, 2)
        
        # Coefficient of variation for coordinates
        for i, axis in enumerate(['x', 'y']):
            coords1 = pts1_flat[:, i]
            coords2 = pts2_flat[:, i]
            
            if np.std(coords1) > 0:
                results[f'cv_{axis}_img1'] = np.std(coords1) / np.mean(coords1)
            if np.std(coords2) > 0:
                results[f'cv_{axis}_img2'] = np.std(coords2) / np.mean(coords2)
        
        # Convex hull area (larger is better for distribution)
        if len(pts1_flat) >= 3:
            hull1 = cv2.convexHull(pts1_flat.astype(np.float32))
            hull2 = cv2.convexHull(pts2_flat.astype(np.float32))
            
            results['convex_hull_area_1'] = float(cv2.contourArea(hull1))
            results['convex_hull_area_2'] = float(cv2.contourArea(hull2))
        
        return results
    
    def _analyze_distance_consistency(self, pts1: np.ndarray, pts2: np.ndarray) -> Dict[str, float]:
        """Analyze consistency of pairwise distances"""
        results = {}
        
        if len(pts1) < 10:  # Need enough points for meaningful analysis
            return results
        
        pts1_flat = pts1.reshape(-1, 2)
        pts2_flat = pts2.reshape(-1, 2)
        
        # Sample random pairs to avoid O(n²) complexity
        num_samples = min(50, len(pts1_flat) * (len(pts1_flat) - 1) // 2)
        indices = np.random.choice(len(pts1_flat), size=(num_samples, 2), replace=True)
        
        # Remove pairs where both indices are the same
        valid_pairs = indices[indices[:, 0] != indices[:, 1]]
        
        if len(valid_pairs) == 0:
            return results
        
        # Calculate distance ratios
        distances1 = np.linalg.norm(pts1_flat[valid_pairs[:, 0]] - pts1_flat[valid_pairs[:, 1]], axis=1)
        distances2 = np.linalg.norm(pts2_flat[valid_pairs[:, 0]] - pts2_flat[valid_pairs[:, 1]], axis=1)
        
        # Avoid division by zero
        valid_distances = (distances1 > 1e-6) & (distances2 > 1e-6)
        if np.sum(valid_distances) > 0:
            ratios = distances2[valid_distances] / distances1[valid_distances]
            
            results['distance_ratio_mean'] = float(np.mean(ratios))
            results['distance_ratio_std'] = float(np.std(ratios))
            results['distance_consistency'] = float(1.0 / (1.0 + np.std(ratios)))
        
        return results
    
    def _compute_overall_quality(self, metrics: Dict[str, float]) -> float:
        """Compute overall quality score from individual metrics"""
        score = 0.0
        weights = {
            'inlier_ratio': 0.3,
            'mean_reprojection_error': -0.2,  # Negative because lower is better
            'distance_consistency': 0.2,
            'convex_hull_area_1': 0.1,
            'mean_corner_error': -0.1
        }
        
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in metrics:
                value = metrics[metric]
                if metric in ['mean_reprojection_error', 'mean_corner_error']:
                    # For error metrics, invert and normalize
                    normalized_value = 1.0 / (1.0 + value)
                else:
                    # For positive metrics, use as-is but clip
                    normalized_value = min(1.0, value)
                
                score += weight * normalized_value
                total_weight += abs(weight)
        
        if total_weight > 0:
            score = score / total_weight
        
        return max(0.0, min(1.0, score))


class StatisticalAnalyzer:
    """Statistical analysis for benchmark results"""
    
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
    
    def compare_methods(self, results1: List[float], results2: List[float], 
                       metric_name: str) -> Dict[str, Any]:
        """
        Statistical comparison between two methods
        
        Args:
            results1: Results from method 1
            results2: Results from method 2
            metric_name: Name of the metric being compared
            
        Returns:
            Statistical comparison results
        """
        comparison = {
            'metric': metric_name,
            'method1_stats': self._compute_descriptive_stats(results1),
            'method2_stats': self._compute_descriptive_stats(results2),
        }
        
        # Perform statistical tests
        if len(results1) >= 3 and len(results2) >= 3:
            # Check normality first
            _, p1 = stats.shapiro(results1) if len(results1) <= 5000 else (None, 0.05)
            _, p2 = stats.shapiro(results2) if len(results2) <= 5000 else (None, 0.05)
            
            normal_distribution = (p1 > 0.05) and (p2 > 0.05)
            
            if normal_distribution:
                # Use t-test for normal distributions
                statistic, p_value = stats.ttest_ind(results1, results2)
                test_type = 'ttest'
            else:
                # Use Mann-Whitney U test for non-normal distributions
                statistic, p_value = stats.mannwhitneyu(results1, results2, alternative='two-sided')
                test_type = 'mannwhitney'
            
            comparison['statistical_test'] = {
                'test_type': test_type,
                'statistic': float(statistic),
                'p_value': float(p_value),
                'significant': p_value < (1 - self.confidence_level),
                'confidence_level': self.confidence_level
            }
            
            # Effect size (Cohen's d for t-test)
            if test_type == 'ttest':
                pooled_std = np.sqrt(((len(results1) - 1) * np.var(results1) + 
                                    (len(results2) - 1) * np.var(results2)) / 
                                   (len(results1) + len(results2) - 2))
                if pooled_std > 0:
                    cohens_d = (np.mean(results1) - np.mean(results2)) / pooled_std
                    comparison['effect_size'] = {
                        'cohens_d': float(cohens_d),
                        'magnitude': self._interpret_cohens_d(abs(cohens_d))
                    }
        
        return comparison
    
    def _compute_descriptive_stats(self, data: List[float]) -> Dict[str, float]:
        """Compute descriptive statistics"""
        if not data:
            return {}
        
        data_array = np.array(data)
        return {
            'count': len(data),
            'mean': float(np.mean(data_array)),
            'std': float(np.std(data_array)),
            'min': float(np.min(data_array)),
            'max': float(np.max(data_array)),
            'median': float(np.median(data_array)),
            'q25': float(np.percentile(data_array, 25)),
            'q75': float(np.percentile(data_array, 75)),
            'cv': float(np.std(data_array) / np.mean(data_array)) if np.mean(data_array) != 0 else 0
        }
    
    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d effect size"""
        if d < 0.2:
            return 'negligible'
        elif d < 0.5:
            return 'small'
        elif d < 0.8:
            return 'medium'
        else:
            return 'large'


class PerformanceBenchmark:
    """ performance benchmarking with statistical analysis"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.image_generator = SyntheticImageGenerator()
        self.statistical_analyzer = StatisticalAnalyzer(config.statistical_confidence)
        
    def create_improved_transform_pair(self, base_image: np.ndarray, 
                                 transformation_type: str = 'perspective') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fixed version of create_transformed_pair with proper transformation matrices
        
        Args:
            base_image: Base image with shape (height, width, channels)
            transformation_type: 'perspective', 'affine', 'rotation', 'scale'
            
        Returns:
            Tuple of (img1, img2, ground_truth_transform)
        """
        # ✅ Get dimensions properly
        height, width = base_image.shape[:2]
        params = self.config.transformation_params
        
        if transformation_type == 'perspective':
            # Perspective transformation with configurable offset
            src_points = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
            offset_min, offset_max = params['perspective_offset_range']
            offset = min(width, height) * np.random.uniform(offset_min, offset_max)
            
            # More varied perspective distortions
            dst_points = np.float32([
                [np.random.uniform(0, offset), np.random.uniform(0, offset/2)], 
                [width - np.random.uniform(0, offset), np.random.uniform(0, offset)], 
                [width - np.random.uniform(0, offset/2), height - np.random.uniform(0, offset)], 
                [np.random.uniform(0, offset/2), height - np.random.uniform(0, offset/2)]
            ])
            
            transform = cv2.getPerspectiveTransform(src_points, dst_points)
            # ✅ warpPerspective expects (width, height) - correct
            img2 = cv2.warpPerspective(base_image, transform, (width, height))
            
        elif transformation_type == 'affine':
            # Affine transformation
            center = (width/2, height/2)  # ✅ (x, y) = (width/2, height/2)
            angle_min, angle_max = params['affine_angle_range']
            scale_min, scale_max = params['affine_scale_range']
            trans_range = params['affine_translation_range']
            
            angle = np.random.uniform(angle_min, angle_max)
            scale = np.random.uniform(scale_min, scale_max)
            
            # Create rotation + scale matrix
            transform = cv2.getRotationMatrix2D(center, angle, scale)
            
            # Add translation
            transform[0, 2] += np.random.uniform(-width*trans_range, width*trans_range)
            transform[1, 2] += np.random.uniform(-height*trans_range, height*trans_range)
            
            # ✅ warpAffine expects (width, height) - correct
            img2 = cv2.warpAffine(base_image, transform, (width, height))
            transform = np.vstack([transform, [0, 0, 1]])
            
        elif transformation_type == 'rotation':
            # Pure rotation
            center = (width/2, height/2)  # ✅ (x, y)
            angle_min, angle_max = params['rotation_angle_range']
            angle = np.random.uniform(angle_min, angle_max)
            
            transform_2x3 = cv2.getRotationMatrix2D(center, angle, 1.0)
            img2 = cv2.warpAffine(base_image, transform_2x3, (width, height))
            transform = np.vstack([transform_2x3, [0, 0, 1]])
            
        elif transformation_type == 'scale':
            # Scale transformation
            scale_min, scale_max = params['scale_factor_range']
            scale_factor = np.random.uniform(scale_min, scale_max)
            
            new_w, new_h = int(width * scale_factor), int(height * scale_factor)
            # ✅ resize expects (width, height) - correct
            img2 = cv2.resize(base_image, (new_w, new_h))
            
            if scale_factor > 1:
                # Crop to original size
                start_x = (new_w - width) // 2
                start_y = (new_h - height) // 2
                img2 = img2[start_y:start_y+height, start_x:start_x+width]
                
                transform = np.array([
                    [scale_factor, 0, -start_x],
                    [0, scale_factor, -start_y],
                    [0, 0, 1]
                ])
            else:
                # Pad to original size
                pad_x = (width - new_w) // 2
                pad_y = (height - new_h) // 2
                remaining_pad_x = width - new_w - pad_x
                remaining_pad_y = height - new_h - pad_y
                
                img2 = cv2.copyMakeBorder(img2, pad_y, remaining_pad_y, 
                                        pad_x, remaining_pad_x, cv2.BORDER_CONSTANT)
                
                transform = np.array([
                    [scale_factor, 0, pad_x],
                    [0, scale_factor, pad_y],
                    [0, 0, 1]
                ])
        
        return base_image, img2, transform
    
    def detailed_memory_profile(self, method: str, image: np.ndarray) -> Dict[str, float]:
        """ memory profiling"""
        if not self.config.memory_profiling:
            return {}
        
        # Start memory tracing
        tracemalloc.start()
        process = psutil.Process()
        
        # Baseline memory
        memory_baseline = process.memory_info().rss / 1024 / 1024  # MB
        
        try:
            # Run detection
            if method in ['SIFT', 'ORB', 'AKAZE', 'BRISK', 'Harris', 'GoodFeatures']:
                detector = create_traditional_detector(method, max_features=2000)
                features = detector.detect(image)
            elif method in ['SuperPoint', 'DISK', 'ALIKED']:
                from .deep_learning_detectors import create_deep_learning_detector
                detector = create_deep_learning_detector(method, max_features=2000)
                features = detector.detect(image)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            # Peak memory during operation
            memory_peak = process.memory_info().rss / 1024 / 1024  # MB
            
            # Get traced memory
            current, peak_traced = tracemalloc.get_traced_memory()
            current_mb = current / 1024 / 1024
            peak_traced_mb = peak_traced / 1024 / 1024
            
        finally:
            tracemalloc.stop()
        
        # Final memory
        memory_final = process.memory_info().rss / 1024 / 1024  # MB
        
        return {
            'baseline_mb': memory_baseline,
            'peak_mb': memory_peak,
            'final_mb': memory_final,
            'net_increase_mb': memory_final - memory_baseline,
            'traced_current_mb': current_mb,
            'traced_peak_mb': peak_traced_mb
        }
    
    def run_statistical_benchmark(self, methods: List[str]) -> Dict[str, Any]:
        """Run benchmark with statistical analysis"""
        print("Running Statistical Benchmark...")
        print("=" * 50)
        
        # Create diverse test images
        test_images = []
        for height, width in self.config.image_sizes:
            for density in ['low', 'medium', 'high']:
                for texture in ['low', 'medium']:
                    img = self.image_generator.create_realistic_test_image(
                        height, width, density, texture
                    )
                    test_images.append({
                        'image': img,
                        'size': f"{width}x{height}",
                        'density': density,
                        'texture': texture
                    })
        
        # Run benchmarks
        all_results = {}
        method_performance = defaultdict(list)
        
        for method in methods:
            print(f"Benchmarking {method}...")
            method_results = []
            
            for test_case in test_images:
                image = test_case['image']
                
                # Multiple runs for statistical significance
                run_times = []
                run_features = []
                run_memory = []
                
                for run in range(self.config.num_runs):
                    try:
                        # Memory profiling
                        memory_profile = self.detailed_memory_profile(method, image)
                        
                        # Timing
                        start_time = time.time()
                        
                        if method in ['SIFT', 'ORB', 'AKAZE', 'BRISK', 'Harris', 'GoodFeatures']:
                            detector = create_traditional_detector(method, max_features=2000)
                            features = detector.detect(image)
                        elif method in ['SuperPoint', 'DISK', 'ALIKED']:
                            from .deep_learning_detectors import create_deep_learning_detector
                            detector = create_deep_learning_detector(method, max_features=2000)
                            features = detector.detect(image)
                        else:
                            continue
                        
                        detection_time = time.time() - start_time
                        
                        run_times.append(detection_time)
                        run_features.append(len(features))
                        if memory_profile:
                            run_memory.append(memory_profile.get('peak_mb', 0))
                        
                    except Exception as e:
                        print(f"  Error in run {run}: {e}")
                        continue
                
                if run_times:  # If we have successful runs
                    # Statistical analysis of runs
                    stats_data = {
                        'method': method,
                        'test_case': test_case,
                        'time_stats': self.statistical_analyzer._compute_descriptive_stats(run_times),
                        'feature_stats': self.statistical_analyzer._compute_descriptive_stats(run_features),
                        'memory_stats': self.statistical_analyzer._compute_descriptive_stats(run_memory) if run_memory else {},
                        'raw_data': {
                            'times': run_times,
                            'features': run_features,
                            'memory': run_memory
                        }
                    }
                    
                    method_results.append(stats_data)
                    method_performance[method].extend(run_times)
            
            all_results[method] = method_results
        
        # Cross-method statistical comparisons
        statistical_comparisons = {}
        for i, method1 in enumerate(methods):
            for method2 in methods[i+1:]:
                if method1 in method_performance and method2 in method_performance:
                    comparison = self.statistical_analyzer.compare_methods(
                        method_performance[method1],
                        method_performance[method2],
                        'detection_time'
                    )
                    statistical_comparisons[f"{method1}_vs_{method2}"] = comparison
        
        return {
            'detailed_results': all_results,
            'statistical_comparisons': statistical_comparisons,
            'summary': self._create_statistical_summary(all_results),
            'config': self.config
        }
    
    def _create_statistical_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Create statistical summary of results"""
        summary = {}
        
        for method, method_results in results.items():
            if method_results:
                # Aggregate across all test cases
                all_times = []
                all_features = []
                all_memory = []
                
                for result in method_results:
                    all_times.extend(result['raw_data']['times'])
                    all_features.extend(result['raw_data']['features'])
                    all_memory.extend(result['raw_data']['memory'])
                
                summary[method] = {
                    'overall_time_stats': self.statistical_analyzer._compute_descriptive_stats(all_times),
                    'overall_feature_stats': self.statistical_analyzer._compute_descriptive_stats(all_features),
                    'overall_memory_stats': self.statistical_analyzer._compute_descriptive_stats(all_memory),
                    'total_test_cases': len(method_results),
                    'total_runs': sum(len(r['raw_data']['times']) for r in method_results)
                }
        
        return summary


# Quick test functions with enhancements
def quick_test(methods: List[str] = None, 
                       image_size: Tuple[int, int] = (480, 640),
                       num_runs: int = 5) -> Dict[str, Any]:
    """ quick test with statistical analysis"""
    if methods is None:
        methods = ['SIFT', 'ORB', 'AKAZE']
    
    config = BenchmarkConfig(num_runs=num_runs, image_sizes=[image_size])
    generator = SyntheticImageGenerator()
    analyzer = StatisticalAnalyzer()
    
    print(f" Quick Test - Image Size: {image_size[1]}x{image_size[0]}")
    print("=" * 60)
    
    # Create realistic test image
    img = generator.create_realistic_test_image(image_size[0], image_size[1], 'medium', 'medium')
    
    results = {}
    all_times = {}
    
    for method in methods:
        try:
            times = []
            features_counts = []
            
            # Multiple runs for statistics
            for run in range(num_runs):
                start_time = time.time()
                
                if method in ['SIFT', 'ORB', 'AKAZE', 'BRISK']:
                    detector = create_traditional_detector(method, max_features=1000)
                    features = detector.detect(img)
                else:
                    print(f"Method {method} not supported in quick test")
                    break
                
                detection_time = time.time() - start_time
                times.append(detection_time)
                features_counts.append(len(features))
            
            if times:
                time_stats = analyzer._compute_descriptive_stats(times)
                feature_stats = analyzer._compute_descriptive_stats(features_counts)
                
                results[method] = {
                    'time_stats': time_stats,
                    'feature_stats': feature_stats,
                    'fps_mean': 1.0 / time_stats['mean'] if time_stats['mean'] > 0 else 0
                }
                
                all_times[method] = times
                
                print(f"{method:<8}: {time_stats['mean']:.3f}±{time_stats['std']:.3f}s, "
                      f"{feature_stats['mean']:.0f}±{feature_stats['std']:.0f} features, "
                      f"{results[method]['fps_mean']:6.1f} FPS")
        
        except Exception as e:
            print(f"{method:<8}: ERROR - {e}")
    
    # Statistical comparisons
    comparisons = {}
    method_list = list(all_times.keys())
    for i, method1 in enumerate(method_list):
        for method2 in method_list[i+1:]:
            comparison = analyzer.compare_methods(all_times[method1], all_times[method2], 'detection_time')
            comparisons[f"{method1}_vs_{method2}"] = comparison
    
    return {'results': results, 'comparisons': comparisons}


if __name__ == "__main__":
    # Example usage of  benchmarking
    print(" Feature Detection Benchmarking Suite")
    print("=" * 50)
    
    # Configuration
    config = BenchmarkConfig(
        num_runs=5,
        image_sizes=[(240, 320), (480, 640)],
        statistical_confidence=0.95
    )
    
    #  quick test
    print("\n1.  Quick Test:")
    _results = quick_test(['SIFT', 'ORB', 'AKAZE'], num_runs=5)
    
    # Show statistical comparisons
    print("\nStatistical Comparisons:")
    for comparison_name, comparison in _results['comparisons'].items():
        if 'statistical_test' in comparison:
            test = comparison['statistical_test']
            significance = "significant" if test['significant'] else "not significant"
            print(f"{comparison_name}: p={test['p_value']:.4f} ({significance})")