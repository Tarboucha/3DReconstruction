import numpy as np
import os
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import time

@dataclass
class CameraPattern:
    """Stores learned patterns from successfully processed cameras"""
    camera_name: str
    image_width: int
    image_height: int
    focal_length: float
    focal_ratio: float  # f / width
    aspect_ratio: float  # width / height
    resolution: int     # width * height
    processing_order: int
    initial_error: float  # How far off the initial estimate was
    confidence: float   # How reliable this pattern is
    timestamp: float

class ProgressiveLearningIntrinsicsEstimator:
    """
    Progressive learning system for camera intrinsics estimation.
    Gets better as more cameras are successfully processed.
    """
    
    def __init__(self):
        self.learning_database: List[CameraPattern] = []
        self.processing_order = 0
        
        # Learning parameters
        self.min_cameras_for_learning = 2
        self.max_database_size = 50  # Keep recent patterns
        self.confidence_threshold = 0.6
        
        # Heuristic parameters
        self.default_fov_degrees = 55.0  # Moderate field of view assumption
        self.phone_fov_degrees = 65.0    # Phones typically have wider FOV
        self.dslr_fov_degrees = 50.0     # DSLRs typically narrower FOV
    
    def estimate_intrinsics_with_progressive_learning(self, selected_image: str, 
                                                    reconstruction_state: Dict) -> np.ndarray:
        """
        Main function: Progressive learning intrinsics estimation.
        
        Args:
            selected_image: Name of the image to estimate intrinsics for
            reconstruction_state: Current reconstruction state with existing cameras
            
        Returns:
            np.ndarray: 3x3 intrinsic matrix estimate
        """
        
        try:
            # Get image dimensions
            width, height = self._get_image_dimensions(selected_image)
            existing_cameras = reconstruction_state.get('cameras', {})
            num_existing = len(existing_cameras)
            
            print(f"Progressive intrinsics estimation for {selected_image}")
            print(f"  Image size: {width}x{height}")
            print(f"  Existing cameras: {num_existing}")
            
            # Progressive learning based on number of existing cameras
            if num_existing == 0:
                # First camera: Pure geometric heuristics
                K_estimate = self._geometric_heuristic_estimation(width, height)
                method = "geometric_heuristic"
                confidence = 0.4
                
            elif num_existing == 1:
                # Second camera: Learn from first + geometric blend
                K_estimate = self._learn_from_first_camera(
                    width, height, existing_cameras
                )
                method = "first_camera_learning"
                confidence = 0.5
                
            elif num_existing < 5:
                # Early learning phase: Pattern recognition begins
                K_estimate = self._early_learning_phase(
                    width, height, existing_cameras
                )
                method = "early_learning"
                confidence = 0.7
                
            else:
                # Mature learning phase: Advanced pattern analysis
                K_estimate = self._mature_learning_phase(
                    width, height, existing_cameras
                )
                method = "mature_learning"  
                confidence = 0.8
            
            print(f"  Method: {method}")
            print(f"  Estimated f: {K_estimate[0,0]:.1f}px (confidence: {confidence:.2f})")
            
            return K_estimate
            
        except Exception as e:
            print(f"  Progressive learning failed: {e}")
            return self._get_default_intrinsics(width, height)
    
    def _geometric_heuristic_estimation(self, width: int, height: int) -> np.ndarray:
        """Pure geometric heuristics for first camera"""
        
        # Estimate camera type from resolution
        total_pixels = width * height
        aspect_ratio = width / height
        
        # Different heuristics based on likely camera type
        if total_pixels < 2_000_000:  # < 2MP (old phone/webcam)
            fov_degrees = self.phone_fov_degrees + 10  # Wider FOV
        elif total_pixels < 8_000_000:  # 2-8MP (modern phone)
            fov_degrees = self.phone_fov_degrees
        elif total_pixels < 20_000_000:  # 8-20MP (consumer camera)
            fov_degrees = self.default_fov_degrees
        else:  # > 20MP (professional camera)
            fov_degrees = self.dslr_fov_degrees
        
        # Adjust for aspect ratio
        if aspect_ratio > 2.0:  # Panoramic
            fov_degrees += 15  # Wider FOV for panoramic
        elif aspect_ratio < 0.8:  # Portrait
            fov_degrees -= 5   # Slightly narrower
        
        # Calculate focal length from field of view
        # f = (image_width / 2) / tan(fov/2)
        fov_radians = np.deg2rad(fov_degrees)
        f = (width / 2.0) / np.tan(fov_radians / 2.0)
        
        # Principal point at image center
        cx, cy = width / 2.0, height / 2.0
        
        K = np.array([[f, 0, cx],
                     [0, f, cy],
                     [0, 0, 1]], dtype=np.float32)
        
        print(f"    Geometric heuristic: FOV={fov_degrees:.1f}°, f={f:.1f}px")
        return K
    
    def _learn_from_first_camera(self, width: int, height: int, 
                                existing_cameras: Dict) -> np.ndarray:
        """Learn from the first successfully processed camera"""
        
        # Get first camera data
        first_camera_name = list(existing_cameras.keys())[0]
        first_camera_data = existing_cameras[first_camera_name]
        
        if 'K' not in first_camera_data:
            return self._geometric_heuristic_estimation(width, height)
        
        # Get first camera's properties
        first_width, first_height = self._get_image_dimensions(first_camera_name)
        first_K = first_camera_data['K']
        first_f = first_K[0, 0]
        first_focal_ratio = first_f / first_width
        
        # Compare image properties
        size_similarity = min(width * height, first_width * first_height) / max(width * height, first_width * first_height)
        aspect_similarity = min(width/height, first_width/first_height) / max(width/height, first_width/first_height)
        
        # If images are similar, use first camera's focal ratio
        if size_similarity > 0.5 and aspect_similarity > 0.8:
            # Images are similar - apply first camera's focal ratio
            predicted_f = first_focal_ratio * width
            weight_learned = 0.7
            print(f"    Similar to first camera: using ratio {first_focal_ratio:.3f}")
        else:
            # Images are different - blend with geometric heuristic
            predicted_f = first_focal_ratio * width
            weight_learned = 0.4
            print(f"    Different from first camera: reduced confidence")
        
        # Blend with geometric heuristic
        geometric_K = self._geometric_heuristic_estimation(width, height)
        geometric_f = geometric_K[0, 0]
        
        blended_f = weight_learned * predicted_f + (1 - weight_learned) * geometric_f
        
        # Build intrinsic matrix
        cx, cy = width / 2.0, height / 2.0
        K = np.array([[blended_f, 0, cx],
                     [0, blended_f, cy],
                     [0, 0, 1]], dtype=np.float32)
        
        print(f"    First camera f: {first_f:.1f}px, ratio: {first_focal_ratio:.3f}")
        print(f"    Predicted f: {predicted_f:.1f}px, geometric f: {geometric_f:.1f}px")
        print(f"    Blended f: {blended_f:.1f}px")
        
        return K
    
    def _early_learning_phase(self, width: int, height: int, 
                             existing_cameras: Dict) -> np.ndarray:
        """Early learning: Pattern recognition from 2-4 cameras"""
        
        # Collect patterns from existing cameras
        focal_ratios = []
        similar_cameras = []
        
        current_aspect = width / height
        current_resolution = width * height
        
        for cam_name, cam_data in existing_cameras.items():
            if 'K' not in cam_data:
                continue
                
            cam_width, cam_height = self._get_image_dimensions(cam_name)
            cam_f = cam_data['K'][0, 0]
            cam_focal_ratio = cam_f / cam_width
            cam_aspect = cam_width / cam_height
            cam_resolution = cam_width * cam_height
            
            focal_ratios.append(cam_focal_ratio)
            
            # Find cameras with similar properties
            aspect_similarity = min(current_aspect, cam_aspect) / max(current_aspect, cam_aspect)
            resolution_similarity = min(current_resolution, cam_resolution) / max(current_resolution, cam_resolution)
            
            if aspect_similarity > 0.8 and resolution_similarity > 0.3:
                similar_cameras.append({
                    'focal_ratio': cam_focal_ratio,
                    'similarity_score': aspect_similarity * resolution_similarity,
                    'camera_name': cam_name
                })
        
        # Prediction strategy
        if similar_cameras:
            # Use similar cameras with weighted average
            total_weight = sum(cam['similarity_score'] for cam in similar_cameras)
            weighted_ratio = sum(cam['focal_ratio'] * cam['similarity_score'] 
                               for cam in similar_cameras) / total_weight
            
            predicted_f = weighted_ratio * width
            confidence = 0.8
            
            print(f"    Found {len(similar_cameras)} similar cameras")
            print(f"    Weighted focal ratio: {weighted_ratio:.3f}")
            
        elif focal_ratios:
            # Use recent cameras with exponential weighting
            weights = [np.exp(-i * 0.3) for i in range(len(focal_ratios))]
            weighted_ratio = np.average(focal_ratios, weights=weights)
            
            predicted_f = weighted_ratio * width
            confidence = 0.6
            
            print(f"    Recent-weighted focal ratio: {weighted_ratio:.3f}")
            
        else:
            # Fallback to geometric heuristic
            return self._geometric_heuristic_estimation(width, height)
        
        # Blend with geometric heuristic based on confidence
        geometric_K = self._geometric_heuristic_estimation(width, height)
        geometric_f = geometric_K[0, 0]
        
        final_f = confidence * predicted_f + (1 - confidence) * geometric_f
        
        # Build intrinsic matrix
        cx, cy = width / 2.0, height / 2.0
        K = np.array([[final_f, 0, cx],
                     [0, final_f, cy],
                     [0, 0, 1]], dtype=np.float32)
        
        print(f"    Predicted f: {predicted_f:.1f}px, geometric f: {geometric_f:.1f}px")
        print(f"    Final blended f: {final_f:.1f}px")
        
        return K
    
    def _mature_learning_phase(self, width: int, height: int,
                              existing_cameras: Dict) -> np.ndarray:
        """Mature learning: Advanced pattern analysis with trend detection"""
        
        # Collect comprehensive camera data
        camera_patterns = []
        
        for i, (cam_name, cam_data) in enumerate(existing_cameras.items()):
            if 'K' not in cam_data:
                continue
                
            cam_width, cam_height = self._get_image_dimensions(cam_name)
            cam_f = cam_data['K'][0, 0]
            
            pattern = {
                'name': cam_name,
                'width': cam_width,
                'height': cam_height,
                'focal_length': cam_f,
                'focal_ratio': cam_f / cam_width,
                'aspect_ratio': cam_width / cam_height,
                'resolution': cam_width * cam_height,
                'processing_order': i
            }
            camera_patterns.append(pattern)
        
        if not camera_patterns:
            return self._geometric_heuristic_estimation(width, height)
        
        # Multiple prediction methods
        predictions = []
        confidences = []
        
        # Method 1: Similarity-based prediction
        similarity_pred, similarity_conf = self._similarity_based_prediction(
            width, height, camera_patterns
        )
        if similarity_pred is not None:
            predictions.append(similarity_pred)
            confidences.append(similarity_conf)
        
        # Method 2: Trend analysis
        trend_pred, trend_conf = self._trend_analysis_prediction(
            width, height, camera_patterns
        )
        if trend_pred is not None:
            predictions.append(trend_pred)
            confidences.append(trend_conf)
        
        # Method 3: Resolution-based prediction
        resolution_pred, resolution_conf = self._resolution_based_prediction(
            width, height, camera_patterns
        )
        if resolution_pred is not None:
            predictions.append(resolution_pred)
            confidences.append(resolution_conf)
        
        # Method 4: Camera type classification
        type_pred, type_conf = self._camera_type_prediction(
            width, height, camera_patterns
        )
        if type_pred is not None:
            predictions.append(type_pred)
            confidences.append(type_conf)
        
        # Combine predictions with confidence weighting
        if predictions:
            total_confidence = sum(confidences)
            final_f = sum(p * c for p, c in zip(predictions, confidences)) / total_confidence
            overall_confidence = min(0.9, total_confidence / len(predictions))
            
            print(f"    Combined {len(predictions)} prediction methods")
            print(f"    Predictions: {[f'{p:.0f}' for p in predictions]}")
            print(f"    Confidences: {[f'{c:.2f}' for c in confidences]}")
            print(f"    Final prediction: {final_f:.1f}px (confidence: {overall_confidence:.2f})")
            
        else:
            # Fallback to geometric heuristic
            return self._geometric_heuristic_estimation(width, height)
        
        # Build intrinsic matrix
        cx, cy = width / 2.0, height / 2.0
        K = np.array([[final_f, 0, cx],
                     [0, final_f, cy],
                     [0, 0, 1]], dtype=np.float32)
        
        return K
    
    def _similarity_based_prediction(self, width: int, height: int, 
                                   camera_patterns: List[Dict]) -> Tuple[Optional[float], float]:
        """Predict based on cameras with similar properties"""
        
        current_aspect = width / height
        current_resolution = width * height
        
        similar_cameras = []
        
        for pattern in camera_patterns:
            # Calculate similarity scores
            aspect_sim = min(current_aspect, pattern['aspect_ratio']) / max(current_aspect, pattern['aspect_ratio'])
            resolution_sim = min(current_resolution, pattern['resolution']) / max(current_resolution, pattern['resolution'])
            
            # Combined similarity (both aspect and resolution matter)
            combined_similarity = aspect_sim * resolution_sim
            
            if combined_similarity > 0.3:  # Threshold for "similar"
                similar_cameras.append({
                    'focal_ratio': pattern['focal_ratio'],
                    'similarity': combined_similarity,
                    'name': pattern['name']
                })
        
        if not similar_cameras:
            return None, 0.0
        
        # Weight by similarity scores
        total_weight = sum(cam['similarity'] for cam in similar_cameras)
        weighted_ratio = sum(cam['focal_ratio'] * cam['similarity'] 
                           for cam in similar_cameras) / total_weight
        
        predicted_f = weighted_ratio * width
        confidence = min(0.8, len(similar_cameras) / 3.0)  # More similar cameras = higher confidence
        
        print(f"      Similarity method: {len(similar_cameras)} similar cameras, ratio={weighted_ratio:.3f}")
        
        return predicted_f, confidence
    
    def _trend_analysis_prediction(self, width: int, height: int,
                                 camera_patterns: List[Dict]) -> Tuple[Optional[float], float]:
        """Analyze trends in recent cameras"""
        
        if len(camera_patterns) < 3:
            return None, 0.0
        
        # Sort by processing order (most recent first)
        sorted_patterns = sorted(camera_patterns, key=lambda x: x['processing_order'], reverse=True)
        recent_patterns = sorted_patterns[:min(4, len(sorted_patterns))]  # Last 4 cameras
        
        # Extract focal ratios
        focal_ratios = [p['focal_ratio'] for p in recent_patterns]
        
        # Linear trend analysis
        x = np.arange(len(focal_ratios))
        coeffs = np.polyfit(x, focal_ratios, 1)
        slope, intercept = coeffs
        
        # Predict next focal ratio (extrapolate trend)
        next_ratio = intercept + slope * len(focal_ratios)
        
        # Confidence based on trend strength and consistency
        trend_strength = abs(slope)
        ratio_std = np.std(focal_ratios)
        
        if ratio_std < 0.1:  # Consistent ratios
            confidence = 0.7
        elif trend_strength > 0.02:  # Strong trend
            confidence = 0.6
        else:  # Weak trend
            confidence = 0.3
        
        # Sanity check - don't extrapolate too far
        min_ratio = min(focal_ratios)
        max_ratio = max(focal_ratios)
        ratio_range = max_ratio - min_ratio
        
        # Clamp prediction
        safe_min = min_ratio - 0.5 * ratio_range
        safe_max = max_ratio + 0.5 * ratio_range
        next_ratio = np.clip(next_ratio, safe_min, safe_max)
        
        predicted_f = next_ratio * width
        
        print(f"      Trend method: slope={slope:.4f}, predicted_ratio={next_ratio:.3f}")
        
        return predicted_f, confidence
    
    def _resolution_based_prediction(self, width: int, height: int,
                                   camera_patterns: List[Dict]) -> Tuple[Optional[float], float]:
        """Predict based on resolution patterns"""
        
        current_resolution = width * height
        
        # Group patterns by resolution similarity
        resolution_groups = []
        for pattern in camera_patterns:
            resolution_ratio = min(current_resolution, pattern['resolution']) / max(current_resolution, pattern['resolution'])
            if resolution_ratio > 0.5:  # Similar resolution
                resolution_groups.append(pattern)
        
        if not resolution_groups:
            return None, 0.0
        
        # Average focal ratio for similar resolutions
        avg_focal_ratio = np.mean([p['focal_ratio'] for p in resolution_groups])
        predicted_f = avg_focal_ratio * width
        
        confidence = min(0.6, len(resolution_groups) / 2.0)
        
        print(f"      Resolution method: {len(resolution_groups)} similar resolution cameras, ratio={avg_focal_ratio:.3f}")
        
        return predicted_f, confidence
    
    def _camera_type_prediction(self, width: int, height: int,
                              camera_patterns: List[Dict]) -> Tuple[Optional[float], float]:
        """Predict based on camera type classification"""
        
        # Classify current image
        current_type = self._classify_camera_type(width, height)
        
        # Find cameras of similar type
        type_patterns = []
        for pattern in camera_patterns:
            pattern_type = self._classify_camera_type(pattern['width'], pattern['height'])
            if pattern_type == current_type:
                type_patterns.append(pattern)
        
        if not type_patterns:
            return None, 0.0
        
        # Average focal ratio for this camera type
        avg_focal_ratio = np.mean([p['focal_ratio'] for p in type_patterns])
        predicted_f = avg_focal_ratio * width
        
        confidence = min(0.7, len(type_patterns) / 2.0)
        
        print(f"      Camera type method: type={current_type}, {len(type_patterns)} similar cameras, ratio={avg_focal_ratio:.3f}")
        
        return predicted_f, confidence
    
    def _classify_camera_type(self, width: int, height: int) -> str:
        """Classify camera type based on resolution and aspect ratio"""
        
        total_pixels = width * height
        aspect_ratio = width / height
        
        if total_pixels < 2_000_000:
            base_type = 'phone_old'
        elif total_pixels < 8_000_000:
            base_type = 'phone_modern'
        elif total_pixels < 20_000_000:
            base_type = 'consumer_camera'
        else:
            base_type = 'professional_camera'
        
        if aspect_ratio > 2.0:
            return base_type + '_panoramic'
        else:
            return base_type
    
    def _get_image_dimensions(self, image_name: str) -> Tuple[int, int]:
        """Get image dimensions (width, height)"""
        # This should be implemented based on your image loading system
        # For now, return default dimensions
        # You should replace this with actual image dimension extraction
        
        try:
            # Example implementation - replace with your actual method
            from PIL import Image
            image_path = self._get_image_path(image_name)
            with Image.open(image_path) as img:
                return img.width, img.height
        except:
            # Fallback default
            return 1920, 1080
    
    def _get_image_path(self, image_name: str) -> str:
        """Get full path to image file"""
        # Replace with your actual image path construction
        return os.path.join("images/statue_of_liberty_images", image_name)
    
    def _get_default_intrinsics(self, width: int, height: int) -> np.ndarray:
        """Fallback default intrinsics"""
        f = max(width, height) * 0.9
        cx, cy = width / 2.0, height / 2.0
        
        return np.array([[f, 0, cx],
                        [0, f, cy],
                        [0, 0, 1]], dtype=np.float32)
    
    def update_learning_database(self, camera_name: str, initial_K: np.ndarray, 
                                refined_K: np.ndarray) -> None:
        """Update learning database after successful bundle adjustment"""
        
        try:
            width, height = self._get_image_dimensions(camera_name)
            
            initial_f = initial_K[0, 0]
            refined_f = refined_K[0, 0]
            
            # Calculate how good the initial estimate was
            improvement_ratio = abs(refined_f - initial_f) / refined_f
            
            # Only store if the initial estimate was reasonably good
            if improvement_ratio < 0.5:  # Less than 50% error
                pattern = CameraPattern(
                    camera_name=camera_name,
                    image_width=width,
                    image_height=height,
                    focal_length=refined_f,
                    focal_ratio=refined_f / width,
                    aspect_ratio=width / height,
                    resolution=width * height,
                    processing_order=self.processing_order,
                    initial_error=improvement_ratio,
                    confidence=1.0 - improvement_ratio,  # Better initial estimate = higher confidence
                    timestamp=time.time()
                )
                
                self.learning_database.append(pattern)
                self.processing_order += 1
                
                # Keep database size manageable
                if len(self.learning_database) > self.max_database_size:
                    # Keep most recent and highest confidence patterns
                    self.learning_database.sort(key=lambda x: (x.timestamp, x.confidence), reverse=True)
                    self.learning_database = self.learning_database[:self.max_database_size]
                
                print(f"Updated learning database: {camera_name}, error={improvement_ratio:.3f}")
            
        except Exception as e:
            print(f"Failed to update learning database: {e}")