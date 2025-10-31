"""
Multi-Method Feature Detector - Updated to use matchers module

This class manages multiple feature detectors using the new matchers/ module.
Each method stays independent (no combining/merging).
"""

import numpy as np
import cv2
from typing import List, Dict, Optional, Union

from .core_data_structures import FeatureData, DetectorType
from .logger import get_logger

# Module logger
logger = get_logger("detector")

# Import from new matchers module
from .matchers.detectors.traditional import SIFTDetector, ORBDetector, AKAZEDetector
from .matchers.detectors.base_detector import BaseDetector

# Try to import deep learning detectors
try:
    from .matchers.detectors.deep_learning import SuperPointDetector, ALIKEDDetector
    HAS_DEEP_LEARNING = True
except ImportError:
    HAS_DEEP_LEARNING = False

# Try to import dense matcher adapter
try:
    from .matchers.dense_adapter import DenseDetectorAdapter
    HAS_DENSE = True
except ImportError:
    HAS_DENSE = False


class DetectorAdapter:
    """
    Adapter to make matchers/ detectors compatible with FeatureData format
    """
    def __init__(self, detector: BaseDetector):
        self.detector = detector
        self.name = detector.name

    def detect(self, image: np.ndarray) -> FeatureData:
        """
        Detect features and convert to FeatureData format

        Args:
            image: Input image (RGB or grayscale)

        Returns:
            FeatureData compatible with existing pipeline
        """
        import time
        start_time = time.time()

        # Detect using new matchers module
        detection_result = self.detector.detect(image)

        # Convert keypoints array to cv2.KeyPoint objects
        keypoints = []
        for kp_array in detection_result.keypoints:
            x, y = kp_array
            # Use scores if available, otherwise default
            score = detection_result.scores[len(keypoints)] if len(detection_result.scores) > len(keypoints) else 1.0
            kp = cv2.KeyPoint(x=float(x), y=float(y), size=1.0, response=float(score))
            keypoints.append(kp)

        detection_time = time.time() - start_time

        # Create FeatureData
        feature_data = FeatureData(
            keypoints=keypoints,
            descriptors=detection_result.descriptors,
            detector_name=self.name,
            raw_image=image,
            detection_time=detection_time,
            image_size=(image.shape[1], image.shape[0])  # (width, height)
        )

        return feature_data


class MultiMethodFeatureDetector:
    """
    Manages multiple feature detection methods using matchers/ module

    Simplified for new API - no combining, no merging.
    Each method stays independent.

    Usage:
        detector = MultiMethodFeatureDetector(
            methods=['SIFT', 'ORB'],
            max_features_per_method=2000,
            detector_params={'SIFT': {'max_features': 3000}}
        )

        # Detect with all methods
        all_features = detector.detect_all(image)
        # Returns: {'SIFT': FeatureData(...), 'ORB': FeatureData(...)}

        # Or detect with specific method
        sift_features = detector.detectors['SIFT'].detect(image)
    """

    def __init__(self,
                 methods: List[Union[str, DetectorType]],
                 max_features_per_method: int = 2000,
                 combine_strategy: str = "independent",  # Only for backward compatibility
                 detector_params: Optional[Dict[str, Dict]] = None):
        """
        Initialize multi-method detector

        Args:
            methods: List of detector names (e.g., ['SIFT', 'ORB', 'AKAZE'])
            max_features_per_method: Max features per method
            combine_strategy: Ignored (kept for backward compatibility)
            detector_params: Per-method detector parameters
        """
        self.methods = [str(m) for m in methods]
        self.max_features_per_method = max_features_per_method
        self.detector_params = detector_params or {}

        # Initialize detectors
        self.detectors = self._initialize_detectors()

        # Validate that we have at least one working detector
        if not self.detectors:
            raise ValueError("No detectors could be initialized!")

        logger.info(f"Initialized {len(self.detectors)} detectors: {list(self.detectors.keys())}")

    def _initialize_detectors(self) -> Dict[str, DetectorAdapter]:
        """
        Initialize all requested detectors using matchers/ module

        Returns:
            Dictionary mapping method name to detector adapter
        """
        detectors = {}

        for method_str in self.methods:
            try:
                # Get parameters for this method
                params = self.detector_params.get(method_str, {}).copy()

                # Set max_features parameter
                max_features = params.pop('max_features', self.max_features_per_method)

                # Create detector from matchers module
                if method_str == 'SIFT':
                    detector = SIFTDetector(max_features=max_features)

                elif method_str == 'ORB':
                    detector = ORBDetector(max_features=max_features)

                elif method_str == 'AKAZE':
                    detector = AKAZEDetector()

                elif method_str == 'SuperPoint':
                    if not HAS_DEEP_LEARNING:
                        logger.warning(f"{method_str} requires deep learning dependencies (torch, lightglue)")
                        continue
                    device = params.pop('device', 'cuda')
                    max_keypoints = params.pop('max_keypoints', max_features)
                    detector = SuperPointDetector(max_keypoints=max_keypoints, device=device)

                elif method_str == 'ALIKED':
                    if not HAS_DEEP_LEARNING:
                        logger.warning(f"{method_str} requires deep learning dependencies (torch)")
                        continue
                    device = params.pop('device', 'cuda')
                    max_keypoints = params.pop('max_keypoints', max_features)
                    weights_path = params.pop('weights_path', None)
                    try:
                        detector = ALIKEDDetector(
                            max_keypoints=max_keypoints,
                            device=device,
                            weights_path=weights_path
                        )
                    except FileNotFoundError as e:
                        logger.warning(f"{method_str} weights not found: {e}")
                        continue

                elif method_str == 'DKM':
                    if not HAS_DENSE:
                        logger.warning(f"{method_str} requires dense matching dependencies (torch, gim)")
                        continue
                    # DKM is a dense matcher - uses special adapter
                    detector = DenseDetectorAdapter(method='DKM')

                else:
                    logger.warning(f"Unknown detector: {method_str}")
                    continue

                # Wrap detector in adapter
                detectors[method_str] = DetectorAdapter(detector)
                logger.info(f"  {method_str} detector initialized")

            except Exception as e:
                logger.error(f"Failed to initialize {method_str}: {e}")

        return detectors

    def detect_all(self, image: np.ndarray) -> Dict[str, FeatureData]:
        """
        Detect features using all initialized methods

        Each method runs independently and returns its own FeatureData.

        Args:
            image: Input image (RGB or grayscale)

        Returns:
            Dictionary mapping method name to FeatureData

        Example:
            >>> all_features = detector.detect_all(image)
            >>> sift_features = all_features['SIFT']
            >>> print(f"SIFT found {len(sift_features)} keypoints")
        """
        results = {}

        for name, detector in self.detectors.items():
            logger.debug(f"Detecting with {name}...")
            try:
                features = detector.detect(image)
                results[name] = features
                logger.debug(f"  {len(features)} features in {features.detection_time:.3f}s")
            except Exception as e:
                logger.error(f"Detection failed with {name}: {e}")
                # Store empty FeatureData for failed methods
                results[name] = FeatureData([], None, name, raw_image=image)

        return results

    def get_detector(self, method_name: str) -> Optional[DetectorAdapter]:
        """
        Get a specific detector by name

        Args:
            method_name: Name of the method (e.g., 'SIFT')

        Returns:
            Detector adapter or None if not found
        """
        return self.detectors.get(method_name)

    def has_method(self, method_name: str) -> bool:
        """Check if a method is available"""
        return method_name in self.detectors

    def __repr__(self) -> str:
        """String representation"""
        methods_str = ', '.join(self.detectors.keys())
        return f"MultiMethodFeatureDetector(methods=[{methods_str}])"


# =============================================================================
# Convenience Function
# =============================================================================

def create_multi_detector(methods: List[str],
                         max_features: int = 2000,
                         **detector_params) -> MultiMethodFeatureDetector:
    """
    Convenience function to create a multi-method detector

    Args:
        methods: List of method names
        max_features: Max features per method
        **detector_params: Parameters for specific detectors

    Returns:
        Configured MultiMethodFeatureDetector

    Example:
        >>> detector = create_multi_detector(
        ...     methods=['SIFT', 'ORB'],
        ...     max_features=2000,
        ...     SIFT={'max_features': 3000},
        ...     ORB={'max_features': 2500}
        ... )
    """
    return MultiMethodFeatureDetector(
        methods=methods,
        max_features_per_method=max_features,
        detector_params=detector_params
    )
