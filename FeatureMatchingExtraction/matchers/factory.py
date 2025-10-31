# matchers/factory.py

from .base_matcher import BaseMatcher
from .sparse_pipeline import SparsePipeline
from .dense_matcher import DenseMatcher

# Traditional detectors
from .detectors.traditional import SIFTDetector, ORBDetector, AKAZEDetector

# Deep learning detectors
from .detectors.deep_learning import SuperPointDetector, ALIKEDDetector

# Descriptor matchers
from .matching_algorithms.descriptor_matcher import BruteForceMatcher, FLANNMatcher
from .matching_algorithms.lightglue_matcher import LightGlueMatcher

# Dense matcher adapter for pipeline integration
from .dense_adapter import DenseMatcherAdapter


def create_matcher(method: str, **kwargs) -> BaseMatcher:
    """
    Create a matcher by name.

    Supports traditional methods (SIFT, ORB, AKAZE), deep learning sparse methods
    (SuperPoint, ALIKED, LightGlue), and dense methods (DKM, RoMa, LoFTR).

    Args:
        method: Matcher method name
            - Traditional sparse: 'SIFT', 'ORB', 'AKAZE'
            - Deep learning sparse: 'SuperPoint', 'ALIKED', 'LightGlue'
            - Dense: 'DKM', 'ROMA', 'LOFTR'

        **kwargs: Method-specific parameters
            For traditional sparse methods (SIFT, ORB, AKAZE):
                - max_features: Maximum features to detect (default: 8000)
                - ratio_test: Lowe's ratio test threshold (default: 0.75)
                - matcher_type: 'FLANN' or 'BruteForce' (default: 'FLANN')

            For deep learning sparse methods (SuperPoint, ALIKED):
                - max_keypoints: Maximum keypoints to detect (default: 2048 for SuperPoint, 4096 for ALIKED)
                - device: 'cuda' or 'cpu' (default: 'cuda')
                - matcher_type: 'LightGlue', 'FLANN', or 'BruteForce'
                    - SuperPoint default: 'LightGlue' (attention-based, most accurate)
                    - ALIKED default: 'FLANN'
                - ratio_test: For FLANN/BruteForce only (default: 0.75)
                - filter_threshold: For LightGlue only (default: 0.1)

            For LightGlue shorthand:
                - max_keypoints: Maximum keypoints (default: 2048)
                - device: 'cuda' or 'cpu' (default: 'cuda')
                - filter_threshold: Match filtering threshold (default: 0.1)

            For dense methods (DKM, ROMA, LOFTR):
                - device: 'cuda' or 'cpu' (default: 'cuda')
                - symmetric: Bidirectional matching (default: True)
                - resize_to: (width, height) tuple (default: (896, 672) for DKM)
                - max_matches: Maximum matches to sample (default: 8000)

    Returns:
        Configured matcher (BaseMatcher)

    Examples:
        >>> # Traditional sparse matchers
        >>> matcher = create_matcher('SIFT', ratio_test=0.7)
        >>> matcher = create_matcher('ORB', matcher_type='BruteForce', max_features=5000)

        >>> # Deep learning sparse - SuperPoint with LightGlue (best quality)
        >>> matcher = create_matcher('SuperPoint')  # Uses LightGlue by default
        >>> matcher = create_matcher('SuperPoint', matcher_type='LightGlue', device='cuda')
        >>> matcher = create_matcher('LightGlue')  # Shorthand for SuperPoint + LightGlue

        >>> # SuperPoint with traditional matchers (faster but less accurate)
        >>> matcher = create_matcher('SuperPoint', matcher_type='FLANN')
        >>> matcher = create_matcher('SuperPoint', matcher_type='BruteForce')

        >>> # ALIKED with traditional matchers
        >>> matcher = create_matcher('ALIKED', matcher_type='FLANN')
        >>> matcher = create_matcher('ALIKED', max_keypoints=5000, device='cuda')

        >>> # Dense matchers (end-to-end)
        >>> matcher = create_matcher('DKM', device='cuda')
        >>> matcher = create_matcher('DKM', symmetric=False, resize_to=(640, 480))

    Raises:
        ValueError: If method is not recognized or invalid parameter combination
    """
    method = method.upper()

    # ========================================================================
    # TRADITIONAL SPARSE METHODS (SIFT, ORB, AKAZE)
    # ========================================================================
    if method in ['SIFT', 'ORB', 'AKAZE']:
        # Extract parameters
        max_features = kwargs.pop('max_features', 8000)
        ratio_test = kwargs.pop('ratio_test', 0.75)
        matcher_type = kwargs.pop('matcher_type', 'FLANN')

        # Handle None matcher_type
        if matcher_type is None:
            matcher_type = 'FLANN'

        # Create detector
        if method == 'SIFT':
            detector = SIFTDetector(max_features=max_features)
        elif method == 'ORB':
            detector = ORBDetector(max_features=max_features)
        elif method == 'AKAZE':
            detector = AKAZEDetector()

        # Create descriptor matcher
        descriptor_type = detector.descriptor_type
        if matcher_type.upper() == 'FLANN':
            desc_matcher = FLANNMatcher(
                ratio_test=ratio_test,
                descriptor_type=descriptor_type
            )
        elif matcher_type.upper() == 'BRUTEFORCE':
            norm = 'HAMMING' if descriptor_type == 'uint8' else 'L2'
            desc_matcher = BruteForceMatcher(
                ratio_test=ratio_test,
                norm_type=norm
            )
        else:
            raise ValueError(
                f"Invalid matcher_type '{matcher_type}' for {method}. "
                f"Use 'FLANN' or 'BruteForce'"
            )

        return SparsePipeline(detector=detector, matcher=desc_matcher)

    # ========================================================================
    # DEEP LEARNING SPARSE METHODS (SuperPoint, ALIKED)
    # ========================================================================
    elif method in ['SUPERPOINT', 'ALIKED']:
        # Extract parameters
        max_keypoints = kwargs.pop('max_keypoints', 2048 if method == 'SUPERPOINT' else 4096)
        device = kwargs.pop('device', 'cuda')

        # Default matcher type: LightGlue for SuperPoint, FLANN for ALIKED
        default_matcher = 'LightGlue' if method == 'SUPERPOINT' else 'FLANN'
        matcher_type = kwargs.pop('matcher_type', default_matcher)

        # Create detector
        if method == 'SUPERPOINT':
            detector = SuperPointDetector(
                max_keypoints=max_keypoints,
                device=device
            )
        elif method == 'ALIKED':
            weights_path = kwargs.pop('weights_path', None)
            detector = ALIKEDDetector(
                max_keypoints=max_keypoints,
                device=device,
                weights_path=weights_path
            )

        # Create matcher
        descriptor_type = detector.descriptor_type
        if matcher_type.upper() == 'LIGHTGLUE':
            # LightGlue only works with SuperPoint
            if method != 'SUPERPOINT':
                raise ValueError(
                    f"LightGlue only works with SuperPoint detector, not {method}. "
                    f"Use matcher_type='FLANN' or 'BruteForce' instead."
                )
            filter_threshold = kwargs.pop('filter_threshold', 0.1)
            desc_matcher = LightGlueMatcher(
                device=device,
                filter_threshold=filter_threshold
            )
        elif matcher_type.upper() == 'FLANN':
            ratio_test = kwargs.pop('ratio_test', 0.75)
            desc_matcher = FLANNMatcher(
                ratio_test=ratio_test,
                descriptor_type=descriptor_type
            )
        elif matcher_type.upper() == 'BRUTEFORCE':
            ratio_test = kwargs.pop('ratio_test', 0.75)
            norm = 'HAMMING' if descriptor_type == 'uint8' else 'L2'
            desc_matcher = BruteForceMatcher(
                ratio_test=ratio_test,
                norm_type=norm
            )
        else:
            raise ValueError(
                f"Invalid matcher_type '{matcher_type}' for {method}. "
                f"Use 'LightGlue', 'FLANN', or 'BruteForce'"
            )

        return SparsePipeline(detector=detector, matcher=desc_matcher)

    # ========================================================================
    # LIGHTGLUE SHORTHAND (SuperPoint + LightGlue)
    # ========================================================================
    elif method == 'LIGHTGLUE':
        # Shorthand for SuperPoint + LightGlue
        max_keypoints = kwargs.pop('max_keypoints', 2048)
        device = kwargs.pop('device', 'cuda')
        filter_threshold = kwargs.pop('filter_threshold', 0.1)

        detector = SuperPointDetector(
            max_keypoints=max_keypoints,
            device=device
        )
        desc_matcher = LightGlueMatcher(
            device=device,
            filter_threshold=filter_threshold
        )

        return SparsePipeline(detector=detector, matcher=desc_matcher)

    # ========================================================================
    # DENSE METHODS (DKM, RoMa, LoFTR)
    # ========================================================================
    elif method in ['DKM', 'ROMA', 'LOFTR']:
        # Return adapter that makes dense matcher compatible with sparse pipeline
        return DenseMatcherAdapter(method=method, **kwargs)

    # ========================================================================
    # UNKNOWN METHOD
    # ========================================================================
    else:
        raise ValueError(
            f"Unknown method: '{method}'. Available methods:\n"
            f"  Traditional sparse: SIFT, ORB, AKAZE\n"
            f"  Deep learning sparse: SuperPoint, ALIKED, LightGlue\n"
            f"  Dense: DKM, ROMA, LOFTR"
        )
