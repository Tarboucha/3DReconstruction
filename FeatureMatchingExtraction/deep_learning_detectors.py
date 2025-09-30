"""
Deep learning-based feature detectors (SuperPoint, etc.).

This module contains implementations of neural network-based
feature detection algorithms.
"""

import cv2
import numpy as np
import time
import os
import urllib.request
from typing import List, Optional, Dict
from .base_classes import BaseFeatureDetector
from .core_data_structures import FeatureData

# Try to import PyTorch - gracefully handle if not available
try:
    import torch
    import torch.nn.functional as F
    from torchvision import transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Deep learning detectors will not work.")


class SuperPointDetector(BaseFeatureDetector):
    """SuperPoint deep learning feature detector"""
    
    def __init__(self, max_features: int = 2048, weights_path: Optional[str] = None,
                 nms_radius: int = 4, keypoint_threshold: float = 0.005):
        """
        Initialize SuperPoint detector
        
        Args:
            max_features: Maximum number of keypoints to extract
            weights_path: Path to pretrained weights (will download if None)
            nms_radius: Non-maximum suppression radius
            keypoint_threshold: Keypoint detection threshold
        """
        super().__init__(max_features)
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for SuperPoint. Install with: pip install torch torchvision")
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.nms_radius = nms_radius
        self.keypoint_threshold = keypoint_threshold
        self.model = None
        self.weights_path = weights_path
        self._load_model()
    
    def _load_model(self):
        """Load SuperPoint model"""
        try:
            # Try to import from LightGlue first
            try:
                from .LightGlue.lightglue import SuperPoint
                self.model = SuperPoint(
                    max_num_keypoints=self.max_features,
                    keypoint_threshold=self.keypoint_threshold,
                    nms_radius=self.nms_radius
                ).eval().to(self.device)
                print("SuperPoint loaded from LightGlue")
                return
            except ImportError:
                pass
            
            # Fallback to manual implementation
            if self.weights_path is None:
                self.weights_path = self._download_weights()
            
            # Note: This would require a custom SuperPoint implementation
            # For now, we'll use a placeholder that shows the structure
            print("SuperPoint model structure loaded (requires custom implementation)")
            
        except Exception as e:
            print(f"Failed to load SuperPoint: {e}")
            self.model = None
    
    def _download_weights(self) -> str:
        """Download SuperPoint pretrained weights"""
        weights_url = "https://github.com/magicleap/SuperPointPretrainedNetwork/raw/master/superpoint_v1.pth"
        weights_path = "superpoint_v1.pth"
        
        if not os.path.exists(weights_path):
            print("Downloading SuperPoint weights...")
            try:
                urllib.request.urlretrieve(weights_url, weights_path)
                print("SuperPoint weights downloaded successfully")
            except Exception as e:
                print(f"Failed to download weights: {e}")
                return None
        
        return weights_path
    
    def _image_to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """Convert image to tensor format for SuperPoint"""
        gray = self.preprocess_image(image)
        
        # Normalize to [0, 1] and add batch dimension
        tensor = torch.from_numpy(gray).float() / 255.0
        tensor = tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        return tensor.to(self.device)
    
    def detect(self, image: np.ndarray) -> FeatureData:
        start_time = time.time()
        
        if self.model is None:
            print("SuperPoint model not available")
            return FeatureData([], None, "SuperPoint", raw_image=image)
        
        try:
            # Convert image to tensor
            tensor = self._image_to_tensor(image)
            
            with torch.no_grad():
                # Extract features using SuperPoint
                result = self.model.extract(tensor)
            
            # Convert results to OpenCV format
            keypoints_np = result['keypoints'][0].cpu().numpy()
            descriptors_np = result['descriptors'][0].cpu().numpy().T
            scores = result['keypoint_scores'][0].cpu().numpy()
            
            # Convert to cv2.KeyPoint format
            keypoints = []
            for i, (kp, score) in enumerate(zip(keypoints_np, scores)):
                x, y = kp
                keypoint = cv2.KeyPoint(
                    x=float(x), y=float(y), size=8, response=float(score)
                )
                keypoints.append(keypoint)
            
            return FeatureData(
                keypoints=keypoints,
                descriptors=descriptors_np,
                method="SuperPoint",
                confidence_scores=scores.tolist(),
                detection_time=time.time() - start_time,
                raw_image=image
            )
            
        except Exception as e:
            print(f"SuperPoint detection failed: {e}")
            return FeatureData([], None, "SuperPoint", raw_image=image)


class DISKDetector(BaseFeatureDetector):
    """DISK (DIStilled feature description using Knowledge distillation) detector"""
    
    def __init__(self, max_features: int = 2048, weights_path: Optional[str] = None):
        """
        Initialize DISK detector
        
        Args:
            max_features: Maximum number of keypoints to extract
            weights_path: Path to pretrained weights
        """
        super().__init__(max_features)
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for DISK. Install with: pip install torch torchvision")
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.weights_path = weights_path
        self._load_model()
    
    def _load_model(self):
        """Load DISK model"""
        try:
            from .LightGlue.lightglue import DISK
            self.model = DISK(max_num_keypoints=self.max_features).eval().to(self.device)
            print("DISK loaded successfully")
        except ImportError:
            print("DISK not available. Install LightGlue: pip install lightglue")
            self.model = None
        except Exception as e:
            print(f"Failed to load DISK: {e}")
            self.model = None
    
    def detect(self, image: np.ndarray) -> FeatureData:
        start_time = time.time()
        
        if self.model is None:
            print("DISK model not available")
            return FeatureData([], None, "DISK", raw_image=image)
        
        try:
            # Convert image to tensor
            gray = self.preprocess_image(image)
            tensor = torch.from_numpy(gray).float() / 255.0
            tensor = tensor.unsqueeze(0).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                result = self.model.extract(tensor)
            
            # Convert to OpenCV format
            keypoints_np = result['keypoints'][0].cpu().numpy()
            descriptors_np = result['descriptors'][0].cpu().numpy().T
            scores = result['keypoint_scores'][0].cpu().numpy()
            
            keypoints = []
            for kp, score in zip(keypoints_np, scores):
                x, y = kp
                keypoint = cv2.KeyPoint(
                    x=float(x), y=float(y), size=8, response=float(score)
                )
                keypoints.append(keypoint)
            
            return FeatureData(
                keypoints=keypoints,
                descriptors=descriptors_np,
                method="DISK",
                confidence_scores=scores.tolist(),
                detection_time=time.time() - start_time,
                raw_image=image
            )
            
        except Exception as e:
            print(f"DISK detection failed: {e}")
            return FeatureData([], None, "DISK", raw_image=image)


class ALIKEDDetector(BaseFeatureDetector):
    """ALIKED (A Lightweight Keypoint and Descriptor Extractor) detector"""
    
    def __init__(self, max_features: int = 2048, model_name: str = 'aliked-n16'):
        """
        Initialize ALIKED detector
        
        Args:
            max_features: Maximum number of keypoints to extract
            model_name: Model variant ('aliked-n16', 'aliked-n32', etc.)
        """
        super().__init__(max_features)
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for ALIKED. Install with: pip install torch torchvision")
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load ALIKED model"""
        try:
            from .LightGlue.lightglue import ALIKED
            self.model = ALIKED(
                model_name=self.model_name,
                max_num_keypoints=self.max_features
            ).eval().to(self.device)
            print(f"ALIKED ({self.model_name}) loaded successfully")
        except ImportError:
            print("ALIKED not available. Install LightGlue: pip install lightglue")
            self.model = None
        except Exception as e:
            print(f"Failed to load ALIKED: {e}")
            self.model = None
    
    def detect(self, image: np.ndarray) -> FeatureData:
        start_time = time.time()
        
        if self.model is None:
            print("ALIKED model not available")
            return FeatureData([], None, "ALIKED", raw_image=image)
        
        try:
            # Convert image to tensor
            gray = self.preprocess_image(image)
            tensor = torch.from_numpy(gray).float() / 255.0
            tensor = tensor.unsqueeze(0).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                result = self.model.extract(tensor)
            
            # Convert to OpenCV format
            keypoints_np = result['keypoints'][0].cpu().numpy()
            descriptors_np = result['descriptors'][0].cpu().numpy().T
            scores = result['keypoint_scores'][0].cpu().numpy()
            
            keypoints = []
            for kp, score in zip(keypoints_np, scores):
                x, y = kp
                keypoint = cv2.KeyPoint(
                    x=float(x), y=float(y), size=8, response=float(score)
                )
                keypoints.append(keypoint)
            
            return FeatureData(
                keypoints=keypoints,
                descriptors=descriptors_np,
                method="ALIKED",
                confidence_scores=scores.tolist(),
                detection_time=time.time() - start_time,
                raw_image=image
            )
            
        except Exception as e:
            print(f"ALIKED detection failed: {e}")
            return FeatureData([], None, "ALIKED", raw_image=image)


# Factory function for deep learning detectors
def create_deep_learning_detector(detector_type: str, **kwargs) -> BaseFeatureDetector:
    """
    Factory function to create deep learning feature detectors
    
    Args:
        detector_type: Type of detector ('SuperPoint', 'DISK', 'ALIKED')
        **kwargs: Additional parameters for the detector
        
    Returns:
        Initialized detector instance
        
    Raises:
        ValueError: If detector_type is not supported
        ImportError: If required dependencies are not available
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for deep learning detectors. Install with: pip install torch torchvision")
    
    detector_map = {
        'SuperPoint': SuperPointDetector,
        'DISK': DISKDetector,
        'ALIKED': ALIKEDDetector
    }
    
    if detector_type not in detector_map:
        available = ', '.join(detector_map.keys())
        raise ValueError(f"Unknown detector type: {detector_type}. Available: {available}")
    
    return detector_map[detector_type](**kwargs)