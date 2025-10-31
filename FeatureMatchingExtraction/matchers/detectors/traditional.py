# matchers/detectors/traditional.py

import cv2
import numpy as np
from .base_detector import BaseDetector, DetectionResult


class SIFTDetector(BaseDetector):
    def __init__(self, max_features=8000):
        self.detector = cv2.SIFT_create(nfeatures=max_features)
    
    @property
    def name(self):
        return "SIFT"
    
    @property
    def descriptor_type(self):
        return "float32"
    
    def detect(self, image: np.ndarray) -> DetectionResult:
        kp, desc = self.detector.detectAndCompute(image, None)
        
        keypoints = np.array([k.pt for k in kp], dtype=np.float32)
        scores = np.array([k.response for k in kp], dtype=np.float32)
        
        return DetectionResult(
            keypoints=keypoints,
            descriptors=desc,
            scores=scores,
            descriptor_type="float32",
            descriptor_dim=128
        )


class ORBDetector(BaseDetector):
    def __init__(self, max_features=8000):
        self.detector = cv2.ORB_create(nfeatures=max_features)
    
    @property
    def name(self):
        return "ORB"
    
    @property
    def descriptor_type(self):
        return "uint8"
    
    def detect(self, image: np.ndarray) -> DetectionResult:
        kp, desc = self.detector.detectAndCompute(image, None)
        
        keypoints = np.array([k.pt for k in kp], dtype=np.float32)
        scores = np.array([k.response for k in kp], dtype=np.float32)
        
        return DetectionResult(
            keypoints=keypoints,
            descriptors=desc,
            scores=scores,
            descriptor_type="uint8",
            descriptor_dim=32  # 256 bits = 32 bytes
        )


class AKAZEDetector(BaseDetector):
    def __init__(self):
        self.detector = cv2.AKAZE_create()
    
    @property
    def name(self):
        return "AKAZE"
    
    @property
    def descriptor_type(self):
        return "uint8"
    
    def detect(self, image: np.ndarray) -> DetectionResult:
        kp, desc = self.detector.detectAndCompute(image, None)
        
        keypoints = np.array([k.pt for k in kp], dtype=np.float32)
        scores = np.array([k.response for k in kp], dtype=np.float32)
        
        return DetectionResult(
            keypoints=keypoints,
            descriptors=desc,
            scores=scores,
            descriptor_type="uint8",
            descriptor_dim=61
        )