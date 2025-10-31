
from abc import ABC, abstractmethod
import numpy as np
from dataclasses import dataclass


@dataclass
class DetectionResult:
    """Result from feature detection"""
    keypoints: np.ndarray      # (N, 2) - x, y coordinates
    descriptors: np.ndarray    # (N, D) - feature descriptors
    scores: np.ndarray         # (N,) - detection confidence (optional)
    descriptor_type: str       # 'float32' or 'uint8' (binary)
    descriptor_dim: int        # Descriptor dimension


class BaseDetector(ABC):
    """Base interface for all detectors"""
    
    @abstractmethod
    def detect(self, image: np.ndarray) -> DetectionResult:
        """Detect features in image"""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Detector name"""
        pass
    
    @property
    @abstractmethod
    def descriptor_type(self) -> str:
        """Descriptor type: 'float32' or 'uint8'"""
        pass