import torch
import cv2
import numpy as np
from pathlib import Path
from .base_detector import BaseDetector, DetectionResult


class SuperPointDetector(BaseDetector):
    def __init__(self, max_keypoints=2048, device='cuda'):
        from lightglue import SuperPoint
        self.detector = SuperPoint(
            max_num_keypoints=max_keypoints,
            detection_threshold=0.005
        ).eval().to(device)
        self.device = device
    
    @property
    def name(self):
        return "SuperPoint"
    
    @property
    def descriptor_type(self):
        return "float32"
    
    def detect(self, image: np.ndarray) -> DetectionResult:
        # Convert to grayscale tensor
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_float = gray.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(gray_float).unsqueeze(0).unsqueeze(0)
        img_tensor = img_tensor.to(self.device)
        
        with torch.no_grad():
            pred = self.detector.extract(img_tensor)
        
        keypoints = pred['keypoints'][0].cpu().numpy()  # (N, 2)
        descriptors = pred['descriptors'][0].T.cpu().numpy()  # (N, 256)
        scores = pred['keypoint_scores'][0].cpu().numpy()  # (N,)
        
        return DetectionResult(
            keypoints=keypoints,
            descriptors=descriptors,
            scores=scores,
            descriptor_type="float32",
            descriptor_dim=256
        )


class ALIKEDDetector(BaseDetector):
    def __init__(self, max_keypoints=4096, device='cuda', weights_path=None):
        import torch
        from lightglue.aliked import ALIKED as ALIKEDModel

        self.detector = ALIKEDModel(
            c1=32, c2=64, c3=128, c4=128,
            dim=128,
            K=max_keypoints,
            detection_threshold=0.2
        )

        # Use provided path or default to matchers/weights/aliked-n16.pth
        if weights_path is None:
            # Get path relative to this file: matchers/detectors/deep_learning.py
            # Go up to matchers/ then into weights/
            weights_dir = Path(__file__).parent.parent / 'weights'
            weights_path = weights_dir / 'aliked-n16.pth'
        else:
            weights_path = Path(weights_path)

        # Check if weights exist
        if not weights_path.exists():
            raise FileNotFoundError(
                f"ALIKED weights not found at: {weights_path}\n"
                f"Please download from: https://github.com/Shiaoming/ALIKED\n"
                f"Or specify custom path with weights_path parameter"
            )

        state_dict = torch.load(str(weights_path), map_location=device)
        self.detector.load_state_dict(state_dict['model'])
        self.detector = self.detector.eval().to(device)
        self.device = device
    
    @property
    def name(self):
        return "ALIKED"
    
    @property
    def descriptor_type(self):
        return "float32"
    
    def detect(self, image: np.ndarray) -> DetectionResult:
        import torch
        
        # Convert to grayscale tensor
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_float = gray.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(gray_float).unsqueeze(0).unsqueeze(0)
        img_tensor = img_tensor.to(self.device)
        
        with torch.no_grad():
            pred = self.detector.run(img_tensor)
        
        keypoints = pred['keypoints'][0].cpu().numpy()  # (N, 2)
        descriptors = pred['descriptors'][0].cpu().numpy()  # (N, 128)
        scores = pred['scores'][0].cpu().numpy()  # (N,)
        
        return DetectionResult(
            keypoints=keypoints,
            descriptors=descriptors,
            scores=scores,
            descriptor_type="float32",
            descriptor_dim=128
        )