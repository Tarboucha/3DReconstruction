# matchers/lightglue_matcher.py

import torch
import numpy as np
from dataclasses import dataclass


@dataclass  
class MatchingResult:
    """Result from matching"""
    matches_idx: np.ndarray
    confidence: np.ndarray
    num_raw_matches: int


class LightGlueMatcher:
    """
    LightGlue attention-based matcher.
    Only works with SuperPoint features!
    """
    
    def __init__(self, device='cuda', filter_threshold=0.1):
        from lightglue import LightGlue
        
        self.matcher = LightGlue(
            features='superpoint',
            filter_threshold=filter_threshold
        ).eval().to(device)
        self.device = device
    
    @property
    def name(self):
        return "LightGlue"
    
    def match(self, 
              keypoints1: np.ndarray, 
              descriptors1: np.ndarray,
              keypoints2: np.ndarray,
              descriptors2: np.ndarray) -> MatchingResult:
        """
        Match using LightGlue attention.
        
        Args:
            keypoints1: (N1, 2) keypoint coordinates
            descriptors1: (N1, 256) SuperPoint descriptors
            keypoints2: (N2, 2)
            descriptors2: (N2, 256)
        """
        # Convert to tensors
        kpts1_t = torch.from_numpy(keypoints1).unsqueeze(0).to(self.device)
        desc1_t = torch.from_numpy(descriptors1).T.unsqueeze(0).to(self.device)
        kpts2_t = torch.from_numpy(keypoints2).unsqueeze(0).to(self.device)
        desc2_t = torch.from_numpy(descriptors2).T.unsqueeze(0).to(self.device)
        
        # Prepare input format for LightGlue
        feats0 = {
            'keypoints': kpts1_t,
            'descriptors': desc1_t
        }
        feats1 = {
            'keypoints': kpts2_t,
            'descriptors': desc2_t
        }
        
        with torch.no_grad():
            matches01 = self.matcher({'image0': feats0, 'image1': feats1})
        
        # Extract matches
        matches_idx_t = matches01['matches0'][0]
        valid = matches_idx_t > -1
        
        # Build match pairs
        idx0 = torch.where(valid)[0].cpu().numpy()
        idx1 = matches_idx_t[valid].cpu().numpy()
        confidence = matches01['scores'][0][valid].cpu().numpy()
        
        matches_idx = np.stack([idx0, idx1], axis=1)
        
        return MatchingResult(
            matches_idx=matches_idx,
            confidence=confidence,
            num_raw_matches=len(keypoints1)
        )