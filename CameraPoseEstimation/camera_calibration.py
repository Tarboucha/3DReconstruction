"""
Camera Calibration and Intrinsic Parameter Estimation
=====================================================

This module handles camera calibration and estimation of intrinsic parameters
when calibration data is not available.
"""

import cv2
import numpy as np
import json
from typing import Tuple, Dict, List, Optional
from pathlib import Path

class CameraCalibration:
    """Camera calibration utilities"""
    
    def __init__(self):
        """Initialize camera calibration"""
        self.camera_matrix = None
        self.dist_coeffs = None
        self.calibration_error = None
        self.calibration_method = None
    
    def calibrate_from_checkerboard(self, image_paths: List[str],
                                   checkerboard_size: Tuple[int, int],
                                   square_size: float = 1.0) -> Dict:
        """
        Calibrate camera using checkerboard patterns
        
        Args:
            image_paths: List of calibration image paths
            checkerboard_size: (corners_per_row, corners_per_col)
            square_size: Size of checkerboard squares in world units
            
        Returns:
            Calibration results
        """
        print(f"Calibrating camera using {len(image_paths)} checkerboard images...")
        
        # Prepare object points
        objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
        objp *= square_size
        
        # Arrays to store object points and image points
        objpoints = []  # 3D points in real world space
        imgpoints = []  # 2D points in image plane
        
        successful_images = 0
        image_size = None
        
        for img_path in image_paths:
            # Load image
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not load image {img_path}")
                continue
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            if image_size is None:
                image_size = gray.shape[::-1]  # (width, height)
            
