"""
Neural Depth Estimation
=======================

Monocular depth estimation using pretrained neural networks.
Fallback method when COLMAP/OpenMVS are not available.

Supports:
- ZoeDepth (metric depth)
- MiDaS (relative depth)
- DPT (Dense Prediction Transformer)
"""

import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional
import cv2

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class NeuralDepthEstimation:
    """
    Neural monocular depth estimation.

    Uses pretrained models for per-image depth estimation.
    Useful as a fallback when stereo matching is not available.
    """

    def __init__(self, config, model_name: str = 'zoedepth'):
        """
        Initialize neural depth estimator.

        Args:
            config: DenseReconstructionConfig
            model_name: Model to use ('zoedepepth', 'midas', 'dpt')
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for neural depth estimation")

        self.config = config
        self.model_name = model_name
        self.logger = logging.getLogger("DenseReconstruction.Neural")

        self.model = None
        self.transform = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.logger.info(f"Initializing {model_name} on {self.device}")

    def _load_model(self):
        """Lazy load model"""
        if self.model is not None:
            return

        try:
            if self.model_name == 'zoedepth':
                self._load_zoedepth()
            elif self.model_name == 'midas':
                self._load_midas()
            elif self.model_name == 'dpt':
                self._load_dpt()
            else:
                raise ValueError(f"Unknown model: {self.model_name}")

            self.logger.info(f"✓ {self.model_name} loaded")

        except Exception as e:
            self.logger.error(f"Failed to load {self.model_name}: {e}")
            raise

    def _load_zoedepth(self):
        """Load ZoeDepth model"""
        try:
            # Try to import ZoeDepth
            repo = "isl-org/ZoeDepth"
            self.model = torch.hub.load(repo, "ZoeD_NK", pretrained=True)
            self.model.to(self.device)
            self.model.eval()
        except:
            self.logger.warning("ZoeDepth not available, falling back to MiDaS")
            self._load_midas()

    def _load_midas(self):
        """Load MiDaS model"""
        # Use torch hub to load MiDaS
        self.model = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
        self.model.to(self.device)
        self.model.eval()

        # Load transforms
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = midas_transforms.dpt_transform

    def _load_dpt(self):
        """Load DPT model (via MiDaS)"""
        self._load_midas()  # DPT is part of MiDaS

    def compute_depth_maps(self, workspace: Dict, output_dir: Path) -> List[Dict]:
        """
        Compute depth maps for all images using neural estimation.

        Args:
            workspace: MVS workspace with cameras and image paths
            output_dir: Output directory

        Returns:
            List of depth map dictionaries
        """
        self.logger.info(f"Computing neural depth maps with {self.model_name}...")

        # Load model
        self._load_model()

        cameras = workspace['cameras']
        image_folder = workspace['image_folder']

        depth_maps = []

        for i, (image_name, camera_data) in enumerate(cameras.items()):
            self.logger.info(f"Processing {i+1}/{len(cameras)}: {image_name}")

            try:
                # Load image
                image_path = Path(image_folder) / image_name
                image = cv2.imread(str(image_path))

                if image is None:
                    self.logger.warning(f"Failed to load image: {image_path}")
                    continue

                # Convert BGR to RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Estimate depth
                depth_map = self._estimate_depth(image_rgb)

                # Scale depth using camera intrinsics (if available)
                if 'K' in camera_data:
                    depth_map = self._scale_depth_with_intrinsics(
                        depth_map, camera_data['K']
                    )

                # Save depth map
                depth_file = output_dir / f"depth_{Path(image_name).stem}.npy"
                np.save(depth_file, depth_map)

                depth_maps.append({
                    'image_name': image_name,
                    'depth_map': depth_map,
                    'camera': camera_data,
                    'depth_file': str(depth_file)
                })

            except Exception as e:
                self.logger.error(f"Failed to process {image_name}: {e}")
                continue

        self.logger.info(f"✓ Computed {len(depth_maps)} neural depth maps")
        return depth_maps

    def _estimate_depth(self, image_rgb: np.ndarray) -> np.ndarray:
        """
        Estimate depth for a single image.

        Args:
            image_rgb: RGB image (H, W, 3)

        Returns:
            Depth map (H, W)
        """
        with torch.no_grad():
            if self.model_name == 'zoedepth':
                # ZoeDepth expects PIL Image or numpy array
                depth = self.model.infer_pil(image_rgb)
            else:
                # MiDaS/DPT
                # Transform image
                input_batch = self.transform(image_rgb).to(self.device)

                # Predict
                prediction = self.model(input_batch)

                # Resize to original resolution
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=image_rgb.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()

                depth = prediction.cpu().numpy()

        return depth

    def _scale_depth_with_intrinsics(self, depth_map: np.ndarray,
                                    K: np.ndarray) -> np.ndarray:
        """
        Scale relative depth to approximate metric depth using focal length.

        This is a heuristic - neural models typically output relative depth.
        """
        # Use focal length as a scaling hint
        focal_length = (K[0, 0] + K[1, 1]) / 2

        # Heuristic: assume scene is at ~5 meters average depth
        # Scale so median depth is ~5m
        median_depth = np.median(depth_map[depth_map > 0])
        if median_depth > 0:
            scale = 5.0 / median_depth
            depth_map = depth_map * scale

        return depth_map
