# matchers/dense_matcher.py

"""
Dense matching methods: DKM, RoMa, LoFTR
These methods process every pixel and don't have separate detection/matching stages.
"""

import torch
import cv2
import numpy as np
import time
import os
from pathlib import Path
from typing import Optional, Tuple
from .base_matcher import BaseMatcher, MatchResult

# Optional dependency for downloading weights
try:
    import gdown
    HAS_GDOWN = True
except ImportError:
    HAS_GDOWN = False

# Model weights configuration
MODEL_WEIGHTS = {
    "gim_dkm_100h": {
        "filename": "gim_dkm_100h.ckpt",
        "gdrive_id": "1gk97V4IROnR1Nprq10W9NCFUv2mxXR_-",
        "url": "https://drive.google.com/uc?id=1gk97V4IROnR1Nprq10W9NCFUv2mxXR_-"
    },
    "gim_lightglue_100h": {
        "filename": "gim_lightglue_100h.ckpt",
        "url": "https://github.com/xuelunshen/gim/raw/main/weights/gim_lightglue_100h.ckpt"
    }
}


def download_weights(weight_name: str, output_dir: str = "./weights") -> str:
    """
    Download model weights from Google Drive or GitHub.

    Args:
        weight_name: Name from MODEL_WEIGHTS dict (e.g., 'gim_dkm_100h')
        output_dir: Directory to save weights

    Returns:
        Path to downloaded weights file
    """
    if weight_name not in MODEL_WEIGHTS:
        raise ValueError(f"Unknown weight: {weight_name}. Available: {list(MODEL_WEIGHTS.keys())}")

    config = MODEL_WEIGHTS[weight_name]
    output_path = Path(output_dir) / config["filename"]

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"  Downloading {config['filename']}...")

    if not HAS_GDOWN:
        raise RuntimeError(
            "gdown is required for downloading weights but not installed. "
            "Install it with: pip install gdown"
        )

    try:
        # Use gdown for Google Drive
        if "gdrive_id" in config:
            gdown.download(id=config["gdrive_id"], output=str(output_path), quiet=False)
        else:
            # Use gdown for other URLs too (it handles many sources)
            gdown.download(config["url"], str(output_path), quiet=False)

        print(f"  ✓ Downloaded to: {output_path}")
        return str(output_path)

    except Exception as e:
        raise RuntimeError(f"Failed to download weights: {e}")


def find_or_download_weights(weight_name: str = "gim_dkm_100h") -> str:
    """
    Find weights locally or download if not found.

    Searches in (in order):
    1. matchers/weights/ (relative to this file)
    2. ./weights/ (current working directory)
    3. ../weights/
    4. ~/.cache/gim/
    5. Current directory

    Args:
        weight_name: Name from MODEL_WEIGHTS dict

    Returns:
        Path to weights file
    """
    config = MODEL_WEIGHTS[weight_name]
    filename = config["filename"]

    # Get directory where this file (dense_matcher.py) is located
    this_dir = Path(__file__).parent.resolve()

    # Search paths (in order of preference)
    search_paths = [
        this_dir / "weights" / filename,              # matchers/weights/
        Path("./weights") / filename,                 # ./weights/ from CWD
        Path("../weights") / filename,                # ../weights/
        Path(os.path.expanduser("~/.cache/gim")) / filename,  # User cache
        Path(".") / filename,                         # Current directory
    ]

    # Check if file exists locally
    for path in search_paths:
        if path.exists():
            print(f"  Found weights at: {path}")
            return str(path)

    # Not found locally - download to matchers/weights/
    print(f"  Weights not found locally. Downloading...")
    weights_dir = this_dir / "weights"
    return download_weights(weight_name, output_dir=str(weights_dir))


class DenseMatcher(BaseMatcher):
    """
    Dense matching methods that process entire images.
    Supports: DKM, RoMa, LoFTR
    """
    
    def __init__(self,
                 method='DKM',
                 device='cuda',
                 max_matches=8000,
                 resize_to: Optional[Tuple[int, int]] = (896, 672),
                 weights='gim_dkm_100h',
                 auto_download=True,
                 symmetric=True):
        """
        Initialize dense matcher.

        Args:
            method: 'DKM', 'RoMa', or 'LoFTR'
            device: 'cuda' or 'cpu'
            max_matches: Maximum number of matches to sample
            resize_to: Target size (width, height) or None for native resolution
                      Default (896, 672) matches DKM training resolution
            weights: Weight name from MODEL_WEIGHTS or path to .ckpt file
            auto_download: Auto-download weights if not found locally
            symmetric: Use bidirectional matching (slower but more reliable, recommended for SFM)
        """
        self.method = method.upper()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.max_matches = max_matches
        self.resize_to = resize_to
        self.weights = weights
        self.auto_download = auto_download
        self.symmetric = symmetric

        print(f"Initializing {self.method} on {self.device}...")
        self._load_model()
        print(f"✓ {self.method} loaded successfully")
    
    def _load_model(self):
        """Load the appropriate dense matching model"""
        try:
            if self.method == 'DKM':
                from networks.dkm.models.model_zoo.DKMv3 import DKMv3

                # Resolve checkpoint path
                checkpoint_path = None
                if self.weights:
                    # Check if weights is a path to existing file
                    if os.path.exists(self.weights):
                        checkpoint_path = self.weights
                        print(f"  Using weights from: {checkpoint_path}")
                    # Check if it's a known weight name
                    elif self.weights in MODEL_WEIGHTS:
                        if self.auto_download:
                            checkpoint_path = find_or_download_weights(self.weights)
                        else:
                            # Try to find locally only (no auto-download)
                            config = MODEL_WEIGHTS[self.weights]
                            filename = config["filename"]

                            # Get directory where this file is located
                            this_dir = Path(__file__).parent.resolve()

                            search_paths = [
                                this_dir / "weights" / filename,     # matchers/weights/
                                Path("./weights") / filename,        # ./weights/
                                Path(".") / filename,                # current dir
                            ]

                            for path in search_paths:
                                if path.exists():
                                    checkpoint_path = str(path)
                                    print(f"  Found weights at: {path}")
                                    break

                            if not checkpoint_path:
                                print(f"  ⚠️  Weights not found. Set auto_download=True to download automatically.")
                                print(f"      Or place {filename} in: {this_dir / 'weights'}")
                    else:
                        print(f"  ⚠️  Unknown weight name: {self.weights}")

                # DKMv3 signature: DKMv3(weights, h, w, symmetric=True, sample_mode="threshold_balanced")
                self.model = DKMv3(
                    weights=None,  # Will load manually from checkpoint
                    h=self.resize_to[1] if self.resize_to else 480,
                    w=self.resize_to[0] if self.resize_to else 640,
                    symmetric=self.symmetric,  # Use bidirectional matching for more reliable correspondences
                    sample_mode="threshold_balanced"
                )

                # Load checkpoint if available
                if checkpoint_path:
                    print(f"  Loading checkpoint...")
                    checkpoint = torch.load(checkpoint_path, map_location=self.device)

                    # Handle different checkpoint formats
                    if 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                    elif 'model' in checkpoint:
                        state_dict = checkpoint['model']
                    else:
                        state_dict = checkpoint

                    self.model.load_state_dict(state_dict, strict=False)
                    print(f"  ✓ Checkpoint loaded successfully")
                else:
                    print(f"  ⚠️  No checkpoint loaded. Using random initialization.")

                self.model = self.model.eval().to(self.device)
            # elif self.method == 'ROMA':
            #     from gim.roma import RoMa
            #     self.model = RoMa(weights=self.weights, device=self.device)
                
            # elif self.method == 'LOFTR':
            #     from gim.loftr import LoFTR
            #     self.model = LoFTR(weights=self.weights, device=self.device)
                
            else:
                raise ValueError(f"Unknown dense method: {self.method}")
                
        except ImportError as e:
            raise ImportError(
                f"Failed to import {self.method}. "
                f"Install GIM with: pip install gim-dkm\n"
                f"Or from source: git clone https://github.com/xuelunshen/gim.git"
            ) from e
    
    @property
    def name(self) -> str:
        return self.method
    
    def match(self, img1: np.ndarray, img2: np.ndarray) -> MatchResult:
        """
        Match two images using dense method.
        
        Args:
            img1: First image (H, W, 3) BGR
            img2: Second image (H, W, 3) BGR
            
        Returns:
            MatchResult with matched keypoints
        """
        start_time = time.time()
        
        # Store original shapes
        original_shape1 = img1.shape[:2]
        original_shape2 = img2.shape[:2]
        
        # Resize if configured
        if self.resize_to is not None:
            img1_resized, scale1 = self._resize_image(img1, self.resize_to)
            img2_resized, scale2 = self._resize_image(img2, self.resize_to)
        else:
            img1_resized = img1
            img2_resized = img2
            scale1 = scale2 = 1.0
        
        # Convert to tensors
        img1_tensor = self._to_tensor(img1_resized)
        img2_tensor = self._to_tensor(img2_resized)
        
        # Match using appropriate method
        with torch.no_grad():
            if self.method in ['DKM', 'ROMA']:
                kpts1, kpts2, confidence = self._match_dkm_roma(
                    img1_tensor, img2_tensor, 
                    img1_resized.shape[:2], img2_resized.shape[:2]
                )
            elif self.method == 'LOFTR':
                kpts1, kpts2, confidence = self._match_loftr(
                    img1_tensor, img2_tensor
                )
        
        # Scale keypoints back to original resolution
        kpts1 = kpts1 / scale1
        kpts2 = kpts2 / scale2
        
        # Filter by confidence
        mask = confidence > 0
        kpts1 = kpts1[mask]
        kpts2 = kpts2[mask]
        confidence = confidence[mask]
        
        elapsed = time.time() - start_time
        
        return MatchResult(
            keypoints1=kpts1,
            keypoints2=kpts2,
            confidence=confidence,
            method=self.method,
            time_seconds=elapsed,
            num_features1=len(kpts1),  # Dense methods don't separate detection
            num_features2=len(kpts2),
            metadata={
                'resize_to': self.resize_to,
                'scale1': scale1,
                'scale2': scale2,
                'original_shape1': original_shape1,
                'original_shape2': original_shape2
            }
        )
    
    def _match_dkm_roma(self, 
                        img1_tensor: torch.Tensor, 
                        img2_tensor: torch.Tensor,
                        shape1: Tuple[int, int],
                        shape2: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Match using DKM or RoMa.
        
        Returns:
            keypoints1 (N, 2), keypoints2 (N, 2), confidence (N,)
        """
        # Get dense matches and certainty
        dense_matches, dense_certainty = self.model.match(img1_tensor, img2_tensor)
        
        # Sample sparse matches
        sparse_matches, confidence = self.model.sample(
            dense_matches, 
            dense_certainty, 
            self.max_matches
        )
        
        # Convert from normalized coordinates [-1, 1] to pixel coordinates
        H1, W1 = shape1
        H2, W2 = shape2
        
        # Extract normalized coordinates
        kpts1_norm = sparse_matches[:, :2]  # (N, 2) - first 2 dims
        kpts2_norm = sparse_matches[:, 2:]  # (N, 2) - last 2 dims
        
        # Convert to pixel coordinates
        # Formula: pixel = (normalized + 1) * size / 2
        kpts1 = torch.stack([
            W1 * (kpts1_norm[:, 0] + 1) / 2,  # x coordinate
            H1 * (kpts1_norm[:, 1] + 1) / 2   # y coordinate
        ], dim=-1)
        
        kpts2 = torch.stack([
            W2 * (kpts2_norm[:, 0] + 1) / 2,
            H2 * (kpts2_norm[:, 1] + 1) / 2
        ], dim=-1)
        
        # Convert to numpy
        kpts1 = kpts1.cpu().numpy()
        kpts2 = kpts2.cpu().numpy()
        confidence = confidence.cpu().numpy()
        
        return kpts1, kpts2, confidence
    
    def _match_loftr(self,
                     img1_tensor: torch.Tensor,
                     img2_tensor: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Match using LoFTR.
        
        Returns:
            keypoints1 (N, 2), keypoints2 (N, 2), confidence (N,)
        """
        # LoFTR expects a batch dictionary
        batch = {
            'image0': img1_tensor,
            'image1': img2_tensor
        }
        
        # Run LoFTR
        self.model(batch)
        
        # Extract results
        kpts1 = batch['mkpts0_f'].cpu().numpy()  # (N, 2)
        kpts2 = batch['mkpts1_f'].cpu().numpy()  # (N, 2)
        confidence = batch['mconf'].cpu().numpy()  # (N,)
        
        # Limit to max_matches if needed
        if len(kpts1) > self.max_matches:
            # Keep highest confidence matches
            top_indices = np.argsort(confidence)[-self.max_matches:]
            kpts1 = kpts1[top_indices]
            kpts2 = kpts2[top_indices]
            confidence = confidence[top_indices]
        
        return kpts1, kpts2, confidence
    
    def _to_tensor(self, img: np.ndarray) -> torch.Tensor:
        """
        Convert numpy image to tensor.
        
        Args:
            img: BGR image (H, W, 3)
            
        Returns:
            Tensor (1, 3, H, W) normalized to [0, 1]
        """
        # Convert BGR to RGB
        if len(img.shape) == 3 and img.shape[2] == 3:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = img
        
        # Normalize to [0, 1]
        img_float = img_rgb.astype(np.float32) / 255.0
        
        # Convert to tensor: (H, W, C) -> (1, C, H, W)
        img_tensor = torch.from_numpy(img_float).permute(2, 0, 1).unsqueeze(0)
        
        return img_tensor.to(self.device)
    
    def _resize_image(self, 
                     img: np.ndarray, 
                     target_size: Tuple[int, int]) -> Tuple[np.ndarray, float]:
        """
        Resize image preserving aspect ratio.
        
        Args:
            img: Input image (H, W, 3)
            target_size: (width, height)
            
        Returns:
            Resized image and scale factor
        """
        h, w = img.shape[:2]
        target_w, target_h = target_size
        
        # Calculate scale to fit within target size
        scale = min(target_w / w, target_h / h)
        
        # New size (ensure divisible by 8 for CNNs)
        new_w = int(w * scale)
        new_h = int(h * scale)
        new_w = (new_w // 8) * 8
        new_h = (new_h // 8) * 8
        
        # Resize
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Pad to exact target size if needed
        if (new_h, new_w) != (target_h, target_w):
            padded = np.zeros((target_h, target_w, 3), dtype=img.dtype)
            padded[:new_h, :new_w] = resized
            return padded, scale
        
        return resized, scale


# Convenience function
def create_dense_matcher(method: str, **kwargs) -> DenseMatcher:
    """
    Create a dense matcher with automatic weight management.

    Args:
        method: 'DKM', 'RoMa', or 'LoFTR'
        **kwargs: Additional parameters for DenseMatcher
            - weights: 'gim_dkm_100h' (default) or path to .ckpt file
            - auto_download: True (default) to auto-download weights
            - device: 'cuda' or 'cpu'
            - max_matches: Maximum number of matches (default: 8000)
            - resize_to: (width, height) tuple or None (default: (896, 672) for DKM training resolution)
            - symmetric: True (default) for bidirectional matching (recommended for SFM)

    Returns:
        Configured DenseMatcher

    Examples:
        >>> # Default: uses training resolution (896x672), symmetric matching
        >>> matcher = create_dense_matcher('DKM', device='cuda')
        >>> result = matcher.match(img1, img2)

        >>> # Memory-constrained: disable symmetric to halve VRAM usage
        >>> matcher = create_dense_matcher('DKM', symmetric=False)

        >>> # Custom resolution (may reduce accuracy)
        >>> matcher = create_dense_matcher('DKM', resize_to=(640, 480))

        >>> # Use local weights
        >>> matcher = create_dense_matcher('DKM', weights='./my_weights.ckpt')

    Note:
        Requires: pip install gdown

        Default resolution (896×672) matches DKM training and requires ~5-8GB VRAM
        with symmetric=True. For limited VRAM, use symmetric=False (~2.5-4GB).

        For 3D reconstruction, symmetric=True (default) is recommended for more
        reliable correspondences, though it's ~2x slower than symmetric=False.
    """
    return DenseMatcher(method=method, **kwargs)