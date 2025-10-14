"""
Result Converters - Transform results for specific use cases

Converts MatchingResult to:
- VisualizationData: For plotting and display
- MultiMethodReconstruction: For 3D reconstruction

Version: 3.0.0 (Complete rewrite)
"""

import cv2
import numpy as np
import pickle
import json
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from pathlib import Path

from .core_data_structures import ScoreType
from .result_types import MatchingResult, MethodResult, ImagePairInfo


# =============================================================================
# VISUALIZATION DATA
# =============================================================================

@dataclass(frozen=True)
class VisualMatch:
    """Single match for visualization"""
    query_idx: int  # Index in unified keypoints1
    train_idx: int  # Index in unified keypoints2
    score: float
    score_type: ScoreType
    color: Tuple[int, int, int]  # RGB
    source_method: str
    
    def to_cv2_dmatch(self) -> cv2.DMatch:
        """Convert to cv2.DMatch"""
        m = cv2.DMatch()
        m.queryIdx = self.query_idx
        m.trainIdx = self.train_idx
        m.distance = self.score if self.score_type == ScoreType.DISTANCE else (1.0 - self.score)
        return m


@dataclass(frozen=True)
class VisualizationData:
    """
    Visualization data with unified keypoint lists
    
    All matches reference the same unified keypoint lists.
    No offset management needed by users!
    
    Attributes:
        keypoints1: All keypoints from image 1
        keypoints2: All keypoints from image 2  
        matches: All matches (indices adjusted for unified lists)
        method_info: Which methods contributed which matches
        image_pair_info: Image metadata
    """
    # Unified keypoint lists
    keypoints1: Tuple[cv2.KeyPoint, ...]
    keypoints2: Tuple[cv2.KeyPoint, ...]
    
    # All matches (with adjusted indices)
    matches: Tuple[VisualMatch, ...]
    
    # Method tracking
    method_info: Dict[str, Dict[str, Any]]
    
    # Image info
    image_pair_info: ImagePairInfo
    
    # Display metadata
    title: str = "Feature Matches"
    subtitle: str = ""
    
    def __getstate__(self):
        """Custom pickle serialization for frozen dataclass"""
        from .utils import keypoints_to_list
        
        # Convert KeyPoints to lists
        state = {
            'keypoints1': keypoints_to_list(self.keypoints1),
            'keypoints2': keypoints_to_list(self.keypoints2),
            'matches': self.matches,  # VisualMatch is pickleable
            'method_info': self.method_info,
            'image_pair_info': self.image_pair_info,
            'title': self.title,
            'subtitle': self.subtitle,
        }
        return state
    
    def __setstate__(self, state):
        """Custom pickle deserialization for frozen dataclass"""
        from .utils import list_to_keypoints
        
        # Convert lists back to KeyPoints
        # For frozen dataclass, we need to use object.__setattr__
        object.__setattr__(self, 'keypoints1', tuple(list_to_keypoints(state['keypoints1'])))
        object.__setattr__(self, 'keypoints2', tuple(list_to_keypoints(state['keypoints2'])))
        object.__setattr__(self, 'matches', state['matches'])
        object.__setattr__(self, 'method_info', state['method_info'])
        object.__setattr__(self, 'image_pair_info', state['image_pair_info'])
        object.__setattr__(self, 'title', state['title'])
        object.__setattr__(self, 'subtitle', state['subtitle'])


    def get_methods(self) -> List[str]:
        """Get list of methods"""
        return list(self.method_info.keys())
    
    def get_matches_for_method(self, method_name: str) -> List[VisualMatch]:
        """Get matches from specific method"""
        return [m for m in self.matches if m.source_method == method_name]
    
    def plot(self, max_matches: Optional[int] = None, figsize: Tuple[int, int] = (15, 10)):
        """
        Plot all matches
        
        Args:
            max_matches: Limit number of matches to display
            figsize: Figure size
        """
        import matplotlib.pyplot as plt
        
        if not self.image_pair_info.image1 is not None or self.image_pair_info.image2 is not None:
            print("❌ Images not available for visualization")
            return
        
        img1 = self.image_pair_info.image1
        img2 = self.image_pair_info.image2
        
        # Create side-by-side image
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        h = max(h1, h2)
        combined = np.zeros((h, w1 + w2, 3), dtype=np.uint8)
        combined[:h1, :w1] = img1
        combined[:h2, w1:w1+w2] = img2
        
        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(combined)
        ax.axis('off')
        ax.set_title(f"{self.title}\n{self.subtitle}")
        
        # Draw matches
        matches_to_draw = self.matches[:max_matches] if max_matches else self.matches
        
        for match in matches_to_draw:
            kp1 = self.keypoints1[match.query_idx]
            kp2 = self.keypoints2[match.train_idx]
            
            pt1 = (int(kp1.pt[0]), int(kp1.pt[1]))
            pt2 = (int(kp2.pt[0] + w1), int(kp2.pt[1]))
            
            color = tuple(c/255.0 for c in match.color)  # Normalize for matplotlib
            ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 
                   color=color, linewidth=0.5, alpha=0.6)
        
        # Draw keypoints
        for kp in self.keypoints1:
            ax.plot(kp.pt[0], kp.pt[1], 'o', color='cyan', markersize=2, alpha=0.5)
        for kp in self.keypoints2:
            ax.plot(kp.pt[0] + w1, kp.pt[1], 'o', color='cyan', markersize=2, alpha=0.5)
        
        plt.tight_layout()
        plt.show()
    
    def plot_method(self, method_name: str, max_matches: Optional[int] = None):
        """Plot matches from specific method"""
        matches = self.get_matches_for_method(method_name)
        
        if not matches:
            print(f"❌ No matches found for method: {method_name}")
            return
        
        # Create temporary VisualizationData with only this method's matches
        temp_viz = VisualizationData(
            keypoints1=self.keypoints1,
            keypoints2=self.keypoints2,
            matches=tuple(matches),
            method_info={method_name: self.method_info[method_name]},
            image_pair_info=self.image_pair_info,
            title=f"{method_name} Matches",
            subtitle=f"{len(matches)} matches"
        )
        temp_viz.plot(max_matches=max_matches)
    
    def save(self, filepath: Union[str, Path], format: str = 'pickle'):
        """Save result"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'pickle':
            with open(filepath, 'wb') as f:
                pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)  # Now works!
        else:
            raise ValueError(f"Unknown format: {format}")
        
        print(f"✅ Saved MatchingResult to {filepath}")
    
    @staticmethod
    def load(filepath: Union[str, Path]) -> 'VisualizationData':
        """Load visualization data"""
        with open(filepath, 'rb') as f:
            viz = pickle.load(f)
        print(f"✅ Loaded VisualizationData from {filepath}")
        return viz


# =============================================================================
# RECONSTRUCTION DATA
# =============================================================================

@dataclass(frozen=True)
class MethodReconstructionData:
    """Reconstruction data for a single method"""
    method_name: str
    
    # Keypoints and descriptors
    keypoints1: Tuple[cv2.KeyPoint, ...]
    keypoints2: Tuple[cv2.KeyPoint, ...]
    descriptors1: Optional[np.ndarray]
    descriptors2: Optional[np.ndarray]
    
    # Matches
    query_indices: np.ndarray
    train_indices: np.ndarray
    match_scores: np.ndarray
    score_type: ScoreType
    
    # Geometry
    homography: Optional[np.ndarray] = None
    fundamental_matrix: Optional[np.ndarray] = None
    essential_matrix: Optional[np.ndarray] = None
    inlier_mask: Optional[np.ndarray] = None
    inlier_ratio: Optional[float] = None
    reprojection_error: Optional[float] = None
    
    # Correspondences (Nx4: x1, y1, x2, y2)
    correspondences: Optional[np.ndarray] = None
    
    @property
    def num_matches(self) -> int:
        return len(self.query_indices)
    
    @property
    def num_inliers(self) -> int:
        if self.inlier_mask is not None:
            return int(np.sum(self.inlier_mask))
        return 0
    
    def get_quality_score(self) -> float:
        """Quality score for ranking"""
        score = 0.0
        if self.num_matches > 0:
            score += min(self.num_matches / 1000.0, 1.0) * 0.3
        if self.inlier_ratio is not None:
            score += self.inlier_ratio * 0.4
        if self.reprojection_error is not None:
            score += max(0, 1.0 - self.reprojection_error / 10.0) * 0.3
        return score
    
    def get_inlier_correspondences(self) -> np.ndarray:
        """Get correspondences for inliers only"""
        if self.correspondences is None or self.inlier_mask is None:
            return self.correspondences
        return self.correspondences[self.inlier_mask.ravel()]
    
    def export_to_colmap(self) -> Dict[str, Any]:
        """Export COLMAP-compatible format"""
        return {
            'keypoints1': [(kp.pt[0], kp.pt[1]) for kp in self.keypoints1],
            'keypoints2': [(kp.pt[0], kp.pt[1]) for kp in self.keypoints2],
            'matches': list(zip(self.query_indices.tolist(), self.train_indices.tolist())),
            'descriptors1': self.descriptors1,
            'descriptors2': self.descriptors2
        }


class MultiMethodReconstruction:
    """
    Multi-method reconstruction data
    
    All methods treated equally. Dictionary-like interface.
    
    Usage:
        recon['SIFT']  # Get method data
        recon.get_best()  # Get best method
        recon.export_method('SIFT', 'output/')
        recon.export_all_methods('output/')
    """
    
    def __init__(self,
                 methods: Dict[str, MethodReconstructionData],
                 image_pair_info: ImagePairInfo,
                 K1: Optional[np.ndarray] = None,
                 K2: Optional[np.ndarray] = None):
        self.methods = methods
        self.image_pair_info = image_pair_info
        self.K1 = K1
        self.K2 = K2
    
    # -------------------------------------------------------------------------
    # Dictionary Interface
    # -------------------------------------------------------------------------
    
    def __getstate__(self):
        """Custom pickle serialization"""
        from .utils import keypoints_to_list
        
        serialized_methods = {}
        for name, method_data in self.methods.items():
            serialized_methods[name] = {
                'method_name': method_data.method_name,
                'keypoints1': keypoints_to_list(method_data.keypoints1),
                'keypoints2': keypoints_to_list(method_data.keypoints2),
                'descriptors1': method_data.descriptors1,
                'descriptors2': method_data.descriptors2,
                'query_indices': method_data.query_indices,
                'train_indices': method_data.train_indices,
                'match_scores': method_data.match_scores,
                'score_type': method_data.score_type,
                'homography': method_data.homography,
                'fundamental_matrix': method_data.fundamental_matrix,
                'essential_matrix': method_data.essential_matrix,
                'inlier_mask': method_data.inlier_mask,
                'inlier_ratio': method_data.inlier_ratio,
                'reprojection_error': method_data.reprojection_error,
                'correspondences': method_data.correspondences,
            }
        
        return {
            'methods': serialized_methods,
            'image_pair_info': self.image_pair_info,
            'K1': self.K1,
            'K2': self.K2,
        }
    
    def __setstate__(self, state):
        """Custom pickle deserialization"""
        from .utils import list_to_keypoints
        
        deserialized_methods = {}
        for name, method_data in state['methods'].items():
            method_recon = MethodReconstructionData(
                method_name=method_data['method_name'],
                keypoints1=tuple(list_to_keypoints(method_data['keypoints1'])),
                keypoints2=tuple(list_to_keypoints(method_data['keypoints2'])),
                descriptors1=method_data['descriptors1'],
                descriptors2=method_data['descriptors2'],
                query_indices=method_data['query_indices'],
                train_indices=method_data['train_indices'],
                match_scores=method_data['match_scores'],
                score_type=method_data['score_type'],
                homography=method_data['homography'],
                fundamental_matrix=method_data['fundamental_matrix'],
                essential_matrix=method_data['essential_matrix'],
                inlier_mask=method_data['inlier_mask'],
                inlier_ratio=method_data['inlier_ratio'],
                reprojection_error=method_data['reprojection_error'],
                correspondences=method_data['correspondences'],
            )
            deserialized_methods[name] = method_recon
        
        self.methods = deserialized_methods
        self.image_pair_info = state['image_pair_info']
        self.K1 = state['K1']
        self.K2 = state['K2']
    

    def __getitem__(self, method_name: str) -> MethodReconstructionData:
        return self.methods[method_name]
    
    def __contains__(self, method_name: str) -> bool:
        return method_name in self.methods
    
    def __len__(self) -> int:
        return len(self.methods)
    
    def __iter__(self):
        return iter(self.methods)
    
    def items(self):
        return self.methods.items()
    
    def keys(self):
        return self.methods.keys()
    
    def values(self):
        return self.methods.values()
    
    # -------------------------------------------------------------------------
    # Method Selection
    # -------------------------------------------------------------------------
    
    def get_method(self, name: str) -> Optional[MethodReconstructionData]:
        return self.methods.get(name)
    
    def get_best(self, criterion: str = 'quality') -> Optional[MethodReconstructionData]:
        """Get best method by criterion"""
        if not self.methods:
            return None
        
        if criterion == 'quality':
            return max(self.methods.values(), key=lambda m: m.get_quality_score())
        elif criterion == 'num_matches':
            return max(self.methods.values(), key=lambda m: m.num_matches)
        elif criterion == 'inlier_ratio':
            return max(self.methods.values(), 
                      key=lambda m: m.inlier_ratio if m.inlier_ratio else 0)
        elif criterion == 'inliers':
            return max(self.methods.values(), key=lambda m: m.num_inliers)
        else:
            raise ValueError(f"Unknown criterion: {criterion}")
    
    def get_best_name(self, criterion: str = 'quality') -> Optional[str]:
        best = self.get_best(criterion)
        return best.method_name if best else None
    
    def rank_methods(self, criterion: str = 'quality') -> List[Tuple[str, MethodReconstructionData]]:
        """Rank all methods"""
        if criterion == 'quality':
            key_func = lambda x: x[1].get_quality_score()
        elif criterion == 'num_matches':
            key_func = lambda x: x[1].num_matches
        elif criterion == 'inlier_ratio':
            key_func = lambda x: x[1].inlier_ratio if x[1].inlier_ratio else 0
        else:
            raise ValueError(f"Unknown criterion: {criterion}")
        
        return sorted(self.methods.items(), key=key_func, reverse=True)
    
    # -------------------------------------------------------------------------
    # Filtering
    # -------------------------------------------------------------------------
    
    def filter_methods(self,
                      min_matches: Optional[int] = None,
                      min_inlier_ratio: Optional[float] = None,
                      min_quality: Optional[float] = None,
                      method_names: Optional[List[str]] = None) -> 'MultiMethodReconstruction':
        """Create new reconstruction with filtered methods"""
        filtered = {}
        
        for name, data in self.methods.items():
            if method_names and name not in method_names:
                continue
            if min_matches and data.num_matches < min_matches:
                continue
            if min_inlier_ratio and (not data.inlier_ratio or data.inlier_ratio < min_inlier_ratio):
                continue
            if min_quality and data.get_quality_score() < min_quality:
                continue
            
            filtered[name] = data
        
        return MultiMethodReconstruction(
            methods=filtered,
            image_pair_info=self.image_pair_info,
            K1=self.K1,
            K2=self.K2
        )
    
    # -------------------------------------------------------------------------
    # Export
    # -------------------------------------------------------------------------
    
    def export_method(self, method_name: str, output_dir: Union[str, Path]):
        """Export specific method to COLMAP format"""
        if method_name not in self.methods:
            raise ValueError(f"Method {method_name} not found")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        method_data = self.methods[method_name]
        colmap_data = method_data.export_to_colmap()
        
        # Save keypoints
        kp1_file = output_dir / f"{self.image_pair_info.image1_id}_keypoints.txt"
        kp2_file = output_dir / f"{self.image_pair_info.image2_id}_keypoints.txt"
        
        with open(kp1_file, 'w') as f:
            for x, y in colmap_data['keypoints1']:
                f.write(f"{x} {y}\n")
        
        with open(kp2_file, 'w') as f:
            for x, y in colmap_data['keypoints2']:
                f.write(f"{x} {y}\n")
        
        # Save matches
        matches_file = output_dir / "matches.txt"
        with open(matches_file, 'w') as f:
            for idx1, idx2 in colmap_data['matches']:
                f.write(f"{idx1} {idx2}\n")
        
        print(f"✅ Exported {method_name} to {output_dir}")
    
    def export_all_methods(self, base_dir: Union[str, Path], separate_dirs: bool = True):
        """Export all methods"""
        base_dir = Path(base_dir)
        
        for method_name in self.methods:
            if separate_dirs:
                output_dir = base_dir / f"colmap_{method_name}"
            else:
                output_dir = base_dir / method_name
            
            self.export_method(method_name, output_dir)
    
    # -------------------------------------------------------------------------
    # Analysis
    # -------------------------------------------------------------------------
    
    def summary(self) -> str:
        """Human-readable summary"""
        lines = [
            f"MultiMethodReconstruction: {len(self.methods)} methods",
            f"Images: {self.image_pair_info.image1_id} ↔ {self.image_pair_info.image2_id}",
            ""
        ]
        
        if not self.methods:
            lines.append("⚠️  No methods available")
            return "\n".join(lines)
        
        ranked = self.rank_methods('quality')
        for i, (name, data) in enumerate(ranked, 1):
            inlier_pct = data.inlier_ratio * 100 if data.inlier_ratio else 0
            
            status = ""
            if data.num_matches == 0:
                status = " ⚠️  NO MATCHES"
            
            lines.append(
                f"{i}. {name:12s}: "
                f"{data.num_matches:4d} matches, "
                f"{data.num_inliers:4d} inliers ({inlier_pct:.1f}%), "
                f"quality={data.get_quality_score():.3f}{status}"
            )
        
        return "\n".join(lines)
    
    def compare_methods(self, method_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Compare methods"""
        methods_to_compare = (
            {name: self.methods[name] for name in method_names if name in self.methods}
            if method_names else self.methods
        )
        
        comparison = {'methods': {}, 'rankings': {}}
        
        for name, data in methods_to_compare.items():
            comparison['methods'][name] = {
                'num_matches': data.num_matches,
                'num_inliers': data.num_inliers,
                'inlier_ratio': data.inlier_ratio,
                'reprojection_error': data.reprojection_error,
                'quality_score': data.get_quality_score()
            }
        
        for criterion in ['quality', 'num_matches', 'inlier_ratio']:
            ranked = self.rank_methods(criterion)
            comparison['rankings'][criterion] = [name for name, _ in ranked]
        
        return comparison
    
    # -------------------------------------------------------------------------
    # Save/Load
    # -------------------------------------------------------------------------
    
    def save(self, filepath: Union[str, Path]):
        """Save reconstruction data"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"✅ Saved MultiMethodReconstruction to {filepath}")
    
    @staticmethod
    def load(filepath: Union[str, Path]) -> 'MultiMethodReconstruction':
        """Load reconstruction data"""
        with open(filepath, 'rb') as f:
            recon = pickle.load(f)
        print(f"✅ Loaded MultiMethodReconstruction from {filepath}")
        return recon


# =============================================================================
# RESULT CONVERTER
# =============================================================================

class ResultConverter:
    """Converts MatchingResult to specialized formats"""
    
    # Method colors for visualization
    METHOD_COLORS = {
        'SIFT': (255, 0, 0), 'ORB': (0, 255, 0), 'AKAZE': (0, 0, 255),
        'BRISK': (255, 255, 0), 'SuperPoint': (255, 0, 255), 
        'DISK': (0, 255, 255), 'ALIKED': (255, 128, 0), 
        'LightGlue': (128, 0, 255)
    }
    
    def to_visualization(self, 
                        result: MatchingResult,
                        include_images: bool = True) -> VisualizationData:
        """
        Convert to VisualizationData
        
        Handles all offset management internally. Creates unified
        keypoint lists and adjusts match indices.
        """
        # Collect all keypoints
        all_keypoints1 = []
        all_keypoints2 = []
        all_matches = []
        method_info = {}
        
        offset1 = 0
        offset2 = 0
        
        for method_name, method_result in result.methods.items():
            kps1 = method_result.features1.keypoints
            kps2 = method_result.features2.keypoints
            matches = method_result.match_data.get_best_matches()
            
            # Add keypoints
            all_keypoints1.extend(kps1)
            all_keypoints2.extend(kps2)
            
            # Add matches with adjusted indices
            color = self._get_method_color(method_name)
            for match in matches:
                visual_match = VisualMatch(
                    query_idx=match.queryIdx + offset1,
                    train_idx=match.trainIdx + offset2,
                    score=match.distance if not hasattr(match, 'score') else match.score,
                    score_type=method_result.match_data.score_type,
                    color=color,
                    source_method=method_name
                )
                all_matches.append(visual_match)
            
            # Track method info
            method_info[method_name] = {
                'num_features1': len(kps1),
                'num_features2': len(kps2),
                'num_matches': len(matches),
                'offset1': offset1,
                'offset2': offset2,
                'color': color
            }
            
            offset1 += len(kps1)
            offset2 += len(kps2)
        
        # Handle images
        image_pair_info = result.image_pair_info
        if not include_images:
            image_pair_info = ImagePairInfo(
                image1_id=image_pair_info.image1_id,
                image2_id=image_pair_info.image2_id,
                image1_size=image_pair_info.image1_size,
                image2_size=image_pair_info.image2_size
            )
        
        return VisualizationData(
            keypoints1=tuple(all_keypoints1),
            keypoints2=tuple(all_keypoints2),
            matches=tuple(all_matches),
            method_info=method_info,
            image_pair_info=image_pair_info,
            title=f"Feature Matches ({len(all_matches)} total)",
            subtitle=f"Methods: {', '.join(result.methods.keys())}"
        )
    
    def to_reconstruction(self,
                         result: MatchingResult,
                         methods: Optional[List[str]] = None,
                         min_matches: int = 0,
                         exclude_empty: bool = True,  # NEW: Automatically exclude methods with 0 matches
                         K1: Optional[np.ndarray] = None,
                         K2: Optional[np.ndarray] = None) -> MultiMethodReconstruction:
        """
        Convert to MultiMethodReconstruction
        
        Args:
            result: MatchingResult to convert
            methods: Specific methods to include (None = all)
            min_matches: Minimum matches filter
            exclude_empty: Automatically exclude methods with 0 matches (default: True)
            K1: Camera intrinsics for image 1
            K2: Camera intrinsics for image 2
        """
        # Determine which methods to convert
        if methods:
            methods_to_convert = {
                name: result.methods[name]
                for name in methods if name in result.methods
            }
        else:
            methods_to_convert = result.methods
        
        # Apply filters
        if exclude_empty:
            methods_to_convert = {
                name: method for name, method in methods_to_convert.items()
                if method.num_matches > 0
            }
        
        if min_matches > 0:
            methods_to_convert = {
                name: method for name, method in methods_to_convert.items()
                if method.num_matches >= min_matches
            }
        
        # Convert each method
        reconstruction_methods = {}
        for name, method_result in methods_to_convert.items():
            reconstruction_methods[name] = self._convert_method(method_result)
        
        return MultiMethodReconstruction(
            methods=reconstruction_methods,
            image_pair_info=result.image_pair_info,
            K1=K1,
            K2=K2
        )
    
    def _convert_method(self, method_result: MethodResult) -> MethodReconstructionData:
        """Convert single MethodResult to MethodReconstructionData"""
        matches = method_result.match_data.get_best_matches()
        
        # Handle empty matches case
        if len(matches) == 0:
            return MethodReconstructionData(
                method_name=method_result.method_name,
                keypoints1=tuple(method_result.features1.keypoints),
                keypoints2=tuple(method_result.features2.keypoints),
                descriptors1=method_result.features1.descriptors,
                descriptors2=method_result.features2.descriptors,
                query_indices=np.array([], dtype=np.int32),
                train_indices=np.array([], dtype=np.int32),
                match_scores=np.array([]),
                score_type=method_result.match_data.score_type,
                homography=method_result.homography,
                fundamental_matrix=method_result.fundamental_matrix,
                inlier_ratio=method_result.inlier_ratio,
                reprojection_error=method_result.reprojection_error,
                correspondences=None
            )
        
        # Extract indices and scores
        query_indices = np.array([m.queryIdx for m in matches], dtype=np.int32)
        train_indices = np.array([m.trainIdx for m in matches], dtype=np.int32)
        
        # Determine score type by checking first match
        if hasattr(matches[0], 'score'):
            match_scores = np.array([m.score for m in matches])
        else:
            match_scores = np.array([m.distance for m in matches])
        
        # Build correspondences
        pts1 = np.array([method_result.features1.keypoints[m.queryIdx].pt for m in matches])
        pts2 = np.array([method_result.features2.keypoints[m.trainIdx].pt for m in matches])
        correspondences = np.hstack([pts1, pts2])
        
        return MethodReconstructionData(
            method_name=method_result.method_name,
            keypoints1=tuple(method_result.features1.keypoints),
            keypoints2=tuple(method_result.features2.keypoints),
            descriptors1=method_result.features1.descriptors,
            descriptors2=method_result.features2.descriptors,
            query_indices=query_indices,
            train_indices=train_indices,
            match_scores=match_scores,
            score_type=method_result.match_data.score_type,
            homography=method_result.homography,
            fundamental_matrix=method_result.fundamental_matrix,
            inlier_ratio=method_result.inlier_ratio,
            reprojection_error=method_result.reprojection_error,
            correspondences=correspondences
        )
    
    def _get_method_color(self, method_name: str) -> Tuple[int, int, int]:
        """Get color for method"""
        if method_name in self.METHOD_COLORS:
            return self.METHOD_COLORS[method_name]
        
        # Generate color from hash
        import hashlib
        hash_val = int(hashlib.md5(method_name.encode()).hexdigest()[:6], 16)
        return ((hash_val >> 16) & 0xFF, (hash_val >> 8) & 0xFF, hash_val & 0xFF)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def save_for_reconstruction(result: MatchingResult,
                           output_dir: Union[str, Path],
                           methods: Optional[List[str]] = None,
                           exclude_empty: bool = True,
                           export_all: bool = True):
    """
    Save for reconstruction
    
    Args:
        result: MatchingResult
        output_dir: Output directory
        methods: Specific methods to save (None = all)
        exclude_empty: Exclude methods with 0 matches (default: True)
        export_all: Export all methods to separate COLMAP dirs
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save MatchingResult
    result.save(output_dir / 'matching_result.pkl')
    
    # Convert and save reconstruction
    recon = result.to_reconstruction(methods=methods, exclude_empty=exclude_empty)
    
    if len(recon.methods) == 0:
        print("⚠️  No methods with matches to save for reconstruction")
        return
    
    recon.save(output_dir / 'reconstruction.pkl')
    
    # Export COLMAP
    if export_all:
        recon.export_all_methods(output_dir)
    else:
        best_name = recon.get_best_name()
        if best_name:
            recon.export_method(best_name, output_dir / 'colmap')
    
    # Save metadata
    metadata = {
        'methods': list(recon.methods.keys()),
        'best_method': recon.get_best_name(),
        'num_methods': len(recon.methods),
        'image_pair': {
            'image1': result.image_pair_info.image1_id,
            'image2': result.image_pair_info.image2_id
        }
    }
    
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Saved reconstruction data to {output_dir}")
    print(recon.summary())


def load_for_reconstruction(input_dir: Union[str, Path]) -> Tuple[MatchingResult, MultiMethodReconstruction]:
    """Load reconstruction data"""
    input_dir = Path(input_dir)
    
    result = MatchingResult.load(input_dir / 'matching_result.pkl')
    recon = MultiMethodReconstruction.load(input_dir / 'reconstruction.pkl')
    
    return result, recon