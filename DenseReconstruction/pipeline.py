"""
Dense Reconstruction Pipeline
==============================

Main orchestrator for the dense reconstruction process.
Automatically selects the best available method and manages the workflow.
"""

import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Literal
from dataclasses import dataclass, field
import time

# Check available methods
try:
    import pycolmap
    HAS_COLMAP = True
except ImportError:
    HAS_COLMAP = False

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False


@dataclass
class DenseReconstructionConfig:
    """Configuration for dense reconstruction pipeline"""

    # Method selection
    method: Literal['auto', 'colmap', 'openmvs', 'neural'] = 'auto'

    # TSDF Fusion parameters
    voxel_size: float = 0.01  # 1cm voxels
    sdf_trunc: float = 0.04   # Truncation distance (4cm)

    # Mesh extraction
    extract_mesh: bool = True
    mesh_method: Literal['marching_cubes', 'poisson'] = 'marching_cubes'

    # Mesh post-processing
    remove_outliers: bool = True
    outlier_nb_neighbors: int = 20
    outlier_std_ratio: float = 2.0

    simplify_mesh: bool = True
    target_triangles: int = 100000

    smooth_mesh: bool = True
    smooth_iterations: int = 1

    # Texturing
    apply_texture: bool = False

    # Quality settings
    depth_map_quality: Literal['low', 'medium', 'high', 'ultra'] = 'medium'

    # Output
    export_depth_maps: bool = False
    export_point_cloud: bool = True
    export_mesh: bool = True

    # Logging
    verbose: bool = True
    log_file: Optional[str] = None


class DenseReconstructionPipeline:
    """
    Main dense reconstruction pipeline.

    Automatically selects and uses the best available dense reconstruction method:
    1. COLMAP PatchMatch MVS (if CUDA available)
    2. OpenMVS (CPU-based fallback)
    3. Neural depth estimation (if no MVS available)

    Then performs TSDF fusion, mesh extraction, and post-processing using Open3D.
    """

    def __init__(self, config: Optional[DenseReconstructionConfig] = None):
        """
        Initialize dense reconstruction pipeline.

        Args:
            config: Configuration object. If None, uses defaults.
        """
        self.config = config or DenseReconstructionConfig()
        self.logger = self._setup_logger()

        # Check available methods
        self.available_methods = self._check_available_methods()
        self.selected_method = self._select_method()

        # Statistics
        self.stats = {
            'method': self.selected_method,
            'num_depth_maps': 0,
            'num_points': 0,
            'num_triangles': 0,
            'processing_time': {}
        }

        self.logger.info(f"Dense reconstruction initialized with method: {self.selected_method}")
        self.logger.info(f"Available methods: {', '.join(self.available_methods)}")

    def _setup_logger(self) -> logging.Logger:
        """Setup logging"""
        logger = logging.getLogger("DenseReconstruction")

        if not logger.handlers:
            logger.setLevel(logging.DEBUG if self.config.verbose else logging.INFO)

            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter(
                '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

            # File handler
            if self.config.log_file:
                file_handler = logging.FileHandler(self.config.log_file)
                file_handler.setLevel(logging.DEBUG)
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)

        return logger

    def _check_available_methods(self) -> List[str]:
        """Check which MVS methods are available"""
        methods = []

        # Check COLMAP
        if HAS_COLMAP:
            try:
                # Check if CUDA is available for patch_match_stereo
                # Note: pycolmap.patch_match_stereo requires CUDA
                methods.append('colmap')
                self.logger.debug("✓ COLMAP available (requires CUDA for dense reconstruction)")
            except Exception as e:
                self.logger.debug(f"✗ COLMAP check failed: {e}")

        # Check OpenMVS (command-line)
        import shutil
        if shutil.which('DensifyPointCloud'):
            methods.append('openmvs')
            self.logger.debug("✓ OpenMVS available")
        else:
            self.logger.debug("✗ OpenMVS not found in PATH")

        # Neural depth is always available if transformers is installed
        try:
            import torch
            methods.append('neural')
            self.logger.debug("✓ Neural depth estimation available")
        except ImportError:
            self.logger.debug("✗ PyTorch not available for neural depth")

        return methods

    def _select_method(self) -> str:
        """Select the best available dense reconstruction method"""
        if self.config.method != 'auto':
            # User specified a method
            if self.config.method in self.available_methods:
                return self.config.method
            else:
                self.logger.warning(
                    f"Requested method '{self.config.method}' not available. "
                    f"Falling back to auto-selection."
                )

        # Auto-select best method
        if 'colmap' in self.available_methods:
            return 'colmap'
        elif 'openmvs' in self.available_methods:
            return 'openmvs'
        elif 'neural' in self.available_methods:
            return 'neural'
        else:
            raise RuntimeError(
                "No dense reconstruction method available. Please install one of:\n"
                "  - pycolmap (pip install pycolmap) + CUDA\n"
                "  - OpenMVS (https://github.com/cdcseacave/openMVS)\n"
                "  - PyTorch (pip install torch) for neural depth"
            )

    def run(self,
            sparse_reconstruction,
            image_folder: Union[str, Path],
            output_dir: Union[str, Path],
            image_list: Optional[List[str]] = None) -> Dict:
        """
        Run the complete dense reconstruction pipeline.

        Args:
            sparse_reconstruction: Sparse reconstruction result from CameraPoseEstimation
                                  (can be Reconstruction object or dict)
            image_folder: Path to folder containing images
            output_dir: Path to output directory
            image_list: Optional list of image filenames to use (if None, uses all)

        Returns:
            Dictionary with results:
                - 'success': bool
                - 'point_cloud_path': Path to dense point cloud PLY
                - 'mesh_path': Path to mesh file
                - 'statistics': Dict with reconstruction statistics
        """
        self.logger.info("="*70)
        self.logger.info("DENSE RECONSTRUCTION PIPELINE")
        self.logger.info("="*70)

        start_time = time.time()

        # Setup paths
        image_folder = Path(image_folder)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Stage 1: Prepare data and export to MVS format
        self.logger.info("\n--- Stage 1: Data Preparation ---")
        stage_start = time.time()

        try:
            mvs_workspace = self._prepare_mvs_workspace(
                sparse_reconstruction, image_folder, output_dir, image_list
            )
            self.stats['processing_time']['preparation'] = time.time() - stage_start
            self.logger.info(f"✓ Data preparation completed in {self.stats['processing_time']['preparation']:.2f}s")
        except Exception as e:
            self.logger.error(f"✗ Data preparation failed: {e}")
            return {'success': False, 'error': str(e)}

        # Stage 2: Depth map estimation
        self.logger.info("\n--- Stage 2: Depth Map Estimation ---")
        stage_start = time.time()

        try:
            depth_maps = self._estimate_depth_maps(mvs_workspace, output_dir)
            self.stats['num_depth_maps'] = len(depth_maps)
            self.stats['processing_time']['depth_estimation'] = time.time() - stage_start
            self.logger.info(
                f"✓ Estimated {len(depth_maps)} depth maps in "
                f"{self.stats['processing_time']['depth_estimation']:.2f}s"
            )
        except Exception as e:
            self.logger.error(f"✗ Depth estimation failed: {e}")
            return {'success': False, 'error': str(e)}

        # Stage 3: TSDF fusion (if Open3D available)
        point_cloud_path = None
        if HAS_OPEN3D:
            self.logger.info("\n--- Stage 3: TSDF Fusion ---")
            stage_start = time.time()

            try:
                from .fusion import TSDFFusion

                tsdf_fusion = TSDFFusion(
                    voxel_size=self.config.voxel_size,
                    sdf_trunc=self.config.sdf_trunc
                )

                point_cloud = tsdf_fusion.fuse_depth_maps(depth_maps, mvs_workspace['cameras'])

                if point_cloud is not None:
                    self.stats['num_points'] = len(point_cloud.points)
                    self.stats['processing_time']['fusion'] = time.time() - stage_start
                    self.logger.info(
                        f"✓ TSDF fusion completed: {self.stats['num_points']} points in "
                        f"{self.stats['processing_time']['fusion']:.2f}s"
                    )

                    # Export point cloud
                    if self.config.export_point_cloud:
                        point_cloud_path = output_dir / "dense_point_cloud.ply"
                        o3d.io.write_point_cloud(str(point_cloud_path), point_cloud)
                        self.logger.info(f"✓ Point cloud saved to {point_cloud_path}")

            except Exception as e:
                self.logger.error(f"✗ TSDF fusion failed: {e}")
                point_cloud = None
        else:
            self.logger.warning("Open3D not available, skipping TSDF fusion")
            point_cloud = None

        # Stage 4: Mesh extraction
        mesh_path = None
        if self.config.extract_mesh and point_cloud is not None and HAS_OPEN3D:
            self.logger.info("\n--- Stage 4: Mesh Extraction ---")
            stage_start = time.time()

            try:
                from .mesh import MeshProcessor

                mesh_processor = MeshProcessor()
                mesh = mesh_processor.extract_mesh(
                    point_cloud,
                    method=self.config.mesh_method
                )

                if mesh is not None:
                    self.stats['num_triangles'] = len(mesh.triangles)
                    self.stats['processing_time']['mesh_extraction'] = time.time() - stage_start
                    self.logger.info(
                        f"✓ Mesh extracted: {self.stats['num_triangles']} triangles in "
                        f"{self.stats['processing_time']['mesh_extraction']:.2f}s"
                    )

                    # Post-processing
                    if self.config.remove_outliers or self.config.simplify_mesh or self.config.smooth_mesh:
                        self.logger.info("\n--- Stage 5: Mesh Post-Processing ---")
                        mesh = mesh_processor.post_process_mesh(
                            mesh,
                            remove_outliers=self.config.remove_outliers,
                            simplify=self.config.simplify_mesh,
                            target_triangles=self.config.target_triangles,
                            smooth=self.config.smooth_mesh,
                            smooth_iterations=self.config.smooth_iterations
                        )
                        self.logger.info(f"✓ Final mesh: {len(mesh.triangles)} triangles")

                    # Export mesh
                    if self.config.export_mesh:
                        mesh_path = output_dir / "dense_mesh.ply"
                        o3d.io.write_triangle_mesh(str(mesh_path), mesh)
                        self.logger.info(f"✓ Mesh saved to {mesh_path}")

            except Exception as e:
                self.logger.error(f"✗ Mesh extraction failed: {e}")
                mesh = None

        # Final statistics
        total_time = time.time() - start_time
        self.stats['total_time'] = total_time

        self.logger.info("\n" + "="*70)
        self.logger.info("RECONSTRUCTION COMPLETE")
        self.logger.info("="*70)
        self.logger.info(f"Method: {self.selected_method}")
        self.logger.info(f"Depth maps: {self.stats['num_depth_maps']}")
        self.logger.info(f"Points: {self.stats['num_points']:,}")
        self.logger.info(f"Triangles: {self.stats['num_triangles']:,}")
        self.logger.info(f"Total time: {total_time:.2f}s")

        return {
            'success': True,
            'point_cloud_path': str(point_cloud_path) if point_cloud_path else None,
            'mesh_path': str(mesh_path) if mesh_path else None,
            'statistics': self.stats
        }

    def _prepare_mvs_workspace(self, sparse_reconstruction, image_folder, output_dir, image_list):
        """Prepare MVS workspace from sparse reconstruction"""
        from .io import SparseReconstructionConverter

        converter = SparseReconstructionConverter()
        workspace = converter.convert(
            sparse_reconstruction,
            image_folder,
            output_dir,
            method=self.selected_method,
            image_list=image_list
        )

        return workspace

    def _estimate_depth_maps(self, workspace, output_dir):
        """Estimate depth maps using selected method"""
        if self.selected_method == 'colmap':
            from .methods import COLMAPDenseReconstruction
            method = COLMAPDenseReconstruction(self.config)
        elif self.selected_method == 'openmvs':
            from .methods import OpenMVSDenseReconstruction
            method = OpenMVSDenseReconstruction(self.config)
        elif self.selected_method == 'neural':
            from .methods import NeuralDepthEstimation
            method = NeuralDepthEstimation(self.config)
        else:
            raise ValueError(f"Unknown method: {self.selected_method}")

        depth_maps = method.compute_depth_maps(workspace, output_dir)
        return depth_maps
