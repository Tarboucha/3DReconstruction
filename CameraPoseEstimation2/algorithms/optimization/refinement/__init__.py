
# Pose refinement
from .pose_refiner import (
    PoseRefiner,
    PoseRefinerConfig,
    refine_camera_pose
)

# Intrinsics learning
from .intrinsics_refiner import (
    ProgressiveIntrinsicsLearner,
    ProgressiveIntrinsicsLearnerConfig,
    IntrinsicsEstimate
)

# Structure refinement
from .structure_refiner import (
    StructureRefiner,
    StructureRefinerConfig,
    StructureQualityMetrics
)

# Progressive pipeline
from .progressive_pipeline import (
    ProgressiveRefinementPipeline,
    ProgressiveRefinementConfig
)

# Essential matrix refinement
from .essential_matrix_refiner import (
    EssentialMatrixRefiner,
    EssentialMatrixRefinerConfig
)


__all__ = [
    # Pose refinement
    'PoseRefiner',
    'PoseRefinerConfig',
    'refine_camera_pose',
    
    # Intrinsics learning
    'ProgressiveIntrinsicsLearner',
    'ProgressiveIntrinsicsLearnerConfig',
    'IntrinsicsEstimate',
    
    # Structure refinement
    'StructureRefiner',
    'StructureRefinerConfig',
    'StructureQualityMetrics',
    
    # Progressive pipeline
    'ProgressiveRefinementPipeline',
    'ProgressiveRefinementConfig',
    
    # Essential matrix refinement
    'EssentialMatrixRefiner',
    'EssentialMatrixRefinerConfig',
]


__version__ = '1.0.0'


# Module metadata
__description__ = 'Iterative refinement algorithms for 3D reconstruction'