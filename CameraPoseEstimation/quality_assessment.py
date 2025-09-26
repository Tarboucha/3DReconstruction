import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
from dataclasses import dataclass
from enum import Enum

class QualityLevel(Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    FAILED = "failed"

@dataclass
class QualityMetrics:
    """Container for reconstruction quality metrics"""
    
    # Reprojection metrics
    mean_reprojection_error: float
    median_reprojection_error: float
    max_reprojection_error: float
    reprojection_error_std: float
    inlier_ratio: float  # Fraction of observations with error < threshold
    
    # Coverage metrics
    num_cameras: int
    num_3d_points: int
    num_observations: int
    observations_per_camera: float
    observations_per_point: float
    
    # Geometric metrics
    reconstruction_volume: float
    camera_spread: float
    point_density: float
    baseline_diversity: float
    
    # Camera calibration metrics
    focal_length_consistency: float
    principal_point_consistency: float
    intrinsic_reliability: float
    
    # Bundle adjustment metrics
    optimization_convergence: bool
    final_cost: float
    cost_reduction_ratio: float
    
    # Overall quality
    overall_score: float
    quality_level: QualityLevel
    
    # Detailed statistics
    per_camera_errors: Dict[str, float]
    per_point_observations: Dict[int, int]
    camera_positions: Dict[str, np.ndarray]

def assess_reconstruction_quality(reconstruction_state: Dict[str, Any], 
                                 optimization_history: Optional[List[Dict]] = None,
                                 reprojection_threshold: float = 2.0) -> QualityMetrics:
    """
    Comprehensive assessment of reconstruction quality.
    
    Args:
        reconstruction_state: Complete reconstruction state
        optimization_history: History from bundle adjustment (optional)
        reprojection_threshold: Pixel threshold for inlier classification
    
    Returns:
        QualityMetrics object with comprehensive quality assessment
    """
    
    print("\n" + "="*60)
    print("RECONSTRUCTION QUALITY ASSESSMENT")
    print("="*60)
    
    try:
        # Extract reconstruction components
        cameras = reconstruction_state.get('cameras', {})
        points_3d = reconstruction_state.get('points_3d', np.array([]))
        observations = reconstruction_state.get('observations', {})
        
        # Validate input data
        if not _validate_reconstruction_data(cameras, points_3d, observations):
            return _create_failed_quality_metrics("Invalid reconstruction data")
        
        print(f"Assessing reconstruction with {len(cameras)} cameras, {points_3d.shape[1]} 3D points")
        
        # Calculate individual metric categories
        reprojection_metrics = _calculate_reprojection_metrics(
            cameras, points_3d, observations, reprojection_threshold
        )
        
        coverage_metrics = _calculate_coverage_metrics(
            cameras, points_3d, observations
        )
        
        geometric_metrics = _calculate_geometric_metrics(
            cameras, points_3d, observations
        )
        
        calibration_metrics = _calculate_calibration_metrics(cameras)
        
        optimization_metrics = _calculate_optimization_metrics(optimization_history)
        
        # Calculate overall quality score
        overall_score, quality_level = _calculate_overall_quality(
            reprojection_metrics, coverage_metrics, geometric_metrics,
            calibration_metrics, optimization_metrics
        )
        
        # Create comprehensive quality metrics
        quality_metrics = QualityMetrics(
            # Reprojection metrics
            mean_reprojection_error=reprojection_metrics['mean_error'],
            median_reprojection_error=reprojection_metrics['median_error'],
            max_reprojection_error=reprojection_metrics['max_error'],
            reprojection_error_std=reprojection_metrics['error_std'],
            inlier_ratio=reprojection_metrics['inlier_ratio'],
            
            # Coverage metrics
            num_cameras=coverage_metrics['num_cameras'],
            num_3d_points=coverage_metrics['num_3d_points'],
            num_observations=coverage_metrics['num_observations'],
            observations_per_camera=coverage_metrics['obs_per_camera'],
            observations_per_point=coverage_metrics['obs_per_point'],
            
            # Geometric metrics
            reconstruction_volume=geometric_metrics['volume'],
            camera_spread=geometric_metrics['camera_spread'],
            point_density=geometric_metrics['point_density'],
            baseline_diversity=geometric_metrics['baseline_diversity'],
            
            # Calibration metrics
            focal_length_consistency=calibration_metrics['focal_consistency'],
            principal_point_consistency=calibration_metrics['pp_consistency'],
            intrinsic_reliability=calibration_metrics['intrinsic_reliability'],
            
            # Optimization metrics
            optimization_convergence=optimization_metrics['converged'],
            final_cost=optimization_metrics['final_cost'],
            cost_reduction_ratio=optimization_metrics['cost_reduction'],
            
            # Overall assessment
            overall_score=overall_score,
            quality_level=quality_level,
            
            # Detailed data
            per_camera_errors=reprojection_metrics['per_camera_errors'],
            per_point_observations=coverage_metrics['per_point_observations'],
            camera_positions=geometric_metrics['camera_positions']
        )
        
        # Print quality report
        _print_quality_report(quality_metrics)
        
        return quality_metrics
        
    except Exception as e:
        print(f"Quality assessment failed: {e}")
        return _create_failed_quality_metrics(f"Assessment error: {str(e)}")


def _calculate_reprojection_metrics(cameras: Dict, points_3d: np.ndarray, 
                                   observations: Dict, threshold: float) -> Dict[str, Any]:
    """Calculate reprojection error metrics"""
    
    print("Calculating reprojection metrics...")
    
    all_errors = []
    per_camera_errors = {}
    
    for camera_name, camera_data in cameras.items():
        if camera_name not in observations:
            continue
            
        K = camera_data['K']
        R = camera_data['R'] 
        t = camera_data['t']
        
        camera_errors = []
        
        for obs in observations[camera_name]:
            point_id = obs['point_id']
            observed_2d = np.array(obs['image_point'])
            
            if point_id < points_3d.shape[1]:
                # Project 3D point to 2D
                point_3d = points_3d[:, point_id]
                projected_2d = _project_point_to_image(point_3d, K, R, t)
                
                # Calculate reprojection error
                error = np.linalg.norm(observed_2d - projected_2d)
                camera_errors.append(error)
                all_errors.append(error)
        
        if camera_errors:
            per_camera_errors[camera_name] = np.mean(camera_errors)
    
    if not all_errors:
        return {
            'mean_error': float('inf'),
            'median_error': float('inf'),
            'max_error': float('inf'),
            'error_std': float('inf'),
            'inlier_ratio': 0.0,
            'per_camera_errors': {}
        }
    
    all_errors = np.array(all_errors)
    inliers = all_errors < threshold
    
    return {
        'mean_error': np.mean(all_errors),
        'median_error': np.median(all_errors),
        'max_error': np.max(all_errors),
        'error_std': np.std(all_errors),
        'inlier_ratio': np.sum(inliers) / len(all_errors),
        'per_camera_errors': per_camera_errors
    }


def _calculate_coverage_metrics(cameras: Dict, points_3d: np.ndarray, 
                               observations: Dict) -> Dict[str, Any]:
    """Calculate coverage and completeness metrics"""
    
    print("Calculating coverage metrics...")
    
    num_cameras = len(cameras)
    num_3d_points = points_3d.shape[1] if points_3d.size > 0 else 0
    
    # Count observations
    total_observations = 0
    per_point_observations = {}
    
    for camera_name, camera_obs in observations.items():
        total_observations += len(camera_obs)
        
        for obs in camera_obs:
            point_id = obs['point_id']
            per_point_observations[point_id] = per_point_observations.get(point_id, 0) + 1
    
    # Calculate averages
    obs_per_camera = total_observations / max(num_cameras, 1)
    obs_per_point = total_observations / max(num_3d_points, 1)
    
    return {
        'num_cameras': num_cameras,
        'num_3d_points': num_3d_points,
        'num_observations': total_observations,
        'obs_per_camera': obs_per_camera,
        'obs_per_point': obs_per_point,
        'per_point_observations': per_point_observations
    }


def _calculate_geometric_metrics(cameras: Dict, points_3d: np.ndarray, 
                                observations: Dict) -> Dict[str, Any]:
    """Calculate geometric quality metrics"""
    
    print("Calculating geometric metrics...")
    
    if not cameras or points_3d.size == 0:
        return {
            'volume': 0.0,
            'camera_spread': 0.0,
            'point_density': 0.0,
            'baseline_diversity': 0.0,
            'camera_positions': {}
        }
    
    # Extract camera positions
    camera_positions = {}
    camera_centers = []
    
    for camera_name, camera_data in cameras.items():
        R = camera_data['R']
        t = camera_data['t']
        
        # Camera center in world coordinates: C = -R^T * t
        camera_center = -R.T @ t
        camera_positions[camera_name] = camera_center.flatten()
        camera_centers.append(camera_center.flatten())
    
    camera_centers = np.array(camera_centers)
    
    # Calculate reconstruction volume (bounding box of 3D points)
    if points_3d.shape[1] > 0:
        point_min = np.min(points_3d, axis=1)
        point_max = np.max(points_3d, axis=1)
        reconstruction_volume = np.prod(point_max - point_min)
    else:
        reconstruction_volume = 0.0
    
    # Calculate camera spread (variance of camera positions)
    if len(camera_centers) > 1:
        camera_spread = np.mean(np.var(camera_centers, axis=0))
    else:
        camera_spread = 0.0
    
    # Calculate point density (points per unit volume)
    point_density = points_3d.shape[1] / max(reconstruction_volume, 1e-6)
    
    # Calculate baseline diversity (variety of camera-camera distances)
    baseline_diversity = _calculate_baseline_diversity(camera_centers)
    
    return {
        'volume': reconstruction_volume,
        'camera_spread': camera_spread,
        'point_density': point_density,
        'baseline_diversity': baseline_diversity,
        'camera_positions': camera_positions
    }


def _calculate_calibration_metrics(cameras: Dict) -> Dict[str, Any]:
    """Calculate camera calibration quality metrics"""
    
    print("Calculating calibration metrics...")
    
    if not cameras:
        return {
            'focal_consistency': 0.0,
            'pp_consistency': 0.0,
            'intrinsic_reliability': 0.0
        }
    
    # Extract intrinsic parameters
    focal_lengths = []
    principal_points = []
    
    for camera_data in cameras.values():
        K = camera_data['K']
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        
        focal_lengths.append([fx, fy])
        principal_points.append([cx, cy])
    
    focal_lengths = np.array(focal_lengths)
    principal_points = np.array(principal_points)
    
    # Calculate consistency (lower variance = higher consistency)
    if len(focal_lengths) > 1:
        focal_variance = np.mean(np.var(focal_lengths, axis=0))
        focal_consistency = 1.0 / (1.0 + focal_variance / np.mean(focal_lengths)**2)
        
        pp_variance = np.mean(np.var(principal_points, axis=0))
        pp_consistency = 1.0 / (1.0 + pp_variance / (np.mean(principal_points)**2 + 1e-6))
    else:
        focal_consistency = 1.0
        pp_consistency = 1.0
    
    # Overall intrinsic reliability
    intrinsic_reliability = (focal_consistency + pp_consistency) / 2.0
    
    return {
        'focal_consistency': focal_consistency,
        'pp_consistency': pp_consistency,
        'intrinsic_reliability': intrinsic_reliability
    }


def _calculate_optimization_metrics(optimization_history: Optional[List[Dict]]) -> Dict[str, Any]:
    """Calculate bundle adjustment optimization metrics"""
    
    if not optimization_history:
        return {
            'converged': True,  # Assume success if no history
            'final_cost': 0.0,
            'cost_reduction': 1.0
        }
    
    final_optimization = optimization_history[-1]
    
    initial_cost = final_optimization.get('initial_cost', 1.0)
    final_cost = final_optimization.get('final_cost', initial_cost)
    converged = final_optimization.get('converged', True)
    
    # Calculate cost reduction ratio
    if initial_cost > 0:
        cost_reduction = (initial_cost - final_cost) / initial_cost
    else:
        cost_reduction = 0.0
    
    return {
        'converged': converged,
        'final_cost': final_cost,
        'cost_reduction': max(0.0, cost_reduction)  # Ensure non-negative
    }


def _calculate_overall_quality(reprojection_metrics: Dict, coverage_metrics: Dict,
                              geometric_metrics: Dict, calibration_metrics: Dict,
                              optimization_metrics: Dict) -> Tuple[float, QualityLevel]:
    """Calculate overall quality score and level"""
    
    # Individual component scores (0-1 scale)
    
    # Reprojection score (lower error = higher score)
    if reprojection_metrics['mean_error'] < 1.0:
        reprojection_score = 1.0
    elif reprojection_metrics['mean_error'] < 2.0:
        reprojection_score = 0.8
    elif reprojection_metrics['mean_error'] < 4.0:
        reprojection_score = 0.6
    elif reprojection_metrics['mean_error'] < 8.0:
        reprojection_score = 0.4
    else:
        reprojection_score = 0.2
    
    # Inlier ratio score
    inlier_score = reprojection_metrics['inlier_ratio']
    
    # Coverage score
    if coverage_metrics['num_cameras'] >= 10 and coverage_metrics['obs_per_point'] >= 3.0:
        coverage_score = 1.0
    elif coverage_metrics['num_cameras'] >= 5 and coverage_metrics['obs_per_point'] >= 2.5:
        coverage_score = 0.8
    elif coverage_metrics['num_cameras'] >= 3 and coverage_metrics['obs_per_point'] >= 2.0:
        coverage_score = 0.6
    else:
        coverage_score = 0.4
    
    # Geometric score
    geometry_components = [
        min(1.0, geometric_metrics['camera_spread'] / 10.0),  # Normalize camera spread
        min(1.0, geometric_metrics['baseline_diversity']),
        min(1.0, geometric_metrics['point_density'] / 100.0)  # Normalize point density
    ]
    geometric_score = np.mean(geometry_components)
    
    # Calibration score
    calibration_score = calibration_metrics['intrinsic_reliability']
    
    # Optimization score
    opt_score = 0.5 * float(optimization_metrics['converged']) + 0.5 * optimization_metrics['cost_reduction']
    
    # Weighted overall score
    weights = {
        'reprojection': 0.3,
        'inlier': 0.2,
        'coverage': 0.2,
        'geometric': 0.15,
        'calibration': 0.1,
        'optimization': 0.05
    }
    
    overall_score = (
        weights['reprojection'] * reprojection_score +
        weights['inlier'] * inlier_score +
        weights['coverage'] * coverage_score +
        weights['geometric'] * geometric_score +
        weights['calibration'] * calibration_score +
        weights['optimization'] * opt_score
    )
    
    # Determine quality level
    if overall_score >= 0.85:
        quality_level = QualityLevel.EXCELLENT
    elif overall_score >= 0.7:
        quality_level = QualityLevel.GOOD
    elif overall_score >= 0.5:
        quality_level = QualityLevel.FAIR
    elif overall_score >= 0.3:
        quality_level = QualityLevel.POOR
    else:
        quality_level = QualityLevel.FAILED
    
    return overall_score, quality_level


def _calculate_baseline_diversity(camera_centers: np.ndarray) -> float:
    """Calculate baseline diversity (variety of camera-camera distances)"""
    
    if len(camera_centers) < 2:
        return 0.0
    
    # Calculate all pairwise distances
    distances = []
    for i in range(len(camera_centers)):
        for j in range(i + 1, len(camera_centers)):
            dist = np.linalg.norm(camera_centers[i] - camera_centers[j])
            distances.append(dist)
    
    if not distances:
        return 0.0
    
    # Diversity = coefficient of variation of distances
    distances = np.array(distances)
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)
    
    if mean_dist > 0:
        diversity = std_dist / mean_dist
    else:
        diversity = 0.0
    
    return min(1.0, diversity)  # Cap at 1.0


def _project_point_to_image(point_3d: np.ndarray, K: np.ndarray, 
                           R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Project 3D point to 2D image coordinates"""
    
    # Transform to camera coordinates
    point_cam = R @ point_3d + t.flatten()
    
    # Project to image plane
    if point_cam[2] != 0:
        point_2d_homo = K @ point_cam
        point_2d = point_2d_homo[:2] / point_2d_homo[2]
    else:
        point_2d = np.array([0, 0])
    
    return point_2d


def _validate_reconstruction_data(cameras: Dict, points_3d: np.ndarray, 
                                 observations: Dict) -> bool:
    """Validate reconstruction data for quality assessment"""
    
    if not cameras:
        print("No cameras in reconstruction")
        return False
    
    if points_3d.size == 0:
        print("No 3D points in reconstruction")
        return False
    
    if not observations:
        print("No observations in reconstruction")
        return False
    
    # Check camera data validity
    for camera_name, camera_data in cameras.items():
        if 'K' not in camera_data or 'R' not in camera_data or 't' not in camera_data:
            print(f"Camera {camera_name} missing required parameters")
            return False
    
    return True


def _create_failed_quality_metrics(error_message: str) -> QualityMetrics:
    """Create quality metrics object for failed assessment"""
    
    return QualityMetrics(
        mean_reprojection_error=float('inf'),
        median_reprojection_error=float('inf'),
        max_reprojection_error=float('inf'),
        reprojection_error_std=float('inf'),
        inlier_ratio=0.0,
        num_cameras=0,
        num_3d_points=0,
        num_observations=0,
        observations_per_camera=0.0,
        observations_per_point=0.0,
        reconstruction_volume=0.0,
        camera_spread=0.0,
        point_density=0.0,
        baseline_diversity=0.0,
        focal_length_consistency=0.0,
        principal_point_consistency=0.0,
        intrinsic_reliability=0.0,
        optimization_convergence=False,
        final_cost=float('inf'),
        cost_reduction_ratio=0.0,
        overall_score=0.0,
        quality_level=QualityLevel.FAILED,
        per_camera_errors={},
        per_point_observations={},
        camera_positions={}
    )


def _print_quality_report(metrics: QualityMetrics) -> None:
    """Print formatted quality assessment report"""
    
    print(f"\n📊 RECONSTRUCTION QUALITY: {metrics.quality_level.value.upper()}")
    print(f"   Overall Score: {metrics.overall_score:.3f}/1.000")
    
    print(f"\n📐 REPROJECTION ACCURACY:")
    print(f"   Mean error: {metrics.mean_reprojection_error:.2f} pixels")
    print(f"   Median error: {metrics.median_reprojection_error:.2f} pixels")
    print(f"   Max error: {metrics.max_reprojection_error:.2f} pixels")
    print(f"   Inlier ratio: {metrics.inlier_ratio:.1%}")
    
    print(f"\n📈 COVERAGE STATISTICS:")
    print(f"   Cameras: {metrics.num_cameras}")
    print(f"   3D Points: {metrics.num_3d_points}")
    print(f"   Observations: {metrics.num_observations}")
    print(f"   Obs/Camera: {metrics.observations_per_camera:.1f}")
    print(f"   Obs/Point: {metrics.observations_per_point:.1f}")
    
    print(f"\n🏗️ GEOMETRIC PROPERTIES:")
    print(f"   Reconstruction volume: {metrics.reconstruction_volume:.2e}")
    print(f"   Camera spread: {metrics.camera_spread:.2f}")
    print(f"   Point density: {metrics.point_density:.2f}")
    print(f"   Baseline diversity: {metrics.baseline_diversity:.3f}")
    
    print(f"\n📷 CALIBRATION QUALITY:")
    print(f"   Focal length consistency: {metrics.focal_length_consistency:.3f}")
    print(f"   Principal point consistency: {metrics.principal_point_consistency:.3f}")
    print(f"   Overall intrinsic reliability: {metrics.intrinsic_reliability:.3f}")
    
    print(f"\n⚙️ OPTIMIZATION STATUS:")
    print(f"   Converged: {'✅' if metrics.optimization_convergence else '❌'}")
    print(f"   Final cost: {metrics.final_cost:.6f}")
    print(f"   Cost reduction: {metrics.cost_reduction_ratio:.1%}")
    
    # Quality interpretation
    print(f"\n💡 QUALITY INTERPRETATION:")
    if metrics.quality_level == QualityLevel.EXCELLENT:
        print("   🟢 Excellent reconstruction suitable for production use")
    elif metrics.quality_level == QualityLevel.GOOD:
        print("   🔵 Good reconstruction with minor issues")
    elif metrics.quality_level == QualityLevel.FAIR:
        print("   🟡 Fair reconstruction - consider more images or refinement")
    elif metrics.quality_level == QualityLevel.POOR:
        print("   🟠 Poor reconstruction - significant improvements needed")
    else:
        print("   🔴 Failed reconstruction - start over with better images")
    
    print("="*60)

