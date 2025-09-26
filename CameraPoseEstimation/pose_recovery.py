"""
Camera Pose Recovery from Essential/Fundamental Matrices
========================================================

This module handles the recovery of camera poses (R, t) from 
essential or fundamental matrices.
"""

import cv2
import numpy as np
from typing import Tuple, Dict, Optional
from scipy.spatial.transform import Rotation



class PoseRecovery:
    """Camera pose recovery from matrices"""

    
    def recover_from_essential(self, essential_matrix: np.ndarray,
                             pts1: np.ndarray, pts2: np.ndarray,
                             camera_matrices: Tuple[np.ndarray, np.ndarray],
                             mask: Optional[np.ndarray] = None) -> Dict:
        """
        Recover camera pose from essential matrix
        
        Args:
            essential_matrix: Essential matrix
            pts1: Points in first image
            pts2: Points in second image
            mask: Inlier mask from matrix estimation
            
        Returns:
            Pose recovery results
        """
        if essential_matrix is None:
            return {'success': False, 'error': 'No essential matrix provided'}
        
        # Use inlier points if mask is provided
        if mask is not None:
            inlier_indices = mask.ravel() == 1
            pts1_inliers = pts1[inlier_indices]
            pts2_inliers = pts2[inlier_indices]
        else:
            pts1_inliers = pts1
            pts2_inliers = pts2
        
        if len(pts1_inliers) < 5:
            return {'success': False, 'error': 'Not enough inlier points'}
        
        camera_matrix_avg = (camera_matrices[0] + camera_matrices[1]) / 2
        # Recover pose using OpenCV
        num_inliers, R, t, pose_mask = cv2.recoverPose(
            essential_matrix, pts1_inliers, pts2_inliers, camera_matrix_avg
        )
        
        # Calculate pose statistics
        baseline_length = np.linalg.norm(t)
        rotation_angle = self._rotation_angle_from_matrix(R)
        
        return {
            'success': True,
            'R': R,
            't': t,
            'num_inliers': num_inliers,
            'pose_inlier_mask': pose_mask,
            'baseline_length': baseline_length,
            'rotation_angle_deg': rotation_angle,
            'method': 'essential_matrix'
        }
    
    def recover_from_fundamental(self, fundamental_matrix: np.ndarray,
                               pts1: np.ndarray, pts2: np.ndarray, camera_matrix: np.ndarray,
                               mask: Optional[np.ndarray] = None) -> Dict:
        """
        Recover camera pose from fundamental matrix
        
        Args:
            fundamental_matrix: Fundamental matrix
            pts1: Points in first image
            pts2: Points in second image
            mask: Inlier mask from matrix estimation
            
        Returns:
            Pose recovery results
        """
        if fundamental_matrix is None:
            return {'success': False, 'error': 'No fundamental matrix provided'}
        
        # Convert fundamental to essential matrix
        essential_matrix = camera_matrix.T @ fundamental_matrix @ camera_matrix
        
        # Use essential matrix recovery
        result = self.recover_from_essential(essential_matrix, pts1, pts2, mask)
        
        if result['success']:
            result['method'] = 'fundamental_to_essential'
            result['fundamental_matrix'] = fundamental_matrix
            result['essential_matrix'] = essential_matrix
        
        return result
    
    def _rotation_angle_from_matrix(self, R: np.ndarray) -> float:
        """
        Calculate rotation angle from rotation matrix
        
        Args:
            R: Rotation matrix
            
        Returns:
            Rotation angle in degrees
        """
        trace = np.trace(R)
        angle = np.arccos(np.clip((trace - 1) / 2, -1, 1))
        return np.degrees(angle)

class PnPSolver:
    """Perspective-n-Point solver for adding new views"""
        
    def solve_pnp(self, points_3d: np.ndarray, points_2d: np.ndarray, camera_matrix: np.ndarray,
                  method: str = 'RANSAC',
                  reprojection_error: float = 5.0,
                  confidence: float = 0.99,
                  max_iterations: int = 1000) -> Dict:
        """
        Solve PnP problem to estimate camera pose
        
        Args:
            points_3d: 3D points in world coordinates (Nx3)
            points_2d: Corresponding 2D points in image (Nx2)
            method: PnP method ('RANSAC', 'ITERATIVE', 'P3P', 'EPNP')
            reprojection_error: RANSAC reprojection error threshold
            confidence: RANSAC confidence
            max_iterations: Maximum RANSAC iterations
            
        Returns:
            PnP solution results
        """
        if len(points_3d) < 4:
            return {'success': False, 'error': 'Need at least 4 3D-2D correspondences'}
        
        # Reshape points for OpenCV
        points_3d_cv = points_3d.reshape(-1, 1, 3).astype(np.float32)
        points_2d_cv = points_2d.reshape(-1, 1, 2).astype(np.float32)
        
        # Run the debug
        self.debug_pnp_data(points_3d_cv, points_2d_cv, camera_matrix)

        if method == 'RANSAC':
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                points_3d_cv, points_2d_cv,
                camera_matrix, None,  # No distortion
                confidence=0.90,
                reprojectionError=reprojection_error,
                iterationsCount=max_iterations
            )
            
            # Convert rotation vector to matrix
            
            if success and rvec is not None:
                R = cv2.Rodrigues(rvec)[0]
                t = tvec.reshape(3, 1)
                
                num_inliers = len(inliers) if inliers is not None else 0
                inlier_ratio = num_inliers / len(points_3d)
                
                # Calculate reprojection errors
                reproj_errors = self._calculate_reprojection_errors(
                    points_3d, points_2d, R, t, inliers
                )
                
                return {
                    'success': True,
                    'R': R,
                    't': t,
                    'rvec': rvec,
                    'tvec': tvec,
                    'inliers': inliers,
                    'num_inliers': num_inliers,
                    'inlier_ratio': inlier_ratio,
                    'mean_reprojection_error': reproj_errors['mean'],
                    'max_reprojection_error': reproj_errors['max'],
                    'method': 'PnP_RANSAC'
                }
            else:
                return {'success': False, 'error': 'PnP RANSAC failed'}
        
        elif method == 'ITERATIVE':
            success, rvec, tvec = cv2.solvePnP(
                points_3d_cv, points_2d_cv,
                camera_matrix, None,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            if success:
                R = cv2.Rodrigues(rvec)[0]
                t = tvec.reshape(3, 1)
                
                # Calculate reprojection errors for all points
                reproj_errors = self._calculate_reprojection_errors(
                    points_3d, points_2d, R, t
                )
                
                return {
                    'success': True,
                    'R': R,
                    't': t,
                    'rvec': rvec,
                    'tvec': tvec,
                    'mean_reprojection_error': reproj_errors['mean'],
                    'max_reprojection_error': reproj_errors['max'],
                    'method': 'PnP_ITERATIVE'
                }
            else:
                return {'success': False, 'error': 'PnP iterative failed'}
        
        elif method == 'P3P':
            success, rvecs, tvecs = cv2.solveP3P(
                points_3d_cv[:3], points_2d_cv[:3],  # Only first 3 points
                camera_matrix, None,
                flags=cv2.SOLVEPNP_P3P
            )
            
            if success and len(rvecs) > 0:
                # Choose best solution based on reprojection error
                best_solution = None
                best_error = float('inf')
                
                for rvec, tvec in zip(rvecs, tvecs):
                    R = cv2.Rodrigues(rvec)[0]
                    t = tvec.reshape(3, 1)
                    
                    reproj_errors = self._calculate_reprojection_errors(
                        points_3d, points_2d, R, t
                    )
                    
                    if reproj_errors['mean'] < best_error:
                        best_error = reproj_errors['mean']
                        best_solution = {
                            'success': True,
                            'R': R,
                            't': t,
                            'rvec': rvec,
                            'tvec': tvec,
                            'mean_reprojection_error': reproj_errors['mean'],
                            'max_reprojection_error': reproj_errors['max'],
                            'method': 'PnP_P3P',
                            'num_solutions': len(rvecs)
                        }
                
                return best_solution if best_solution else {'success': False, 'error': 'No valid P3P solution'}
            else:
                return {'success': False, 'error': 'P3P failed'}
        
        else:
            return {'success': False, 'error': f'Unknown PnP method: {method}'}
    
    def _calculate_reprojection_errors(self, points_3d: np.ndarray, points_2d: np.ndarray, camera_matrix: np.ndarray,
                                     R: np.ndarray, t: np.ndarray,
                                     inliers: Optional[np.ndarray] = None) -> Dict:
        """
        Calculate reprojection errors
        
        Args:
            points_3d: 3D points
            points_2d: 2D points
            R: Rotation matrix
            t: Translation vector
            inliers: Inlier indices (if available)
            
        Returns:
            Reprojection error statistics
        """
        # Project 3D points to image
        rvec = cv2.Rodrigues(R)[0]
        projected_points, _ = cv2.projectPoints(
            points_3d.reshape(-1, 1, 3), rvec, t, camera_matrix, None
        )
        projected_points = projected_points.reshape(-1, 2)
        
        # Calculate errors
        errors = np.linalg.norm(projected_points - points_2d, axis=1)
        
        if inliers is not None:
            inlier_errors = errors[inliers.ravel()]
            return {
                'mean': np.mean(inlier_errors),
                'max': np.max(inlier_errors),
                'std': np.std(inlier_errors),
                'all_errors': errors,
                'inlier_errors': inlier_errors
            }
        else:
            return {
                'mean': np.mean(errors),
                'max': np.max(errors),
                'std': np.std(errors),
                'all_errors': errors
            }

    def debug_pnp_data(self, points_3d_cv, points_2d_cv, camera_matrix):
        print("=== Complete PnP Debug ===")
        
        # Basic checks
        print(f"Number of correspondences: {len(points_3d_cv)}")
        print(f"3D shape: {points_3d_cv.shape}, dtype: {points_3d_cv.dtype}")
        print(f"2D shape: {points_2d_cv.shape}, dtype: {points_2d_cv.dtype}")
        
        # Check for required format
        if points_3d_cv.shape[1] != 1 or points_3d_cv.shape[2] != 3:
            print("❌ 3D points wrong format - should be (N, 1, 3)")
            points_3d_cv = points_3d_cv.reshape(-1, 1, 3)
            
        if points_2d_cv.shape[1] != 1 or points_2d_cv.shape[2] != 2:
            print("❌ 2D points wrong format - should be (N, 1, 2)")
            points_2d_cv = points_2d_cv.reshape(-1, 1, 2)
        
        # Check for NaN/inf
        if np.any(np.isnan(points_3d_cv)) or np.any(np.isinf(points_3d_cv)):
            print("❌ NaN/inf in 3D points")
            return False
            
        if np.any(np.isnan(points_2d_cv)) or np.any(np.isinf(points_2d_cv)):
            print("❌ NaN/inf in 2D points")
            return False
        
        # Check 2D bounds
        x_coords = points_2d_cv[:, 0, 0]
        y_coords = points_2d_cv[:, 0, 1]
        
        print(f"2D point ranges: X=[{np.min(x_coords):.1f}, {np.max(x_coords):.1f}], Y=[{np.min(y_coords):.1f}, {np.max(y_coords):.1f}]")
        
        if np.min(x_coords) < 0 or np.max(x_coords) > 4300:
            print("❌ 2D X coordinates outside image bounds [0, 4300]")
            
        if np.min(y_coords) < 0 or np.max(y_coords) > 6000:
            print("❌ 2D Y coordinates outside image bounds [0, 6000]")
        
        # Check 3D distribution
        z_coords = points_3d_cv[:, 0, 2]
        print(f"3D Z ranges: [{np.min(z_coords):.2f}, {np.max(z_coords):.2f}]")
        
        if np.any(z_coords <= 0):
            print(f"❌ {np.sum(z_coords <= 0)} points behind/at camera (Z <= 0)")
            
        # Check correspondence quality by projecting manually
        print("\n=== Manual Projection Test ===")
        self.test_correspondences(points_3d_cv[:5], points_2d_cv[:5], camera_matrix)
        
        return True

    @staticmethod
    def test_correspondences(points_3d, points_2d, K):
        """Test if 2D points could plausibly be projections of 3D points"""
        for i in range(len(points_3d)):
            p3d = points_3d[i, 0]  # [X, Y, Z]
            p2d_observed = points_2d[i, 0]  # [u, v]
            
            # Simple projection: [u, v] = K * [X/Z, Y/Z, 1]
            if p3d[2] > 0:  # Point in front of camera
                projected_x = K[0,0] * (p3d[0] / p3d[2]) + K[0,2]
                projected_y = K[1,1] * (p3d[1] / p3d[2]) + K[1,2]
                
                error_x = abs(projected_x - p2d_observed[0])
                error_y = abs(projected_y - p2d_observed[1])
                
                print(f"Correspondence {i}:")
                print(f"  3D: ({p3d[0]:.2f}, {p3d[1]:.2f}, {p3d[2]:.2f})")
                print(f"  2D observed: ({p2d_observed[0]:.1f}, {p2d_observed[1]:.1f})")
                print(f"  2D projected: ({projected_x:.1f}, {projected_y:.1f})")
                print(f"  Error: ({error_x:.1f}, {error_y:.1f}) pixels")
                
                if error_x > 100 or error_y > 100:
                    print("  ❌ LARGE PROJECTION ERROR - Bad correspondence!")
            else:
                print(f"Correspondence {i}: ❌ Point behind camera (Z={p3d[2]:.2f})")




class PoseValidation:
    """Validation of estimated camera poses"""
    
    @staticmethod
    def validate_pose(R: np.ndarray, t: np.ndarray,
                     baseline_range: Tuple[float, float] = (0.01, 10.0),
                     rotation_range: Tuple[float, float] = (0.1, 180.0)) -> Dict:
        """
        Validate estimated camera pose
        
        Args:
            R: Rotation matrix
            t: Translation vector
            baseline_range: Valid baseline length range
            rotation_range: Valid rotation angle range (degrees)
            
        Returns:
            Validation results
        """
        validation_results = {
            'valid': True,
            'warnings': [],
            'errors': []
        }
        
        # Check rotation matrix properties
        if not PoseValidation._is_valid_rotation_matrix(R):
            validation_results['valid'] = False
            validation_results['errors'].append('Invalid rotation matrix')
        
        # Check baseline length
        baseline = np.linalg.norm(t)
        if not (baseline_range[0] <= baseline <= baseline_range[1]):
            validation_results['warnings'].append(
                f'Baseline length {baseline:.3f} outside expected range {baseline_range}'
            )
        
        # Check rotation angle
        trace = np.trace(R)
        angle = np.degrees(np.arccos(np.clip((trace - 1) / 2, -1, 1)))
        
        if not (rotation_range[0] <= angle <= rotation_range[1]):
            validation_results['warnings'].append(
                f'Rotation angle {angle:.1f}° outside expected range {rotation_range}'
            )
        
        # Check for degenerate cases
        if baseline < 1e-6:
            validation_results['errors'].append('Degenerate case: zero baseline')
            validation_results['valid'] = False
        
        if angle < 0.1:
            validation_results['warnings'].append('Very small rotation angle - may be unstable')
        
        return validation_results
    
    @staticmethod
    def _is_valid_rotation_matrix(R: np.ndarray, tolerance: float = 1e-3) -> bool:
        """
        Check if matrix is a valid rotation matrix
        
        Args:
            R: Matrix to check
            tolerance: Numerical tolerance
            
        Returns:
            True if valid rotation matrix
        """
        # Check if matrix is 3x3
        if R.shape != (3, 3):
            return False
        
        # Check if R^T * R = I
        should_be_identity = R.T @ R
        identity = np.eye(3)
        
        if not np.allclose(should_be_identity, identity, atol=tolerance):
            return False
        
        # Check if det(R) = 1
        det = np.linalg.det(R)
        if not np.isclose(det, 1.0, atol=tolerance):
            return False
        
        return True

def recover_pose_auto(matrix: np.ndarray, matrix_type: str,
                     pts1: np.ndarray, pts2: np.ndarray,
                     camera_matrix: np.ndarray,
                     mask: Optional[np.ndarray] = None) -> Dict:
    """
    Automatically recover pose from essential or fundamental matrix
    
    Args:
        matrix: Essential or fundamental matrix
        matrix_type: 'essential' or 'fundamental'
        pts1: Points in first image
        pts2: Points in second image
        camera_matrix: Camera intrinsic matrix
        mask: Inlier mask
        
    Returns:
        Pose recovery results
    """
    pose_recovery = PoseRecovery(camera_matrix)
    
    if matrix_type == 'essential':
        result = pose_recovery.recover_from_essential(matrix, pts1, pts2, mask)
    elif matrix_type == 'fundamental':
        result = pose_recovery.recover_from_fundamental(matrix, pts1, pts2, mask)
    else:
        return {'success': False, 'error': f'Unknown matrix type: {matrix_type}'}
    
    # Add pose validation
    if result['success']:
        validation = PoseValidation.validate_pose(result['R'], result['t'])
        result['validation'] = validation
    
    return result