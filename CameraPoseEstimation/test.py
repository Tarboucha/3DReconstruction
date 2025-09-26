def _initialize_two_view_with_ba(self, chosen_images=None):
        """
        Initialize reconstruction with two views, including:
        1. Initial triangulation with conservative thresholds
        2. Bundle adjustment to optimize cameras and points
        3. Re-triangulation of initially rejected points with relaxed thresholds
        4. Validation and filtering of all points
        5. Bootstrap triangulation with unprocessed images
        """
        
        # Initialize Reconstruction object
        self.reconstruction = Reconstruction()
        
        # 1. Select best initialization pair
        if chosen_images is None:
            best_pair = self.pair_selector.get_best_pair_for_pipeline(self.matches_pickle['matches_data'])
        else: 
            best_pair = self.pair_selector.get_selected_pair_for_pipeline(
                self.matches_pickle['matches_data'], selected_pair=chosen_images
            )
        
        image1, image2 = best_pair['image_pair']
        
        # 2. Initial essential matrix estimation (for initial camera matrices)
        print("\n=== INITIAL ESSENTIAL MATRIX ESTIMATION ===")
        essential_result = self.essential_estimator.estimate(
            best_pair['pts1'], best_pair['pts2'], 
            image_size1=self.matches_pickle['image_info'][image1]['size'], 
            image_size2=self.matches_pickle['image_info'][image2]['size']
        )
        
        if not essential_result['success']:
            print(f"Initial estimation failed: {essential_result.get('error', 'Unknown error')}")
            print("Attempting with default camera matrices...")
            
            # Fallback: use default camera matrices
            image_size1 = self.matches_pickle['image_info'][image1]['size']
            image_size2 = self.matches_pickle['image_info'][image2]['size']
            K1_init = self.essential_estimator.estimate_camera_matrix(image_size1)
            K2_init = self.essential_estimator.estimate_camera_matrix(image_size2)
        else:
            # Use initial estimate as starting point
            K1_init = essential_result['camera_matrices'][0]
            K2_init = essential_result['camera_matrices'][1]
            
            print(f"Initial essential matrix estimation:")
            print(f"  Method: {essential_result['method']}")
            print(f"  Points: {essential_result['num_points']}")
            print(f"  Inliers: {essential_result['num_inliers']} ({essential_result['inlier_ratio']:.2%})")
        
        # 3. ITERATIVE REFINEMENT - This is the key addition
        print("\n=== ITERATIVE REFINEMENT ===")
        
        # Get inlier points from initial estimation (or use all if initial failed)
        if essential_result['success']:
            mask = essential_result['inlier_mask'].ravel().astype(bool)
            inlier_pts1 = best_pair['pts1'][mask]
            inlier_pts2 = best_pair['pts2'][mask]
        else:
            inlier_pts1 = best_pair['pts1']
            inlier_pts2 = best_pair['pts2']
                
        original_inlier_pts1 = inlier_pts1.copy()
        original_inlier_pts2 = inlier_pts2.copy()

        # Run iterative refinement
        refinement_result = self.iterative_adjuster.iterative_refinement_with_relaxation(
            inlier_pts1, inlier_pts2,
            K1_init=K1_init,
            K2_init=K2_init,
            image_size1=self.matches_pickle['image_info'][image1]['size'],
            image_size2=self.matches_pickle['image_info'][image2]['size']
        )
        
        triangulated_mask = np.zeros(len(original_inlier_pts1), dtype=bool)

        # 4. Use refined results or fall back to initial
        if refinement_result and refinement_result.get('success'):
            print("\n✓ Using refined camera matrices and pose")
            
            if essential_result['success']:
                initial_inlier_mask = essential_result['inlier_mask'].ravel().astype(bool)
            else:
                initial_inlier_mask = np.ones(len(best_pair['pts1']), dtype=bool)
            
            # Now check which inliers made it through refinement
            if 'final_mask' in refinement_result:
                # This mask is relative to inlier_pts1/pts2
                refinement_mask = refinement_result['final_mask']
                inlier_indices = np.where(initial_inlier_mask)[0]
                for i, survived in enumerate(refinement_mask):
                    if survived and i < len(inlier_indices):
                        final_point_mask[inlier_indices[i]] = True
            else:
                # Fallback: mark the points we have as valid
                inlier_indices = np.where(initial_inlier_mask)[0]
                for i in range(min(len(valid_pts1), len(inlier_indices))):
                    final_point_mask[inlier_indices[i]] = True

            # Extract refined parameters
            K1_final = refinement_result['K1']
            K2_final = refinement_result['K2']
            R_final = refinement_result['R']
            t_final = refinement_result['t']
            
            # Use refined points if available
            if 'points_3d' in refinement_result:
                points_3d = refinement_result['points_3d']
                valid_pts1 = refinement_result.get('valid_pts1', inlier_pts1[:points_3d.shape[1]])
                valid_pts2 = refinement_result.get('valid_pts2', inlier_pts2[:points_3d.shape[1]])
                initial_point_count = points_3d.shape[1]
                
                if 'mask_inliers' in refinement_result:
                    # The refinement mask tells us which of the inlier points were successfully triangulated
                    refinement_mask = refinement_result['mask_inliers']
                    # Map back to original inliers
                    triangulated_indices = np.where(refinement_mask)[0]
                    for idx in triangulated_indices[:initial_point_count]:
                        if idx < len(triangulated_mask):
                            triangulated_mask[idx] = True
                else:
                    # Assume first N points were triangulated
                    triangulated_mask[:initial_point_count] = True


                print(f"Refined triangulation: {initial_point_count} points")
                print(f"Refined K1 focal: {K1_final[0,0]:.1f}")
                print(f"Refined K2 focal: {K2_final[0,0]:.1f}")
                
                # Skip additional triangulation since refinement already did it
                skip_triangulation = True
            else:
                skip_triangulation = False
                
            # Update essential result with refined values
            essential_result['camera_matrices'] = [K1_final, K2_final]
            essential_result['camera_estimated'] = [True, True]
            
            # Create pose result with refined values
            pose_result = {
                'R': R_final,
                't': t_final,
                'success': True
            }
            
        else:
            print("\n⚠ Refinement failed, falling back to initial estimation")
            
            if not essential_result['success']:
                raise RuntimeError("Both initial estimation and refinement failed")
            
            # Fall back to initial estimation
            skip_triangulation = False
            
            # Recover pose from initial essential matrix
            pose_result = self.pose_recovery.recover_from_essential(
                essential_result['essential_matrix'],
                inlier_pts1, 
                inlier_pts2,
                essential_result['camera_matrices']
            )
            
            K1_final = essential_result['camera_matrices'][0]
            K2_final = essential_result['camera_matrices'][1]
            R_final = pose_result['R']
            t_final = pose_result['t']
        
        # 5. Add cameras to reconstruction with final parameters
        print("\n=== ADDING CAMERAS TO RECONSTRUCTION ===")
        
        # First camera at origin
        self.reconstruction.add_camera(
            camera_id=image1,
            R=np.eye(3),
            t=np.zeros((3,1)),
            K=K1_final
        )
        
        # Second camera with relative pose
        self.reconstruction.add_camera(
            camera_id=image2,
            R=R_final,
            t=t_final,
            K=K2_final
        )
        
        print(f"Added camera {image1} at origin")
        print(f"Added camera {image2} with relative pose")
        
        # 6. Handle triangulation (if not already done by refinement)
        if not skip_triangulation:
            print("\n=== TRIANGULATION ===")
            print(f"Triangulating {len(inlier_pts1)} inlier correspondences...")
            
            triangulated_result = self.triangulation_engine.triangulate_initial_points(
                inlier_pts1,
                inlier_pts2,
                np.eye(3),
                np.zeros((3,1)),
                R_final,
                t_final,
                [K1_final, K2_final],
                best_pair['image_pair']
            )
            
            # Extract triangulated points
            if isinstance(triangulated_result, dict) and 'points_3d' in triangulated_result:
                points_3d = triangulated_result['points_3d']
                initial_stats = triangulated_result.get('statistics', {})
            else:
                points_3d = triangulated_result
                initial_stats = {}
            
            initial_point_count = points_3d.shape[1]
            valid_pts1 = inlier_pts1[:initial_point_count]  
            valid_pts2 = inlier_pts2[:initial_point_count]
            
            triangulated_mask[:initial_point_count] = True

            print(f"Successfully triangulated: {initial_point_count}/{len(inlier_pts1)} points")
            print(f"Initial success rate: {100.0 * initial_point_count / len(inlier_pts1):.1f}%")
        
        # 7. Add points and observations to reconstruction
        for i in range(points_3d.shape[1]):
            point_id = self.reconstruction.add_point(points_3d[:, i])
            
            # Add observations
            if i < len(valid_pts1):
                self.reconstruction.add_observation(image1, point_id, valid_pts1[i])
                self.reconstruction.add_observation(image2, point_id, valid_pts2[i])
        
        # 8. Store initialization info
        self.reconstruction.initialization_info = {
            'method': 'iterative_refinement' if refinement_result and refinement_result.get('success') else essential_result.get('method', 'unknown'),
            'num_inliers': len(inlier_pts1),
            'initial_triangulated': initial_point_count,
            'initial_rejected': len(inlier_pts1) - initial_point_count,
            'refinement_iterations': refinement_result.get('iteration', 0) if refinement_result else 0,
            'final_score': refinement_result.get('score', 0) if refinement_result else 0,
            'camera_estimated': [True, True],  # Both were estimated/refined
            'quality_assessment': essential_result.get('quality_assessment', {})
        }
        
        all_original_correspondences = {
            'pts1': best_pair['pts1'].copy(),
            'pts2': best_pair['pts2'].copy(),
            'inlier_pts1': original_inlier_pts1.copy(),
            'inlier_pts2': original_inlier_pts2.copy(),
            'triangulated_mask': triangulated_mask,  # Now properly defined
            'image1': image1,
            'image2': image2
        }

        # 10. Convert to legacy format for bundle adjustment
        reconstruction_state = self.reconstruction.to_legacy_format()
        
        print(f"\n=== INITIALIZATION SUMMARY ===")
        print(f"Cameras: {len(self.reconstruction.cameras)}")
        print(f"3D Points: {len(self.reconstruction.points)}")
        print(f"Observations: {len(self.reconstruction.observations)}")
        
        # 10. INITIAL BUNDLE ADJUSTMENT
        print("\n=== BUNDLE ADJUSTMENT ===")
        print(f"Optimizing {len(self.reconstruction.cameras)} cameras and {initial_point_count} points...")
        
        reconstruction_state = self.incremental_bundle_adjuster.adjust_after_new_view(
            reconstruction_state,
            image2,
            optimize_intrinsics=essential_result['camera_estimated']
        )
        
        print("Bundle adjustment complete")
        
        # 11. RE-TRIANGULATE INITIALLY REJECTED POINTS
        print("\n=== RE-TRIANGULATION WITH OPTIMIZED CAMERAS ===")
        
        num_rejected = len(inlier_pts1) - initial_point_count
        if num_rejected > 0:
            print(f"Re-evaluating {num_rejected} initially rejected points...")
            
            # Get points that were initially rejected
            rejected_indices = np.where(~triangulated_mask)[0]
            rejected_pts1 = inlier_pts1[rejected_indices]
            rejected_pts2 = inlier_pts2[rejected_indices]
            
            # Get optimized camera parameters
            cameras = reconstruction_state['cameras']
            cam1 = cameras[image1]
            cam2 = cameras[image2]
            
            # Calculate adaptive depth bounds based on existing points
            existing_points = reconstruction_state['points_3d']['points_3d']
            min_depth, max_depth = self._calculate_adaptive_depth_bounds(existing_points, cameras)
            
            # Create triangulation engine with relaxed thresholds
            retriangulation_engine = TriangulationEngine(
                min_triangulation_angle_deg=0.5,  # Relaxed from 2.0
                max_reprojection_error=4.0,       # Relaxed from 2.0
                min_depth=min_depth,               # Adaptive
                max_depth=max_depth                # Adaptive
            )
            
            # Re-triangulate with optimized cameras
            retriangulated_result = retriangulation_engine.triangulate_initial_points(
                pts1=rejected_pts1,
                pts2=rejected_pts2,
                R1=cam1['R'],
                t1=cam1['t'],
                R2=cam2['R'],
                t2=cam2['t'],
                K=[cam1.get('K', essential_result['camera_matrices'][0]),
                cam2.get('K', essential_result['camera_matrices'][1])],
                image_pair=(image1, image2)
            )
            
            recovered_points = retriangulated_result['points_3d']
            num_recovered = recovered_points.shape[1] if recovered_points.size > 0 else 0
            
            print(f"Successfully recovered: {num_recovered}/{num_rejected} points")
            print(f"Recovery rate: {100.0 * num_recovered / num_rejected:.1f}%")
            
            # Add recovered points to reconstruction state
            if num_recovered > 0:
                current_points = reconstruction_state['points_3d']['points_3d']
                current_point_count = current_points.shape[1]
                
                # Append recovered points
                updated_points = np.hstack([current_points, recovered_points])
                reconstruction_state['points_3d']['points_3d'] = updated_points
                
                # Add observations for recovered points
                observations = reconstruction_state.get('observations', {})
                if image1 not in observations:
                    observations[image1] = []
                if image2 not in observations:
                    observations[image2] = []
                
                # Use the actual recovered 2D points (subset of rejected points that succeeded)
                if 'observations' in retriangulated_result:
                    # Extract which rejected points were successfully triangulated
                    for obs in retriangulated_result['observations']:
                        if obs['image_id'] == image1:
                            pt_idx = obs.get('point_id', 0)
                            if pt_idx < len(rejected_pts1):
                                observations[image1].append({
                                    'point_id': current_point_count + pt_idx,
                                    'image_point': rejected_pts1[pt_idx].tolist(),
                                    'source': 'retriangulation'
                                })
                        elif obs['image_id'] == image2:
                            pt_idx = obs.get('point_id', 0)
                            if pt_idx < len(rejected_pts2):
                                observations[image2].append({
                                    'point_id': current_point_count + pt_idx,
                                    'image_point': rejected_pts2[pt_idx].tolist(),
                                    'source': 'retriangulation'
                                })
                
                reconstruction_state['observations'] = observations
        
        # 12. VALIDATE ALL POINTS AFTER BA AND RE-TRIANGULATION
        print("\n=== POINT VALIDATION ===")
        total_points_before_validation = reconstruction_state['points_3d']['points_3d'].shape[1]
        print(f"Validating {total_points_before_validation} total points...")
        
        reconstruction_state = self._validate_points_after_ba(
            reconstruction_state,
            max_reprojection_error=3.0,
            min_triangulation_angle=1.0
        )
        
        # 13. Update reconstruction from validated state
        self._update_reconstruction_from_validated_state(reconstruction_state)
        
        points_after_validation = len(self.reconstruction.points)
        points_removed = total_points_before_validation - points_after_validation
        
        print(f"Validation complete:")
        print(f"  Kept: {points_after_validation} points")
        print(f"  Removed: {points_removed} points")
        
        # 14. Save checkpoint after initialization
        print("\n=== SAVING CHECKPOINT ===")
        with open('saved_variable.pkl', 'wb') as f:
            pickle.dump(self.reconstruction.to_legacy_format(), f)
        print("Checkpoint saved to saved_variable.pkl")
        
        # 15. BOOTSTRAP TRIANGULATION WITH UNPROCESSED IMAGES
        print("\n=== BOOTSTRAP PROGRESSIVE TRIANGULATION ===")
        
        processed_images = set(self.reconstruction.cameras.keys())
        all_images = set(self.matches_pickle['image_info'].keys())
        bootstrap_images = all_images - processed_images
        
        print(f"Bootstrapping with {len(bootstrap_images)} additional unprocessed images...")
        
        if len(bootstrap_images) > 0:
            # Convert to legacy format for bootstrap
            reconstruction_state = self.reconstruction.to_legacy_format()
            
            bootstrap_points_added = self._bootstrap_triangulate_with_both_cameras(
                image1, image2, reconstruction_state, bootstrap_images
            )
            
            # Update reconstruction from bootstrap results
            self._update_reconstruction_from_state(reconstruction_state)
            
            final_points = len(self.reconstruction.points)
            
            print(f"\nBootstrap triangulation complete:")
            print(f"  Points before bootstrap: {points_after_validation}")
            print(f"  Bootstrap points added: {bootstrap_points_added}")
            print(f"  Total points: {final_points}")
            
            # Run BA again if we added significant points
            if bootstrap_points_added > 50:
                print("\nRunning bundle adjustment after bootstrap...")
                reconstruction_state = self.reconstruction.to_legacy_format()
                reconstruction_state = self.incremental_bundle_adjuster.adjust_after_new_view(
                    reconstruction_state,
                    image1,
                    optimize_intrinsics=False
                )
                self._update_reconstruction_from_state(reconstruction_state)
                print("Post-bootstrap BA complete")
        
        # 16. Final summary
        print("\n" + "="*60)
        print("INITIALIZATION COMPLETE")
        print("="*60)
        final_stats = self.reconstruction.get_statistics()
        print(f"Final reconstruction statistics:")
        print(f"  Cameras: {final_stats['num_cameras']}")
        print(f"  3D Points: {final_stats['num_points']}")
        print(f"  Observations: {final_stats['num_observations']}")
        print(f"  Mean track length: {final_stats['mean_track_length']:.2f}")
        
        # Summary of point sources
        init_info = self.reconstruction.initialization_info
        print(f"\nPoint source breakdown:")
        print(f"  Initial triangulation: {init_info.get('initial_triangulated', 0)}")
        print(f"  Recovered after BA: {num_recovered if num_rejected > 0 else 0}")
        print(f"  Bootstrap triangulation: {bootstrap_points_added if len(bootstrap_images) > 0 else 0}")
        print(f"  Removed by validation: {points_removed}")
        print("="*60)