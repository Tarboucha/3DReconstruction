"""
Feature Processing Pipeline

Main pipeline for feature detection and matching with batch processing support.
"""

import time
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path
import cv2
import pickle
import gc
import json

from .core_data_structures import FeatureData, MatchData
from .result_types import (
    MatchingResult, 
    MethodResult, 
    ImagePairInfo, 
    ProcessingMetadata, 
    create_method_result,

)

from .result_converters import (
    save_for_reconstruction
    )

from .matcher_factory import MatcherFactory 
from .utils import (
    calculate_reprojection_error,
    adaptive_match_filtering, 
    enhanced_filter_matches_with_homography
)


class FeatureProcessingPipeline:
    """
    Feature detection and matching pipeline with batch processing
    
    Features:
    - Multi-method feature detection (SIFT, ORB, ALIKED, etc.)
    - Batch processing with smart image caching
    - Checkpointing and resume support
    - Memory efficient (bounded RAM usage)
    - Automatic result saving
    
    Usage:
        >>> pipeline = create_pipeline('balanced')
        >>> 
        >>> # Single pair
        >>> result = pipeline.match(img1, img2)
        >>> 
        >>> # Batch process folder
        >>> results = pipeline.match_folder(
        ...     './images',
        ...     output_dir='./output',
        ...     auto_save=True,
        ...     batch_size=10
        ... )
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize pipeline with configuration
        
        Args:
            config: Configuration dictionary containing:
                - methods: List of method names (e.g., ['SIFT', 'ALIKED'])
                - max_features: Maximum features per method
                - detector_params: Method-specific parameters
                - matcher_config: Matcher configuration per method
                - filtering: Filtering configuration
                - combine_strategy: How to combine multi-method results
        """
        self.config = config
        
        # Force 'independent' strategy for multi-method pipelines
        combine_strategy = config.get('combine_strategy', 'independent')
        if len(config.get('methods', [])) > 1 and combine_strategy not in ['independent', 'best']:
            print(f"⚠️  combine_strategy '{combine_strategy}' not suitable for multi-method")
            print("   Using 'independent' strategy")
            combine_strategy = 'independent'
            config['combine_strategy'] = combine_strategy
        
        # Import here to avoid circular dependencies
        from .multi_method_detector import MultiMethodFeatureDetector
        
        # Initialize detectors
        self.multi_detector = MultiMethodFeatureDetector(
            methods=config.get('methods', ['SIFT']),
            max_features_per_method=config.get('max_features', 2000),
            detector_params=config.get('detector_params', {}),
            combine_strategy=combine_strategy
        )
        
        # Initialize matchers
        self.matchers = {}
        matcher_config = config.get('matcher_config', {})
        matcher_factory = MatcherFactory()
        for method in config.get('methods', ['SIFT']):
            matcher_type = matcher_config.get(method, None)
            self.matchers[method] = matcher_factory.create_matcher(method, matcher_type)
        
        self.filtering_config = config.get('filtering', {})
    
    
    def match(
        self,
        image1: np.ndarray,
        image2: np.ndarray,
        image1_id: str = "image1",
        image2_id: str = "image2",
        image1_size: Optional[Tuple[int, int]] = None,
        image2_size: Optional[Tuple[int, int]] = None,
        filter_matches: bool = True,
        compute_geometry: bool = True,
        visualize: bool = False
    ) -> MatchingResult:
        """
        Match two images using configured methods
        
        Args:
            image1: First image (RGB numpy array)
            image2: Second image (RGB numpy array)
            image1_id: Identifier for first image
            image2_id: Identifier for second image
            image1_size: Original size before any resizing (width, height)
            image2_size: Original size before any resizing (width, height)
            filter_matches: Apply match filtering
            compute_geometry: Compute homography/fundamental matrix
            visualize: Show visualization (for debugging)
        
        Returns:
            MatchingResult containing matches from all methods
        
        Example:
            >>> result = pipeline.match(img1, img2)
            >>> best_method = result.get_best_method()
            >>> print(f"Best: {best_method} with {result.get_method(best_method).num_matches} matches")
        """
        # Get image sizes
        if image1_size is None:
            image1_size = (image1.shape[1], image1.shape[0])
        if image2_size is None:
            image2_size = (image2.shape[1], image2.shape[0])
        
        # Detect features with all methods
        start_time = time.time()
        multi_features1 = self.multi_detector.detect_all(image1)  
        multi_features2 = self.multi_detector.detect_all(image2) 
        detection_time = time.time() - start_time

        # Match with each method independently
        method_results = {}

        for method in self.config.get('methods', ['SIFT']):
            if method not in multi_features1:  
                continue
            
            features1 = multi_features1[method]  
            features2 = multi_features2[method] 
            
            # Match
            matcher = self.matchers[method]
            start_match = time.time()
            match_data = matcher.match(features1, features2)
            matching_time = time.time() - start_match

            # Apply filtering if requested
            if filter_matches and len(match_data.matches) > 0:
                match_data = self._apply_filtering(
                    match_data,
                    features1.keypoints,
                    features2.keypoints
                )

            # Get the best matches (filtered if available, otherwise raw)
            best_matches = match_data.get_best_matches()
            num_matches = len(best_matches)

            # Compute geometry if requested and we have enough matches
            homography = None
            fundamental_matrix = None
            inlier_ratio = None
            reprojection_error = None

            if compute_geometry and num_matches > 4:
                # Homography might already be computed during filtering
                homography = match_data.homography
                fundamental_matrix = match_data.fundamental_matrix
                
                # If homography exists, calculate additional metrics
                if homography is not None and num_matches > 0:
                    
                    errors = calculate_reprojection_error(
                        features1.keypoints,
                        features2.keypoints,
                        best_matches,
                        homography
                    )
                    
                    if len(errors) > 0:
                        inlier_threshold = self.filtering_config.get('ransac_threshold', 4.0)
                        inliers = np.sum(errors < inlier_threshold)
                        inlier_ratio = inliers / len(errors)
                        reprojection_error = float(np.mean(errors))
                
                # If no homography yet (filtering not applied or failed), compute it now
                elif num_matches >= 4:
                    try:
                        pts1 = np.float32([features1.keypoints[m.queryIdx].pt for m in best_matches])
                        pts2 = np.float32([features2.keypoints[m.trainIdx].pt for m in best_matches])
                        
                        H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 
                                                    self.filtering_config.get('ransac_threshold', 4.0))
                        if H is not None:
                            homography = H
                            match_data.homography = H
                            
                            # Calculate metrics
                            errors = calculate_reprojection_error(
                                features1.keypoints,
                                features2.keypoints,
                                best_matches,
                                homography
                            )
                            if len(errors) > 0:
                                inlier_threshold = self.filtering_config.get('ransac_threshold', 4.0)
                                inliers = np.sum(errors < inlier_threshold)
                                inlier_ratio = inliers / len(errors)
                                reprojection_error = float(np.mean(errors))
                            
                    except Exception as e:
                        print(f"  ⚠️  Homography computation failed: {e}")

            # Create MethodResult
            method_result = create_method_result(
                method_name=method,
                features1=features1,
                features2=features2,
                match_data=match_data,
                homography=homography,
                fundamental_matrix=fundamental_matrix,
                inlier_ratio=inlier_ratio,
                reprojection_error=reprojection_error,
                detection_time=detection_time / len(self.config.get('methods', ['SIFT'])),
                matching_time=matching_time
            )

            method_results[method] = method_result
                    
        # Create image pair info
        image_pair_info = ImagePairInfo(
            image1_id=image1_id,
            image2_id=image2_id,
            image1_size=image1_size,
            image2_size=image2_size,
            image1=image1 if visualize else None,
            image2=image2 if visualize else None
        )
        
        # Create processing metadata
        processing_metadata = ProcessingMetadata(
            methods_used=list(method_results.keys()),
            total_processing_time=time.time() - start_time,
            config_used=self.config.copy() 
        )
        
        # Create and return result
        result = MatchingResult(
            methods=method_results,
            image_pair_info=image_pair_info,
            processing_metadata=processing_metadata
        )
        
        # Visualize if requested
        if visualize:
            result.visualize()
        
        return result
    
    def _apply_filtering(
        self,
        match_data: MatchData,
        keypoints1: List[cv2.KeyPoint],
        keypoints2: List[cv2.KeyPoint]
    ) -> MatchData:
        """Apply match filtering"""
        
        use_adaptive = self.filtering_config.get('use_adaptive_filtering', True)
        ransac_threshold = self.filtering_config.get('ransac_threshold', 4.0)
        
        # Get the matches to filter
        matches_to_filter = match_data.get_best_matches()  # ✅ Get matches from MatchData
        
        if use_adaptive:
            # Adaptive filtering
            filtered_matches, H, filter_info = adaptive_match_filtering(
                keypoints1,
                keypoints2,
                matches_to_filter,  # ✅ Pass matches as list
                match_data,  # ✅ Pass MatchData separately
                ransac_threshold=ransac_threshold
            )
            
            if filtered_matches:
                match_data.filtered_matches = filtered_matches
                match_data.homography = H
        else:
            # Simple homography-based filtering
            filtered_matches, H = enhanced_filter_matches_with_homography(
                keypoints1,
                keypoints2,
                matches_to_filter,  # ✅ Pass matches as list
                match_data,
                ransac_threshold=ransac_threshold
            )
            
            if filtered_matches:
                match_data.filtered_matches = filtered_matches
                match_data.homography = H
        
        return match_data
    
    def match_folder(
        self,
        folder_path: str,
        pattern: str = '*.jpg',
        pair_mode: str = 'consecutive',
        resize_to: Optional[Tuple[int, int]] = None,
        
        # Batch Processing Parameters
        batch_size: int = 10,
        resume: bool = True,
        cache_size_mb: float = 500,
        
        # Auto-save Parameters
        output_dir: Optional[Union[str, Path]] = None,
        auto_save: bool = False,
        min_matches_for_save: int = 5,
        export_colmap: bool = True,
        save_visualizations: bool = False,
        
        # Processing Parameters
        save_metadata: bool = True,
        metadata_path: Optional[str] = None,
        filter_matches: bool = True,
        compute_geometry: bool = True,
        visualize: bool = False
    ) -> List[MatchingResult]:
        """
        Match images in folder with batch processing and smart caching
        
        Solution 3: Batch Loading with Smart Cache
        - Scans folder for metadata only (fast, low memory)
        - Loads images in batches (reuses images within batch)
        - Bounded memory usage (cache_size_mb controls max RAM)
        - Optimized for consecutive/first/all pair modes
        
        Memory Profile:
        - Old: 50GB for 1000 4K images (loads all upfront)
        - New: ~500MB (metadata + batch cache) - 100x improvement!
        
        Args:
            folder_path: Path to folder containing images
            pattern: Image file pattern (e.g., '*.jpg')
            pair_mode: 'consecutive', 'first', or 'all'
            resize_to: Optional (width, height) to resize images
            
            batch_size: Number of pairs to process per batch (default: 10)
            resume: Skip already-processed pairs (default: True)
            cache_size_mb: Maximum cache size in MB (default: 500)
            
            output_dir: Directory to save results
            auto_save: Enable automatic saving
            min_matches_for_save: Minimum matches to save
            export_colmap: Export COLMAP format
            save_visualizations: Save visualization images
            
        Returns:
            List[MatchingResult]
        
        Examples:
            >>> # Basic usage
            >>> results = pipeline.match_folder(
            ...     './images',
            ...     output_dir='./output',
            ...     auto_save=True,
            ...     batch_size=10
            ... )
            
            >>> # Resume after crash
            >>> results = pipeline.match_folder(
            ...     './images',
            ...     output_dir='./output',
            ...     resume=True
            ... )
        """
        from .image_manager import (
            FolderImageSource, 
            BatchImageLoader,
            create_pairs_from_metadata,
            analyze_batch_reuse
        )
        from .batch_processor import BatchProcessor
        
        # Validation
        if auto_save and output_dir is None:
            raise ValueError("output_dir must be specified when auto_save=True")
        
        # Setup output directory
        batch_processor = None
        if auto_save:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Test write permission
            test_file = output_dir / '.write_test'
            try:
                test_file.touch()
                test_file.unlink()
            except Exception as e:
                raise PermissionError(f"Cannot write to {output_dir}: {e}")
            
            # Create subdirectories
            (output_dir / 'matching_results').mkdir(exist_ok=True)
            (output_dir / 'reconstruction').mkdir(exist_ok=True)
            if save_visualizations:
                (output_dir / 'visualizations').mkdir(exist_ok=True)
            
            # Initialize checkpoint manager
            batch_processor = BatchProcessor(output_dir, batch_size=batch_size)
            
            if resume and batch_processor.completed_pairs:
                print(f"\n Resume: {len(batch_processor.completed_pairs)} pairs already done")
        
        # Scan folder for metadata
        print(f"\n{'='*70}")
        print(f"BATCH PROCESSING WITH SMART CACHING")
        print(f"{'='*70}")
        
        ext = pattern.replace('*', '').lower()
        if not ext.startswith('.'):
            ext = '.' + ext
        
        try:
            image_source = FolderImageSource(
                folder_path=folder_path,
                resize_to=resize_to,
                image_extensions=[ext]
            )
        except ValueError as e:
            raise ValueError(f"Error with folder: {e}")
        
        print(f"\n Scanning {folder_path}...")
        metadata_list = image_source.get_metadata_list()
        
        if not metadata_list:
            raise ValueError(f"No images found in {folder_path} with pattern {pattern}")
        
        print(f"✓ Found {len(metadata_list)} images")
        
        total_size_mb = sum(m.file_size_bytes for m in metadata_list) / (1024 * 1024)
        print(f"  Total size: {total_size_mb:.2f} MB ({total_size_mb/1024:.2f} GB)")
        print(f"  Metadata size: ~{len(metadata_list) * 0.5 / 1024:.2f} MB")
        
        # Save metadata
        if save_metadata and auto_save:
            meta_path = metadata_path or str(output_dir / 'image_metadata.pkl')
            try:
                metadata_dict = {
                    'folder_path': str(folder_path),
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'num_images': len(metadata_list),
                    'resize_to': resize_to,
                    'images': [
                        {
                            'identifier': m.identifier,
                            'filepath': str(m.filepath),
                            'file_size_bytes': m.file_size_bytes,
                            'source_type': m.source_type.value
                        }
                        for m in metadata_list
                    ]
                }
                with open(meta_path, 'wb') as f:
                    pickle.dump(metadata_dict, f)
                print(f" Metadata saved")
            except Exception as e:
                print(f"  Metadata save failed: {e}")
        
        # Create pairs from metadata
        print(f"\n🔗 Creating pairs...")
        pairs_metadata = create_pairs_from_metadata(metadata_list, pair_mode)
        total_pairs = len(pairs_metadata)
        
        if total_pairs == 0:
            print("  No pairs created")
            return []
        
        print(f"✓ Created {total_pairs} pairs ({pair_mode} mode)")
        
        # Initialize batch loader and tracking
        batch_loader = BatchImageLoader(cache_size_mb=cache_size_mb)
        
        all_results = []
        all_reconstruction_data = []
        
        stats = {
            'total_pairs': total_pairs,
            'processed': 0,
            'saved': 0,
            'skipped_insufficient_matches': 0,
            'skipped_already_done': 0,
            'failed': 0,
            'total_images_loaded': 0,
            'cache_hits': 0,
        }
        
        start_time = time.time()
        
        # Batch processing loop
        print(f"\n{'='*70}")
        print(f"PROCESSING")
        print(f"{'='*70}")
        print(f"Total pairs: {total_pairs}")
        print(f"Batch size: {batch_size}")
        print(f"Cache size: {cache_size_mb} MB")
        print(f"Auto-save: {'ON' if auto_save else 'OFF'}")
        if auto_save and resume:
            print(f"Resume: ON")
        print(f"{'='*70}")
        
        total_batches = (total_pairs + batch_size - 1) // batch_size
        
        for batch_start in range(0, total_pairs, batch_size):
            batch_end = min(batch_start + batch_size, total_pairs)
            batch_num = (batch_start // batch_size) + 1
            
            print(f"\n{'='*70}")
            print(f"BATCH {batch_num}/{total_batches}: Pairs {batch_start}-{batch_end-1}")
            print(f"{'='*70}")
            
            # Get metadata for this batch
            batch_pairs_metadata = pairs_metadata[batch_start:batch_end]
            
            # Analyze batch
            batch_analysis = analyze_batch_reuse(batch_pairs_metadata)
            print(f"  Unique images needed: {batch_analysis['unique_images']}")
            print(f"  Image reuse ratio: {batch_analysis['reuse_ratio']:.1%}")
            
            # Load images for this batch
            print(f"  Loading batch images...")
            cache_before = len(batch_loader.cache)
            num_loaded = batch_loader.load_batch(batch_pairs_metadata, resize_to=resize_to)
            cache_after = len(batch_loader.cache)
            
            stats['total_images_loaded'] += num_loaded
            stats['cache_hits'] += (batch_analysis['unique_images'] - num_loaded)
            
            print(f"  ✓ Loaded {num_loaded} new, {cache_after - num_loaded} from cache")
            print(f"  Cache: {batch_loader.get_cache_stats()}")
            
            # Process pairs in batch
            batch_results = []
            batch_recon_data = []
            
            for i in range(batch_start, batch_end):
                meta1, meta2 = pairs_metadata[i]
                pair_id = f"pair_{i:03d}"
                
                # Check if completed
                if auto_save and resume and batch_processor.is_completed(pair_id):
                    print(f"\n⏭  {pair_id}: Already completed")
                    stats['skipped_already_done'] += 1
                    continue
                
                print(f"\n{'-'*60}")
                print(f"[{i+1}/{total_pairs}] {pair_id}")
                print(f"  {meta1.identifier} → {meta2.identifier}")
                print(f"{'-'*60}")
                
                try:
                    # Get images from cache
                    img1_info = batch_loader.get_image(meta1.identifier)
                    img2_info = batch_loader.get_image(meta2.identifier)
                    
                    if img1_info is None or img2_info is None:
                        print(f" Images not in cache!")
                        stats['failed'] += 1
                        continue
                    
                    # Match
                    result = self.match(
                        img1_info.image,
                        img2_info.image,
                        image1_id=img1_info.identifier,
                        image2_id=img2_info.identifier,
                        image1_size=img1_info.metadata.get('original_size', img1_info.size),
                        image2_size=img2_info.metadata.get('original_size', img2_info.size),
                        filter_matches=filter_matches,
                        compute_geometry=compute_geometry,
                        visualize=visualize
                    )
                    
                    batch_results.append(result)
                    stats['processed'] += 1
                    
                    # Get stats
                    best_result = result.get_best()
                    if best_result:
                        num_matches = best_result.num_matches
                        best_method = best_result.method_name  # Get the name if you need it
                        print(f"  ✓ {best_method}: {num_matches} matches")
                    else:
                        num_matches = 0
                        print(f"  ⚠️  No matches found")
                    
                    
                    # Save if enough matches
                    if auto_save:
                        if num_matches < min_matches_for_save:
                            print(f" Skipped: {num_matches} < {min_matches_for_save}")
                            stats['skipped_insufficient_matches'] += 1
                        else:
                            try:
                                # Save MatchingResult
                                result_path = output_dir / 'matching_results' / f'{pair_id}_result.pkl'
                                result.save(str(result_path), format='pickle')
                                
                                # Save reconstruction
                                recon_dir = output_dir / 'reconstruction' / pair_id
                                save_for_reconstruction(result, recon_dir)
                                
                                # Save visualization
                                if save_visualizations:
                                    from .visualization import save_visualization
                                    viz_path = output_dir / 'visualizations' / f'{pair_id}_matches.png'
                                    save_visualization(result, str(viz_path))
                                
                                # Track
                                recon_data = result.to_reconstruction()
                                batch_recon_data.append((pair_id, recon_data))
                                stats['saved'] += 1
                                
                                # Checkpoint
                                batch_processor.save_progress(pair_id)
                                print(f"  ✓ Saved & checkpointed")
                                
                            except Exception as e:
                                print(f"  ⚠️  Save failed: {e}")
                
                except Exception as e:
                    print(f" Error: {e}")
                    stats['failed'] += 1
                    import traceback
                    traceback.print_exc()
                    continue
            
            # End of batch
            print(f"\n{'='*70}")
            print(f"BATCH {batch_num} COMPLETE")
            print(f"  Processed: {len(batch_results)}")
            if auto_save:
                print(f"  Saved: {len(batch_recon_data)}")
            print(f"{'='*70}")
            
            all_results.extend(batch_results)
            all_reconstruction_data.extend(batch_recon_data)
            
            # Clear batch (keep cache for next batch)
            del batch_results
            del batch_recon_data
            gc.collect()
            
            print(f"🧹 Batch memory cleared (cache retained)")
        
        # Cleanup
        batch_loader.clear_batch()
        print(f"\n🧹 Image cache cleared")
        
        total_time = time.time() - start_time
        
        # Cache efficiency
        print(f"\n Cache Efficiency:")
        print(f"  Images loaded from disk: {stats['total_images_loaded']}")
        print(f"  Cache hits (reused): {stats['cache_hits']}")
        total_refs = stats['total_images_loaded'] + stats['cache_hits']
        if total_refs > 0:
            print(f"  Cache hit rate: {100 * stats['cache_hits'] / total_refs:.1f}%")
        
        # Create summary
        if auto_save:
            try:
                summary = self._create_batch_summary(
                    all_results, all_reconstruction_data,
                    output_dir, total_time, min_matches_for_save, stats
                )
                self._print_final_summary(summary, output_dir, stats)
            except Exception as e:
                print(f"\n  Summary failed: {e}")
        else:
            print(f"\n{'='*70}")
            print(f"COMPLETE")
            print(f"{'='*70}")
            print(f"  Pairs: {stats['total_pairs']}")
            print(f"  Processed: {stats['processed']}")
            print(f"  Failed: {stats['failed']}")
            print(f"  Time: {total_time:.2f}s ({total_time/60:.1f}m)")
            if stats['processed'] > 0:
                print(f"  Avg: {total_time/stats['processed']:.2f}s/pair")
            print(f"{'='*70}")
        
        return all_results
    
        
    def _create_batch_summary(
        self,
        results: List[MatchingResult],
        reconstruction_data: list,
        output_dir: Path,
        total_time: float,
        min_matches: int,
        stats: Dict
    ) -> Dict:
        """Create comprehensive batch summary"""
        
        summary = {
            'processing_stats': stats,
            'configuration': {
                'methods': self.config.get('methods', []),
                'max_features': self.config.get('max_features', 2000),
                'min_matches_threshold': min_matches,
            },
            'timing': {
                'total_time': round(total_time, 2),
                'average_per_pair': round(
                    total_time / stats['processed'], 2
                ) if stats['processed'] > 0 else 0,
            },
            'pairs': []
        }
        
        # Add per-pair information (v3.0 API)
        for pair_id, recon_data in reconstruction_data:
            # v3.0: Get best method instead of primary
            best = recon_data.get_best('quality')
            
            if best:
                pair_info = {
                    'pair_id': pair_id,
                    'image1': recon_data.image_pair_info.image1_id,
                    'image2': recon_data.image_pair_info.image2_id,
                    'methods': list(recon_data.keys()),  # v3.0: All methods
                    'best_method': best.method_name,  # v3.0: Best instead of primary
                    'num_correspondences': best.num_matches,
                    'num_inliers': best.num_inliers,
                    'inlier_ratio': float(best.inlier_ratio) if best.inlier_ratio else None,
                    'reprojection_error': float(best.reprojection_error) if best.reprojection_error else None,
                    'has_homography': best.homography is not None,
                    'has_fundamental_matrix': best.fundamental_matrix is not None,
                }
                
                # v3.0: Add info for all methods
                pair_info['methods_detail'] = {}
                for method_name, method_data in recon_data.items():
                    pair_info['methods_detail'][method_name] = {
                        'num_matches': method_data.num_matches,
                        'num_inliers': method_data.num_inliers,
                        'inlier_ratio': method_data.inlier_ratio,
                        'quality_score': method_data.get_quality_score()
                    }
                
                summary['pairs'].append(pair_info)
        
        return summary

    
    
    def _print_final_summary(
        self,
        summary: Dict,
        output_dir: Path,
        stats: Dict
    ):
        """Print final summary to console"""
        print(f"\n{'='*70}")
        print(f"FINAL SUMMARY")
        print(f"{'='*70}")
        
        # Stats
        print(f"\n Processing Statistics:")
        print(f"  Total pairs: {stats['total_pairs']}")
        print(f"  Successfully processed: {stats['processed']}")
        print(f"  Saved for reconstruction: {stats['saved']}")
        
        if stats['skipped_already_done'] > 0:
            print(f"  Skipped (already done): {stats['skipped_already_done']}")
        if stats['skipped_insufficient_matches'] > 0:
            print(f"  Skipped (low matches): {stats['skipped_insufficient_matches']}")
        if stats['failed'] > 0:
            print(f"  Failed: {stats['failed']}")
        
        # Timing
        timing = summary['timing']
        print(f"\n⏱  Timing:")
        print(f"  Total: {timing['total_time']}s ({timing['total_time']/60:.1f} min)")
        print(f"  Avg per pair: {timing['average_per_pair']}s")
        
        # Output
        print(f"\n📁 Output Structure:")
        print(f"  {output_dir}/")
        print(f"  ├── matching_results/       ({stats['saved']} pairs)")
        print(f"  ├── reconstruction/         ({stats['saved']} pairs)")
        print(f"  ├── progress.json           (checkpoint)")
        print(f"  ├── image_metadata.pkl")
        print(f"  └── batch_summary.json")
        
        print(f"\n{'='*70}")
        print(f" ALL PROCESSING COMPLETE")
        print(f"{'='*70}")


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_pipeline(
    preset: str = 'balanced',
    methods: Optional[List[str]] = None,
    max_features: Optional[int] = None,
    **kwargs
) -> FeatureProcessingPipeline:
    """
    Create a feature processing pipeline with preset or custom configuration
    
    Args:
        preset: Preset name ('fast', 'balanced', 'accurate', 'custom')
        methods: List of method names (overrides preset)
        max_features: Max features per method (overrides preset)
        **kwargs: Additional configuration parameters
    
    Returns:
        FeatureProcessingPipeline instance
    
    Examples:
        >>> # Use preset
        >>> pipeline = create_pipeline('balanced')
        
        >>> # Custom methods
        >>> pipeline = create_pipeline('custom', methods=['SIFT', 'ALIKED'])
        
        >>> # Override preset
        >>> pipeline = create_pipeline('fast', max_features=5000)
    """
    from .config import create_config_from_preset
    
    # Get preset config
    if preset != 'custom':
        config = create_config_from_preset(preset)
    else:
        config = {}
    
    # Override with custom parameters
    if methods is not None:
        config['methods'] = methods
    if max_features is not None:
        config['max_features'] = max_features
    
    # Merge additional kwargs
    config.update(kwargs)
    
    # Ensure we have methods
    if 'methods' not in config:
        config['methods'] = ['SIFT']
    
    return FeatureProcessingPipeline(config)