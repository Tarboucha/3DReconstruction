"""
Unit tests for data structures module.

File: CameraPoseEstimation/tests/test_structures.py
"""

import unittest
import numpy as np
import cv2
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from CameraPoseEstimation2.structures import (
    ScoreType,
    EnhancedDMatch,
    StructuredMatchData,
    keypoints_to_serializable,
    keypoints_from_serializable,
    create_minimal_match_data
)

class TestScoreType(unittest.TestCase):
    """Test ScoreType enumeration"""
    
    def test_values(self):
        """Test enum values"""
        self.assertEqual(ScoreType.DISTANCE.value, "distance")
        self.assertEqual(ScoreType.CONFIDENCE.value, "confidence")
        self.assertEqual(ScoreType.SIMILARITY.value, "similarity")
    
    def test_from_string(self):
        """Test creating from string"""
        self.assertEqual(ScoreType.from_string("distance"), ScoreType.DISTANCE)
        self.assertEqual(ScoreType.from_string("CONFIDENCE"), ScoreType.CONFIDENCE)
        self.assertEqual(ScoreType.from_string("similarity"), ScoreType.SIMILARITY)
    
    def test_string_representation(self):
        """Test string conversion"""
        self.assertEqual(str(ScoreType.DISTANCE), "distance")


class TestEnhancedDMatch(unittest.TestCase):
    """Test EnhancedDMatch class"""
    
    def test_creation(self):
        """Test basic creation"""
        match = EnhancedDMatch(
            queryIdx=0,
            trainIdx=5,
            score=0.85,
            score_type=ScoreType.CONFIDENCE,
            standardized_quality=0.87
        )
        
        self.assertEqual(match.queryIdx, 0)
        self.assertEqual(match.trainIdx, 5)
        self.assertEqual(match.score, 0.85)
        self.assertEqual(match.score_type, ScoreType.CONFIDENCE)
        self.assertEqual(match.standardized_quality, 0.87)
    
    def test_distance_property_confidence(self):
        """Test distance property with confidence score"""
        match = EnhancedDMatch(
            queryIdx=0,
            trainIdx=1,
            score=0.9,
            score_type=ScoreType.CONFIDENCE
        )
        
        # Confidence score should be inverted to distance
        self.assertAlmostEqual(match.distance, 0.1, places=5)
    
    def test_distance_property_distance(self):
        """Test distance property with distance score"""
        match = EnhancedDMatch(
            queryIdx=0,
            trainIdx=1,
            score=50.0,
            score_type=ScoreType.DISTANCE
        )
        
        # Distance score should be returned as-is
        self.assertEqual(match.distance, 50.0)
    
    def test_get_quality_score(self):
        """Test quality score retrieval"""
        match = EnhancedDMatch(
            queryIdx=0,
            trainIdx=1,
            score=0.85,
            score_type=ScoreType.CONFIDENCE,
            standardized_quality=0.87
        )
        
        # Should return standardized quality
        self.assertEqual(match.get_quality_score(higher_is_better=True), 0.87)
        self.assertAlmostEqual(match.get_quality_score(higher_is_better=False), 0.13, places=5)
    
    def test_to_cv2_dmatch(self):
        """Test conversion to cv2.DMatch"""
        match = EnhancedDMatch(
            queryIdx=10,
            trainIdx=20,
            score=0.8,
            score_type=ScoreType.CONFIDENCE
        )
        
        cv2_match = match.to_cv2_dmatch()
        
        self.assertIsInstance(cv2_match, cv2.DMatch)
        self.assertEqual(cv2_match.queryIdx, 10)
        self.assertEqual(cv2_match.trainIdx, 20)
        self.assertAlmostEqual(cv2_match.distance, 0.2, places=5)


class TestStructuredMatchData(unittest.TestCase):
    """Test StructuredMatchData class"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create test keypoints
        self.kp1 = [
            cv2.KeyPoint(x=10, y=20, size=5, angle=45, response=0.8),
            cv2.KeyPoint(x=30, y=40, size=5, angle=90, response=0.7),
            cv2.KeyPoint(x=50, y=60, size=5, angle=135, response=0.9)
        ]
        
        self.kp2 = [
            cv2.KeyPoint(x=15, y=25, size=5, angle=50, response=0.75),
            cv2.KeyPoint(x=35, y=45, size=5, angle=95, response=0.65),
            cv2.KeyPoint(x=55, y=65, size=5, angle=140, response=0.85)
        ]
        
        # Create test matches
        self.matches = [
            EnhancedDMatch(
                queryIdx=i,
                trainIdx=i,
                score=0.8 + i*0.05,
                score_type=ScoreType.CONFIDENCE,
                standardized_quality=0.8 + i*0.05
            )
            for i in range(3)
        ]
        
        # Create match data
        self.match_data = StructuredMatchData(
            matches=self.matches,
            keypoints1=self.kp1,
            keypoints2=self.kp2,
            method="test",
            score_type=ScoreType.CONFIDENCE,
            num_matches=3,
            standardized_pair_quality=0.85,
            match_quality_stats={
                'mean': 0.85,
                'std': 0.05,
                'min': 0.8,
                'max': 0.9,
                'median': 0.85
            }
        )
    
    def test_creation(self):
        """Test basic creation"""
        self.assertEqual(self.match_data.num_matches, 3)
        self.assertEqual(self.match_data.method, "test")
        self.assertEqual(self.match_data.score_type, ScoreType.CONFIDENCE)
        self.assertEqual(len(self.match_data.matches), 3)
    
    def test_correspondences_property(self):
        """Test correspondences property"""
        corr = self.match_data.correspondences
        
        self.assertEqual(corr.shape, (3, 4))
        self.assertAlmostEqual(corr[0, 0], 10, places=1)
        self.assertAlmostEqual(corr[0, 1], 20, places=1)
        self.assertAlmostEqual(corr[0, 2], 15, places=1)
        self.assertAlmostEqual(corr[0, 3], 25, places=1)
    
    def test_pts_properties(self):
        """Test pts1 and pts2 properties"""
        pts1 = self.match_data.pts1
        pts2 = self.match_data.pts2
        
        self.assertEqual(pts1.shape, (3, 2))
        self.assertEqual(pts2.shape, (3, 2))
        
        self.assertAlmostEqual(pts1[0, 0], 10, places=1)
        self.assertAlmostEqual(pts1[0, 1], 20, places=1)
        self.assertAlmostEqual(pts2[0, 0], 15, places=1)
        self.assertAlmostEqual(pts2[0, 1], 25, places=1)
    
    def test_match_scores_property(self):
        """Test match_scores property"""
        scores = self.match_data.match_scores
        
        self.assertEqual(len(scores), 3)
        self.assertAlmostEqual(scores[0], 0.8, places=2)
        self.assertAlmostEqual(scores[1], 0.85, places=2)
        self.assertAlmostEqual(scores[2], 0.9, places=2)
    
    def test_match_qualities_property(self):
        """Test match_qualities property"""
        qualities = self.match_data.match_qualities
        
        self.assertEqual(len(qualities), 3)
        self.assertAlmostEqual(qualities[0], 0.8, places=2)
    
    def test_get_top_k_matches(self):
        """Test filtering top-k matches"""
        top_2 = self.match_data.get_top_k_matches(2)
        
        self.assertEqual(top_2.num_matches, 2)
        self.assertEqual(len(top_2.matches), 2)
        
        # Should be sorted by quality (descending)
        self.assertGreaterEqual(top_2.matches[0].standardized_quality,
                               top_2.matches[1].standardized_quality)
    
    def test_filter_by_quality(self):
        """Test filtering by quality threshold"""
        high_quality = self.match_data.filter_by_quality(0.85)
        
        # Should have 2 matches (0.85 and 0.9)
        self.assertEqual(high_quality.num_matches, 2)
        
        # All matches should meet threshold
        for match in high_quality.matches:
            self.assertGreaterEqual(match.standardized_quality, 0.85)
    
    def test_to_cv2_matches(self):
        """Test conversion to cv2.DMatch list"""
        cv2_matches = self.match_data.to_cv2_matches()
        
        self.assertEqual(len(cv2_matches), 3)
        self.assertIsInstance(cv2_matches[0], cv2.DMatch)
    
    def test_to_dict(self):
        """Test conversion to dictionary"""
        data_dict = self.match_data.to_dict()
        
        self.assertIsInstance(data_dict, dict)
        self.assertEqual(data_dict['num_matches'], 3)
        self.assertEqual(data_dict['method'], "test")
        self.assertEqual(data_dict['standardized_pair_quality'], 0.85)
        self.assertIn('correspondences', data_dict)
        self.assertIn('match_scores', data_dict)
    
    def test_summary(self):
        """Test summary string generation"""
        summary = self.match_data.summary()
        
        self.assertIsInstance(summary, str)
        self.assertIn("StructuredMatchData Summary", summary)
        self.assertIn("test", summary)
        self.assertIn("3", summary)
    
    def test_empty_match_data(self):
        """Test handling of empty match data"""
        empty_data = StructuredMatchData(
            matches=[],
            keypoints1=[],
            keypoints2=[],
            method="test",
            score_type=ScoreType.CONFIDENCE,
            num_matches=0,
            standardized_pair_quality=0.0,
            match_quality_stats={'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'median': 0}
        )
        
        self.assertEqual(len(empty_data.correspondences), 0)
        self.assertEqual(len(empty_data.pts1), 0)
        self.assertEqual(len(empty_data.pts2), 0)


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions"""
    
    def test_keypoints_serialization(self):
        """Test keypoint serialization and deserialization"""
        kp = [
            cv2.KeyPoint(x=10, y=20, size=5, angle=45, response=0.8, octave=1, class_id=0),
            cv2.KeyPoint(x=30, y=40, size=7, angle=90, response=0.9, octave=2, class_id=1)
        ]
        
        # Serialize
        serialized = keypoints_to_serializable(kp)
        
        self.assertIsInstance(serialized, list)
        self.assertEqual(len(serialized), 2)
        self.assertIsInstance(serialized[0], dict)
        self.assertEqual(serialized[0]['pt'], (10.0, 20.0))
        
        # Deserialize
        reconstructed = keypoints_from_serializable(serialized)
        
        self.assertEqual(len(reconstructed), 2)
        self.assertIsInstance(reconstructed[0], cv2.KeyPoint)
        self.assertAlmostEqual(reconstructed[0].pt[0], 10, places=5)
        self.assertAlmostEqual(reconstructed[0].pt[1], 20, places=5)
    
    def test_create_minimal_match_data(self):
        """Test creating minimal match data"""
        correspondences = np.array([
            [10, 20, 15, 25],
            [30, 40, 35, 45],
            [50, 60, 55, 65]
        ], dtype=np.float32)
        
        match_data = create_minimal_match_data(
            correspondences,
            method="SIFT",
            quality=0.8
        )
        
        self.assertEqual(match_data.num_matches, 3)
        self.assertEqual(match_data.method, "SIFT")
        self.assertEqual(match_data.standardized_pair_quality, 0.8)
        self.assertEqual(len(match_data.keypoints1), 3)
        self.assertEqual(len(match_data.keypoints2), 3)
        
        # Check correspondences match
        np.testing.assert_array_almost_equal(
            match_data.correspondences,
            correspondences
        )
    
    def test_create_minimal_match_data_empty(self):
        """Test creating minimal match data with empty array"""
        correspondences = np.array([]).reshape(0, 4)
        
        match_data = create_minimal_match_data(correspondences)
        
        self.assertEqual(match_data.num_matches, 0)
        self.assertEqual(len(match_data.matches), 0)


def run_tests():
    """Run all tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestScoreType))
    suite.addTests(loader.loadTestsFromTestCase(TestEnhancedDMatch))
    suite.addTests(loader.loadTestsFromTestCase(TestStructuredMatchData))
    suite.addTests(loader.loadTestsFromTestCase(TestUtilityFunctions))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    print("="*70)
    print("RUNNING STRUCTURE TESTS")
    print("="*70)
    print()
    
    success = run_tests()
    
    print()
    print("="*70)
    if success:
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    print("="*70)
    
    sys.exit(0 if success else 1)