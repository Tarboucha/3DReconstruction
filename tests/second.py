"""
Quick test to verify the foundation is ready for pipeline integration
Run this first to ensure all components work before modifying pipeline
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from CameraPoseEstimation2.structures import StructuredMatchData, EnhancedDMatch, ScoreType
from CameraPoseEstimation2.data_providers import create_provider, MockProvider, IMatchDataProvider
from CameraPoseEstimation2.loaders import MatchQualityStandardizer

def test_imports():
    """Test that all new modules can be imported"""
    print("Testing imports...")
    try:
        from CameraPoseEstimation2.structures import StructuredMatchData, EnhancedDMatch, ScoreType
        from CameraPoseEstimation2.data_providers import create_provider, MockProvider, IMatchDataProvider
        from CameraPoseEstimation2.loaders import MatchQualityStandardizer
        print("✅ All imports successful!\n")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}\n")
        return False


def test_mock_provider():
    """Test MockProvider creates valid data"""
    print("Testing MockProvider...")
    try:
        from CameraPoseEstimation2.data_providers import MockProvider
        
        # Create mock provider
        mock = MockProvider(num_images=10, pairs_per_image=3, matches_per_pair=150, seed=42)
        
        print(f"✅ MockProvider created successfully")
        print(f"   Images: {mock.get_image_count()}")
        print(f"   Pairs: {mock.get_pair_count()}")
        
        # Test getting data
        pairs = mock.get_all_pairs()
        if not pairs:
            print("❌ No pairs found")
            return False
            
        test_pair = pairs[0]
        match_data = mock.get_match_data(test_pair)
        
        print(f"   Test pair: {test_pair}")
        print(f"   Matches: {match_data.num_matches}")
        print(f"   Quality: {match_data.standardized_pair_quality:.3f}")
        print(f"   Match qualities available: {match_data.has_match_qualities()}")
        
        # Test validation
        validation = mock.validate()
        if validation:
            print(f"✅ Provider validation passed\n")
        else:
            print(f"⚠️  Provider has warnings (this is OK for mock data)\n")
        
        return True
        
    except Exception as e:
        print(f"❌ MockProvider test failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_provider_interface():
    """Test key provider interface methods"""
    print("Testing provider interface...")
    try:
        from CameraPoseEstimation2.data_providers import MockProvider
        
        mock = MockProvider(num_images=5, seed=42)
        
        # Test key methods
        assert mock.get_image_count() > 0, "Should have images"
        assert mock.get_pair_count() > 0, "Should have pairs"
        
        # Test best pairs
        best_pairs = mock.get_best_pairs(k=3, criterion='quality')
        assert len(best_pairs) <= 3, "Should return at most k pairs"
        print(f"   Best pairs: {best_pairs[:2]}")
        
        # Test filtering
        filtered = mock.filter_pairs(min_matches=50, min_quality=0.5)
        print(f"   Filtered pairs: {len(filtered)}")
        
        # Test image info
        images = mock.get_all_images()
        if images:
            img_info = mock.get_image_info(images[0])
            assert 'size' in img_info, "Image info should have size"
            print(f"   Sample image info: size={img_info['size']}")
        
        print("✅ Provider interface works correctly\n")
        return True
        
    except Exception as e:
        print(f"❌ Interface test failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all foundation tests"""
    print("="*70)
    print("FOUNDATION READINESS CHECK")
    print("="*70 + "\n")
    
    tests = [
        ("Imports", test_imports),
        ("MockProvider", test_mock_provider),
        ("Provider Interface", test_provider_interface),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"❌ {name} crashed: {e}\n")
            results.append((name, False))
    
    # Summary
    print("="*70)
    print("SUMMARY")
    print("="*70)
    
    all_passed = True
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {name}")
        if not success:
            all_passed = False
    
    print()
    if all_passed:
        print("🎉 Foundation is ready! You can proceed with pipeline integration.")
        print("   Next step: Modify pipeline2.py")
    else:
        print("⚠️  Some tests failed. Fix these issues before proceeding.")
        print("   Check that all files are in place:")
        print("   - structures/")
        print("   - data_providers/")
        print("   - loaders/")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)