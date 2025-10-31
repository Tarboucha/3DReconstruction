#!/usr/bin/env python3
"""
Test script for factory.py - verifies all matcher creation options work correctly.

Tests:
- Traditional sparse: SIFT, ORB, AKAZE
- Deep learning sparse: SuperPoint, ALIKED, LightGlue
- Dense: DKM
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from FeatureMatchingExtraction.matchers import create_matcher


def test_traditional_matchers():
    """Test traditional sparse matchers"""
    print("\n" + "="*70)
    print("Testing Traditional Sparse Matchers")
    print("="*70)

    methods = ['SIFT', 'ORB', 'AKAZE']

    for method in methods:
        try:
            # Default (FLANN)
            matcher = create_matcher(method)
            print(f"✅ {method} + FLANN: {matcher.name}")

            # BruteForce
            matcher = create_matcher(method, matcher_type='BruteForce')
            print(f"✅ {method} + BruteForce: {matcher.name}")

        except Exception as e:
            print(f"❌ {method} failed: {e}")


def test_deep_learning_sparse():
    """Test deep learning sparse matchers"""
    print("\n" + "="*70)
    print("Testing Deep Learning Sparse Matchers")
    print("="*70)

    # SuperPoint tests
    print("\n--- SuperPoint ---")
    try:
        # SuperPoint + LightGlue (default)
        matcher = create_matcher('SuperPoint', device='cpu')
        print(f"✅ SuperPoint (default): {matcher.name}")

        # SuperPoint + FLANN
        matcher = create_matcher('SuperPoint', matcher_type='FLANN', device='cpu')
        print(f"✅ SuperPoint + FLANN: {matcher.name}")

        # SuperPoint + BruteForce
        matcher = create_matcher('SuperPoint', matcher_type='BruteForce', device='cpu')
        print(f"✅ SuperPoint + BruteForce: {matcher.name}")

        # LightGlue shorthand
        matcher = create_matcher('LightGlue', device='cpu')
        print(f"✅ LightGlue shorthand: {matcher.name}")

    except Exception as e:
        print(f"❌ SuperPoint failed: {e}")
        import traceback
        traceback.print_exc()

    # ALIKED tests
    print("\n--- ALIKED ---")

    # Check if ALIKED weights exist first
    aliked_weights = Path(__file__).parent / 'weights' / 'aliked-n16.pth'

    if not aliked_weights.exists():
        print(f"⚠️  ALIKED weights not found at: {aliked_weights}")
        print(f"   This is expected - ALIKED weights must be downloaded separately")
        print(f"   Download from: https://github.com/Shiaoming/ALIKED")
        print(f"   Skipping ALIKED tests...")
    else:
        try:
            # ALIKED + FLANN (default)
            matcher = create_matcher('ALIKED', device='cpu')
            print(f"✅ ALIKED (default): {matcher.name}")

            # ALIKED + BruteForce
            matcher = create_matcher('ALIKED', matcher_type='BruteForce', device='cpu')
            print(f"✅ ALIKED + BruteForce: {matcher.name}")

            # This should fail - ALIKED doesn't support LightGlue
            try:
                matcher = create_matcher('ALIKED', matcher_type='LightGlue', device='cpu')
                print(f"❌ ALIKED + LightGlue should have failed!")
            except ValueError as e:
                print(f"✅ ALIKED + LightGlue correctly rejected: {str(e)[:60]}...")

        except Exception as e:
            print(f"❌ ALIKED failed: {e}")
            import traceback
            traceback.print_exc()


def test_dense_matchers():
    """Test dense matchers"""
    print("\n" + "="*70)
    print("Testing Dense Matchers")
    print("="*70)

    try:
        # DKM
        matcher = create_matcher('DKM', device='cpu', symmetric=False)
        print(f"✅ DKM: {matcher.name}")

    except Exception as e:
        print(f"❌ DKM failed: {e}")
        import traceback
        traceback.print_exc()


def test_error_handling():
    """Test error handling"""
    print("\n" + "="*70)
    print("Testing Error Handling")
    print("="*70)

    # Unknown method
    try:
        matcher = create_matcher('UNKNOWN')
        print("❌ Should have raised ValueError for unknown method")
    except ValueError as e:
        print(f"✅ Unknown method correctly rejected")

    # Invalid matcher type
    try:
        matcher = create_matcher('SIFT', matcher_type='INVALID')
        print("❌ Should have raised ValueError for invalid matcher_type")
    except ValueError as e:
        print(f"✅ Invalid matcher_type correctly rejected")


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("FACTORY MATCHER TESTS")
    print("="*70)

    test_traditional_matchers()
    test_deep_learning_sparse()
    test_dense_matchers()
    test_error_handling()

    print("\n" + "="*70)
    print("TESTS COMPLETE")
    print("="*70)
    print("\nNote: Some tests may show warnings or missing weights - this is expected.")
    print("The important thing is that the factory creates matchers without crashing.")


if __name__ == '__main__':
    main()
