#!/usr/bin/env python3
"""
Quick verification that DKM can find and load weights from matchers/weights/
"""

from matchers.dense_matcher import DenseMatcher

print("Testing DKM weight loading from matchers/weights/...\n")

try:
    matcher = DenseMatcher(
        method='DKM',
        device='cuda',  # or 'cpu'
        weights='gim_dkm_100h',
        auto_download=False  # Should find local weights
    )

    print("\n✓ SUCCESS! DKM loaded weights from matchers/weights/")
    print(f"  Device: {matcher.device}")
    print(f"  Resize to: {matcher.resize_to}")

except Exception as e:
    print(f"\n✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
