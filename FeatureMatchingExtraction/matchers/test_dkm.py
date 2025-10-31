import cv2
import sys
from pathlib import Path
from dense_matcher import create_dense_matcher

# Construct paths relative to the repository root
repo_root = Path(__file__).parent.parent.parent  # Go up to 3Dreconstruction/
images_dir = repo_root / 'images' / 'statue_of_liberty_images'

# Image filenames
img1_name = 'g01f09eae8523020ec8f9d6b1452305c1c02b066eda4ddbd5135158740033e45302806e322816a39cc80f6dbb123bf1fb33f9f2d3e7e4f4b2ea28030c5efd731f_1280.jpg'
img2_name = 'gcd1467f565bc12140098b6ccb6bf28976106640446360cb017c442d5ef376db1da0033abcf3b9e4e8ca78ff8ebcfab098a1241bf4cbc7f4617da46a45fd9f4e7_1280.jpg'

img1_path = images_dir / img1_name
img2_path = images_dir / img2_name

# Load images
print("Loading images...")
print(f"  Image 1: {img1_path}")
print(f"  Image 2: {img2_path}")

img1 = cv2.imread(str(img1_path))
img2 = cv2.imread(str(img2_path))

# Check if images loaded successfully
if img1 is None:
    print(f"\n✗ ERROR: Failed to load image 1!")
    print(f"  Path: {img1_path}")
    print(f"  Exists: {img1_path.exists()}")
    if not images_dir.exists():
        print(f"\n  Images directory not found: {images_dir}")
        print(f"  Please check the path or place test images in this directory.")
    sys.exit(1)

if img2 is None:
    print(f"\n✗ ERROR: Failed to load image 2!")
    print(f"  Path: {img2_path}")
    print(f"  Exists: {img2_path.exists()}")
    sys.exit(1)

print(f"✓ Images loaded successfully")
print(f"  Image 1: {img1.shape[1]}x{img1.shape[0]}")
print(f"  Image 2: {img2.shape[1]}x{img2.shape[0]}")

# Create matcher
matcher = create_dense_matcher(method='DKM', device='cpu', max_matches=30000)

# Match
result = matcher.match(img1, img2)

# Print results
print(f"✓ Found {len(result.keypoints1)} matches")
print(f"✓ Time: {result.time_seconds:.3f}s")
print(f"✓ Avg confidence: {result.confidence.mean():.3f}")

# Simple visualization
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(15, 7))

axes[0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
axes[0].scatter(result.keypoints1[:, 0], result.keypoints1[:, 1], 
                c='red', s=10, alpha=0.5)
axes[0].set_title(f'Image 1 ({len(result.keypoints1)} keypoints)')
axes[0].axis('off')

axes[1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
axes[1].scatter(result.keypoints2[:, 0], result.keypoints2[:, 1], 
                c='blue', s=10, alpha=0.5)
axes[1].set_title(f'Image 2 ({len(result.keypoints2)} keypoints)')
axes[1].axis('off')

plt.tight_layout()
plt.savefig('quick_test_result.png', dpi=150)
print("✓ Saved to quick_test_result.png")