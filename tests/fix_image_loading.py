# save as: fix_image_loading.py

"""
Diagnostic and fix for image loading issues
"""

import gradio as gr
from gradio_image_annotation import image_annotator
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import sys

def test_image_loading(images_dir: str):
    """Test different methods of loading images"""
    
    images_dir = Path(images_dir)
    image_files = sorted(list(images_dir.glob("*.jpg")) + 
                        list(images_dir.glob("*.png")) + 
                        list(images_dir.glob("*.jpeg")))
    
    if not image_files:
        print(f"❌ No images found in {images_dir}")
        return
    
    test_image = image_files[0]
    print(f"Testing with: {test_image}")
    print("="*70)
    
    # Test 1: Path as string
    print("\n1. Testing absolute path as string:")
    abs_path = str(test_image.absolute())
    print(f"   Path: {abs_path}")
    print(f"   Exists: {Path(abs_path).exists()}")
    
    # Test 2: PIL Image
    print("\n2. Testing PIL Image:")
    try:
        pil_img = Image.open(test_image)
        print(f"   ✓ Loaded: {pil_img.size}, {pil_img.mode}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 3: OpenCV + NumPy
    print("\n3. Testing OpenCV/NumPy:")
    try:
        cv_img = cv2.imread(str(test_image))
        if cv_img is not None:
            cv_img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            print(f"   ✓ Loaded: {cv_img_rgb.shape}")
        else:
            print(f"   ❌ cv2.imread returned None")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 4: Create test annotator
    print("\n4. Testing image_annotator component:")
    
    test_cases = [
        ("Absolute path string", str(test_image.absolute())),
        ("Relative path string", str(test_image)),
        ("PIL Image", pil_img),
        ("NumPy array", cv_img_rgb),
    ]
    
    for name, value in test_cases:
        try:
            print(f"\n   Testing {name}...")
            test_data = {
                "image": value,
                "boxes": []
            }
            annotator = image_annotator(
                value=test_data,
                label_list=["test"],
            )
            print(f"   ✓ {name} works!")
        except Exception as e:
            print(f"   ❌ {name} failed: {e}")
    
    print("\n" + "="*70)
    print("Recommendation: Use the method that works above")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fix_image_loading.py /path/to/images")
        sys.exit(1)
    
    test_image_loading(sys.argv[1])