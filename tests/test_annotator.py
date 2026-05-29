# save as: test_gradio6_setup.py

"""
Test script to verify Gradio 6.x setup
"""

def test_setup():
    """Test that all dependencies are properly installed"""
    print("="*70)
    print("TESTING GRADIO 6.x SETUP")
    print("="*70)
    
    success = True
    
    # Test Python version
    import sys
    print(f"\n✓ Python: {sys.version}")
    
    if sys.version_info < (3, 12):
        print("  ⚠️  Python 3.12+ recommended")
    
    # Test Gradio
    try:
        import gradio as gr
        print(f"✓ gradio: {gr.__version__}")
        
        if not gr.__version__.startswith("6."):
            print(f"  ⚠️  Expected gradio 6.x, found {gr.__version__}")
    except ImportError as e:
        print(f"❌ gradio not installed: {e}")
        print("   Install: pip install gradio==6.10.0")
        success = False
    
    # Test gradio_client
    try:
        import gradio_client
        print(f"✓ gradio_client: {gradio_client.__version__}")
    except ImportError:
        print("❌ gradio_client not installed")
        print("   Install: pip install gradio-client==2.4.0")
        success = False
    
    # Test gradio_image_annotation
    try:
        from gradio_image_annotation import image_annotator
        print("✓ gradio_image_annotation: installed")
        
        # Test if we can create an annotator
        test_annotator = image_annotator(
            value={"image": None, "boxes": []},
            label_list=["test"],
        )
        print("  ✓ image_annotator component works")
        
    except ImportError as e:
        print(f"❌ gradio_image_annotation not installed: {e}")
        print("   Install: pip install gradio-image-annotation==0.5.0")
        success = False
    except Exception as e:
        print(f"  ⚠️  Warning: {e}")
    
    # Test OpenCV
    try:
        import cv2
        print(f"✓ opencv-python: {cv2.__version__}")
    except ImportError:
        print("❌ opencv-python not installed")
        print("   Install: pip install opencv-python")
        success = False
    
    # Test NumPy
    try:
        import numpy as np
        print(f"✓ numpy: {np.__version__}")
    except ImportError:
        print("❌ numpy not installed")
        print("   Install: pip install numpy")
        success = False
    
    print("\n" + "="*70)
    if success:
        print("✅ ALL DEPENDENCIES INSTALLED CORRECTLY!")
        print("\nYou can now run:")
        print("  python minimal_annotator_updated.py /path/to/images")
    else:
        print("❌ SOME DEPENDENCIES ARE MISSING")
        print("\nInstall all at once:")
        print("  pip install gradio==6.10.0 gradio-client==2.4.0 \\")
        print("              gradio-image-annotation==0.5.0 \\")
        print("              opencv-python numpy")
    print("="*70)
    
    return success

if __name__ == "__main__":
    test_setup()