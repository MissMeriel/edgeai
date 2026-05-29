# save as: auto_tag_and_annotate.py

"""
Complete pipeline: Auto-tag images and launch annotation

Usage:
    python auto_tag_and_annotate.py /path/to/images
"""

import subprocess
import sys
from pathlib import Path
import argparse

def run_complete_pipeline(images_dir: str, model: str = "yolov5", 
                         confidence: float = 0.3):
    """Run complete auto-tagging and annotation pipeline"""
    
    images_path = Path(images_dir)
    
    if not images_path.exists():
        print(f"❌ Directory not found: {images_dir}")
        return
    
    print("="*70)
    print("COMPLETE AUTO-TAGGING AND ANNOTATION PIPELINE")
    print("="*70)
    
    # Step 1: Auto-tag images with scene classifier
    print("\n" + "="*70)
    print("STEP 1: Auto-tagging images with Places365")
    print("="*70)
    
    tags_file = "auto_tags.json"
    tags_with_splits = "auto_tags_with_splits.json"
    
    cmd = [
        sys.executable,
        "scene_classifier_places365.py",
        str(images_dir),
        "--output", tags_file,
        "--confidence", str(confidence)
    ]
    
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print("❌ Auto-tagging failed")
        return
    
    # Step 2: Add distribution splits
    print("\n" + "="*70)
    print("STEP 2: Adding distribution splits")
    print("="*70)
    
    cmd = [
        sys.executable,
        "scene_classifier_places365.py",
        "--balanced-split", tags_file,
        "--stratify", "scene",
        "--output", tags_with_splits
    ]
    
    subprocess.run(cmd)
    
    # Step 3: Launch annotation app with tags
    print("\n" + "="*70)
    print("STEP 3: Launching FiftyOne annotation app")
    print("="*70)
    
    cmd = [
        sys.executable,
        "fiftyone_annotation_app_fixed.py",
        str(images_dir),
        "--model", model,
        "--import-tags", tags_with_splits
    ]
    
    subprocess.run(cmd)
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)
    print(f"\nGenerated files:")
    print(f"  - {tags_file}: Raw auto-generated tags")
    print(f"  - {tags_with_splits}: Tags with train/val/test splits")
    print(f"\nYou can now:")
    print(f"  1. Review annotations in FiftyOne")
    print(f"  2. Export with: python fiftyone_annotation_app_fixed.py {images_dir} --export output --format yolo")


def main():
    parser = argparse.ArgumentParser(
        description="Complete auto-tagging and annotation pipeline"
    )
    
    parser.add_argument('images_dir', help='Directory containing images')
    parser.add_argument('--model', default='yolov5',
                       choices=['yolov5', 'yolov8', 'faster-rcnn', 'none'],
                       help='Object detection model')
    parser.add_argument('--confidence', type=float, default=0.3,
                       help='Scene classification confidence threshold')
    
    args = parser.parse_args()
    
    run_complete_pipeline(args.images_dir, args.model, args.confidence)


if __name__ == "__main__":
    main()