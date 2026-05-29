# save as: quick_annotate.py

"""
Quick start script for FiftyOne annotation

Usage:
    python quick_annotate.py /path/to/images
"""

import fiftyone as fo
from version_graveyard.fiftyone_annotation_app import FiftyOneAnnotationApp
import sys

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python quick_annotate.py <images_directory>")
        sys.exit(1)
    
    # Create app with YOLO predictions
    app = FiftyOneAnnotationApp(
        images_dir=sys.argv[1],
        dataset_name="quick_annotation",
        model_type="yolov5",
        auto_predict=True
    )
    
    # Create helpful views
    app.create_annotation_views()
    
    # Copy high-confidence predictions to ground truth
    print("\n📋 Copying high-confidence predictions as starting point...")
    app.copy_predictions_to_ground_truth(confidence_threshold=0.7)
    
    # Launch
    print("\n🚀 Launching annotation interface...")
    session = app.launch_annotation_session()
    
    print("\nAnnotation Tips:")
    print("  1. Review samples in 'needs_review' view")
    print("  2. Correct any incorrect predictions")
    print("  3. Add missing detections")
    print("  4. Annotations save automatically")
    
    session.wait()
    
    # Export when done
    print("\n📤 Exporting annotations...")
    app.export_annotations("./annotations_export", format="yolo")
    
    print("\n✅ Done! Annotations exported to ./annotations_export")