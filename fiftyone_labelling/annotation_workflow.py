# save as: annotation_workflow.py

"""
Complete annotation workflow with proper saving

Usage:
    python annotation_workflow.py annotation_dataset
"""

import fiftyone as fo
from fiftyone import ViewField as F
import argparse

def launch_annotation_workflow(dataset_name: str):
    """Launch FiftyOne with proper annotation configuration"""
    
    dataset = fo.load_dataset(dataset_name)
    
    print("\n" + "="*70)
    print("ANNOTATION WORKFLOW")
    print("="*70)
    
    # Ensure all samples have ground_truth field (even if empty)
    print("\nEnsuring all samples have ground_truth field...")
    for sample in dataset.match(F("ground_truth").exists() == False).iter_samples():
        # Copy from predictions if available
        if sample.predictions and len(sample.predictions.detections) > 0:
            new_dets = []
            for pred in sample.predictions.detections:
                new_dets.append(fo.Detection(
                    label=pred.label,
                    bounding_box=pred.bounding_box
                ))
            sample["ground_truth"] = fo.Detections(detections=new_dets)
        else:
            # Create empty
            sample["ground_truth"] = fo.Detections(detections=[])
        sample.save()
    
    print("✅ All samples ready for annotation")
    
    # Launch app
    print("\n🚀 Launching FiftyOne...")
    print("\n" + "="*70)
    print("HOW TO ANNOTATE IN FIFTYONE")
    print("="*70)
    print("""
METHOD 1: Edit Existing Boxes
  1. Click on a sample to open modal
  2. Press 'a' key
  3. Select 'ground_truth' field when prompted
  4. Click and drag boxes to move/resize
  5. Click box and press Delete to remove
  6. Press Escape when done
  ✅ Changes save automatically!

METHOD 2: Add New Boxes
  1. Open sample modal
  2. Press 'a' key
  3. Select 'ground_truth' field
  4. Click "Add detection" button
  5. Draw new bounding box
  6. Type label name
  7. Press Escape when done
  ✅ New box is saved!

METHOD 3: Bulk Edit
  1. Select multiple samples (Ctrl+Click)
  2. Press 'a' key
  3. Edit in batch mode
  
IMPORTANT: Always select 'ground_truth' as the label field!

Press 't' → type 'fixed' → Enter to mark as reviewed
    """)
    
    print("="*70)
    
    session = fo.launch_app(dataset)
    
    print("\n✅ FiftyOne launched")
    print("Press Ctrl+C when done annotating\n")
    
    session.wait()
    
    # Show what was annotated
    print("\n" + "="*70)
    print("ANNOTATION SESSION COMPLETE")
    print("="*70)
    
    total = len(dataset)
    with_dets = len(dataset.match(F("ground_truth.detections").length() > 0))
    
    print(f"\nAnnotated samples: {with_dets}/{total} ({with_dets/total*100:.1f}%)")
    
    if with_dets > 0:
        print(f"\n✅ Ready to export!")
        print(f"Run: python fix_export.py {dataset_name} output_dir --format yolo")
    else:
        print(f"\n⚠️  No annotations found")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_name', help='Dataset name')
    args = parser.parse_args()
    
    launch_annotation_workflow(args.dataset_name)