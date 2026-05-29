# save as: copy_all_to_ground_truth.py

"""
Copy ALL data (predictions or existing annotations) to ground_truth
This ensures everything is exportable

Usage:
    python copy_all_to_ground_truth.py annotation_dataset
"""

import fiftyone as fo
from fiftyone import ViewField as F
import sys

def copy_all_to_ground_truth(dataset_name: str, min_confidence: float = 0.0):
    """Copy all predictions to ground_truth"""
    
    dataset = fo.load_dataset(dataset_name)
    
    print(f"\n{'='*70}")
    print(f"COPYING ALL DATA TO GROUND_TRUTH")
    print(f"{'='*70}")
    print(f"Min confidence: {min_confidence}")
    
    copied = 0
    already_had = 0
    no_data = 0
    
    for sample in dataset.iter_samples(progress=True):
        
        # Skip if already has ground truth with detections
        if sample.ground_truth and len(sample.ground_truth.detections) > 0:
            already_had += 1
            continue
        
        # Copy from predictions
        if sample.predictions and len(sample.predictions.detections) > 0:
            new_dets = []
            for pred in sample.predictions.detections:
                if pred.confidence >= min_confidence:
                    new_dets.append(fo.Detection(
                        label=pred.label,
                        bounding_box=pred.bounding_box
                    ))
            
            if new_dets:
                sample["ground_truth"] = fo.Detections(detections=new_dets)
                sample.save()
                copied += 1
            else:
                # Create empty
                sample["ground_truth"] = fo.Detections(detections=[])
                sample.save()
                no_data += 1
        else:
            # No predictions either - create empty
            sample["ground_truth"] = fo.Detections(detections=[])
            sample.save()
            no_data += 1
    
    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    print(f"✅ Copied to ground_truth: {copied}")
    print(f"ℹ️  Already had ground_truth: {already_had}")
    print(f"⚠️  No detections: {no_data}")
    
    print(f"\n✅ Now you can:")
    print(f"  1. Annotate: Press 'a' in FiftyOne, edit ground_truth field")
    print(f"  2. Export: python fix_export.py {dataset_name} output_dir")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python copy_all_to_ground_truth.py <dataset_name> [min_confidence]")
        sys.exit(1)
    
    dataset_name = sys.argv[1]
    min_conf = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    
    copy_all_to_ground_truth(dataset_name, min_conf)