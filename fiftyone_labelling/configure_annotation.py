# save as: configure_annotation.py

"""
Configure FiftyOne to save annotations to ground_truth field

This ensures your manual edits are saved and exportable
"""

import fiftyone as fo
from fiftyone import ViewField as F
import argparse

def setup_annotation_config(dataset_name: str):
    """Configure dataset for proper annotation"""
    
    dataset = fo.load_dataset(dataset_name)
    
    print("\n" + "="*70)
    print("CONFIGURING ANNOTATION SETTINGS")
    print("="*70)
    
    # Option 1: Copy all predictions to ground_truth as starting point
    print("\n1️⃣  Copying predictions to ground_truth (if not already done)...")
    
    count = 0
    for sample in dataset.match(
        (F("predictions").exists() == True) &
        (F("ground_truth").exists() == False)
    ).iter_samples(progress=True):
        
        if sample.predictions and len(sample.predictions.detections) > 0:
            # Copy predictions to ground truth
            new_detections = []
            for pred in sample.predictions.detections:
                det = fo.Detection(
                    label=pred.label,
                    bounding_box=pred.bounding_box
                )
                new_detections.append(det)
            
            sample["ground_truth"] = fo.Detections(detections=new_detections)
            sample.save()
            count += 1
    
    print(f"   ✅ Copied predictions to ground_truth for {count} samples")
    
    # Option 2: Initialize empty ground_truth for samples without predictions
    print("\n2️⃣  Initializing empty ground_truth for remaining samples...")
    
    count = 0
    for sample in dataset.match(F("ground_truth").exists() == False).iter_samples(progress=True):
        sample["ground_truth"] = fo.Detections(detections=[])
        sample.save()
        count += 1
    
    print(f"   ✅ Initialized {count} samples with empty ground_truth")
    
    print("\n" + "="*70)
    print("✅ CONFIGURATION COMPLETE")
    print("="*70)
    print("\nNow in FiftyOne:")
    print("  1. Select a sample")
    print("  2. Press 'a' key to annotate")
    print("  3. You'll see a dialog - select 'ground_truth' field")
    print("  4. Edit/add/delete bounding boxes")
    print("  5. Changes save automatically!")
    print("\n💡 All annotations now save to 'ground_truth' and will export properly")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_name', help='Dataset name')
    args = parser.parse_args()
    
    setup_annotation_config(args.dataset_name)