# save as: fix_export.py

"""
Debug and fix FiftyOne export issues

This script:
1. Checks what data exists in your dataset
2. Exports data properly with verbose output
3. Handles edge cases

Usage:
    python fix_export.py annotation_dataset /path/to/output
"""

import fiftyone as fo
from fiftyone import ViewField as F
from pathlib import Path
import shutil
import cv2
import json
import argparse

def diagnose_dataset(dataset_name: str):
    """Diagnose what data exists in dataset"""
    
    dataset = fo.load_dataset(dataset_name)
    
    print("\n" + "="*70)
    print("DATASET DIAGNOSIS")
    print("="*70)
    
    total = len(dataset)
    print(f"\nTotal samples: {total}")
    
    # Check what fields exist
    with_gt = len(dataset.match(F("ground_truth").exists() == True))
    with_pred = len(dataset.match(F("predictions").exists() == True))
    with_tags = len(dataset.match(F("tag_review_status").exists() == True))
    
    print(f"\nData breakdown:")
    print(f"  ✅ With ground_truth: {with_gt} ({with_gt/total*100:.1f}%)")
    print(f"  ✅ With predictions: {with_pred} ({with_pred/total*100:.1f}%)")
    print(f"  ✅ With tag_review_status: {with_tags} ({with_tags/total*100:.1f}%)")
    
    if with_gt == 0:
        print(f"\n⚠️  WARNING: No samples have ground_truth!")
        print(f"   This is why export is empty.")
        print(f"\n   Solutions:")
        print(f"   1. Copy predictions to ground_truth:")
        print(f"      python fiftyone_annotation_app_fixed_tags.py /path/to/images \\")
        print(f"             --dataset-name {dataset_name} --copy-predictions")
        print(f"   2. Manually annotate in FiftyOne (press 'a' key)")
    
    # Show sample data
    if total > 0:
        sample = dataset.first()
        print(f"\n📝 First sample example:")
        print(f"   File: {Path(sample.filepath).name}")
        print(f"   Has ground_truth: {sample.ground_truth is not None}")
        print(f"   Has predictions: {sample.predictions is not None}")
        
        if sample.ground_truth:
            print(f"   Ground truth detections: {len(sample.ground_truth.detections)}")
        
        if sample.predictions:
            print(f"   Predicted detections: {len(sample.predictions.detections)}")
        
        # Show tags
        print(f"\n🏷️  Tags on sample:")
        for field in sample.field_names:
            if field.startswith('tag_'):
                value = getattr(sample, field)
                if value:
                    print(f"   {field}: {value.label}")
    
    return with_gt > 0


def export_with_debugging(dataset_name: str, output_dir: str, 
                         format: str = "yolo", use_predictions: bool = False):
    """Export with verbose debugging output"""
    
    dataset = fo.load_dataset(dataset_name)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"EXPORTING DATASET")
    print(f"{'='*70}")
    print(f"Output: {output_path}")
    print(f"Format: {format}")
    print(f"Use predictions if no GT: {use_predictions}")
    
    # Determine what to export
    if use_predictions:
        # Export samples with ground_truth OR predictions
        export_view = dataset.match(
            (F("ground_truth").exists() == True) | 
            (F("predictions").exists() == True)
        )
        print(f"\nExporting samples with ground_truth OR predictions: {len(export_view)}")
    else:
        # Only export samples with ground_truth
        export_view = dataset.match(F("ground_truth").exists() == True)
        print(f"\nExporting samples with ground_truth only: {len(export_view)}")
    
    if len(export_view) == 0:
        print("\n❌ No samples to export!")
        print("\nTroubleshooting:")
        print("  1. Check if you have ground truth: diagnose_dataset(dataset_name)")
        print("  2. Copy predictions: --copy-predictions")
        print("  3. Manually annotate: Press 'a' in FiftyOne")
        return False
    
    # Export based on format
    if format == "yolo":
        return export_yolo_verbose(export_view, output_path, use_predictions)
    elif format == "coco":
        return export_coco_verbose(export_view, output_path, use_predictions)
    elif format == "json":
        return export_json_verbose(export_view, output_path)
    else:
        print(f"Unknown format: {format}")
        return False


def export_yolo_verbose(view: fo.DatasetView, output_dir: Path, 
                       use_predictions: bool):
    """Export YOLO format with verbose output"""
    
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    images_dir.mkdir(exist_ok=True)
    labels_dir.mkdir(exist_ok=True)
    
    print(f"\n📂 Created directories:")
    print(f"   {images_dir}")
    print(f"   {labels_dir}")
    
    # Collect all classes
    classes = set()
    
    for sample in view:
        label_field = sample.ground_truth if sample.ground_truth else sample.predictions
        if label_field:
            for det in label_field.detections:
                classes.add(det.label)
    
    classes = sorted(list(classes))
    
    print(f"\n🏷️  Found {len(classes)} classes: {classes}")
    
    # Save classes
    with open(output_dir / "classes.txt", 'w') as f:
        f.write('\n'.join(classes))
    
    print(f"   ✅ Saved classes.txt")
    
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    
    # Export samples
    exported_count = 0
    skipped_count = 0
    
    print(f"\n📤 Exporting {len(view)} samples...")
    
    for sample in view.iter_samples(progress=True):
        try:
            # Determine which field to use
            if sample.ground_truth:
                detections = sample.ground_truth.detections
                source = "ground_truth"
            elif use_predictions and sample.predictions:
                detections = sample.predictions.detections
                source = "predictions"
            else:
                skipped_count += 1
                continue
            
            if not detections:
                skipped_count += 1
                continue
            
            # Copy image
            img_filename = Path(sample.filepath).name
            img_dest = images_dir / img_filename
            
            if not img_dest.exists():
                shutil.copy(sample.filepath, img_dest)
            
            # Get image dimensions
            img = cv2.imread(sample.filepath)
            if img is None:
                print(f"   ⚠️  Could not read {img_filename}")
                skipped_count += 1
                continue
            
            h, w = img.shape[:2]
            
            # Create label file
            label_filename = Path(sample.filepath).stem + ".txt"
            label_path = labels_dir / label_filename
            
            with open(label_path, 'w') as f:
                for det in detections:
                    # Get bounding box in YOLO format
                    x, y, bw, bh = det.bounding_box
                    center_x = x + bw/2
                    center_y = y + bh/2
                    
                    class_id = class_to_idx.get(det.label, 0)
                    f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {bw:.6f} {bh:.6f}\n")
            
            exported_count += 1
            
        except Exception as e:
            print(f"   ❌ Error exporting {Path(sample.filepath).name}: {e}")
            skipped_count += 1
    
    # Export tags
    tags_data = {}
    for sample in view:
        filename = Path(sample.filepath).name
        tags = {}
        
        for field in sample.field_names:
            if field.startswith('tag_'):
                tag_obj = getattr(sample, field, None)
                if tag_obj:
                    category = field.replace('tag_', '')
                    tags[category] = tag_obj.label
        
        # Also include simple tags
        if sample.tags:
            tags['simple_tags'] = sample.tags
        
        if tags:
            tags_data[filename] = tags
    
    if tags_data:
        with open(output_dir / "image_tags.json", 'w') as f:
            json.dump(tags_data, f, indent=2)
        print(f"\n✅ Saved image_tags.json with {len(tags_data)} entries")
    
    # Create dataset.yaml
    yaml_content = f"""# YOLO Dataset
path: {output_dir.absolute()}
train: images
val: images

nc: {len(classes)}
names: {classes}

# Exported: {exported_count} images
# Skipped: {skipped_count} images
# Tags: image_tags.json
"""
    
    with open(output_dir / "dataset.yaml", 'w') as f:
        f.write(yaml_content)
    
    print(f"\n{'='*70}")
    print(f"EXPORT COMPLETE")
    print(f"{'='*70}")
    print(f"✅ Exported: {exported_count} images")
    print(f"⚠️  Skipped: {skipped_count} images (no detections)")
    print(f"\n📁 Output structure:")
    print(f"   {output_dir}/")
    print(f"   ├── images/          ({exported_count} images)")
    print(f"   ├── labels/          ({exported_count} label files)")
    print(f"   ├── classes.txt      ({len(classes)} classes)")
    print(f"   ├── image_tags.json  (image-level tags)")
    print(f"   └── dataset.yaml     (YOLO config)")
    
    # Verify files exist
    image_count = len(list(images_dir.glob("*")))
    label_count = len(list(labels_dir.glob("*.txt")))
    
    print(f"\n🔍 Verification:")
    print(f"   Images in folder: {image_count}")
    print(f"   Labels in folder: {label_count}")
    
    if image_count == 0:
        print(f"\n❌ ERROR: No images exported!")
        return False
    
    return True


def export_coco_verbose(view: fo.DatasetView, output_dir: Path,
                       use_predictions: bool):
    """Export COCO format with verbose output"""
    
    print(f"\n📤 Exporting COCO format...")
    
    try:
        # Use FiftyOne's built-in COCO export
        label_field = "ground_truth"
        
        # Check if we should use predictions
        if use_predictions:
            # Count samples
            with_gt = len(view.match(F("ground_truth").exists() == True))
            with_pred = len(view.match(F("predictions").exists() == True))
            
            if with_gt == 0 and with_pred > 0:
                print(f"   Using predictions field (no ground truth found)")
                label_field = "predictions"
        
        view.export(
            export_dir=str(output_dir),
            dataset_type=fo.types.COCODetectionDataset,
            label_field=label_field
        )
        
        print(f"✅ COCO export complete: {output_dir}")
        
        # Verify
        labels_file = output_dir / "labels.json"
        if labels_file.exists():
            with open(labels_file) as f:
                coco_data = json.load(f)
            print(f"   Images: {len(coco_data.get('images', []))}")
            print(f"   Annotations: {len(coco_data.get('annotations', []))}")
        
        return True
        
    except Exception as e:
        print(f"❌ Export failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def export_json_verbose(view: fo.DatasetView, output_dir: Path):
    """Export as simple JSON with all data"""
    
    print(f"\n📤 Exporting JSON format...")
    
    all_data = []
    
    for sample in view.iter_samples(progress=True):
        sample_data = {
            'filepath': sample.filepath,
            'filename': Path(sample.filepath).name,
            'tags': {},
            'ground_truth': [],
            'predictions': []
        }
        
        # Extract tags
        for field in sample.field_names:
            if field.startswith('tag_'):
                tag_obj = getattr(sample, field, None)
                if tag_obj:
                    category = field.replace('tag_', '')
                    sample_data['tags'][category] = tag_obj.label
        
        # Add simple tags
        if sample.tags:
            sample_data['tags']['simple_tags'] = sample.tags
        
        # Extract ground truth
        if sample.ground_truth:
            for det in sample.ground_truth.detections:
                sample_data['ground_truth'].append({
                    'label': det.label,
                    'bounding_box': det.bounding_box
                })
        
        # Extract predictions
        if sample.predictions:
            for det in sample.predictions.detections:
                sample_data['predictions'].append({
                    'label': det.label,
                    'bounding_box': det.bounding_box,
                    'confidence': det.confidence
                })
        
        all_data.append(sample_data)
    
    # Save
    output_file = output_dir / "complete_export.json"
    with open(output_file, 'w') as f:
        json.dump(all_data, f, indent=2)
    
    print(f"\n✅ JSON export complete: {output_file}")
    print(f"   Samples: {len(all_data)}")
    print(f"   Size: {output_file.stat().st_size / 1024:.2f} KB")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Debug and fix export issues")
    parser.add_argument('dataset_name', help='FiftyOne dataset name')
    parser.add_argument('output_dir', nargs='?', default='export_output',
                       help='Output directory')
    parser.add_argument('--format', default='yolo',
                       choices=['yolo', 'coco', 'json'],
                       help='Export format')
    parser.add_argument('--use-predictions', action='store_true',
                       help='Export predictions if no ground truth')
    parser.add_argument('--diagnose-only', action='store_true',
                       help='Only diagnose, do not export')
    
    args = parser.parse_args()
    
    # First diagnose
    has_data = diagnose_dataset(args.dataset_name)
    
    if args.diagnose_only:
        return
    
    if not has_data:
        print("\n⚠️  No ground truth found. Use --use-predictions to export predictions instead")
        response = input("\nExport predictions anyway? (y/n): ")
        if response.lower() == 'y':
            args.use_predictions = True
        else:
            print("Export cancelled")
            return
    
    # Export
    success = export_with_debugging(
        args.dataset_name,
        args.output_dir,
        args.format,
        args.use_predictions
    )
    
    if success:
        print(f"\n✅ Export successful! Check: {args.output_dir}")
    else:
        print(f"\n❌ Export failed")


if __name__ == "__main__":
    main()