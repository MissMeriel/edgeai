# save as: batch_annotate.py

"""
Batch process multiple directories with FiftyOne

Usage:
    python batch_annotate.py /path/to/base/directory --model yolov5
    python batch_annotate.py /path/to/base/directory --model yolov5 --export
"""

from version_graveyard.fiftyone_annotation_app import FiftyOneAnnotationApp
from pathlib import Path
import argparse
import json
from datetime import datetime

def batch_process(base_dir: str, model: str = "yolov5", 
                 export: bool = False, confidence: float = 0.7):
    """Process all subdirectories in base directory"""
    
    base_path = Path(base_dir)
    subdirs = [d for d in base_path.iterdir() if d.is_dir()]
    
    if not subdirs:
        print(f"No subdirectories found in {base_dir}")
        return
    
    print(f"\n{'='*70}")
    print(f"BATCH PROCESSING")
    print(f"{'='*70}")
    print(f"Base directory: {base_dir}")
    print(f"Subdirectories: {len(subdirs)}")
    print(f"Model: {model}")
    print(f"{'='*70}\n")
    
    results = {}
    
    for idx, subdir in enumerate(subdirs, 1):
        print(f"\n{'='*70}")
        print(f"Processing [{idx}/{len(subdirs)}]: {subdir.name}")
        print(f"{'='*70}")
        
        dataset_name = f"batch_{subdir.name}"
        
        try:
            # Create app for this directory
            app = FiftyOneAnnotationApp(
                images_dir=str(subdir),
                dataset_name=dataset_name,
                model_type=model,
                auto_predict=True
            )
            
            # Create views
            app.create_annotation_views()
            
            # Copy high-confidence predictions
            print(f"\n📋 Copying predictions (confidence > {confidence})...")
            app.copy_predictions_to_ground_truth(confidence_threshold=confidence)
            
            # Get statistics
            total = len(app.dataset)
            with_gt = len(app.dataset.match(
                app.dataset.view()._dataset._handle.ViewField("ground_truth").exists() == True
            ))
            
            result = {
                'status': 'success',
                'dataset_name': dataset_name,
                'total_samples': total,
                'with_ground_truth': with_gt,
                'completion': f"{with_gt/total*100:.1f}%" if total > 0 else "0%"
            }
            
            # Export if requested
            if export:
                export_dir = subdir / "annotations_export"
                print(f"\n📤 Exporting to {export_dir}...")
                app.export_annotations(str(export_dir), format="yolo")
                result['export_dir'] = str(export_dir)
            
            results[subdir.name] = result
            print(f"\n✅ Completed {subdir.name}")
            
        except Exception as e:
            print(f"\n❌ Error processing {subdir.name}: {e}")
            results[subdir.name] = {
                'status': 'error',
                'error': str(e)
            }
    
    # Print summary
    print(f"\n{'='*70}")
    print("BATCH PROCESSING SUMMARY")
    print(f"{'='*70}")
    
    success_count = sum(1 for r in results.values() if r['status'] == 'success')
    error_count = len(results) - success_count
    
    print(f"\nTotal directories: {len(results)}")
    print(f"Successful: {success_count}")
    print(f"Errors: {error_count}")
    
    print("\nDetailed Results:")
    for dir_name, result in results.items():
        if result['status'] == 'success':
            print(f"\n  ✅ {dir_name}:")
            print(f"     Dataset: {result['dataset_name']}")
            print(f"     Samples: {result['total_samples']}")
            print(f"     Annotated: {result['with_ground_truth']} ({result['completion']})")
            if 'export_dir' in result:
                print(f"     Exported: {result['export_dir']}")
        else:
            print(f"\n  ❌ {dir_name}:")
            print(f"     Error: {result['error']}")
    
    # Save report
    report_file = Path(base_dir) / f"batch_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📄 Report saved to: {report_file}")
    print(f"{'='*70}\n")
    
    return results


def merge_datasets(base_dir: str, output_name: str = "merged_dataset"):
    """Merge all batch datasets into one"""
    import fiftyone as fo
    
    base_path = Path(base_dir)
    subdirs = [d for d in base_path.iterdir() if d.is_dir()]
    
    print(f"\n🔗 Merging datasets from {len(subdirs)} directories...")
    
    # Create merged dataset
    merged = fo.Dataset(output_name)
    merged.persistent = True
    
    total_samples = 0
    
    for subdir in subdirs:
        dataset_name = f"batch_{subdir.name}"
        
        try:
            if dataset_name in fo.list_datasets():
                dataset = fo.load_dataset(dataset_name)
                
                # Add samples to merged dataset
                merged.add_samples(dataset)
                total_samples += len(dataset)
                
                print(f"  ✅ Added {len(dataset)} samples from {dataset_name}")
            else:
                print(f"  ⚠️  Dataset {dataset_name} not found")
                
        except Exception as e:
            print(f"  ❌ Error merging {dataset_name}: {e}")
    
    print(f"\n✅ Merged dataset created: {output_name}")
    print(f"   Total samples: {total_samples}")
    
    return merged


def review_batch_results(base_dir: str):
    """Launch FiftyOne to review all batch results"""
    import fiftyone as fo
    
    base_path = Path(base_dir)
    subdirs = [d for d in base_path.iterdir() if d.is_dir()]
    
    # Get all batch datasets
    batch_datasets = []
    for subdir in subdirs:
        dataset_name = f"batch_{subdir.name}"
        if dataset_name in fo.list_datasets():
            batch_datasets.append(dataset_name)
    
    if not batch_datasets:
        print("No batch datasets found")
        return
    
    print(f"\nFound {len(batch_datasets)} batch datasets:")
    for i, name in enumerate(batch_datasets, 1):
        dataset = fo.load_dataset(name)
        total = len(dataset)
        annotated = len(dataset.match(
            fo.ViewField("ground_truth").exists() == True
        ))
        print(f"  {i}. {name}: {annotated}/{total} annotated")
    
    # Ask which to review
    print("\nEnter dataset number to review (or 'all' to merge and review all):")
    choice = input("> ")
    
    if choice.lower() == 'all':
        # Merge all and review
        print("\nMerging all datasets...")
        merged = merge_datasets(base_dir, "merged_batch_review")
        session = fo.launch_app(merged)
    else:
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(batch_datasets):
                dataset_name = batch_datasets[idx]
                dataset = fo.load_dataset(dataset_name)
                session = fo.launch_app(dataset)
            else:
                print("Invalid choice")
                return
        except ValueError:
            print("Invalid input")
            return
    
    print("\n🚀 FiftyOne app launched for review")
    print("Close browser or press Ctrl+C to exit")
    session.wait()


def cleanup_batch_datasets(base_dir: str, keep_merged: bool = True):
    """Clean up individual batch datasets"""
    import fiftyone as fo
    
    base_path = Path(base_dir)
    subdirs = [d for d in base_path.iterdir() if d.is_dir()]
    
    print("\n🗑️  Cleaning up batch datasets...")
    
    deleted = 0
    for subdir in subdirs:
        dataset_name = f"batch_{subdir.name}"
        
        if dataset_name in fo.list_datasets():
            try:
                fo.delete_dataset(dataset_name)
                deleted += 1
                print(f"  ✅ Deleted {dataset_name}")
            except Exception as e:
                print(f"  ❌ Error deleting {dataset_name}: {e}")
    
    if not keep_merged:
        if "merged_batch_review" in fo.list_datasets():
            fo.delete_dataset("merged_batch_review")
            print(f"  ✅ Deleted merged_batch_review")
    
    print(f"\n✅ Cleaned up {deleted} batch datasets")


def export_all_batch_results(base_dir: str, format: str = "yolo"):
    """Export all batch datasets"""
    import fiftyone as fo
    
    base_path = Path(base_dir)
    subdirs = [d for d in base_path.iterdir() if d.is_dir()]
    
    print(f"\n📤 Exporting all batch results in {format.upper()} format...")
    
    exported = 0
    for subdir in subdirs:
        dataset_name = f"batch_{subdir.name}"
        
        if dataset_name in fo.list_datasets():
            try:
                dataset = fo.load_dataset(dataset_name)
                
                # Check if has annotations
                annotated_view = dataset.match(
                    fo.ViewField("ground_truth").exists() == True
                )
                
                if len(annotated_view) == 0:
                    print(f"  ⚠️  {dataset_name}: No annotations to export")
                    continue
                
                # Export
                export_dir = subdir / f"export_{format}"
                export_dir.mkdir(exist_ok=True)
                
                app = FiftyOneAnnotationApp(
                    images_dir=str(subdir),
                    dataset_name=dataset_name,
                    model_type="none",
                    auto_predict=False
                )
                
                app.export_annotations(str(export_dir), format=format)
                exported += 1
                print(f"  ✅ Exported {dataset_name} to {export_dir}")
                
            except Exception as e:
                print(f"  ❌ Error exporting {dataset_name}: {e}")
    
    print(f"\n✅ Exported {exported} datasets")


def main():
    parser = argparse.ArgumentParser(
        description="Batch process multiple image directories with FiftyOne",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all subdirectories with YOLO predictions
  python batch_annotate.py /path/to/base/directory --model yolov5
  
  # Process and export immediately
  python batch_annotate.py /path/to/base/directory --model yolov5 --export
  
  # Review batch results
  python batch_annotate.py /path/to/base/directory --review
  
  # Merge all batch datasets
  python batch_annotate.py /path/to/base/directory --merge
  
  # Export all batch results
  python batch_annotate.py /path/to/base/directory --export-all --format yolo
  
  # Cleanup batch datasets
  python batch_annotate.py /path/to/base/directory --cleanup

Directory Structure Expected:
  base_directory/
    ├── category1/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    ├── category2/
    │   ├── image1.jpg
    │   └── ...
    └── category3/
        └── ...
        """
    )
    
    parser.add_argument('base_dir', help='Base directory containing subdirectories of images')
    parser.add_argument('--model', default='yolov5',
                       choices=['yolov5', 'yolov8', 'faster-rcnn', 'none'],
                       help='Model for predictions')
    parser.add_argument('--confidence', type=float, default=0.7,
                       help='Confidence threshold for copying predictions')
    parser.add_argument('--export', action='store_true',
                       help='Export annotations after processing')
    parser.add_argument('--review', action='store_true',
                       help='Launch FiftyOne to review batch results')
    parser.add_argument('--merge', action='store_true',
                       help='Merge all batch datasets into one')
    parser.add_argument('--export-all', action='store_true',
                       help='Export all batch results')
    parser.add_argument('--format', default='yolo',
                       choices=['yolo', 'coco', 'voc'],
                       help='Export format')
    parser.add_argument('--cleanup', action='store_true',
                       help='Delete batch datasets after processing')
    parser.add_argument('--keep-merged', action='store_true',
                       help='Keep merged dataset when cleaning up')
    
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    
    if not base_dir.exists():
        print(f"❌ Directory not found: {base_dir}")
        return
    
    if not base_dir.is_dir():
        print(f"❌ Not a directory: {base_dir}")
        return
    
    # Handle different modes
    if args.review:
        review_batch_results(args.base_dir)
    
    elif args.merge:
        merged = merge_datasets(args.base_dir)
        print("\n🚀 Launching merged dataset in FiftyOne...")
        import fiftyone as fo
        session = fo.launch_app(merged)
        session.wait()
    
    elif args.export_all:
        export_all_batch_results(args.base_dir, args.format)
    
    elif args.cleanup:
        cleanup_batch_datasets(args.base_dir, args.keep_merged)
    
    else:
        # Run batch processing
        results = batch_process(
            base_dir=args.base_dir,
            model=args.model,
            export=args.export,
            confidence=args.confidence
        )
        
        # Cleanup if requested
        if args.cleanup:
            print("\n" + "="*70)
            cleanup_batch_datasets(args.base_dir, args.keep_merged)


if __name__ == "__main__":
    main()