# save as: view_in_fiftyone.py

"""View extracted frames and annotations in FiftyOne"""

import argparse
import json
from pathlib import Path
import fiftyone as fo

def load_frames_to_fiftyone(frames_dir: Path, dataset_name: str = None) -> fo.Dataset:
    """
    Load frames and annotations into FiftyOne
    
    Args:
        frames_dir: Directory containing frames and annotations
        dataset_name: Name for the dataset (default: auto-generated)
    
    Returns:
        FiftyOne dataset
    """
    
    frames_dir = Path(frames_dir)
    
    # Generate dataset name if not provided
    if dataset_name is None:
        dataset_name = f"video_frames_{frames_dir.name}"
    
    # Delete existing dataset if it exists
    if dataset_name in fo.list_datasets():
        print(f"Deleting existing dataset: {dataset_name}")
        fo.delete_dataset(dataset_name)
    
    # Create new dataset
    dataset = fo.Dataset(dataset_name)
    dataset.persistent = True
    
    # Load metadata
    metadata_file = frames_dir / "metadata.json"
    if metadata_file.exists():
        with open(metadata_file) as f:
            metadata = json.load(f)
            dataset.info = metadata
    
    # Load detections
    detections_file = frames_dir / "detections.json"
    detections = {}
    if detections_file.exists():
        with open(detections_file) as f:
            detections = json.load(f)
    
    # Load manual annotations
    annotations_file = frames_dir / "manual_annotations.json"
    manual_annotations = {}
    if annotations_file.exists():
        with open(annotations_file) as f:
            manual_annotations = json.load(f)
    
    # Get all image files
    image_files = sorted(list(frames_dir.glob("*.jpg")) + list(frames_dir.glob("*.png")))
    
    print(f"Loading {len(image_files)} images into FiftyOne...")
    
    samples = []
    for image_path in image_files:
        filename = image_path.name
        
        # Create sample
        sample = fo.Sample(filepath=str(image_path))
        
        # Add auto detections
        if filename in detections:
            auto_detections = []
            for det in detections[filename]:
                bbox = det['bbox']
                # Convert to relative coordinates [x, y, width, height]
                # Assuming bbox is [x1, y1, x2, y2] in absolute coordinates
                # Need to get image dimensions
                import cv2
                img = cv2.imread(str(image_path))
                h, w = img.shape[:2]
                
                x1, y1, x2, y2 = bbox
                rel_x = x1 / w
                rel_y = y1 / h
                rel_w = (x2 - x1) / w
                rel_h = (y2 - y1) / h
                
                auto_detections.append(
                    fo.Detection(
                        label=det['label'],
                        bounding_box=[rel_x, rel_y, rel_w, rel_h],
                        confidence=det['confidence']
                    )
                )
            
            sample["auto_detections"] = fo.Detections(detections=auto_detections)
        
        # Add manual annotations
        if filename in manual_annotations:
            manual_dets = []
            for ann in manual_annotations[filename]:
                bbox = ann['bbox']
                import cv2
                img = cv2.imread(str(image_path))
                h, w = img.shape[:2]
                
                x1, y1, x2, y2 = bbox
                rel_x = x1 / w
                rel_y = y1 / h
                rel_w = (x2 - x1) / w
                rel_h = (y2 - y1) / h
                
                manual_dets.append(
                    fo.Detection(
                        label=ann['label'],
                        bounding_box=[rel_x, rel_y, rel_w, rel_h],
                        confidence=ann.get('confidence', 1.0)
                    )
                )
            
            sample["manual_annotations"] = fo.Detections(detections=manual_dets)
        
        samples.append(sample)
    
    dataset.add_samples(samples)
    
    print(f"\nDataset created: {dataset_name}")
    print(f"Total samples: {len(dataset)}")
    print(f"Samples with auto detections: {len([s for s in dataset if 'auto_detections' in s])}")
    print(f"Samples with manual annotations: {len([s for s in dataset if 'manual_annotations' in s])}")
    
    return dataset

def main():
    parser = argparse.ArgumentParser(description="View extracted frames in FiftyOne")
    parser.add_argument('frames_dir', help='Directory containing extracted frames')
    parser.add_argument('--dataset-name', help='Custom dataset name')
    parser.add_argument('--no-launch', action='store_true', help='Don\'t launch FiftyOne app')
    
    args = parser.parse_args()
    
    # Load into FiftyOne
    dataset = load_frames_to_fiftyone(
        frames_dir=Path(args.frames_dir),
        dataset_name=args.dataset_name
    )
    
    # Launch app
    if not args.no_launch:
        print("\nLaunching FiftyOne app...")
        session = fo.launch_app(dataset)
        
        print("\nFiftyOne app running!")
        print("Press Ctrl+C to exit")
        
        session.wait()

if __name__ == "__main__":
    main()