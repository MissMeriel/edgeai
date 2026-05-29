# save as: fiftyone_annotation_app_fixed_tags.py

"""
FiftyOne Object Detection Labeling Application with Image-Level Tags
Includes review status tracking (todo/fixed/needs_work)

Installation:
    pip install fiftyone torch torchvision opencv-python

Usage:
    python fiftyone_annotation_app_fixed_tags.py /path/to/images --model yolov5
"""

import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.brain as fob
from fiftyone import ViewField as F
import cv2
import numpy as np
from pathlib import Path
import json
from typing import List, Dict, Optional, Tuple, Set
import argparse
import torch
from datetime import datetime

class FiftyOneAnnotationApp:
    """Complete annotation application with image tags and review status"""
    
    def __init__(self, images_dir: str, dataset_name: str = "annotation_dataset",
                 model_type: str = "yolov5", auto_predict: bool = True,
                 tag_categories: Optional[Dict[str, List[str]]] = None):
        """
        Initialize FiftyOne annotation application with tagging
        
        Args:
            images_dir: Directory containing images
            dataset_name: Name for the FiftyOne dataset
            model_type: Pretrained model ('yolov5', 'yolov8', 'faster-rcnn', 'none')
            auto_predict: Run predictions on all images at startup
            tag_categories: Dictionary of tag categories and their options
        """
        self.images_dir = Path(images_dir)
        self.dataset_name = dataset_name
        self.model_type = model_type.lower()
        self.auto_predict = auto_predict
        
        # Default tag categories if none provided
        self.tag_categories = tag_categories or {
            'scene': ['city_street', 'highway', 'parking_lot', 'residential', 'rural', 'indoor'],
            'time': ['day', 'night', 'dawn', 'dusk'],
            'weather': ['clear', 'cloudy', 'rainy', 'snowy', 'foggy'],
            'quality': ['high_quality', 'low_quality', 'blurry', 'dark', 'overexposed'],
            'distribution': ['train', 'val', 'test'],
            'review_status': ['todo', 'fixed', 'needs_work', 'reviewed', 'skip']  # NEW
        }
        
        # Load or create dataset
        self.dataset = self._load_or_create_dataset()
        
        # Initialize tag fields and review status
        self._initialize_tag_fields()
        self._initialize_review_status()
        
        # Load model
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if self.model_type != 'none':
            self._load_model()
        
        # Run auto-predictions
        if self.auto_predict and self.model is not None:
            self._run_auto_predictions()
        
        print(f"\n{'='*70}")
        print(f"FiftyOne Annotation App with Review Status Tracking")
        print(f"{'='*70}")
        print(f"Dataset: {self.dataset_name}")
        print(f"Samples: {len(self.dataset)}")
        print(f"Model: {self.model_type if self.model else 'Manual only'}")
        print(f"Device: {self.device}")
        print(f"\nTag Categories:")
        for category, tags in self.tag_categories.items():
            print(f"  {category}: {', '.join(tags)}")
        print(f"{'='*70}\n")
    
    def _load_or_create_dataset(self) -> fo.Dataset:
        """Load existing dataset or create new one"""
        
        if self.dataset_name in fo.list_datasets():
            print(f"Loading existing dataset: {self.dataset_name}")
            dataset = fo.load_dataset(self.dataset_name)
            return dataset
        
        print(f"Creating new dataset: {self.dataset_name}")
        dataset = fo.Dataset(self.dataset_name)
        dataset.persistent = True
        
        # Load images
        samples = []
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        for ext in extensions:
            for img_path in self.images_dir.glob(f"*{ext}"):
                sample = fo.Sample(filepath=str(img_path))
                samples.append(sample)
            for img_path in self.images_dir.glob(f"*{ext.upper()}"):
                sample = fo.Sample(filepath=str(img_path))
                samples.append(sample)
        
        if samples:
            dataset.add_samples(samples)
            print(f"Added {len(samples)} images to dataset")
        else:
            print("Warning: No images found in directory")
        
        return dataset
    
    def _initialize_tag_fields(self):
        """Initialize classification fields for each tag category"""
        print(f"Initialized tag fields for categories: {list(self.tag_categories.keys())}")
    
    def _initialize_review_status(self):
        """Initialize all images without review status to 'todo'"""
        print("\n🔖 Initializing review status...")
        
        # Get samples without review status
        uninitialized = self.dataset.match(F("tag_review_status").exists() == False)
        
        if len(uninitialized) == 0:
            print("All samples already have review status")
            
            # Print current status distribution
            status_counts = self.dataset.count_values("tag_review_status.label")
            print("\nCurrent review status:")
            for status, count in sorted(status_counts.items()):
                if status:
                    print(f"  {status}: {count}")
            return
        
        print(f"Setting {len(uninitialized)} samples to 'todo'...")
        
        # Set all to 'todo'
        for sample in uninitialized.iter_samples(progress=True):
            sample["tag_review_status"] = fo.Classification(label='todo')
            sample.save()
        
        print(f"✅ Initialized {len(uninitialized)} samples with review_status='todo'")
    
    def mark_as_fixed(self, sample: fo.Sample):
        """Mark a sample as fixed/reviewed"""
        sample["tag_review_status"] = fo.Classification(label='fixed')
        sample.save()
    
    def mark_as_needs_work(self, sample: fo.Sample):
        """Mark a sample as needing more work"""
        sample["tag_review_status"] = fo.Classification(label='needs_work')
        sample.save()
    
    def mark_batch_as_fixed(self, view: fo.DatasetView):
        """Mark multiple samples as fixed"""
        count = 0
        for sample in view.iter_samples(progress=True):
            sample["tag_review_status"] = fo.Classification(label='fixed')
            sample.save()
            count += 1
        
        print(f"✅ Marked {count} samples as 'fixed'")
        return count
    
    def reset_review_status(self):
        """Reset all samples back to 'todo'"""
        print("\n⚠️  Resetting all review statuses to 'todo'...")
        
        response = input("This will reset ALL samples. Continue? (y/n): ")
        if response.lower() != 'y':
            print("Cancelled")
            return
        
        for sample in self.dataset.iter_samples(progress=True):
            sample["tag_review_status"] = fo.Classification(label='todo')
            sample.save()
        
        print(f"✅ Reset all samples to 'todo'")
    
    def get_review_statistics(self) -> Dict[str, int]:
        """Get review status statistics"""
        status_counts = self.dataset.count_values("tag_review_status.label")
        
        # Ensure all statuses are represented
        for status in ['todo', 'fixed', 'needs_work', 'reviewed', 'skip']:
            if status not in status_counts:
                status_counts[status] = 0
        
        return status_counts
    
    def _load_model(self):
        """Load pretrained detection model"""
        try:
            if self.model_type in ["yolo", "yolov5"]:
                print("Loading YOLOv5 model...")
                self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', device=self.device)
                self.model.conf = 0.25
                print("✅ YOLOv5 loaded")
                
            elif self.model_type == "yolov8":
                print("Loading YOLOv8 model...")
                try:
                    from ultralytics import YOLO
                    self.model = YOLO('yolov8n.pt')
                    print("✅ YOLOv8 loaded")
                except ImportError:
                    print("⚠️  YOLOv8 requires 'ultralytics'. Falling back to YOLOv5")
                    self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', device=self.device)
                    self.model_type = "yolov5"
                    
            elif self.model_type == "faster-rcnn":
                print("Loading Faster R-CNN (COCO pretrained)...")
                import torchvision
                self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
                    weights='DEFAULT'
                )
                self.model.to(self.device)
                self.model.eval()
                print("✅ Faster R-CNN loaded")
            
            else:
                print(f"Unknown model: {self.model_type}")
                self.model = None
                
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.model = None
    
    def _run_auto_predictions(self):
        """Run predictions on samples without predictions"""
        print("\n🔮 Running auto-predictions...")
        
        view = self.dataset.match(F("predictions").exists() == False)
        
        if len(view) == 0:
            print("All samples already have predictions")
            return
        
        print(f"Predicting on {len(view)} samples...")
        
        for sample in view.iter_samples(progress=True):
            try:
                detections = self._predict_sample(sample.filepath)
                
                if detections:
                    sample["predictions"] = fo.Detections(detections=detections)
                    sample.save()
            except Exception as e:
                print(f"Error predicting {sample.filepath}: {e}")
        
        print("✅ Auto-predictions complete")
    
    def _predict_sample(self, image_path: str, confidence: float = 0.25) -> List[fo.Detection]:
        """Run prediction on a single image"""
        if self.model is None:
            return []
        
        try:
            if self.model_type in ["yolo", "yolov5"]:
                return self._predict_yolov5(image_path, confidence)
            elif self.model_type == "yolov8":
                return self._predict_yolov8(image_path, confidence)
            elif self.model_type == "faster-rcnn":
                return self._predict_faster_rcnn(image_path, confidence)
        except Exception as e:
            print(f"Prediction error: {e}")
            return []
    
    def _predict_yolov5(self, image_path: str, threshold: float) -> List[fo.Detection]:
        """YOLOv5 prediction"""
        results = self.model(image_path)
        detections = []
        
        img = cv2.imread(image_path)
        h, w = img.shape[:2]
        
        for *box, conf, cls in results.xyxy[0].cpu().numpy():
            if conf >= threshold:
                x1, y1, x2, y2 = box
                rel_box = [x1/w, y1/h, (x2-x1)/w, (y2-y1)/h]
                
                detection = fo.Detection(
                    label=results.names[int(cls)],
                    bounding_box=rel_box,
                    confidence=float(conf),
                    model="yolov5"
                )
                detections.append(detection)
        
        return detections
    
    def _predict_yolov8(self, image_path: str, threshold: float) -> List[fo.Detection]:
        """YOLOv8 prediction"""
        results = self.model(image_path)
        detections = []
        
        img = cv2.imread(image_path)
        h, w = img.shape[:2]
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                conf = float(box.conf[0])
                if conf >= threshold:
                    xyxy = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = xyxy
                    rel_box = [x1/w, y1/h, (x2-x1)/w, (y2-y1)/h]
                    
                    detection = fo.Detection(
                        label=result.names[int(box.cls[0])],
                        bounding_box=rel_box,
                        confidence=conf,
                        model="yolov8"
                    )
                    detections.append(detection)
        
        return detections
    
    def _predict_faster_rcnn(self, image_path: str, threshold: float) -> List[fo.Detection]:
        """Faster R-CNN prediction"""
        from PIL import Image as PILImage
        import torchvision.transforms as T
        
        COCO_CLASSES = [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
            'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
            'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        
        image = PILImage.open(image_path).convert("RGB")
        w, h = image.size
        
        transform = T.Compose([T.ToTensor()])
        image_tensor = transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(image_tensor)
        
        detections = []
        pred = predictions[0]
        
        for box, label, score in zip(pred['boxes'], pred['labels'], pred['scores']):
            if score >= threshold:
                label_id = int(label)
                label_name = COCO_CLASSES[label_id] if label_id < len(COCO_CLASSES) else 'unknown'
                
                if label_name not in ['N/A', '__background__']:
                    x1, y1, x2, y2 = box.cpu().numpy()
                    rel_box = [x1/w, y1/h, (x2-x1)/w, (y2-y1)/h]
                    
                    detection = fo.Detection(
                        label=label_name,
                        bounding_box=rel_box,
                        confidence=float(score),
                        model="faster-rcnn"
                    )
                    detections.append(detection)
        
        return detections
    
    # ========================================================================
    # IMAGE TAGGING METHODS
    # ========================================================================
    
    def add_tag_to_sample(self, sample: fo.Sample, category: str, tag: str):
        """Add a tag to a sample"""
        if category not in self.tag_categories:
            print(f"Warning: Unknown category '{category}', adding it")
            self.tag_categories[category] = [tag]
        
        if tag not in self.tag_categories[category]:
            print(f"Warning: Adding new tag '{tag}' to category '{category}'")
            self.tag_categories[category].append(tag)
        
        field_name = f"tag_{category}"
        sample[field_name] = fo.Classification(label=tag)
        sample.save()
    
    def add_tags_batch(self, view: fo.DatasetView, category: str, tag: str):
        """Add same tag to multiple samples"""
        count = 0
        for sample in view.iter_samples(progress=True):
            self.add_tag_to_sample(sample, category, tag)
            count += 1
        
        print(f"✅ Added tag '{category}:{tag}' to {count} samples")
    
    def remove_tag_from_sample(self, sample: fo.Sample, category: str):
        """Remove a tag category from a sample"""
        field_name = f"tag_{category}"
        if field_name in sample:
            sample[field_name] = None
            sample.save()
    
    def get_sample_tags(self, sample: fo.Sample) -> Dict[str, str]:
        """Get all tags for a sample"""
        tags = {}
        for category in self.tag_categories.keys():
            field_name = f"tag_{category}"
            if field_name in sample and sample[field_name] is not None:
                tags[category] = sample[field_name].label
        return tags
    
    def auto_tag_by_filename(self):
        """Auto-tag samples based on filename patterns"""
        print("\n🏷️  Auto-tagging based on filenames...")
        
        patterns = {
            'scene': {
                'city_street': ['city', 'urban', 'street'],
                'highway': ['highway', 'freeway', 'motorway'],
                'parking_lot': ['parking', 'lot'],
                'residential': ['residential', 'neighborhood'],
                'indoor': ['indoor', 'interior']
            },
            'time': {
                'day': ['day', 'daytime'],
                'night': ['night', 'nighttime'],
                'dawn': ['dawn', 'sunrise'],
                'dusk': ['dusk', 'sunset']
            },
            'weather': {
                'rainy': ['rain', 'rainy', 'wet'],
                'snowy': ['snow', 'snowy'],
                'foggy': ['fog', 'foggy'],
                'cloudy': ['cloud', 'cloudy', 'overcast']
            }
        }
        
        tagged_count = 0
        
        for sample in self.dataset.iter_samples(progress=True):
            filename = Path(sample.filepath).stem.lower()
            
            for category, tag_patterns in patterns.items():
                for tag, keywords in tag_patterns.items():
                    if any(keyword in filename for keyword in keywords):
                        self.add_tag_to_sample(sample, category, tag)
                        tagged_count += 1
                        break
        
        print(f"✅ Auto-tagged {tagged_count} samples based on filenames")
    
    def create_tag_views(self):
        """Create filtered views based on tags"""
        print("\n📂 Creating tag-based views...")
        
        for category, tags in self.tag_categories.items():
            field_name = f"tag_{category}"
            
            for tag in tags:
                view_name = f"{category}_{tag}"
                
                # Create view for this tag
                view = self.dataset.match(
                    F(field_name + ".label") == tag
                )
                
                if len(view) > 0:
                    self.dataset.save_view(view_name, view, overwrite=True)
                    print(f"  Created view '{view_name}': {len(view)} samples")
        
        # Create combination views
        for scene_tag in ['city_street', 'highway']:
            for time_tag in ['day', 'night']:
                view_name = f"{scene_tag}_{time_tag}"
                view = self.dataset.match(
                    (F("tag_scene.label") == scene_tag) &
                    (F("tag_time.label") == time_tag)
                )
                
                if len(view) > 0:
                    self.dataset.save_view(view_name, view, overwrite=True)
                    print(f"  Created view '{view_name}': {len(view)} samples")
    
    def split_by_distribution_tags(self, train_ratio: float = 0.7, 
                                   val_ratio: float = 0.15):
        """Automatically split dataset and tag with train/val/test"""
        print(f"\n📊 Splitting dataset (train:{train_ratio}, val:{val_ratio}, test:{1-train_ratio-val_ratio})...")
        
        import random
        
        # Get samples without distribution tag
        untagged = self.dataset.match(
            F("tag_distribution").exists() == False
        )
        
        if len(untagged) == 0:
            print("All samples already have distribution tags")
            return
        
        # Shuffle samples
        samples = list(untagged)
        random.shuffle(samples)
        
        # Calculate splits
        total = len(samples)
        train_end = int(total * train_ratio)
        val_end = int(total * (train_ratio + val_ratio))
        
        # Assign tags
        for i, sample in enumerate(samples):
            if i < train_end:
                tag = 'train'
            elif i < val_end:
                tag = 'val'
            else:
                tag = 'test'
            
            self.add_tag_to_sample(sample, 'distribution', tag)
        
        print(f"✅ Tagged {total} samples with distribution splits")
        print(f"   Train: {train_end}")
        print(f"   Val: {val_end - train_end}")
        print(f"   Test: {total - val_end}")
    
    # ========================================================================
    # ANNOTATION AND VIEW METHODS
    # ========================================================================
    
    def launch_annotation_session(self):
        """Launch FiftyOne annotation session"""
        print("\n🚀 Launching FiftyOne annotation interface...")
        print("\n" + "="*70)
        print("ANNOTATION AND REVIEW WORKFLOW")
        print("="*70)
        print("\n📋 REVIEW STATUS WORKFLOW:")
        print("  1. Start with 'review_status_todo' view (all images needing review)")
        print("  2. Annotate/correct objects in each image")
        print("  3. When done with an image, tag it:")
        print("     - Select the sample")
        print("     - In sidebar, find 'tag_review_status'")
        print("     - Change from 'todo' to 'fixed'")
        print("  4. Or use Python: app.mark_as_fixed(sample)")
        print("\n📦 OBJECT DETECTION:")
        print("  - Press 'a' to annotate objects")
        print("  - Draw bounding boxes and label them")
        print("  - Review AI predictions (if available)")
        print("\n🏷️  IMAGE TAGGING:")
        print("  - Select samples")
        print("  - Add/modify tags in sidebar")
        print("  - Use tag_scene, tag_time, tag_weather, etc.")
        print("\n⌨️  KEYBOARD SHORTCUTS:")
        print("  'a' - Annotate objects")
        print("  't' - Add tags")
        print("  'e' - Evaluate predictions")
        print("  'tab' - Toggle sidebar")
        print("  'n' - Next sample")
        print("  'p' - Previous sample")
        print("\n" + "="*70)
        
        session = fo.launch_app(self.dataset)
        return session
    
    def create_annotation_views(self):
        """Create useful views for annotation workflow"""
        
        print("\n📁 Creating annotation views...")
        
        # Object detection views
        unannotated = self.dataset.match(F("ground_truth").exists() == False)
        self.dataset.save_view("unannotated", unannotated, overwrite=True)
        print(f"  Created 'unannotated': {len(unannotated)} samples")
        
        needs_review = self.dataset.match(
            (F("predictions").exists() == True) & 
            (F("ground_truth").exists() == False)
        )
        self.dataset.save_view("needs_review", needs_review, overwrite=True)
        print(f"  Created 'needs_review': {len(needs_review)} samples")
        
        # Review status views (NEW)
        todo_view = self.dataset.match(F("tag_review_status.label") == "todo")
        self.dataset.save_view("review_status_todo", todo_view, overwrite=True)
        print(f"  Created 'review_status_todo': {len(todo_view)} samples ⭐ START HERE")
        
        fixed_view = self.dataset.match(F("tag_review_status.label") == "fixed")
        self.dataset.save_view("review_status_fixed", fixed_view, overwrite=True)
        print(f"  Created 'review_status_fixed': {len(fixed_view)} samples")
        
        needs_work_view = self.dataset.match(F("tag_review_status.label") == "needs_work")
        self.dataset.save_view("review_status_needs_work", needs_work_view, overwrite=True)
        print(f"  Created 'review_status_needs_work': {len(needs_work_view)} samples")
        
        # Tagging views
        untagged = self.dataset.match(F("tag_scene").exists() == False)
        self.dataset.save_view("untagged_scene", untagged, overwrite=True)
        print(f"  Created 'untagged_scene': {len(untagged)} samples")
        
        # Create tag-based views
        self.create_tag_views()
    
    def copy_predictions_to_ground_truth(self, confidence_threshold: float = 0.5,
                                         overwrite: bool = False):
        """
        Copy high-confidence predictions to ground truth
        
        Args:
            confidence_threshold: Minimum confidence to copy
            overwrite: If True, overwrite existing ground truth annotations
                      If False, skip samples that already have ground truth
        """
        print(f"\n📋 Copying predictions with confidence > {confidence_threshold}...")
        print(f"   Overwrite existing: {overwrite}")
        
        count = 0
        for sample in self.dataset.match(F("predictions").exists() == True).iter_samples(progress=True):
            if not overwrite and sample.ground_truth is not None:
                continue
            
            high_conf_dets = []
            if sample.predictions:
                for det in sample.predictions.detections:
                    if det.confidence >= confidence_threshold:
                        new_det = fo.Detection(
                            label=det.label,
                            bounding_box=det.bounding_box
                        )
                        high_conf_dets.append(new_det)
            
            if high_conf_dets:
                sample["ground_truth"] = fo.Detections(detections=high_conf_dets)
                sample.save()
                count += 1
        
        print(f"✅ Copied predictions to ground truth for {count} samples")
    
    # ========================================================================
    # EXPORT METHODS WITH TAGS
    # ========================================================================
    
    def export_annotations(self, output_dir: str, format: str = "yolo",
                          include_tags: bool = True, only_fixed: bool = False):
        """
        Export annotations with image tags
        
        Args:
            output_dir: Output directory
            format: Export format ('yolo', 'coco', 'voc', 'csv')
            include_tags: Include image tags in export
            only_fixed: Only export samples marked as 'fixed'
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n📤 Exporting annotations in {format.upper()} format...")
        print(f"   Include tags: {include_tags}")
        print(f"   Only fixed: {only_fixed}")
        
        # Get annotated samples
        annotated_view = self.dataset.match(F("ground_truth").exists() == True)
        
        # Filter to only fixed if requested
        if only_fixed:
            annotated_view = annotated_view.match(F("tag_review_status.label") == "fixed")
        
        if len(annotated_view) == 0:
            print("No annotated samples to export")
            return
        
        print(f"Exporting {len(annotated_view)} samples...")
        
        if format.lower() == "yolo":
            self._export_yolo_with_tags(annotated_view, output_dir, include_tags)
        elif format.lower() == "coco":
            self._export_coco_with_tags(annotated_view, output_dir, include_tags)
        elif format.lower() == "voc":
            self._export_voc_with_tags(annotated_view, output_dir, include_tags)
        elif format.lower() == "csv":
            self._export_csv(annotated_view, output_dir)
        else:
            print(f"Unknown format: {format}")
    
    # Add this to your fiftyone_annotation_app_fixed_tags.py
# Replace the existing _export_yolo_with_tags method

def _export_yolo_with_tags(self, view: fo.DatasetView, output_dir: Path, 
                           include_tags: bool):
    """Export in YOLO format with tags - FIXED VERSION"""
    import shutil
    
    print(f"\n{'='*70}")
    print("YOLO EXPORT - VERBOSE MODE")
    print(f"{'='*70}")
    
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    
    print(f"Creating directories...")
    images_dir.mkdir(exist_ok=True, parents=True)
    labels_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"   ✅ {images_dir}")
    print(f"   ✅ {labels_dir}")
    
    # Get classes from ground truth
    print(f"\nCollecting classes...")
    classes = view.distinct("ground_truth.detections.label")
    classes = sorted([c for c in classes if c])
    
    if not classes:
        print(f"   ⚠️  No classes found in ground_truth")
        print(f"   Checking predictions...")
        classes = view.distinct("predictions.detections.label")
        classes = sorted([c for c in classes if c])
    
    if not classes:
        print(f"   ❌ No classes found at all!")
        return
    
    print(f"   ✅ Found {len(classes)} classes: {classes}")
    
    with open(output_dir / "classes.txt", 'w') as f:
        f.write('\n'.join(classes))
    print(f"   ✅ Saved classes.txt")
    
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    
    # Export samples
    tags_data = {}
    exported_images = 0
    exported_labels = 0
    skipped = 0
    
    print(f"\n📦 Exporting {len(view)} samples...")
    
    for sample in view.iter_samples(progress=True):
        try:
            # Get detections
            if sample.ground_truth and len(sample.ground_truth.detections) > 0:
                detections = sample.ground_truth.detections
            elif sample.predictions and len(sample.predictions.detections) > 0:
                detections = sample.predictions.detections
            else:
                skipped += 1
                continue
            
            # Copy image
            img_filename = Path(sample.filepath).name
            img_source = Path(sample.filepath)
            img_dest = images_dir / img_filename
            
            if not img_source.exists():
                print(f"   ⚠️  Source image not found: {img_source}")
                skipped += 1
                continue
            
            shutil.copy(img_source, img_dest)
            exported_images += 1
            
            # Read image for dimensions
            img = cv2.imread(str(img_source))
            if img is None:
                print(f"   ⚠️  Could not read image: {img_filename}")
                skipped += 1
                continue
            
            h, w = img.shape[:2]
            
            # Create label file
            label_filename = Path(sample.filepath).stem + ".txt"
            label_path = labels_dir / label_filename
            
            with open(label_path, 'w') as f:
                for det in detections:
                    # YOLO format: class_id center_x center_y width height (normalized)
                    x, y, bw, bh = det.bounding_box
                    center_x = x + bw/2
                    center_y = y + bh/2
                    
                    class_id = class_to_idx.get(det.label, 0)
                    f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {bw:.6f} {bh:.6f}\n")
            
            exported_labels += 1
            
            # Collect tags
            if include_tags:
                tags = {}
                for field in sample.field_names:
                    if field.startswith('tag_'):
                        tag_obj = getattr(sample, field, None)
                        if tag_obj:
                            category = field.replace('tag_', '')
                            tags[category] = tag_obj.label
                
                if sample.tags:
                    tags['simple_tags'] = sample.tags
                
                if tags:
                    tags_data[img_filename] = tags
            
        except Exception as e:
            print(f"   ❌ Error processing {Path(sample.filepath).name}: {e}")
            skipped += 1
    
    # Save tags
    if include_tags and tags_data:
        tags_file = output_dir / "image_tags.json"
        with open(tags_file, 'w') as f:
            json.dump(tags_data, f, indent=2)
        print(f"\n✅ Saved {tags_file} ({len(tags_data)} images)")
    
    # Create dataset.yaml
    yaml_content = f"""# YOLO Dataset Configuration
path: {output_dir.absolute()}
train: images
val: images

# Classes
nc: {len(classes)}
names: {classes}

# Export info
exported_images: {exported_images}
exported_labels: {exported_labels}
skipped: {skipped}
date: {datetime.now().isoformat()}
"""
    
    yaml_file = output_dir / "dataset.yaml"
    with open(yaml_file, 'w') as f:
        f.write(yaml_content)
    print(f"✅ Saved {yaml_file}")
    
    # Summary
    print(f"\n{'='*70}")
    print(f"EXPORT SUMMARY")
    print(f"{'='*70}")
    print(f"✅ Images exported: {exported_images}")
    print(f"✅ Labels exported: {exported_labels}")
    print(f"⚠️  Skipped: {skipped}")
    
    print(f"\n📁 Output files:")
    print(f"   {output_dir}/")
    print(f"   ├── images/ ({len(list(images_dir.glob('*')))} files)")
    print(f"   ├── labels/ ({len(list(labels_dir.glob('*.txt')))} files)")
    print(f"   ├── classes.txt")
    print(f"   ├── dataset.yaml")
    if include_tags:
        print(f"   └── image_tags.json")
    
    # Verify files actually exist
    actual_images = len(list(images_dir.glob("*")))
    actual_labels = len(list(labels_dir.glob("*.txt")))
    
    if actual_images == 0:
        print(f"\n❌ ERROR: No images in output directory!")
        print(f"   Check that source images exist and are readable")
        return False
    
    print(f"\n✅ Verified: {actual_images} images, {actual_labels} labels in output")
    
    return True

    def _create_split_files(self, view: fo.DatasetView, output_dir: Path,
                           images_dir: Path):
        """Create train.txt, val.txt, test.txt based on distribution tags"""
        
        splits = {'train': [], 'val': [], 'test': []}
        
        for sample in view:
            img_filename = Path(sample.filepath).name
            img_path = str(images_dir / img_filename)
            
            # Get distribution tag
            if hasattr(sample, 'tag_distribution') and sample.tag_distribution:
                split = sample.tag_distribution.label
                if split in splits:
                    splits[split].append(img_path)
            else:
                # Default to train if no tag
                splits['train'].append(img_path)
        
        # Write split files
        for split_name, paths in splits.items():
            if paths:
                split_file = output_dir / f"{split_name}.txt"
                with open(split_file, 'w') as f:
                    f.write('\n'.join(paths))
                print(f"   Created {split_name}.txt with {len(paths)} images")
    
    def _export_coco_with_tags(self, view: fo.DatasetView, output_dir: Path,
                               include_tags: bool):
        """Export in COCO format with tags as image metadata"""
        
        # Use FiftyOne's COCO export
        view.export(
            export_dir=str(output_dir),
            dataset_type=fo.types.COCODetectionDataset,
            label_field="ground_truth"
        )
        
        # Add tags to COCO JSON
        if include_tags:
            coco_json_path = output_dir / "labels.json"
            
            if coco_json_path.exists():
                with open(coco_json_path, 'r') as f:
                    coco_data = json.load(f)
                
                # Add tags to image entries
                for sample in view:
                    img_filename = Path(sample.filepath).name
                    tags = self.get_sample_tags(sample)
                    
                    # Find image in COCO data
                    for img_entry in coco_data['images']:
                        if img_entry['file_name'] == img_filename:
                            img_entry['tags'] = tags
                            break
                
                # Save updated COCO JSON
                with open(coco_json_path, 'w') as f:
                    json.dump(coco_data, f, indent=2)
                
                print(f"   Added tags to COCO JSON")
        
        print(f"✅ Exported {len(view)} samples to {output_dir}")
    
    def _export_voc_with_tags(self, view: fo.DatasetView, output_dir: Path,
                              include_tags: bool):
        """Export in Pascal VOC XML format with tags"""
        import shutil
        
        annotations_dir = output_dir / "Annotations"
        images_dir = output_dir / "JPEGImages"
        annotations_dir.mkdir(parents=True, exist_ok=True)
        images_dir.mkdir(parents=True, exist_ok=True)
        
        tags_data = {}
        
        for sample in view.iter_samples(progress=True):
            # Copy image
            img_filename = Path(sample.filepath).name
            shutil.copy(sample.filepath, images_dir / img_filename)
            
            # Get image dimensions
            img = cv2.imread(sample.filepath)
            h, w, c = img.shape
            
            # Create XML
            xml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<annotation>
    <folder>JPEGImages</folder>
    <filename>{img_filename}</filename>
    <source>
        <database>FiftyOne Annotations</database>
    </source>
    <size>
        <width>{w}</width>
        <height>{h}</height>
        <depth>{c}</depth>
    </size>
    <segmented>0</segmented>
"""
            
            # Add tags as metadata if requested
            if include_tags:
                tags = self.get_sample_tags(sample)
                if tags:
                    xml_content += "    <tags>\n"
                    for category, tag in tags.items():
                        xml_content += f"        <{category}>{tag}</{category}>\n"
                    xml_content += "    </tags>\n"
                    tags_data[img_filename] = tags
            
            # Add objects
            for det in sample.ground_truth.detections:
                x, y, bw, bh = det.bounding_box
                x1 = int(x * w)
                y1 = int(y * h)
                x2 = int((x + bw) * w)
                y2 = int((y + bh) * h)
                
                xml_content += f"""    <object>
        <name>{det.label}</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>{x1}</xmin>
            <ymin>{y1}</ymin>
            <xmax>{x2}</xmax>
            <ymax>{y2}</ymax>
        </bndbox>
    </object>
"""
            
            xml_content += "</annotation>\n"
            
            # Save XML
            xml_filename = Path(sample.filepath).stem + ".xml"
            with open(annotations_dir / xml_filename, 'w', encoding='utf-8') as f:
                f.write(xml_content)
        
        # Save tags separately as JSON
        if include_tags and tags_data:
            with open(output_dir / "image_tags.json", 'w') as f:
                json.dump(tags_data, f, indent=2)
            print(f"   Tags saved to: image_tags.json")
        
        print(f"✅ Exported {len(view)} samples to {output_dir}")
    
    def _export_csv(self, view: fo.DatasetView, output_dir: Path):
        """Export image tags and metadata to CSV"""
        import csv
        
        csv_path = output_dir / "annotations_with_tags.csv"
        
        # Prepare data
        rows = []
        
        for sample in view.iter_samples(progress=True):
            row = {
                'filename': Path(sample.filepath).name,
                'filepath': sample.filepath,
            }
            
            # Add tags
            tags = self.get_sample_tags(sample)
            for category in self.tag_categories.keys():
                row[f'tag_{category}'] = tags.get(category, '')
            
            # Add detection count
            if sample.ground_truth:
                row['num_detections'] = len(sample.ground_truth.detections)
                
                # Count by class
                class_counts = {}
                for det in sample.ground_truth.detections:
                    class_counts[det.label] = class_counts.get(det.label, 0) + 1
                
                row['detected_classes'] = ', '.join(
                    f"{cls}:{count}" for cls, count in class_counts.items()
                )
            else:
                row['num_detections'] = 0
                row['detected_classes'] = ''
            
            rows.append(row)
        
        # Write CSV
        if rows:
            fieldnames = rows[0].keys()
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            
            print(f"✅ Exported CSV to {csv_path}")
            print(f"   {len(rows)} samples with {len(fieldnames)} columns")
    
    def evaluate_predictions(self):
        """Evaluate predictions against ground truth"""
        print("\n📊 Evaluating predictions...")
        
        eval_view = self.dataset.match(
            (F("predictions").exists() == True) & 
            (F("ground_truth").exists() == True)
        )
        
        if len(eval_view) == 0:
            print("No samples with both predictions and ground truth")
            return
        
        # Run evaluation
        results = eval_view.evaluate_detections(
            "predictions",
            gt_field="ground_truth",
            eval_key="eval",
            compute_mAP=True
        )
        
        # Print results
        print("\n" + "="*70)
        print("EVALUATION RESULTS")
        print("="*70)
        results.print_report()
        
        # Plot confusion matrix
        print("\nGenerating confusion matrix...")
        plot = results.plot_confusion_matrix()
        plot.show()
        
        return results
    
    def get_statistics(self):
        """Print comprehensive dataset statistics"""
        print("\n" + "="*70)
        print("DATASET STATISTICS")
        print("="*70)
        
        total = len(self.dataset)
        with_predictions = len(self.dataset.match(F("predictions").exists() == True))
        with_ground_truth = len(self.dataset.match(F("ground_truth").exists() == True))
        
        print(f"\nSample Counts:")
        print(f"  Total samples: {total}")
        print(f"  With predictions: {with_predictions} ({with_predictions/total*100:.1f}%)")
        print(f"  With ground truth: {with_ground_truth} ({with_ground_truth/total*100:.1f}%)")
        print(f"  Remaining to annotate: {total - with_ground_truth}")
        
        # Review status statistics (NEW)
        print("\n📋 Review Status:")
        review_stats = self.get_review_statistics()
        for status in ['todo', 'needs_work', 'fixed', 'reviewed', 'skip']:
            count = review_stats.get(status, 0)
            percentage = (count / total * 100) if total > 0 else 0
            emoji = "⏳" if status == "todo" else "✅" if status == "fixed" else "⚠️" if status == "needs_work" else "👁️" if status == "reviewed" else "⏭️"
            print(f"  {emoji} {status}: {count} ({percentage:.1f}%)")
        
        # Detection class distribution
        if with_ground_truth > 0:
            print("\nDetection Class Distribution (ground truth):")
            counts = self.dataset.count_values("ground_truth.detections.label")
            for label, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
                if label:
                    print(f"  {label}: {count}")
        
        # Tag statistics
        print("\nImage Tag Statistics:")
        for category in self.tag_categories.keys():
            if category == 'review_status':
                continue  # Skip review_status, already printed above
            
            field_name = f"tag_{category}"
            tagged = len(self.dataset.match(F(field_name).exists() == True))
            
            if tagged > 0:
                print(f"\n  {category.upper()} ({tagged} tagged):")
                tag_counts = self.dataset.count_values(f"{field_name}.label")
                for tag, count in sorted(tag_counts.items(), key=lambda x: x[1], reverse=True):
                    if tag:
                        print(f"    {tag}: {count} ({count/total*100:.1f}%)")
            else:
                print(f"\n  {category.upper()}: No samples tagged")
        
        print("="*70)
    
    def import_tags_from_json(self, json_file: str):
        """Import tags from JSON file"""
        print(f"\n📥 Importing tags from {json_file}...")
        
        json_path = Path(json_file)
        if not json_path.exists():
            print(f"❌ File not found: {json_file}")
            return
        
        with open(json_path, 'r') as f:
            tags_data = json.load(f)
        
        print(f"Loaded tags for {len(tags_data)} images")
        
        imported_count = 0
        not_found_count = 0
        
        for filename, tags in tags_data.items():
            # Find sample by filename
            samples = self.dataset.match(F("filepath").ends_with(filename))
            
            if len(samples) == 0:
                not_found_count += 1
                continue
            
            sample = samples.first()
            
            # Add tags
            for category, tag in tags.items():
                if category in self.tag_categories:
                    # Add tag to category if it doesn't exist
                    if tag not in self.tag_categories[category]:
                        self.tag_categories[category].append(tag)
                    
                    self.add_tag_to_sample(sample, category, tag)
                    imported_count += 1
                else:
                    # Create new category
                    self.tag_categories[category] = [tag]
                    field_name = f"tag_{category}"
                    sample[field_name] = fo.Classification(label=tag)
                    sample.save()
                    imported_count += 1
        
        print(f"✅ Imported {imported_count} tags from {len(tags_data)} images")
        if not_found_count > 0:
            print(f"⚠️  {not_found_count} images not found in dataset")
    
    def export_tags_to_json(self, output_file: str):
        """Export all tags to JSON file"""
        print(f"\n📤 Exporting tags to {output_file}...")
        
        tags_data = {}
        
        for sample in self.dataset.iter_samples(progress=True):
            filename = Path(sample.filepath).name
            tags = self.get_sample_tags(sample)
            
            if tags:
                tags_data[filename] = tags
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(tags_data, f, indent=2)
        
        print(f"✅ Exported tags for {len(tags_data)} samples to {output_path}")
    
    def create_balanced_splits(self, stratify_by: str = 'scene'):
        """Create balanced train/val/test splits stratified by a tag category"""
        print(f"\n⚖️  Creating balanced splits stratified by tag_{stratify_by}...")
        
        if stratify_by not in self.tag_categories:
            print(f"❌ Unknown category: {stratify_by}")
            return
        
        field_name = f"tag_{stratify_by}"
        
        # Get all unique tags in this category
        tagged_samples = self.dataset.match(F(field_name).exists() == True)
        unique_tags = tagged_samples.distinct(f"{field_name}.label")
        
        if not unique_tags:
            print(f"No samples tagged with {stratify_by}")
            return
        
        import random
        
        print(f"Found {len(unique_tags)} unique tags: {unique_tags}")
        
        # For each tag, split samples
        for tag in unique_tags:
            tag_view = tagged_samples.match(F(f"{field_name}.label") == tag)
            samples = list(tag_view)
            random.shuffle(samples)
            
            total = len(samples)
            train_end = int(total * 0.7)
            val_end = int(total * 0.85)
            
            for i, sample in enumerate(samples):
                if i < train_end:
                    dist = 'train'
                elif i < val_end:
                    dist = 'val'
                else:
                    dist = 'test'
                
                self.add_tag_to_sample(sample, 'distribution', dist)
            
            print(f"  {tag}: train={train_end}, val={val_end-train_end}, test={total-val_end}")
        
        print(f"✅ Created balanced splits")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="FiftyOne Object Detection & Tagging Application with Review Status",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic: Launch with YOLO predictions
  python fiftyone_annotation_app_fixed_tags.py /path/to/images --model yolov5
  
  # Import auto-generated tags from Places365
  python fiftyone_annotation_app_fixed_tags.py /path/to/images --import-tags tags.json --model faster-rcnn
  
  # View only samples needing review (todo status)
  # Then launch app - look for 'review_status_todo' view in sidebar
  
  # Mark all samples with high-quality predictions as fixed
  python fiftyone_annotation_app_fixed_tags.py /path/to/images --mark-high-conf-fixed --confidence 0.8
  
  # Export only reviewed/fixed samples
  python fiftyone_annotation_app_fixed_tags.py /path/to/images --export output --only-fixed
  
  # Reset all review statuses back to todo
  python fiftyone_annotation_app_fixed_tags.py /path/to/images --reset-review-status

Review Status Tags:
  - todo: Not yet reviewed (DEFAULT for all new images)
  - fixed: Reviewed and annotations are correct
  - needs_work: Reviewed but needs more annotation work
  - reviewed: Reviewed but not corrected
  - skip: Skip this image (e.g., poor quality, irrelevant)

Tag Categories:
  scene: city_street, highway, parking_lot, residential, rural, indoor
  time: day, night, dawn, dusk
  weather: clear, cloudy, rainy, snowy, foggy
  quality: high_quality, low_quality, blurry, dark, overexposed
  distribution: train, val, test
  review_status: todo, fixed, needs_work, reviewed, skip
        """
    )
    
    parser.add_argument('images_dir', help='Directory containing images')
    parser.add_argument('--dataset-name', default='annotation_dataset',
                       help='Name for FiftyOne dataset (default: annotation_dataset)')
    parser.add_argument('--model', default='yolov5',
                       choices=['yolov5', 'yolov8', 'faster-rcnn', 'none'],
                       help='Pretrained model for auto-prediction (default: yolov5)')
    parser.add_argument('--no-auto-predict', action='store_true',
                       help='Disable automatic predictions on startup')
    
    # Tagging options
    parser.add_argument('--auto-tag', action='store_true',
                       help='Auto-tag images based on filename patterns')
    parser.add_argument('--split', action='store_true',
                       help='Automatically split into train/val/test (70/15/15)')
    parser.add_argument('--balanced-split', type=str,
                       help='Create balanced split stratified by tag category (e.g., "scene")')
    parser.add_argument('--import-tags', type=str,
                       help='Import tags from JSON file (e.g., from Places365)')
    parser.add_argument('--export-tags', type=str,
                       help='Export tags to JSON file')
    
    # Review status options (NEW)
    parser.add_argument('--mark-high-conf-fixed', action='store_true',
                       help='Auto-mark high-confidence predictions as "fixed"')
    parser.add_argument('--reset-review-status', action='store_true',
                       help='Reset all review statuses back to "todo"')
    parser.add_argument('--only-fixed', action='store_true',
                       help='Only export samples with review_status="fixed"')
    
    # Annotation options
    parser.add_argument('--copy-predictions', action='store_true',
                       help='Copy high-confidence predictions to ground truth')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Confidence threshold for copying predictions (default: 0.5)')
    parser.add_argument('--overwrite', action='store_true',
                       help='Overwrite existing ground truth when copying predictions (CAREFUL!)')
    
    # Evaluation and export
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate predictions vs ground truth and show metrics')
    parser.add_argument('--export', type=str,
                       help='Export annotations to directory')
    parser.add_argument('--format', default='yolo',
                       choices=['yolo', 'coco', 'voc', 'csv'],
                       help='Export format (default: yolo)')
    parser.add_argument('--no-tags', action='store_true',
                       help='Exclude tags from export')
    
    # Statistics
    parser.add_argument('--stats', action='store_true',
                       help='Print statistics and exit (no UI launch)')
    
    args = parser.parse_args()
    
    print("="*70)
    print("FIFTYONE ANNOTATION APP WITH REVIEW STATUS TRACKING")
    print("="*70)
    print(f"Images: {args.images_dir}")
    print(f"Dataset: {args.dataset_name}")
    print(f"Model: {args.model}")
    print("="*70)
    
    # Create application
    try:
        app = FiftyOneAnnotationApp(
            images_dir=args.images_dir,
            dataset_name=args.dataset_name,
            model_type=args.model,
            auto_predict=not args.no_auto_predict
        )
    except Exception as e:
        print(f"\n❌ Error creating application: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Handle review status operations (NEW)
    if args.reset_review_status:
        try:
            app.reset_review_status()
        except Exception as e:
            print(f"\n❌ Error resetting review status: {e}")
    
    if args.mark_high_conf_fixed:
        try:
            print(f"\n✨ Marking high-confidence samples as 'fixed'...")
            view = app.dataset.match(
                (F("predictions").exists() == True) &
                (F("tag_review_status.label") == "todo")
            )
            
            marked_count = 0
            for sample in view.iter_samples(progress=True):
                # Check if all predictions are above threshold
                if sample.predictions:
                    all_high_conf = all(
                        det.confidence >= args.confidence 
                        for det in sample.predictions.detections
                    )
                    
                    if all_high_conf and len(sample.predictions.detections) > 0:
                        app.mark_as_fixed(sample)
                        marked_count += 1
            
            print(f"✅ Marked {marked_count} samples as 'fixed'")
            
        except Exception as e:
            print(f"\n❌ Error marking samples: {e}")
    
    # Create annotation views
    try:
        app.create_annotation_views()
    except Exception as e:
        print(f"\n⚠️  Error creating views (may already exist): {e}")
    
    # Handle tagging operations
    if args.import_tags:
        try:
            app.import_tags_from_json(args.import_tags)
        except Exception as e:
            print(f"\n❌ Error importing tags: {e}")
            import traceback
            traceback.print_exc()
    
    if args.auto_tag:
        try:
            app.auto_tag_by_filename()
        except Exception as e:
            print(f"\n❌ Error auto-tagging: {e}")
    
    if args.split:
        try:
            app.split_by_distribution_tags()
        except Exception as e:
            print(f"\n❌ Error creating splits: {e}")
    
    if args.balanced_split:
        try:
            app.create_balanced_splits(stratify_by=args.balanced_split)
        except Exception as e:
            print(f"\n❌ Error creating balanced splits: {e}")
    
    if args.export_tags:
        try:
            app.export_tags_to_json(args.export_tags)
        except Exception as e:
            print(f"\n❌ Error exporting tags: {e}")
    
    # Copy predictions if requested
    if args.copy_predictions:
        try:
            app.copy_predictions_to_ground_truth(
                confidence_threshold=args.confidence,
                overwrite=args.overwrite
            )
        except Exception as e:
            print(f"\n❌ Error copying predictions: {e}")
    
    # Evaluate if requested
    if args.evaluate:
        try:
            app.evaluate_predictions()
        except Exception as e:
            print(f"\n❌ Error evaluating: {e}")
    
    # Export if requested
    if args.export:
        try:
            app.export_annotations(
                args.export, 
                args.format,
                include_tags=not args.no_tags,
                only_fixed=args.only_fixed
            )
        except Exception as e:
            print(f"\n❌ Error exporting: {e}")
            import traceback
            traceback.print_exc()
    
    # Print statistics
    if args.stats:
        app.get_statistics()
        return
    
    # Launch annotation interface
    try:
        session = app.launch_annotation_session()
    except Exception as e:
        print(f"\n❌ Error launching app: {e}")
        return
    
    # Print usage information
    print("\n" + "="*70)
    print("ANNOTATION SESSION ACTIVE - REVIEW WORKFLOW")
    print("="*70)
    
    print("\n🔄 REVIEW WORKFLOW:")
    print("  1. Open 'review_status_todo' view in FiftyOne sidebar")
    print("  2. Review each image and its annotations")
    print("  3. Correct any errors")
    print("  4. Mark as complete:")
    print("     - Select sample(s)")
    print("     - Find 'tag_review_status' in sidebar")
    print("     - Change to 'fixed'")
    print("\n  Or use Python API:")
    print("     >>> sample = dataset.first()")
    print("     >>> sample['tag_review_status'] = fo.Classification(label='fixed')")
    print("     >>> sample.save()")
    
    print("\n📁 Useful Views:")
    print("  ⭐ review_status_todo: Images needing review (START HERE)")
    print("  ✅ review_status_fixed: Completed and verified images")
    print("  ⚠️  review_status_needs_work: Images needing more work")
    print("  - unannotated: Samples without object annotations")
    print("  - needs_review: Samples with predictions to review")
    
    print("\n🏷️  Tag-based Views:")
    print("  - scene_<tag>: Filter by scene (city_street, highway, etc.)")
    print("  - time_<tag>: Filter by time (day, night, etc.)")
    print("  - distribution_<tag>: Filter by split (train, val, test)")
    
    print("\n💡 Quick Actions:")
    print("  # Mark all high-conf predictions as fixed")
    print(f"  python {__file__} {args.images_dir} --mark-high-conf-fixed --confidence 0.8")
    print("\n  # Export only fixed/reviewed samples")
    print(f"  python {__file__} {args.images_dir} --export output --only-fixed")
    print("\n  # View progress")
    print(f"  python {__file__} {args.images_dir} --stats")
    
    print("\n" + "="*70)
    print("Close browser or press Ctrl+C to exit")
    print("All changes are saved automatically")
    print("="*70)
    
    # Wait for session
    try:
        session.wait()
    except KeyboardInterrupt:
        print("\n\n👋 Annotation session ended")
    
    # Final statistics
    print("\n")
    app.get_statistics()


if __name__ == "__main__":
    main()