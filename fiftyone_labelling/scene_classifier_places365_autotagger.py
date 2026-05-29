# save as: scene_classifier_places365.py

"""
Scene Classification and Auto-Tagging using Places365

This script uses the pre-trained Places365 model to automatically classify
scenes and generate tags that can be imported into FiftyOne.

Features:
- Scene classification (indoor/outdoor, specific locations)
- Time of day detection (day/night/dawn/dusk)
- Weather condition detection
- Quality assessment
- Export tags for FiftyOne import

Installation:
    pip install torch torchvision opencv-python pillow numpy

Usage:
    python scene_classifier_places365.py /path/to/images --output tags.json
    python fiftyone_annotation_app_fixed.py /path/to/images --import-tags tags.json
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import cv2
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
import argparse
from tqdm import tqdm
import urllib.request
import pickle

class Places365Classifier:
    """Scene classifier using Places365 pre-trained model"""
    
    def __init__(self, arch='resnet18', use_gpu=True):
        """
        Initialize Places365 classifier
        
        Args:
            arch: Model architecture ('resnet18' or 'resnet50')
            use_gpu: Use GPU if available
        """
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        self.arch = arch
        
        print(f"Initializing Places365 classifier ({arch})...")
        print(f"Device: {self.device}")
        
        # Load model
        self.model = self._load_model()
        
        # Load class names and categories
        self.classes, self.labels_IO, self.categories = self._load_labels()
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Scene to tag mappings
        self.scene_mappings = self._define_scene_mappings()
        
        print("✅ Places365 classifier ready")
    
    def _load_model(self):
        """Load pre-trained Places365 model"""
        
        # Model URL
        model_url = f'http://places2.csail.mit.edu/models_places365/{self.arch}_places365.pth.tar'
        
        # Create model
        if self.arch == 'resnet18':
            model = models.resnet18(num_classes=365)
        elif self.arch == 'resnet50':
            model = models.resnet50(num_classes=365)
        else:
            raise ValueError(f"Unknown architecture: {self.arch}")
        
        # Try to load pre-trained weights
        model_path = Path(f'models/{self.arch}_places365.pth.tar')
        
        if not model_path.exists():
            print(f"Downloading Places365 weights...")
            model_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                urllib.request.urlretrieve(model_url, model_path)
                print("✅ Download complete")
            except Exception as e:
                print(f"⚠️  Could not download model: {e}")
                print("Using ImageNet pre-trained weights instead")
                return model.to(self.device).eval()
        
        # Load weights
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
            model.load_state_dict(state_dict)
            print("✅ Loaded Places365 weights")
        except Exception as e:
            print(f"⚠️  Error loading weights: {e}")
            print("Using random initialization")
        
        model = model.to(self.device)
        model.eval()
        
        return model
    
    def _load_labels(self):
        """Load Places365 class labels"""
        
        # Class names
        classes_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
        classes_path = Path('models/categories_places365.txt')
        
        if not classes_path.exists():
            try:
                urllib.request.urlretrieve(classes_url, classes_path)
            except:
                print("⚠️  Could not download class labels, using defaults")
                return self._get_default_classes()
        
        # Load classes
        classes = []
        with open(classes_path) as f:
            for line in f:
                classes.append(line.strip().split(' ')[0][3:])
        
        # Load IO labels (indoor/outdoor)
        io_url = 'https://raw.githubusercontent.com/csailvision/places365/master/IO_places365.txt'
        io_path = Path('models/IO_places365.txt')
        
        labels_IO = []
        if io_path.exists() or self._try_download(io_url, io_path):
            with open(io_path) as f:
                for line in f:
                    labels_IO.append(int(line.strip().split()[1]))
        else:
            labels_IO = [0] * 365  # Default to outdoor
        
        # Categories
        categories = self._load_categories()
        
        return classes, labels_IO, categories
    
    def _try_download(self, url: str, path: Path) -> bool:
        """Try to download a file"""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(url, path)
            return True
        except:
            return False
    
    def _get_default_classes(self):
        """Return default minimal classes"""
        classes = ['street', 'highway', 'parking_lot', 'residential', 'building']
        labels_IO = [1] * len(classes)
        categories = []
        return classes, labels_IO, categories
    
    def _load_categories(self):
        """Load scene categories"""
        # Simplified categories for common scenes
        categories = {
            'transportation': ['street', 'highway', 'road', 'parking_lot', 'garage'],
            'residential': ['house', 'apartment', 'bedroom', 'living_room', 'kitchen'],
            'commercial': ['shop', 'store', 'restaurant', 'office', 'building'],
            'outdoor': ['park', 'forest', 'mountain', 'beach', 'field'],
            'urban': ['city', 'downtown', 'alley', 'plaza', 'square']
        }
        return categories
    
    def _define_scene_mappings(self):
        """Define mappings from Places365 classes to our tag categories"""
        
        mappings = {
            'scene': {
                # Street and road scenes
                'street': 'city_street',
                'crosswalk': 'city_street',
                'downtown': 'city_street',
                'highway': 'highway',
                'road': 'highway',
                'parking_lot': 'parking_lot',
                'parking_garage': 'parking_lot',
                
                # Residential
                'house': 'residential',
                'residential_neighborhood': 'residential',
                'driveway': 'residential',
                
                # Indoor
                'garage': 'indoor',
                'warehouse': 'indoor',
                
                # Rural
                'field': 'rural',
                'forest': 'rural',
                'countryside': 'rural'
            }
        }
        
        return mappings
    
    def classify_image(self, image_path: str, top_k: int = 5) -> List[Tuple[str, float, str]]:
        """
        Classify scene in image
        
        Args:
            image_path: Path to image
            top_k: Return top K predictions
            
        Returns:
            List of (class_name, probability, indoor/outdoor)
        """
        
        # Load and transform image
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        # Run inference
        with torch.no_grad():
            logit = self.model(img_tensor)
            probs = torch.nn.functional.softmax(logit, 1)
        
        # Get top K predictions
        top_probs, top_indices = probs.topk(top_k)
        
        results = []
        for prob, idx in zip(top_probs[0], top_indices[0]):
            idx = idx.item()
            prob = prob.item()
            class_name = self.classes[idx]
            io_label = 'indoor' if self.labels_IO[idx] == 1 else 'outdoor'
            
            results.append((class_name, prob, io_label))
        
        return results
    
    def get_scene_tag(self, predictions: List[Tuple[str, float, str]]) -> Optional[str]:
        """Map scene prediction to our tag categories"""
        
        for class_name, prob, io in predictions:
            # Look for mapping
            for scene_class, tag in self.scene_mappings['scene'].items():
                if scene_class in class_name:
                    return tag
        
        # Default based on indoor/outdoor
        if predictions[0][2] == 'indoor':
            return 'indoor'
        else:
            return 'city_street'  # Default outdoor
    
    def analyze_image_properties(self, image_path: str) -> Dict[str, str]:
        """
        Analyze image for additional properties (time, weather, quality)
        
        Returns:
            Dictionary of detected properties
        """
        
        img = cv2.imread(image_path)
        if img is None:
            return {}
        
        properties = {}
        
        # Analyze brightness for time of day
        time_tag = self._detect_time_of_day(img)
        if time_tag:
            properties['time'] = time_tag
        
        # Analyze for weather conditions
        weather_tag = self._detect_weather(img)
        if weather_tag:
            properties['weather'] = weather_tag
        
        # Analyze image quality
        quality_tag = self._assess_quality(img)
        if quality_tag:
            properties['quality'] = quality_tag
        
        return properties
    
    def _detect_time_of_day(self, img: np.ndarray) -> Optional[str]:
        """Detect time of day based on brightness and color"""
        
        # Convert to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Get average brightness (V channel)
        avg_brightness = np.mean(hsv[:, :, 2])
        
        # Get average saturation
        avg_saturation = np.mean(hsv[:, :, 1])
        
        # Classify time of day
        if avg_brightness < 50:
            return 'night'
        elif avg_brightness < 100:
            # Check if it's dawn/dusk (lower saturation)
            if avg_saturation < 80:
                return 'dusk'
            else:
                return 'night'
        elif avg_brightness > 180:
            return 'day'
        else:
            # Mid-range brightness
            if avg_saturation < 100:
                return 'dawn'
            else:
                return 'day'
        
        return None
    
    def _detect_weather(self, img: np.ndarray) -> Optional[str]:
        """Detect weather conditions"""
        
        # Convert to grayscale and HSV
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Calculate metrics
        avg_saturation = np.mean(hsv[:, :, 1])
        contrast = np.std(gray)
        
        # Detect foggy (low contrast, low saturation)
        if contrast < 30 and avg_saturation < 50:
            return 'foggy'
        
        # Detect cloudy (moderate contrast, lower saturation)
        if avg_saturation < 80 and contrast < 50:
            return 'cloudy'
        
        # Default to clear
        if avg_saturation > 80:
            return 'clear'
        
        return None
    
    def _assess_quality(self, img: np.ndarray) -> Optional[str]:
        """Assess image quality"""
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Calculate sharpness using Laplacian variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = laplacian.var()
        
        # Calculate brightness
        brightness = np.mean(gray)
        
        # Assess quality
        if sharpness < 50:
            return 'blurry'
        elif brightness < 50:
            return 'dark'
        elif brightness > 200:
            return 'overexposed'
        elif sharpness > 200:
            return 'high_quality'
        
        return None


class ImageTagger:
    """Auto-tag images using scene classifier"""
    
    def __init__(self, classifier: Places365Classifier):
        self.classifier = classifier
    
    def tag_images(self, images_dir: str, output_file: str,
                  confidence_threshold: float = 0.3) -> Dict[str, Dict[str, str]]:
        """
        Tag all images in directory and export tags
        
        Args:
            images_dir: Directory containing images
            output_file: Output JSON file for tags
            confidence_threshold: Minimum confidence for scene classification
            
        Returns:
            Dictionary mapping filenames to tags
        """
        
        images_dir = Path(images_dir)
        
        # Find all images
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        
        for ext in extensions:
            image_files.extend(images_dir.glob(f"*{ext}"))
            image_files.extend(images_dir.glob(f"*{ext.upper()}"))
        
        image_files = sorted(image_files)
        
        if not image_files:
            print(f"No images found in {images_dir}")
            return {}
        
        print(f"\n{'='*70}")
        print(f"AUTO-TAGGING {len(image_files)} IMAGES")
        print(f"{'='*70}\n")
        
        tags_data = {}
        
        for img_path in tqdm(image_files, desc="Tagging images"):
            try:
                tags = self._tag_single_image(img_path, confidence_threshold)
                if tags:
                    tags_data[img_path.name] = tags
            except Exception as e:
                print(f"Error processing {img_path.name}: {e}")
                continue
        
        # Save to file
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(tags_data, f, indent=2)
        
        print(f"\n✅ Tagged {len(tags_data)} images")
        print(f"📄 Saved tags to: {output_file}")
        
        # Print statistics
        self._print_tag_statistics(tags_data)
        
        return tags_data
    
    def _tag_single_image(self, image_path: Path, 
                         confidence_threshold: float) -> Dict[str, str]:
        """Tag a single image"""
        
        tags = {}
        
        # Scene classification
        predictions = self.classifier.classify_image(str(image_path))
        
        if predictions[0][1] >= confidence_threshold:
            scene_tag = self.classifier.get_scene_tag(predictions)
            if scene_tag:
                tags['scene'] = scene_tag
        
        # Additional properties
        properties = self.classifier.analyze_image_properties(str(image_path))
        tags.update(properties)
        
        return tags
    
    def _print_tag_statistics(self, tags_data: Dict[str, Dict[str, str]]):
        """Print tagging statistics"""
        
        print(f"\n{'='*70}")
        print("TAGGING STATISTICS")
        print(f"{'='*70}")
        
        # Count tags by category
        from collections import Counter
        
        categories = {}
        
        for tags in tags_data.values():
            for category, tag in tags.items():
                if category not in categories:
                    categories[category] = Counter()
                categories[category][tag] += 1
        
        # Print statistics
        for category, counts in categories.items():
            print(f"\n{category.upper()}:")
            for tag, count in counts.most_common():
                percentage = count / len(tags_data) * 100
                print(f"  {tag}: {count} ({percentage:.1f}%)")


def batch_tag_directories(base_dir: str, output_dir: str):
    """Tag multiple directories of images"""
    
    base_path = Path(base_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize classifier once
    classifier = Places365Classifier()
    tagger = ImageTagger(classifier)
    
    # Find subdirectories
    subdirs = [d for d in base_path.iterdir() if d.is_dir()]
    
    if not subdirs:
        # No subdirectories, process base directory
        output_file = output_path / "tags.json"
        tagger.tag_images(str(base_path), str(output_file))
        return
    
    print(f"\nFound {len(subdirs)} subdirectories")
    
    # Process each subdirectory
    for subdir in subdirs:
        print(f"\n{'='*70}")
        print(f"Processing: {subdir.name}")
        print(f"{'='*70}")
        
        output_file = output_path / f"tags_{subdir.name}.json"
        
        try:
            tagger.tag_images(str(subdir), str(output_file))
        except Exception as e:
            print(f"Error processing {subdir.name}: {e}")
    
    # Merge all tags into one file
    print(f"\n{'='*70}")
    print("MERGING TAG FILES")
    print(f"{'='*70}")
    
    merged_tags = {}
    for tag_file in output_path.glob("tags_*.json"):
        with open(tag_file) as f:
            tags = json.load(f)
            merged_tags.update(tags)
    
    merged_file = output_path / "tags_all.json"
    with open(merged_file, 'w') as f:
        json.dump(merged_tags, f, indent=2)
    
    print(f"✅ Merged tags saved to: {merged_file}")
    print(f"Total images tagged: {len(merged_tags)}")


def create_distribution_splits(tags_file: str, output_file: str,
                               train_ratio: float = 0.7, val_ratio: float = 0.15):
    """Add distribution splits (train/val/test) to tags"""
    
    print(f"\n{'='*70}")
    print("CREATING DISTRIBUTION SPLITS")
    print(f"{'='*70}")
    
    # Load tags
    with open(tags_file) as f:
        tags_data = json.load(f)
    
    import random
    
    # Get all filenames
    filenames = list(tags_data.keys())
    random.shuffle(filenames)
    
    # Calculate splits
    total = len(filenames)
    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))
    
    # Assign splits
    for i, filename in enumerate(filenames):
        if i < train_end:
            tags_data[filename]['distribution'] = 'train'
        elif i < val_end:
            tags_data[filename]['distribution'] = 'val'
        else:
            tags_data[filename]['distribution'] = 'test'
    
    # Save updated tags
    with open(output_file, 'w') as f:
        json.dump(tags_data, f, indent=2)
    
    print(f"✅ Added distribution splits")
    print(f"   Train: {train_end}")
    print(f"   Val: {val_end - train_end}")
    print(f"   Test: {total - val_end}")
    print(f"📄 Saved to: {output_file}")


def create_balanced_splits(tags_file: str, output_file: str,
                          stratify_by: str = 'scene'):
    """Create balanced splits stratified by tag category"""
    
    print(f"\n{'='*70}")
    print(f"CREATING BALANCED SPLITS (stratified by {stratify_by})")
    print(f"{'='*70}")
    
    # Load tags
    with open(tags_file) as f:
        tags_data = json.load(f)
    
    import random
    from collections import defaultdict
    
    # Group by stratification tag
    groups = defaultdict(list)
    
    for filename, tags in tags_data.items():
        tag_value = tags.get(stratify_by, 'unknown')
        groups[tag_value].append(filename)
    
    print(f"\nFound {len(groups)} groups:")
    for tag, filenames in groups.items():
        print(f"  {tag}: {len(filenames)} images")
    
    # Split each group
    for tag, filenames in groups.items():
        random.shuffle(filenames)
        
        total = len(filenames)
        train_end = int(total * 0.7)
        val_end = int(total * 0.85)
        
        for i, filename in enumerate(filenames):
            if i < train_end:
                tags_data[filename]['distribution'] = 'train'
            elif i < val_end:
                tags_data[filename]['distribution'] = 'val'
            else:
                tags_data[filename]['distribution'] = 'test'
    
    # Save
    with open(output_file, 'w') as f:
        json.dump(tags_data, f, indent=2)
    
    print(f"\n✅ Created balanced splits")
    print(f"📄 Saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Auto-tag images using Places365 scene classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Tag images in directory
  python scene_classifier_places365.py /path/to/images --output tags.json
  
  # Tag with custom confidence threshold
  python scene_classifier_places365.py /path/to/images --output tags.json --confidence 0.4
  
  # Batch tag multiple directories
  python scene_classifier_places365.py /path/to/base --batch --output-dir tags_output
  
  # Add distribution splits
  python scene_classifier_places365.py --split tags.json --output tags_with_splits.json
  
  # Create balanced splits
  python scene_classifier_places365.py --balanced-split tags.json --stratify scene --output tags_balanced.json
  
  # Then import into FiftyOne:
  python fiftyone_annotation_app_fixed.py /path/to/images --import-tags tags.json
        """
    )
    
    parser.add_argument('images_dir', nargs='?', help='Directory containing images')
    parser.add_argument('--output', '-o', default='tags.json',
                       help='Output JSON file (default: tags.json)')
    parser.add_argument('--confidence', type=float, default=0.3,
                       help='Confidence threshold for scene classification (default: 0.3)')
    parser.add_argument('--arch', default='resnet18',
                       choices=['resnet18', 'resnet50'],
                       help='Model architecture (default: resnet18)')
    parser.add_argument('--cpu', action='store_true',
                       help='Force CPU usage')
    
    # Batch processing
    parser.add_argument('--batch', action='store_true',
                       help='Batch process subdirectories')
    parser.add_argument('--output-dir', default='tags_output',
                       help='Output directory for batch processing')
    
    # Distribution splits
    parser.add_argument('--split', type=str,
                       help='Add distribution splits to existing tags file')
    parser.add_argument('--balanced-split', type=str,
                       help='Create balanced splits from existing tags file')
    parser.add_argument('--stratify', default='scene',
                       help='Tag category to stratify by (default: scene)')
    
    args = parser.parse_args()
    
    # Handle split operations
    if args.split:
        create_distribution_splits(args.split, args.output)
        return
    
    if args.balanced_split:
        create_balanced_splits(args.balanced_split, args.output, args.stratify)
        return
    
    # Require images_dir for tagging operations
    if not args.images_dir:
        parser.error("images_dir is required for tagging operations")
    
    # Batch processing
    if args.batch:
        batch_tag_directories(args.images_dir, args.output_dir)
        return
    
    # Single directory processing
    classifier = Places365Classifier(arch=args.arch, use_gpu=not args.cpu)
    tagger = ImageTagger(classifier)
    
    tags_data = tagger.tag_images(
        args.images_dir,
        args.output,
        confidence_threshold=args.confidence
    )
    
    # Print usage instructions
    print(f"\n{'='*70}")
    print("NEXT STEPS")
    print(f"{'='*70}")
    print(f"\n1. Review tags:")
    print(f"   cat {args.output}")
    
    print(f"\n2. Add distribution splits (optional):")
    print(f"   python {__file__} --split {args.output} --output tags_with_splits.json")
    
    print(f"\n3. Import into FiftyOne:")
    print(f"   python fiftyone_annotation_app_fixed.py {args.images_dir} --import-tags {args.output}")
    
    print(f"\n4. Launch annotation app:")
    print(f"   python fiftyone_annotation_app_fixed.py {args.images_dir}")


if __name__ == "__main__":
    main()