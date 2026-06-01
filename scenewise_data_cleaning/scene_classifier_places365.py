# save as: scene_classifier_places365.py

"""
Scene Classifier using Places365 - The Best Pretrained Model for Scene Recognition

Places365 is specifically trained on 365 scene categories, making it far superior
to VGG, ResNet (ImageNet), or general classification models.

Installation:
    pip install torch torchvision pillow opencv-python numpy tqdm

Usage:
    python scene_classifier_places365.py /path/to/images --scenes park highway night
    python scene_classifier_places365.py /path/to/images --list-available-scenes
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import argparse
from tqdm import tqdm
import shutil
import json
from datetime import datetime
import urllib.request

class Places365Classifier:
    """Scene classifier using Places365-trained models"""
    
    # Mapping of common scene types to Places365 categories
    SCENE_MAPPINGS = {
        'night': ['street', 'highway', 'parking_lot', 'alley', 'downtown'],
        'park': ['park', 'botanical_garden', 'playground', 'field/cultivated', 'lawn'],
        'city_street': ['street', 'downtown', 'shopfront', 'crosswalk', 'plaza'],
        'highway': ['highway', 'freeway', 'viaduct', 'toll_plaza'],
        'beach': ['beach', 'coast', 'sandbar', 'ocean'],
        'mountain': ['mountain', 'mountain_snowy', 'mountain_path', 'cliff', 'canyon'],
        'forest': ['forest_path', 'forest/broadleaf', 'forest_road', 'bamboo_forest', 'rainforest'],
        'indoor': ['bedroom', 'living_room', 'kitchen', 'office', 'classroom', 'restaurant'],
        'outdoor': ['street', 'park', 'beach', 'mountain', 'forest_path', 'highway'],
        'water': ['ocean', 'lake_natural', 'river', 'waterfall', 'pond'],
        'urban': ['street', 'downtown', 'skyscraper', 'building_facade', 'plaza'],
        'nature': ['forest_path', 'mountain', 'lake_natural', 'field_wild', 'valley'],
        'building': ['building_facade', 'skyscraper', 'house', 'apartment_building/outdoor'],
        'desert': ['desert_sand', 'desert_vegetation', 'badlands'],
        'snow': ['mountain_snowy', 'ice_skating_rink/outdoor', 'ski_resort', 'snowfield'],
        'sunset': ['sky', 'ocean', 'beach'],  # Detected by color, not scene type
        'bridge': ['bridge', 'viaduct', 'rope_bridge'],
        'tunnel': ['tunnel'],
        'parking': ['parking_lot', 'parking_garage/indoor', 'parking_garage/outdoor'],
        'stadium': ['stadium/baseball', 'stadium/football', 'stadium/soccer'],
        'airport': ['airport_terminal', 'hangar/outdoor', 'runway'],
        'train_station': ['train_station/platform', 'subway_station/platform'],
        'industrial': ['factory/outdoor', 'construction_site', 'warehouse/outdoor'],
        'residential': ['residential_neighborhood', 'apartment_building/outdoor', 'house']
    }
    
    def __init__(self, device: str = None):
        """
        Initialize Places365 classifier
        
        Args:
            device: 'cuda' or 'cpu'
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        print("Loading Places365-ResNet50 model...")
        self.model = self._load_places365_model()
        
        # Load scene categories
        self.categories = self._load_categories()
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        print(f"✅ Places365 model loaded on {self.device}")
        print(f"   Categories: {len(self.categories)}")
    
    def _load_places365_model(self) -> nn.Module:
        """Load Places365-trained ResNet50"""
        
        # Use ResNet50 architecture
        model = models.resnet50(pretrained=False)
        
        # Modify for Places365 (365 categories)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 365)
        
        # Download weights if not present
        weights_path = Path('resnet50_places365.pth')
        
        if not weights_path.exists():
            print("Downloading Places365 weights (92MB)...")
            url = 'http://places2.csail.mit.edu/models_places365/resnet50_places365.pth.tar'
            
            try:
                urllib.request.urlretrieve(url, weights_path)
                print("✅ Downloaded weights")
            except Exception as e:
                print(f"❌ Could not download weights: {e}")
                print("Please download manually from:")
                print("http://places2.csail.mit.edu/models_places365/")
                raise
        
        # Load weights
        checkpoint = torch.load(weights_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if 'state_dict' in checkpoint:
            state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
        else:
            state_dict = checkpoint
        
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        
        return model
    
    def _load_categories(self) -> List[str]:
        """Load Places365 category names"""
        
        categories_file = Path('categories_places365.txt')
        
        if not categories_file.exists():
            print("Downloading category names...")
            url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
            
            try:
                urllib.request.urlretrieve(url, categories_file)
            except:
                # Fallback: use embedded list
                return self._get_embedded_categories()
        
        # Load categories
        categories = []
        with open(categories_file) as f:
            for line in f:
                category = line.strip().split(' ')[0]
                category = category.split('/')[2:]  # Remove /a/abbey/
                category = '/'.join(category) if category else line.strip()
                categories.append(category)
        
        return categories
    
    def _get_embedded_categories(self) -> List[str]:
        """Embedded category list (top 50 common scenes)"""
        return [
            'airfield', 'airport_terminal', 'alley', 'amphitheater', 'apartment_building/outdoor',
            'arch', 'archive', 'arrival_gate', 'art_gallery', 'art_studio', 'assembly_line',
            'athletic_field/outdoor', 'attic', 'auditorium', 'auto_factory', 'badlands',
            'ballroom', 'bamboo_forest', 'bank_vault', 'bar', 'barn', 'baseball_field',
            'basement', 'basketball_court/outdoor', 'bathroom', 'beach', 'beauty_salon',
            'bedroom', 'boardwalk', 'boat_deck', 'bookstore', 'botanical_garden', 'bridge',
            'building_facade', 'bus_interior', 'cafeteria', 'campsite', 'campus', 'canyon',
            'castle', 'cemetery', 'chalet', 'church/outdoor', 'classroom', 'cliff',
            'closet', 'coast', 'construction_site', 'corridor', 'cottage_garden', 'courthouse',
            'courtyard', 'creek', 'crosswalk', 'dam', 'desert_sand', 'desert_vegetation',
            'diner/outdoor', 'downtown', 'driveway', 'dwelling', 'field/cultivated', 'field/wild',
            'fire_escape', 'forest_path', 'forest_road', 'forest/broadleaf', 'freeway',
            'garage/indoor', 'gas_station', 'golf_course', 'gymnasium/indoor', 'harbor',
            'highway', 'home_office', 'hospital_room', 'hotel_room', 'house', 'ice_skating_rink/outdoor',
            'industrial_area', 'intersection', 'kitchen', 'lake_natural', 'lawn', 'library/outdoor',
            'living_room', 'mansion', 'mountain', 'mountain_path', 'mountain_snowy', 'museum/indoor',
            'ocean', 'office', 'office_building', 'park', 'parking_garage/outdoor', 'parking_lot',
            'pasture', 'patio', 'phone_booth', 'plaza', 'pond', 'promenade', 'rainforest',
            'reception', 'residential_neighborhood', 'restaurant', 'restaurant_kitchen', 'river',
            'road', 'rope_bridge', 'runway', 'sandbar', 'schoolhouse', 'shop', 'shopfront',
            'shower', 'ski_resort', 'ski_slope', 'skyscraper', 'slum', 'snowfield',
            'soccer_field', 'stable', 'stadium/baseball', 'stadium/football', 'staircase',
            'street', 'subway_station/platform', 'supermarket', 'swamp', 'swimming_pool/outdoor',
            'synagogue/outdoor', 'television_studio', 'temple/east_asia', 'tower', 'train_railway',
            'train_station/platform', 'tree_farm', 'trench', 'tunnel', 'underwater/coral_reef',
            'valley', 'vegetable_garden', 'veranda', 'viaduct', 'village', 'vineyard',
            'waiting_room', 'warehouse/outdoor', 'waterfall', 'watering_hole', 'wheat_field',
            'wind_farm', 'windmill', 'yard', 'zen_garden'
        ]
    
    def predict_scene(self, image_path: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Predict scene categories for an image
        
        Returns:
            List of (category, probability) tuples, sorted by probability
        """
        try:
            # Load and preprocess image
            img = Image.open(image_path).convert('RGB')
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            
            # Predict
            with torch.no_grad():
                logits = self.model(img_tensor)
                probs = torch.nn.functional.softmax(logits, dim=1)
            
            # Get top predictions
            top_probs, top_indices = torch.topk(probs, k=min(top_k, len(self.categories)))
            
            results = []
            for prob, idx in zip(top_probs[0], top_indices[0]):
                category = self.categories[idx] if idx < len(self.categories) else 'unknown'
                results.append((category, float(prob)))
            
            return results
            
        except Exception as e:
            print(f"Error predicting {image_path}: {e}")
            return []
    
    def detect_scene_type(self, image_path: str, scene_type: str) -> Tuple[bool, float]:
        """
        Detect if image matches a specific scene type
        
        Args:
            image_path: Path to image
            scene_type: Scene type from SCENE_MAPPINGS
            
        Returns:
            (is_match, confidence)
        """
        # Get top predictions
        predictions = self.predict_scene(image_path, top_k=10)
        
        if not predictions:
            return False, 0.0
        
        # Get relevant categories for this scene type
        if scene_type not in self.SCENE_MAPPINGS:
            print(f"⚠️  Unknown scene type: {scene_type}")
            return False, 0.0
        
        relevant_categories = self.SCENE_MAPPINGS[scene_type]
        
        # Calculate confidence as sum of probabilities for relevant categories
        total_confidence = 0.0
        matches = []
        
        for category, prob in predictions:
            for relevant in relevant_categories:
                if relevant in category or category in relevant:
                    total_confidence += prob
                    matches.append((category, prob))
        
        # Special handling for time-of-day (night) using color analysis
        if scene_type == 'night':
            brightness_conf = self._detect_night_brightness(image_path)
            total_confidence = (total_confidence * 0.4 + brightness_conf * 0.6)
        
        # Special handling for sunset
        if scene_type == 'sunset':
            sunset_conf = self._detect_sunset_colors(image_path)
            total_confidence = max(total_confidence, sunset_conf)
        
        is_match = total_confidence > 0.35  # Lower threshold since we sum probabilities
        
        return is_match, min(total_confidence, 1.0)
    
    def _detect_night_brightness(self, image_path: str) -> float:
        """Detect night based on brightness"""
        import cv2
        
        img = cv2.imread(str(image_path))
        if img is None:
            return 0.0
        
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        brightness = np.mean(hsv[:, :, 2])
        
        # Convert brightness to night confidence
        if brightness < 60:
            return 0.9
        elif brightness < 100:
            return 0.7
        elif brightness < 130:
            return 0.4
        else:
            return 0.1
    
    def _detect_sunset_colors(self, image_path: str) -> float:
        """Detect sunset based on orange/red colors in sky"""
        import cv2
        
        img = cv2.imread(str(image_path))
        if img is None:
            return 0.0
        
        h, w = img.shape[:2]
        
        # Check top third of image (sky)
        sky_region = img[:h//3, :]
        hsv_sky = cv2.cvtColor(sky_region, cv2.COLOR_BGR2HSV)
        
        # Orange/red/pink colors
        orange_mask1 = cv2.inRange(hsv_sky, np.array([0, 50, 50]), np.array([20, 255, 255]))
        orange_mask2 = cv2.inRange(hsv_sky, np.array([160, 50, 50]), np.array([180, 255, 255]))
        orange_mask = cv2.bitwise_or(orange_mask1, orange_mask2)
        
        orange_pct = np.sum(orange_mask > 0) / orange_mask.size
        
        return min(orange_pct * 3, 1.0)
    
    def classify_image(self, image_path: str, scene_types: List[str]) -> Dict[str, Tuple[bool, float]]:
        """Classify image for multiple scene types"""
        
        results = {}
        
        for scene_type in scene_types:
            is_match, confidence = self.detect_scene_type(image_path, scene_type)
            results[scene_type] = (is_match, confidence)
        
        return results
    
    def get_top_scenes(self, image_path: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Get top predicted scene categories"""
        return self.predict_scene(image_path, top_k=top_k)
    
    def list_available_scenes(self):
        """List all scene types that can be detected"""
        print("\n" + "="*70)
        print("AVAILABLE SCENE TYPES (Places365-based)")
        print("="*70)
        
        for scene_type, categories in sorted(self.SCENE_MAPPINGS.items()):
            print(f"\n{scene_type}:")
            print(f"  Mapped to Places365 categories:")
            for cat in categories:
                print(f"    - {cat}")
        
        print("\n" + "="*70)
        print(f"Total scene types available: {len(self.SCENE_MAPPINGS)}")
        print("="*70)
    
    def process_directory(self, input_dir: str, scene_types: List[str],
                         output_dir: str = None, preview: bool = False,
                         copy: bool = False, multi_label: bool = False,
                         force_classify: bool = False) -> Dict:
        """Process directory and organize by scene type"""
        
        input_path = Path(input_dir)
        
        if output_dir is None:
            output_dir = input_path / "classified_places365"
        output_path = Path(output_dir)
        
        # Get images
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        image_files = []
        for ext in extensions:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))
        
        if not image_files:
            print(f"No images found in {input_dir}")
            return {}
        
        print(f"\n{'='*70}")
        print("SCENE CLASSIFICATION - Places365")
        print(f"{'='*70}")
        print(f"Input: {input_dir}")
        print(f"Output: {output_dir}")
        print(f"Images: {len(image_files)}")
        print(f"Scene types: {', '.join(scene_types)}")
        print(f"{'='*70}\n")
        
        # Process
        results = {scene: [] for scene in scene_types}
        results['unclassified'] = []
        results['errors'] = []
        
        print("Classifying images...")
        for image_path in tqdm(image_files, desc="Processing"):
            try:
                classifications = self.classify_image(str(image_path), scene_types)
                
                # Collect scenes that pass threshold, or all scenes if force_classify
                all_confs = [(scene, conf)
                             for scene, (_, conf) in classifications.items()]
                all_confs.sort(key=lambda x: x[1], reverse=True)

                if force_classify:
                    # Assign to the highest-confidence scene unconditionally;
                    # multi_label still works — just no threshold filter
                    matches = all_confs
                else:
                    matches = [(scene, conf)
                               for scene, (is_match, conf) in classifications.items()
                               if is_match]

                if not matches:
                    top_preds = self.get_top_scenes(str(image_path), top_k=3)
                    results['unclassified'].append({
                        'path': str(image_path),
                        'filename': image_path.name,
                        'top_predictions': top_preds
                    })
                else:
                    matches.sort(key=lambda x: x[1], reverse=True)

                    if multi_label:
                        for scene, conf in matches:
                            results[scene].append({
                                'path': str(image_path),
                                'filename': image_path.name,
                                'confidence': float(conf)
                            })
                    else:
                        best_scene, best_conf = matches[0]
                        results[best_scene].append({
                            'path': str(image_path),
                            'filename': image_path.name,
                            'confidence': float(best_conf)
                        })
                    
            except Exception as e:
                results['errors'].append({
                    'path': str(image_path),
                    'error': str(e)
                })
        
        # Summary
        print(f"\n{'='*70}")
        print("CLASSIFICATION SUMMARY")
        print(f"{'='*70}")
        
        for scene in scene_types:
            count = len(results[scene])
            pct = count / len(image_files) * 100 if image_files else 0
            print(f"{scene:20s}: {count:4d} ({pct:5.1f}%)")
        
        unclass_count = len(results['unclassified'])
        print(f"{'Unclassified':20s}: {unclass_count:4d} ({unclass_count/len(image_files)*100:5.1f}%)")
        print(f"{'Errors':20s}: {len(results['errors']):4d}")
        print(f"{'='*70}\n")
        
        # Show top predictions for unclassified
        if results['unclassified'][:5]:
            print("Top predictions for some unclassified images:")
            for item in results['unclassified'][:5]:
                print(f"  {item['filename']}:")
                for cat, prob in item['top_predictions']:
                    print(f"    {cat}: {prob:.1%}")
        
        # Preview
        if preview:
            self._show_preview(results, scene_types)
            response = input("\nProceed with organizing? (y/n): ")
            if response.lower() != 'y':
                print("Cancelled")
                return results
        
        # Organize files
        print(f"\n{'Copying' if copy else 'Moving'} files...")
        
        for scene in scene_types:
            if not results[scene]:
                continue
            
            scene_dir = output_path / scene
            scene_dir.mkdir(parents=True, exist_ok=True)
            
            for item in tqdm(results[scene], desc=f"{scene:15s}"):
                try:
                    src = Path(item['path'])
                    dst = scene_dir / item['filename']
                    
                    if dst.exists():
                        counter = 1
                        while dst.exists():
                            dst = scene_dir / f"{dst.stem}_{counter}{dst.suffix}"
                            counter += 1
                    
                    if copy or multi_label:
                        shutil.copy2(src, dst)
                    else:
                        shutil.move(str(src), str(dst))
                        
                except Exception as e:
                    print(f"\nError: {e}")
        
        # Handle unclassified
        if results['unclassified']:
            unclass_dir = output_path / "unclassified"
            unclass_dir.mkdir(parents=True, exist_ok=True)
            
            for item in results['unclassified']:
                src = Path(item['path'])
                dst = unclass_dir / item['filename']
                
                try:
                    if copy:
                        shutil.copy2(src, dst)
                    else:
                        shutil.move(str(src), str(dst))
                except:
                    pass
        
        print(f"\n✅ Complete! Location: {output_path}")
        
        # Save report
        report_file = output_path / f"places365_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"   Report: {report_file}")
        
        return results
    
    def _show_preview(self, results: Dict, scene_types: List[str], max_per: int = 5):
        """Show preview"""
        print(f"\n{'='*70}")
        print("PREVIEW")
        print(f"{'='*70}")
        
        for scene in scene_types:
            if not results[scene]:
                continue
            
            print(f"\n{scene.upper()}:")
            for i, item in enumerate(results[scene][:max_per], 1):
                print(f"  {i}. {item['filename']} ({item['confidence']:.1%})")
            
            if len(results[scene]) > max_per:
                print(f"  ... and {len(results[scene]) - max_per} more")
        
        print(f"\n{'='*70}")


def test_single_image(image_path: str, classifier: Places365Classifier):
    """Test classification on single image with detailed output"""
    
    print(f"\n{'='*70}")
    print(f"TESTING: {image_path}")
    print(f"{'='*70}\n")
    
    # Get top predictions
    print("Top 10 Places365 predictions:")
    predictions = classifier.get_top_scenes(image_path, top_k=10)
    
    for i, (category, prob) in enumerate(predictions, 1):
        print(f"  {i}. {category:30s}: {prob:.1%}")
    
    # Test against common scene types
    print(f"\n{'='*70}")
    print("Scene Type Matches:")
    print(f"{'='*70}")
    
    common_scenes = ['night', 'park', 'city_street', 'highway', 'mountain', 
                    'beach', 'forest', 'indoor', 'outdoor']
    
    for scene in common_scenes:
        is_match, confidence = classifier.detect_scene_type(image_path, scene)
        status = "✓ MATCH" if is_match else "✗ no match"
        print(f"{scene:15s}: {confidence:.3f} [{status}]")
    
    print(f"\n{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Scene Classifier using Places365",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available scene types
  python scene_classifier_places365.py --list-scenes
  
  # Test single image
  python scene_classifier_places365.py --test-image photo.jpg
  
  # Classify directory
  python scene_classifier_places365.py /photos --scenes park highway city_street
  
  # Classify all common scenes
  python scene_classifier_places365.py /photos --scenes night day park highway beach mountain
  
  # Preview before organizing
  python scene_classifier_places365.py /photos --scenes park highway --preview
  
  # Copy instead of move
  python scene_classifier_places365.py /photos --scenes night --copy
        """
    )
    
    parser.add_argument('input_dir', nargs='?', help='Directory with images')
    parser.add_argument('--scenes', '-s', nargs='+', help='Scene types to detect')
    parser.add_argument('--output', '-o', help='Output directory')
    parser.add_argument('--preview', '-p', action='store_true')
    parser.add_argument('--copy', action='store_true')
    parser.add_argument('--multi-label', action='store_true')
    parser.add_argument('--force-classify', action='store_true',
                        help='Assign every image to its highest-confidence scene regardless '
                             'of threshold — no image goes to unclassified')
    parser.add_argument('--test-image', help='Test single image')
    parser.add_argument('--list-scenes', action='store_true')
    parser.add_argument('--device', choices=['cuda', 'cpu'])
    
    args = parser.parse_args()
    
    # Create classifier
    classifier = Places365Classifier(device=args.device)
    
    # List scenes
    if args.list_scenes:
        classifier.list_available_scenes()
        print("\nAll 365 Places365 categories available for fine-grained classification!")
        return
    
    # Test single image
    if args.test_image:
        if not Path(args.test_image).exists():
            print(f"❌ Image not found: {args.test_image}")
            return
        
        test_single_image(args.test_image, classifier)
        return
    
    # Validate input
    if not args.input_dir:
        parser.print_help()
        return
    
    if not args.scenes:
        print("❌ Please specify --scenes")
        print("Use --list-scenes to see available options")
        return
    
    # Process directory
    results = classifier.process_directory(
        input_dir=args.input_dir,
        scene_types=args.scenes,
        output_dir=args.output,
        preview=args.preview,
        copy=args.copy,
        multi_label=args.multi_label,
        force_classify=args.force_classify,
    )


if __name__ == "__main__":
    main()