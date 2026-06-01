# save as: scene_classifier.py

"""
Advanced Scene Classifier for Images

Detects multiple scene types:
- Night/Day
- Parks
- City Streets
- Highways
- Indoor/Outdoor
- Natural/Urban
- And more...

Installation:
    pip install torch torchvision pillow opencv-python numpy tqdm transformers

Usage:
    python scene_classifier.py /path/to/images --scenes night park highway
    python scene_classifier.py /path/to/images --scenes all --preview
"""

import torch
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
import argparse
from tqdm import tqdm
import shutil
import json
from datetime import datetime
from PIL import Image
import warnings
warnings.filterwarnings('ignore')
import truststore
truststore.inject_into_ssl()

# Scene definitions with detection prompts
SCENE_DEFINITIONS = {
    'night': {
        'name': 'Night Images',
        'clip_prompts': [
            "a photo taken during daytime with bright sunlight",
            "a photo taken at night in darkness"
        ],
        'positive_index': 1,
        'description': 'Images taken at night or in low light'
    },
    'day': {
        'name': 'Daytime Images',
        'clip_prompts': [
            "a photo taken at night in darkness",
            "a photo taken during daytime with bright sunlight"
        ],
        'positive_index': 1,
        'description': 'Images taken during daytime'
    },
    'park': {
        'name': 'Parks',
        'clip_prompts': [
            "a photo of a city street or urban area",
            "a photo of a park with trees and grass"
        ],
        'positive_index': 1,
        'cv_features': ['green_dominance', 'tree_detection'],
        'description': 'Parks, gardens, and green spaces'
    },
    'city_street': {
        'name': 'City Streets',
        'clip_prompts': [
            "a photo of nature or countryside",
            "a photo of a city street with buildings and roads"
        ],
        'positive_index': 1,
        'cv_features': ['edge_density', 'vertical_lines'],
        'description': 'Urban streets with buildings'
    },
    'highway': {
        'name': 'Highways',
        'clip_prompts': [
            "a photo of a city street or residential area",
            "a photo of a highway or freeway with multiple lanes"
        ],
        'positive_index': 1,
        'cv_features': ['horizontal_lines', 'perspective'],
        'description': 'Highways and freeways'
    },
    'indoor': {
        'name': 'Indoor',
        'clip_prompts': [
            "a photo taken outdoors in nature",
            "a photo taken indoors inside a building"
        ],
        'positive_index': 1,
        'description': 'Indoor scenes'
    },
    'outdoor': {
        'name': 'Outdoor',
        'clip_prompts': [
            "a photo taken indoors inside a building",
            "a photo taken outdoors in nature"
        ],
        'positive_index': 1,
        'description': 'Outdoor scenes'
    },
    'urban': {
        'name': 'Urban',
        'clip_prompts': [
            "a photo of natural scenery or wilderness",
            "a photo of urban area with buildings and infrastructure"
        ],
        'positive_index': 1,
        'description': 'Urban and city environments'
    },
    'nature': {
        'name': 'Nature',
        'clip_prompts': [
            "a photo of urban area with buildings",
            "a photo of nature with trees, mountains, or water"
        ],
        'positive_index': 1,
        'cv_features': ['green_dominance', 'sky_detection'],
        'description': 'Natural landscapes'
    },
    'beach': {
        'name': 'Beach',
        'clip_prompts': [
            "a photo of a city or forest",
            "a photo of a beach with sand and water"
        ],
        'positive_index': 1,
        'cv_features': ['sand_detection', 'water_blue'],
        'description': 'Beach and coastal scenes'
    },
    'mountain': {
        'name': 'Mountain',
        'clip_prompts': [
            "a photo of flat landscape or city",
            "a photo of mountains or mountainous terrain"
        ],
        'positive_index': 1,
        'cv_features': ['elevation_profile'],
        'description': 'Mountain landscapes'
    },
    'vehicle': {
        'name': 'Vehicles',
        'clip_prompts': [
            "a photo without any vehicles",
            "a photo with cars, trucks, or other vehicles"
        ],
        'positive_index': 1,
        'description': 'Images containing vehicles'
    },
    'people': {
        'name': 'People',
        'clip_prompts': [
            "a photo without any people",
            "a photo with people or crowds"
        ],
        'positive_index': 1,
        'description': 'Images containing people'
    },
    'building': {
        'name': 'Buildings',
        'clip_prompts': [
            "a photo of nature without buildings",
            "a photo of buildings or architecture"
        ],
        'positive_index': 1,
        'cv_features': ['vertical_lines', 'rectangular_shapes'],
        'description': 'Images with buildings'
    },
    'water': {
        'name': 'Water',
        'clip_prompts': [
            "a photo without water",
            "a photo of water, ocean, lake, or river"
        ],
        'positive_index': 1,
        'cv_features': ['water_blue', 'horizontal_gradient'],
        'description': 'Images with water bodies'
    },
    'sunset': {
        'name': 'Sunset/Sunrise',
        'clip_prompts': [
            "a photo taken at midday",
            "a photo of sunset or sunrise with orange sky"
        ],
        'positive_index': 1,
        'cv_features': ['orange_sky'],
        'description': 'Sunset or sunrise scenes'
    },
    'rain': {
        'name': 'Rain/Wet',
        'clip_prompts': [
            "a photo on a clear sunny day",
            "a photo in rain or with wet surfaces"
        ],
        'positive_index': 1,
        'description': 'Rainy or wet conditions'
    },
    'snow': {
        'name': 'Snow',
        'clip_prompts': [
            "a photo without snow",
            "a photo with snow on the ground"
        ],
        'positive_index': 1,
        'cv_features': ['white_dominance'],
        'description': 'Snowy scenes'
    },
    'forest': {
        'name': 'Forest',
        'clip_prompts': [
            "a photo of open space or urban area",
            "a photo of forest with dense trees"
        ],
        'positive_index': 1,
        'cv_features': ['green_dominance', 'tree_density'],
        'description': 'Forest and woodland'
    },
    'desert': {
        'name': 'Desert',
        'clip_prompts': [
            "a photo of lush vegetation or city",
            "a photo of desert with sand dunes"
        ],
        'positive_index': 1,
        'cv_features': ['sand_color'],
        'description': 'Desert landscapes'
    }
}


class SceneClassifier:
    """Multi-scene classifier for images"""
    
    def __init__(self, scenes: List[str] = None, method: str = "hybrid", device: str = None):
        """
        Initialize scene classifier
        
        Args:
            scenes: List of scene types to detect (None = all)
            method: Detection method ('clip', 'cv', 'hybrid')
            device: Device for torch models
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.method = method
        
        # Validate scenes
        if scenes is None or 'all' in scenes:
            self.scenes = list(SCENE_DEFINITIONS.keys())
        else:
            invalid = [s for s in scenes if s not in SCENE_DEFINITIONS]
            if invalid:
                print(f"⚠️  Unknown scenes: {invalid}")
                print(f"Available: {list(SCENE_DEFINITIONS.keys())}")
            self.scenes = [s for s in scenes if s in SCENE_DEFINITIONS]
        
        if not self.scenes:
            raise ValueError("No valid scenes specified")
        
        # Load models
        self.clip_model = None
        self.clip_processor = None
        
        if method in ['clip', 'hybrid']:
            self._load_clip_model()
        
        print(f"\nInitialized SceneClassifier")
        print(f"  Scenes: {', '.join(self.scenes)}")
        print(f"  Method: {method}")
        print(f"  Device: {self.device}")
    
    def _load_clip_model(self):
        """Load CLIP model"""
        try:
            print("\nLoading CLIP model...")
            from transformers import CLIPProcessor, CLIPModel
            
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_model.to(self.device)
            self.clip_model.eval()
            
            print("✅ CLIP model loaded")
        except Exception as e:
            print(f"⚠️  Could not load CLIP: {e}")
            self.method = 'cv'
    
    def detect_scene_clip(self, image_path: str, scene_type: str) -> Tuple[bool, float]:
        """Detect scene using CLIP"""
        if self.clip_model is None:
            return False, 0.0
        
        try:
            scene_def = SCENE_DEFINITIONS[scene_type]
            
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Process
            inputs = self.clip_processor(
                text=scene_def['clip_prompts'],
                images=image,
                return_tensors="pt",
                padding=True
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.clip_model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)
            
            # Get probability for positive class
            pos_prob = float(probs[0][scene_def['positive_index']])
            is_scene = pos_prob > 0.5
            
            return is_scene, pos_prob
            
        except Exception as e:
            return False, 0.0
    
    def detect_scene_cv(self, image_path: str, scene_type: str) -> Tuple[bool, float]:
        """Detect scene using computer vision features"""
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                return False, 0.0
            
            # Get features for this scene type
            scene_def = SCENE_DEFINITIONS[scene_type]
            cv_features = scene_def.get('cv_features', [])
            
            if not cv_features:
                # No CV features defined, return neutral
                return False, 0.5
            
            # Extract features
            features = self._extract_cv_features(img)
            
            # Score based on required features
            scores = []
            
            for feature in cv_features:
                if feature in features:
                    scores.append(features[feature])
            
            if not scores:
                return False, 0.5
            
            confidence = np.mean(scores)
            is_scene = confidence > 0.5
            
            return is_scene, confidence
            
        except Exception as e:
            return False, 0.0
    
    def _extract_cv_features(self, img: np.ndarray) -> Dict[str, float]:
        """Extract computer vision features from image"""
        features = {}
        
        h, w = img.shape[:2]
        
        # Convert to different color spaces
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Green dominance (for parks, nature, forest)
        green_mask = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
        green_pct = np.sum(green_mask > 0) / green_mask.size
        features['green_dominance'] = min(green_pct * 3, 1.0)
        
        # Tree detection (vertical green structures)
        features['tree_detection'] = green_pct * 0.8 if green_pct > 0.2 else 0.3
        features['tree_density'] = min(green_pct * 2.5, 1.0)
        
        # Edge density (for urban scenes, buildings)
        edges = cv2.Canny(gray, 50, 150)
        edge_pct = np.sum(edges > 0) / edges.size
        features['edge_density'] = min(edge_pct * 5, 1.0)
        
        # Vertical lines (for buildings, city)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        vertical_edges = np.abs(sobelx)
        features['vertical_lines'] = min(np.mean(vertical_edges) / 50, 1.0)
        
        # Horizontal lines (for highways, horizons)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        horizontal_edges = np.abs(sobely)
        features['horizontal_lines'] = min(np.mean(horizontal_edges) / 50, 1.0)
        
        # Perspective convergence (for highways)
        # Detect vanishing point
        features['perspective'] = 0.6 if features['horizontal_lines'] > 0.5 else 0.4
        
        # Sky detection (upper part of image is blue)
        upper_third = img[:h//3, :]
        upper_hsv = cv2.cvtColor(upper_third, cv2.COLOR_BGR2HSV)
        sky_mask = cv2.inRange(upper_hsv, np.array([100, 50, 50]), np.array([130, 255, 255]))
        sky_pct = np.sum(sky_mask > 0) / sky_mask.size
        features['sky_detection'] = min(sky_pct * 2, 1.0)
        
        # Water detection (blue/cyan)
        water_mask = cv2.inRange(hsv, np.array([90, 50, 50]), np.array([130, 255, 255]))
        water_pct = np.sum(water_mask > 0) / water_mask.size
        features['water_blue'] = min(water_pct * 3, 1.0)
        
        # Horizontal gradient (for water)
        features['horizontal_gradient'] = 0.7 if water_pct > 0.2 else 0.3
        
        # Sand detection (yellowish/beige)
        sand_mask = cv2.inRange(hsv, np.array([15, 20, 100]), np.array([35, 150, 255]))
        sand_pct = np.sum(sand_mask > 0) / sand_mask.size
        features['sand_detection'] = min(sand_pct * 2.5, 1.0)
        features['sand_color'] = min(sand_pct * 2, 1.0)
        
        # White dominance (for snow)
        white_mask = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 50, 255]))
        white_pct = np.sum(white_mask > 0) / white_mask.size
        features['white_dominance'] = min(white_pct * 2, 1.0)
        
        # Orange sky (sunset/sunrise)
        orange_mask = cv2.inRange(upper_hsv, np.array([10, 50, 100]), np.array([25, 255, 255]))
        orange_pct = np.sum(orange_mask > 0) / orange_mask.size
        features['orange_sky'] = min(orange_pct * 3, 1.0)
        
        # Rectangular shapes (buildings)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rect_count = 0
        for cnt in contours:
            if cv2.contourArea(cnt) > 100:
                epsilon = 0.02 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                if len(approx) == 4:
                    rect_count += 1
        features['rectangular_shapes'] = min(rect_count / 50, 1.0)
        
        # Elevation profile (for mountains)
        # Analyze top edge profile
        top_edges = edges[:h//2, :]
        profile_variance = np.var(np.sum(top_edges, axis=0))
        features['elevation_profile'] = min(profile_variance / 10000, 1.0)
        
        return features
    
    def classify_image(self, image_path: str, 
                       confidence_threshold: float = 0.5) -> Dict[str, Tuple[bool, float]]:
        """
        Classify image across all configured scenes
        
        Returns:
            Dictionary mapping scene_type -> (is_match, confidence)
        """
        results = {}
        
        for scene_type in self.scenes:
            if self.method == 'clip':
                is_match, conf = self.detect_scene_clip(image_path, scene_type)
            elif self.method == 'cv':
                is_match, conf = self.detect_scene_cv(image_path, scene_type)
            else:  # hybrid
                # Try CV first
                is_cv, conf_cv = self.detect_scene_cv(image_path, scene_type)
                
                # Use CLIP if CV is uncertain or for scenes without CV features
                if 0.3 < conf_cv < 0.7 or conf_cv == 0.5:
                    is_clip, conf_clip = self.detect_scene_clip(image_path, scene_type)
                    conf = (conf_cv * 0.4 + conf_clip * 0.6)
                    is_match = conf > confidence_threshold
                else:
                    is_match, conf = is_cv, conf_cv
            
            results[scene_type] = (is_match, conf)
        
        return results
    
    def process_directory(self, input_dir: str, output_dir: str = None,
                         confidence_threshold: float = 0.5, 
                         preview: bool = False,
                         copy: bool = False,
                         multi_label: bool = True) -> Dict:
        """
        Process directory and organize images by scene type
        
        Args:
            input_dir: Input directory
            output_dir: Output directory (default: input_dir/classified)
            confidence_threshold: Minimum confidence
            preview: Show preview before organizing
            copy: Copy instead of move
            multi_label: Allow images in multiple categories
        """
        input_path = Path(input_dir)
        
        if output_dir is None:
            output_dir = input_path / "classified"
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
        print(f"SCENE CLASSIFICATION")
        print(f"{'='*70}")
        print(f"Input: {input_dir}")
        print(f"Output: {output_dir}")
        print(f"Images: {len(image_files)}")
        print(f"Scenes: {', '.join(self.scenes)}")
        print(f"Method: {self.method}")
        print(f"Confidence: {confidence_threshold}")
        print(f"Multi-label: {multi_label}")
        print(f"{'='*70}\n")
        
        # Process images
        results = {scene: [] for scene in self.scenes}
        results['unclassified'] = []
        results['errors'] = []
        
        print("Classifying images...")
        for image_path in tqdm(image_files, desc="Processing"):
            try:
                classifications = self.classify_image(str(image_path), confidence_threshold)
                
                # Find matching scenes
                matches = [
                    (scene, conf) for scene, (is_match, conf) in classifications.items()
                    if is_match and conf >= confidence_threshold
                ]
                
                if not matches:
                    results['unclassified'].append({
                        'path': str(image_path),
                        'filename': image_path.name,
                        'classifications': {k: v[1] for k, v in classifications.items()}
                    })
                else:
                    # Sort by confidence
                    matches.sort(key=lambda x: x[1], reverse=True)
                    
                    if multi_label:
                        # Add to all matching categories
                        for scene, conf in matches:
                            results[scene].append({
                                'path': str(image_path),
                                'filename': image_path.name,
                                'confidence': float(conf),
                                'all_matches': [{'scene': s, 'conf': c} for s, c in matches]
                            })
                    else:
                        # Add to best matching category only
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
        
        for scene in self.scenes:
            count = len(results[scene])
            pct = count / len(image_files) * 100 if image_files else 0
            print(f"{SCENE_DEFINITIONS[scene]['name']:20s}: {count:4d} ({pct:5.1f}%)")
        
        unclass_count = len(results['unclassified'])
        print(f"{'Unclassified':20s}: {unclass_count:4d} ({unclass_count/len(image_files)*100:5.1f}%)")
        print(f"{'Errors':20s}: {len(results['errors']):4d}")
        print(f"{'='*70}\n")
        
        # Preview
        if preview:
            self._show_preview(results)
            response = input("\nProceed with organizing files? (y/n): ")
            if response.lower() != 'y':
                print("Operation cancelled.")
                return results
        
        # Organize files
        print(f"\n{'Copying' if copy else 'Moving'} files...")
        
        for scene in self.scenes:
            if not results[scene]:
                continue
            
            scene_dir = output_path / scene
            scene_dir.mkdir(parents=True, exist_ok=True)
            
            for item in tqdm(results[scene], desc=f"{scene:15s}"):
                try:
                    src = Path(item['path'])
                    dst = scene_dir / item['filename']
                    
                    # Handle duplicates
                    if dst.exists():
                        stem = dst.stem
                        suffix = dst.suffix
                        counter = 1
                        while dst.exists():
                            dst = scene_dir / f"{stem}_{counter}{suffix}"
                            counter += 1
                    
                    if copy or (multi_label and len(item.get('all_matches', [])) > 1):
                        shutil.copy2(src, dst)
                    else:
                        shutil.move(str(src), str(dst))
                        
                except Exception as e:
                    print(f"\nError organizing {item['filename']}: {e}")
        
        # Handle unclassified
        if results['unclassified']:
            unclass_dir = output_path / "unclassified"
            unclass_dir.mkdir(parents=True, exist_ok=True)
            
            for item in results['unclassified']:
                try:
                    src = Path(item['path'])
                    dst = unclass_dir / item['filename']
                    
                    if copy:
                        shutil.copy2(src, dst)
                    else:
                        shutil.move(str(src), str(dst))
                except:
                    pass
        
        print(f"\n✅ Organization complete!")
        print(f"   Location: {output_path}")
        
        # Save report
        report_file = output_path / f"classification_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Prepare serializable results
        report_data = {}
        for key, items in results.items():
            if isinstance(items, list):
                report_data[key] = items
        
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"   Report: {report_file}")
        
        return results
    
    def _show_preview(self, results: Dict, max_per_scene: int = 5):
        """Show preview of classifications"""
        print(f"\n{'='*70}")
        print("CLASSIFICATION PREVIEW")
        print(f"{'='*70}")
        
        for scene in self.scenes:
            if not results[scene]:
                continue
            
            print(f"\n{SCENE_DEFINITIONS[scene]['name']}:")
            for i, item in enumerate(results[scene][:max_per_scene], 1):
                conf = item['confidence']
                print(f"  {i}. {item['filename']} ({conf:.1%})")
            
            if len(results[scene]) > max_per_scene:
                print(f"  ... and {len(results[scene]) - max_per_scene} more")
        
        print(f"\n{'='*70}")


def create_custom_classifier(scenes_config: str) -> SceneClassifier:
    """Create classifier with custom scene definitions from JSON file"""
    import json
    
    with open(scenes_config) as f:
        custom_scenes = json.load(f)
    
    # Merge with default scenes
    SCENE_DEFINITIONS.update(custom_scenes)
    
    scenes = list(custom_scenes.keys())
    return SceneClassifier(scenes=scenes)


def analyze_scene_distribution(input_dir: str, classifier: SceneClassifier):
    """Analyze scene distribution without organizing files"""
    
    input_path = Path(input_dir)
    extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    for ext in extensions:
        image_files.extend(input_path.glob(f"*{ext}"))
    
    if not image_files:
        print("No images found")
        return
    
    print(f"\nAnalyzing {len(image_files)} images...")
    
    scene_counts = {scene: 0 for scene in classifier.scenes}
    scene_confidences = {scene: [] for scene in classifier.scenes}
    
    for image_path in tqdm(image_files, desc="Analyzing"):
        try:
            classifications = classifier.classify_image(str(image_path))
            
            for scene, (is_match, conf) in classifications.items():
                if is_match:
                    scene_counts[scene] += 1
                    scene_confidences[scene].append(conf)
        except:
            continue
    
    # Print statistics
    print(f"\n{'='*70}")
    print("SCENE DISTRIBUTION ANALYSIS")
    print(f"{'='*70}")
    
    for scene in classifier.scenes:
        count = scene_counts[scene]
        pct = count / len(image_files) * 100
        
        print(f"\n{SCENE_DEFINITIONS[scene]['name']}:")
        print(f"  Count: {count} ({pct:.1f}%)")
        
        if scene_confidences[scene]:
            avg_conf = np.mean(scene_confidences[scene])
            min_conf = np.min(scene_confidences[scene])
            max_conf = np.max(scene_confidences[scene])
            print(f"  Avg confidence: {avg_conf:.1%}")
            print(f"  Range: {min_conf:.1%} - {max_conf:.1%}")
    
    print(f"\n{'='*70}")


def list_available_scenes():
    """List all available scene types"""
    print("\n" + "="*70)
    print("AVAILABLE SCENE TYPES")
    print("="*70)
    
    for scene_type, scene_def in sorted(SCENE_DEFINITIONS.items()):
        print(f"\n{scene_type}:")
        print(f"  Name: {scene_def['name']}")
        print(f"  Description: {scene_def['description']}")
        if 'cv_features' in scene_def:
            print(f"  CV Features: {', '.join(scene_def['cv_features'])}")
    
    print("\n" + "="*70)


# Continuation of scene_classifier.py

def main():
    parser = argparse.ArgumentParser(
        description="Advanced Scene Classifier for Images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Classify into night and day
  python scene_classifier.py /path/to/images --scenes night day
  
  # Classify parks, city streets, and highways
  python scene_classifier.py /path/to/images --scenes park city_street highway
  
  # Classify all scene types
  python scene_classifier.py /path/to/images --scenes all
  
  # Preview before organizing
  python scene_classifier.py /path/to/images --scenes park highway --preview
  
  # Copy instead of move (for multi-label classification)
  python scene_classifier.py /path/to/images --scenes all --copy --multi-label
  
  # Analyze distribution without organizing
  python scene_classifier.py /path/to/images --scenes all --analyze
  
  # Use CV only (faster, no model download)
  python scene_classifier.py /path/to/images --scenes park nature --method cv
  
  # Use CLIP only (most accurate)
  python scene_classifier.py /path/to/images --scenes all --method clip
  
  # Custom output directory
  python scene_classifier.py /path/to/images --scenes night highway --output /organized
  
  # List all available scene types
  python scene_classifier.py --list-scenes

Available Scene Types:
  night, day, park, city_street, highway, indoor, outdoor, urban, nature,
  beach, mountain, vehicle, people, building, water, sunset, rain, snow,
  forest, desert
        """
    )
    
    parser.add_argument('input_dir', nargs='?', help='Directory containing images')
    parser.add_argument('--scenes', '-s', nargs='+', required=False,
                       help='Scene types to detect (or "all")')
    parser.add_argument('--output', '-o', help='Output directory for organized images')
    parser.add_argument('--method', '-m', default='hybrid',
                       choices=['clip', 'cv', 'hybrid'],
                       help='Detection method (default: hybrid)')
    parser.add_argument('--confidence', '-c', type=float, default=0.5,
                       help='Confidence threshold (default: 0.5)')
    parser.add_argument('--preview', '-p', action='store_true',
                       help='Preview classifications before organizing')
    parser.add_argument('--copy', action='store_true',
                       help='Copy files instead of moving')
    parser.add_argument('--multi-label', action='store_true',
                       help='Allow images in multiple categories')
    parser.add_argument('--analyze', '-a', action='store_true',
                       help='Analyze distribution only, don\'t organize')
    parser.add_argument('--list-scenes', action='store_true',
                       help='List all available scene types and exit')
    parser.add_argument('--custom-scenes', help='JSON file with custom scene definitions')
    parser.add_argument('--device', choices=['cuda', 'cpu'],
                       help='Device for models (default: auto-detect)')
    
    args = parser.parse_args()
    
    # List scenes and exit
    if args.list_scenes:
        list_available_scenes()
        return
    
    # Validate input
    if not args.input_dir:
        parser.print_help()
        return
    
    if not Path(args.input_dir).exists():
        print(f"❌ Directory not found: {args.input_dir}")
        return
    
    if not args.scenes:
        print("❌ Please specify --scenes")
        print("Use --list-scenes to see all available scene types")
        return
    
    # Create classifier
    try:
        if args.custom_scenes:
            classifier = create_custom_classifier(args.custom_scenes)
        else:
            classifier = SceneClassifier(
                scenes=args.scenes,
                method=args.method,
                device=args.device
            )
    except Exception as e:
        print(f"❌ Error creating classifier: {e}")
        return
    
    # Analyze or classify
    if args.analyze:
        analyze_scene_distribution(args.input_dir, classifier)
    else:
        results = classifier.process_directory(
            input_dir=args.input_dir,
            output_dir=args.output,
            confidence_threshold=args.confidence,
            preview=args.preview,
            copy=args.copy,
            multi_label=args.multi_label
        )


if __name__ == "__main__":
    main()