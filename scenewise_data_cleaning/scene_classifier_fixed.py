# save as: scene_classifier_fixed.py

"""
Fixed Scene Classifier with improved detection accuracy

Key improvements:
- Better CLIP prompts with more contrast
- Improved CV feature detection
- Per-scene calibrated thresholds
- Debug mode to see detection scores
- Verification step for uncertain classifications
"""

import torch
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import argparse
from tqdm import tqdm
import shutil
import json
from datetime import datetime
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# IMPROVED Scene definitions with better prompts and thresholds
SCENE_DEFINITIONS = {
    'night': {
        'name': 'Night Images',
        'clip_prompts': [
            "a bright photograph taken during sunny daytime with clear visibility",
            "a dark photograph taken at night with low lighting and darkness"
        ],
        'positive_index': 1,
        'threshold': 0.55,  # Calibrated threshold
        'description': 'Images taken at night or in low light'
    },
    'day': {
        'name': 'Daytime Images',
        'clip_prompts': [
            "a dark photograph taken at night with low lighting",
            "a bright photograph taken during sunny daytime with good lighting"
        ],
        'positive_index': 1,
        'threshold': 0.55,
        'description': 'Images taken during daytime'
    },
    'park': {
        'name': 'Parks',
        'clip_prompts': [
            "a photograph of a road, parking lot, or indoor space",
            "a photograph of a park with grass, trees, benches and open green space for recreation"
        ],
        'positive_index': 1,
        'cv_weight': 0.5,
        'threshold': 0.60,
        'description': 'Parks, gardens, and recreational green spaces'
    },
    'city_street': {
        'name': 'City Streets',
        'clip_prompts': [
            "a photograph of nature, highway, or countryside",
            "a photograph of a city street with buildings, sidewalks, shops and pedestrians"
        ],
        'positive_index': 1,
        'cv_weight': 0.3,
        'threshold': 0.58,
        'description': 'Urban streets with buildings'
    },
    'highway': {
        'name': 'Highways',
        'clip_prompts': [
            "a photograph of a city street or narrow residential road",
            "a photograph of a wide multi-lane highway or freeway with road markings"
        ],
        'positive_index': 1,
        'cv_weight': 0.4,
        'threshold': 0.62,
        'description': 'Highways and freeways'
    },
    'indoor': {
        'name': 'Indoor',
        'clip_prompts': [
            "a photograph taken outdoors with sky visible",
            "a photograph taken indoors inside a room or building with ceiling and walls"
        ],
        'positive_index': 1,
        'threshold': 0.60,
        'description': 'Indoor scenes'
    },
    'outdoor': {
        'name': 'Outdoor',
        'clip_prompts': [
            "a photograph taken indoors with ceiling and walls visible",
            "a photograph taken outdoors in open air with sky or horizon visible"
        ],
        'positive_index': 1,
        'threshold': 0.55,
        'description': 'Outdoor scenes'
    },
    'urban': {
        'name': 'Urban',
        'clip_prompts': [
            "a photograph of wilderness, forest, or natural landscape",
            "a photograph of urban cityscape with many buildings, concrete and infrastructure"
        ],
        'positive_index': 1,
        'threshold': 0.60,
        'description': 'Urban and city environments'
    },
    'nature': {
        'name': 'Nature',
        'clip_prompts': [
            "a photograph of city with buildings and roads",
            "a photograph of natural landscape with trees, plants, or wilderness"
        ],
        'positive_index': 1,
        'cv_weight': 0.4,
        'threshold': 0.58,
        'description': 'Natural landscapes'
    },
    'beach': {
        'name': 'Beach',
        'clip_prompts': [
            "a photograph of mountains, forest, or city",
            "a photograph of a sandy beach with ocean or sea water visible"
        ],
        'positive_index': 1,
        'cv_weight': 0.5,
        'threshold': 0.65,
        'description': 'Beach and coastal scenes'
    },
    'mountain': {
        'name': 'Mountain',
        'clip_prompts': [
            "a photograph of flat terrain, city streets, or indoor space",
            "a photograph of tall mountains with peaks, rocky terrain and steep elevation"
        ],
        'positive_index': 1,
        'cv_weight': 0.3,
        'threshold': 0.70,  # Higher threshold to reduce false positives
        'verify': True,  # Require verification
        'description': 'Mountain landscapes with significant elevation'
    },
    'vehicle': {
        'name': 'Vehicles',
        'clip_prompts': [
            "a photograph of landscape without any vehicles",
            "a photograph with cars, trucks, buses or other motor vehicles prominently visible"
        ],
        'positive_index': 1,
        'threshold': 0.62,
        'description': 'Images containing vehicles'
    },
    'people': {
        'name': 'People',
        'clip_prompts': [
            "a photograph of empty landscape without any people",
            "a photograph with people or human figures clearly visible"
        ],
        'positive_index': 1,
        'threshold': 0.65,
        'description': 'Images containing people'
    },
    'building': {
        'name': 'Buildings',
        'clip_prompts': [
            "a photograph of open nature without buildings",
            "a photograph of buildings or architectural structures"
        ],
        'positive_index': 1,
        'cv_weight': 0.4,
        'threshold': 0.60,
        'description': 'Images with buildings'
    },
    'water': {
        'name': 'Water',
        'clip_prompts': [
            "a photograph of dry land without water",
            "a photograph of water body like ocean, lake, river or sea"
        ],
        'positive_index': 1,
        'cv_weight': 0.5,
        'threshold': 0.62,
        'description': 'Images with water bodies'
    },
    'sunset': {
        'name': 'Sunset/Sunrise',
        'clip_prompts': [
            "a photograph taken during midday with sun high in sky",
            "a photograph of sunset or sunrise with orange and red sky near horizon"
        ],
        'positive_index': 1,
        'cv_weight': 0.6,
        'threshold': 0.68,
        'description': 'Sunset or sunrise scenes'
    },
    'forest': {
        'name': 'Forest',
        'clip_prompts': [
            "a photograph of open field, city or desert",
            "a photograph of dense forest with many trees close together"
        ],
        'positive_index': 1,
        'cv_weight': 0.5,
        'threshold': 0.63,
        'description': 'Forest and woodland'
    },
    'desert': {
        'name': 'Desert',
        'clip_prompts': [
            "a photograph of green vegetation or city",
            "a photograph of arid desert with sand, dunes and sparse vegetation"
        ],
        'positive_index': 1,
        'cv_weight': 0.5,
        'threshold': 0.68,
        'description': 'Desert landscapes'
    }
}


class ImprovedSceneClassifier:
    """Improved scene classifier with better accuracy"""
    
    def __init__(self, scenes: List[str] = None, method: str = "hybrid", 
                 device: str = None, debug: bool = False):
        """
        Initialize scene classifier
        
        Args:
            scenes: List of scene types to detect
            method: Detection method ('clip', 'cv', 'hybrid')
            device: Device for models
            debug: Show debug information
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.method = method
        self.debug = debug
        
        # Validate scenes
        if scenes is None or 'all' in scenes:
            self.scenes = list(SCENE_DEFINITIONS.keys())
        else:
            self.scenes = [s for s in scenes if s in SCENE_DEFINITIONS]
        
        # Load models
        self.clip_model = None
        self.clip_processor = None
        
        if method in ['clip', 'hybrid']:
            self._load_clip_model()
        
        print(f"\n{'='*70}")
        print("IMPROVED SCENE CLASSIFIER")
        print(f"{'='*70}")
        print(f"Scenes: {', '.join(self.scenes)}")
        print(f"Method: {method}")
        print(f"Device: {self.device}")
        print(f"Debug mode: {debug}")
        print(f"{'='*70}\n")
    
    def _load_clip_model(self):
        """Load CLIP model"""
        try:
            print("Loading CLIP model...")
            from transformers import CLIPProcessor, CLIPModel
            
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_model.to(self.device)
            self.clip_model.eval()
            
            print("✅ CLIP model loaded\n")
        except Exception as e:
            print(f"⚠️  Could not load CLIP: {e}")
            print("Falling back to CV-only method\n")
            self.method = 'cv'
    
    def detect_scene_clip(self, image_path: str, scene_type: str) -> Tuple[bool, float]:
        """Detect scene using CLIP with improved prompts"""
        if self.clip_model is None:
            return False, 0.0
        
        try:
            scene_def = SCENE_DEFINITIONS[scene_type]
            
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            
            # Resize if too large (for speed)
            max_size = 512
            if max(image.size) > max_size:
                image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
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
            
            pos_prob = float(probs[0][scene_def['positive_index']])
            
            # Use scene-specific threshold
            threshold = scene_def.get('threshold', 0.5)
            is_scene = pos_prob > threshold
            
            if self.debug:
                print(f"  CLIP {scene_type}: {pos_prob:.3f} (threshold: {threshold})")
            
            return is_scene, pos_prob
            
        except Exception as e:
            if self.debug:
                print(f"  CLIP error for {scene_type}: {e}")
            return False, 0.0
    
    def _extract_improved_cv_features(self, img: np.ndarray, scene_type: str) -> float:
        """Extract CV features specific to scene type"""
        
        h, w = img.shape[:2]
        
        # Convert color spaces
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Scene-specific feature extraction
        if scene_type == 'mountain':
            # Much stricter mountain detection
            score = 0.0
            
            # 1. Check for elevation profile (jagged skyline at top)
            top_quarter = gray[:h//4, :]
            edges = cv2.Canny(top_quarter, 50, 150)
            
            # Get skyline (top edge profile)
            edge_rows = np.where(edges > 0)[0]
            if len(edge_rows) > 0:
                # Measure variance in vertical position (jaggedness)
                variance = np.var(edge_rows)
                if variance > 50:  # Significant elevation changes
                    score += 0.3
            else:
                return 0.1  # No skyline = probably not mountain
            
            # 2. Check for rocky/gray colors (mountains often gray/brown)
            gray_mask = cv2.inRange(hsv, np.array([0, 0, 80]), np.array([180, 50, 180]))
            gray_pct = np.sum(gray_mask > 0) / gray_mask.size
            if gray_pct > 0.3:
                score += 0.2
            
            # 3. Check for lack of green (above tree line)
            green_mask = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
            green_pct = np.sum(green_mask > 0) / green_mask.size
            if green_pct < 0.2:  # Little vegetation
                score += 0.2
            else:
                score -= 0.1  # Lots of green = probably not mountain peak
            
            # 4. Check for steep angles (diagonal lines)
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
            angles = np.arctan2(sobely, sobelx)
            steep_angles = np.abs(angles)
            steep_pct = np.sum(steep_angles > 0.5) / steep_angles.size
            if steep_pct > 0.2:
                score += 0.3
            
            return min(score, 1.0)
        
        elif scene_type == 'park':
            score = 0.0
            
            # Green grass
            green_mask = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
            green_pct = np.sum(green_mask > 0) / green_mask.size
            if green_pct > 0.3:
                score += 0.6
            
            # Not too many edges (open space)
            edges = cv2.Canny(gray, 50, 150)
            edge_pct = np.sum(edges > 0) / edges.size
            if edge_pct < 0.1:
                score += 0.4
            
            return min(score, 1.0)
        
        elif scene_type == 'highway':
            score = 0.0
            
            # Strong horizontal lines in bottom half
            bottom_half = gray[h//2:, :]
            sobely = cv2.Sobel(bottom_half, cv2.CV_64F, 0, 1, ksize=3)
            horizontal_strength = np.mean(np.abs(sobely))
            if horizontal_strength > 30:
                score += 0.5
            
            # Perspective lines (vanishing point)
            edges = cv2.Canny(gray, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=50, maxLineGap=10)
            if lines is not None and len(lines) > 10:
                score += 0.5
            
            return min(score, 1.0)
        
        elif scene_type == 'beach':
            score = 0.0
            
            # Sand color (yellowish/beige)
            sand_mask = cv2.inRange(hsv, np.array([15, 20, 100]), np.array([35, 150, 255]))
            sand_pct = np.sum(sand_mask > 0) / sand_mask.size
            if sand_pct > 0.2:
                score += 0.5
            
            # Water (blue)
            water_mask = cv2.inRange(hsv, np.array([90, 50, 50]), np.array([130, 255, 255]))
            water_pct = np.sum(water_mask > 0) / water_mask.size
            if water_pct > 0.2:
                score += 0.5
            
            return min(score, 1.0)
        
        elif scene_type == 'city_street':
            score = 0.0
            
            # High edge density
            edges = cv2.Canny(gray, 50, 150)
            edge_pct = np.sum(edges > 0) / edges.size
            if edge_pct > 0.15:
                score += 0.5
            
            # Vertical lines (buildings)
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            vertical_strength = np.mean(np.abs(sobelx))
            if vertical_strength > 25:
                score += 0.5
            
            return min(score, 1.0)
        
        # Default: return neutral
        return 0.5
    
    def detect_scene_cv(self, image_path: str, scene_type: str) -> Tuple[bool, float]:
        """Detect scene using improved CV features"""
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                return False, 0.0
            
            # Get scene-specific features
            confidence = self._extract_improved_cv_features(img, scene_type)
            
            # Use scene-specific threshold
            scene_def = SCENE_DEFINITIONS[scene_type]
            threshold = scene_def.get('threshold', 0.5)
            is_scene = confidence > threshold
            
            if self.debug:
                print(f"  CV {scene_type}: {confidence:.3f} (threshold: {threshold})")
            
            return is_scene, confidence
            
        except Exception as e:
            if self.debug:
                print(f"  CV error for {scene_type}: {e}")
            return False, 0.0
    
    def verify_detection(self, image_path: str, scene_type: str, initial_confidence: float) -> Tuple[bool, float]:
        """Additional verification for scenes that need it"""
        
        scene_def = SCENE_DEFINITIONS[scene_type]
        
        # If scene doesn't need verification, return as-is
        if not scene_def.get('verify', False):
            return initial_confidence > scene_def.get('threshold', 0.5), initial_confidence
        
        # For mountains, do additional checks
        if scene_type == 'mountain':
            # Run both CLIP and CV
            if self.clip_model:
                is_clip, conf_clip = self.detect_scene_clip(image_path, scene_type)
                is_cv, conf_cv = self.detect_scene_cv(image_path, scene_type)
                
                # Both must agree with high confidence
                if is_clip and is_cv and conf_clip > 0.65 and conf_cv > 0.6:
                    return True, (conf_clip + conf_cv) / 2
                else:
                    if self.debug:
                        print(f"  Verification FAILED: CLIP={conf_clip:.3f}, CV={conf_cv:.3f}")
                    return False, (conf_clip + conf_cv) / 2
            
        return initial_confidence > scene_def.get('threshold', 0.5), initial_confidence
    
    def classify_image(self, image_path: str) -> Dict[str, Tuple[bool, float]]:
        """Classify image across all configured scenes"""
        
        if self.debug:
            print(f"\nClassifying: {Path(image_path).name}")
        
        results = {}
        
        for scene_type in self.scenes:
            scene_def = SCENE_DEFINITIONS[scene_type]
            
            if self.method == 'clip':
                is_match, conf = self.detect_scene_clip(image_path, scene_type)
            elif self.method == 'cv':
                is_match, conf = self.detect_scene_cv(image_path, scene_type)
            else:  # hybrid
                # Get both scores
                is_cv, conf_cv = self.detect_scene_cv(image_path, scene_type)
                
                if self.clip_model:
                    is_clip, conf_clip = self.detect_scene_clip(image_path, scene_type)
                    
                    # Weighted combination based on scene definition
                    cv_weight = scene_def.get('cv_weight', 0.3)
                    clip_weight = 1.0 - cv_weight
                    
                    conf = (conf_cv * cv_weight + conf_clip * clip_weight)
                    threshold = scene_def.get('threshold', 0.5)
                    is_match = conf > threshold
                else:
                    is_match, conf = is_cv, conf_cv
            
            # Verification step for certain scenes
            if is_match and scene_def.get('verify', False):
                is_match, conf = self.verify_detection(image_path, scene_type, conf)
            
            results[scene_type] = (is_match, conf)
        
        if self.debug:
            matches = [s for s, (m, c) in results.items() if m]
            print(f"  Matches: {matches if matches else 'None'}")
        
        return results
    
    def process_directory(self, input_dir: str, output_dir: str = None,
                         preview: bool = False, copy: bool = False,
                         multi_label: bool = False) -> Dict:
        """Process directory and organize images"""
        
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
        print(f"PROCESSING DIRECTORY")
        print(f"{'='*70}")
        print(f"Input: {input_dir}")
        print(f"Output: {output_dir}")
        print(f"Images: {len(image_files)}")
        print(f"Scenes: {', '.join(self.scenes)}")
        print(f"{'='*70}\n")
        
        # Process images
        results = {scene: [] for scene in self.scenes}
        results['unclassified'] = []
        results['errors'] = []
        
        print("Classifying images...")
        for image_path in tqdm(image_files, desc="Processing"):
            try:
                classifications = self.classify_image(str(image_path))
                
                # Find matching scenes
                matches = [
                    (scene, conf) for scene, (is_match, conf) in classifications.items()
                    if is_match
                ]
                
                if not matches:
                    results['unclassified'].append({
                        'path': str(image_path),
                        'filename': image_path.name,
                        'scores': {k: f"{v[1]:.3f}" for k, v in classifications.items()}
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
        
        for scene in self.scenes:
            count = len(results[scene])
            pct = count / len(image_files) * 100 if image_files else 0
            scene_name = SCENE_DEFINITIONS[scene]['name']
            print(f"{scene_name:20s}: {count:4d} ({pct:5.1f}%)")
        
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
                    
                    if dst.exists():
                        stem = dst.stem
                        suffix = dst.suffix
                        counter = 1
                        while dst.exists():
                            dst = scene_dir / f"{stem}_{counter}{suffix}"
                            counter += 1
                    
                    if copy or multi_label:
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
        
        with open(report_file, 'w') as f:
            json.dump(results, f, indent=2)
        
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


def test_single_image(image_path: str, scenes: List[str], method: str = 'hybrid'):
    """Test classification on a single image with debug output"""
    
    print(f"\n{'='*70}")
    print(f"TESTING SINGLE IMAGE")
    print(f"{'='*70}")
    print(f"Image: {image_path}")
    print(f"Scenes to test: {', '.join(scenes)}")
    print(f"{'='*70}\n")
    
    classifier = ImprovedSceneClassifier(
        scenes=scenes,
        method=method,
        debug=True
    )
    
    results = classifier.classify_image(image_path)
    
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    
    # Sort by confidence
    sorted_results = sorted(results.items(), key=lambda x: x[1][1], reverse=True)
    
    for scene, (is_match, conf) in sorted_results:
        threshold = SCENE_DEFINITIONS[scene].get('threshold', 0.5)
        status = "✓ MATCH" if is_match else "✗ no match"
        print(f"{scene:15s}: {conf:.3f} (threshold: {threshold:.2f}) [{status}]")
    
    print(f"{'='*70}\n")


# Continuation of scene_classifier_fixed.py - Complete the main() function

def main():
    parser = argparse.ArgumentParser(
        description="Improved Scene Classifier with Better Accuracy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test on single image with debug output
  python scene_classifier_fixed.py --test-image photo.jpg --scenes mountain park highway
  
  # Classify with debug mode
  python scene_classifier_fixed.py /photos --scenes night park highway --debug
  
  # Preview before organizing
  python scene_classifier_fixed.py /photos --scenes park city_street --preview
  
  # Use CLIP only for best accuracy
  python scene_classifier_fixed.py /photos --scenes all --method clip
  
  # Copy instead of move
  python scene_classifier_fixed.py /photos --scenes night day --copy
        """
    )
    
    parser.add_argument('input_dir', nargs='?', help='Directory containing images')
    parser.add_argument('--scenes', '-s', nargs='+',
                       help='Scene types to detect')
    parser.add_argument('--output', '-o', help='Output directory')
    parser.add_argument('--method', '-m', default='hybrid',
                       choices=['clip', 'cv', 'hybrid'],
                       help='Detection method (default: hybrid)')
    parser.add_argument('--preview', '-p', action='store_true',
                       help='Preview before organizing')
    parser.add_argument('--copy', action='store_true',
                       help='Copy instead of move')
    parser.add_argument('--multi-label', action='store_true',
                       help='Allow images in multiple categories')
    parser.add_argument('--debug', '-d', action='store_true',
                       help='Show debug information')
    parser.add_argument('--test-image', help='Test classification on single image')
    parser.add_argument('--list-scenes', action='store_true',
                       help='List available scenes')
    parser.add_argument('--device', choices=['cuda', 'cpu'],
                       help='Device for models')
    
    args = parser.parse_args()
    
    # List scenes
    if args.list_scenes:
        print("\n" + "="*70)
        print("AVAILABLE SCENE TYPES")
        print("="*70)
        
        for scene_type, scene_def in sorted(SCENE_DEFINITIONS.items()):
            threshold = scene_def.get('threshold', 0.5)
            print(f"\n{scene_type}:")
            print(f"  Name: {scene_def['name']}")
            print(f"  Description: {scene_def['description']}")
            print(f"  Threshold: {threshold}")
            if scene_def.get('verify'):
                print(f"  Verification: Required (strict detection)")
        
        print("\n" + "="*70)
        return
    
    # Test single image
    if args.test_image:
        if not args.scenes:
            print("❌ Please specify --scenes to test")
            return
        
        if not Path(args.test_image).exists():
            print(f"❌ Image not found: {args.test_image}")
            return
        
        test_single_image(args.test_image, args.scenes, args.method)
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
        classifier = ImprovedSceneClassifier(
            scenes=args.scenes,
            method=args.method,
            device=args.device,
            debug=args.debug
        )
    except Exception as e:
        print(f"❌ Error creating classifier: {e}")
        return
    
    # Process directory
    results = classifier.process_directory(
        input_dir=args.input_dir,
        output_dir=args.output,
        preview=args.preview,
        copy=args.copy,
        multi_label=args.multi_label
    )


if __name__ == "__main__":
    main()