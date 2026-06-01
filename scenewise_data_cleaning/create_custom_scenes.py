# save as: create_custom_scenes.py

"""
Create custom scene definitions for the classifier

Usage:
    python create_custom_scenes.py --interactive
    python create_custom_scenes.py --template custom_scenes.json
"""

import json
import argparse
from pathlib import Path

def create_scene_template():
    """Create a template for custom scenes"""
    
    template = {
        "example_scene": {
            "name": "Example Scene Name",
            "clip_prompts": [
                "a photo that is NOT your scene",
                "a photo that IS your scene"
            ],
            "positive_index": 1,
            "cv_features": ["feature1", "feature2"],
            "description": "Description of what this scene represents"
        },
        "parking_lot": {
            "name": "Parking Lots",
            "clip_prompts": [
                "a photo of nature or buildings",
                "a photo of a parking lot with many cars"
            ],
            "positive_index": 1,
            "cv_features": ["horizontal_lines", "rectangular_shapes"],
            "description": "Parking lots and car parks"
        },
        "bridge": {
            "name": "Bridges",
            "clip_prompts": [
                "a photo without any bridge",
                "a photo of a bridge over water or road"
            ],
            "positive_index": 1,
            "cv_features": ["horizontal_lines", "perspective"],
            "description": "Bridge structures"
        },
        "tunnel": {
            "name": "Tunnels",
            "clip_prompts": [
                "a photo in open air",
                "a photo inside a tunnel"
            ],
            "positive_index": 1,
            "description": "Road or rail tunnels"
        },
        "gas_station": {
            "name": "Gas Stations",
            "clip_prompts": [
                "a photo of nature or residential area",
                "a photo of a gas station or fuel station"
            ],
            "positive_index": 1,
            "description": "Gas stations and fuel stops"
        },
        "intersection": {
            "name": "Intersections",
            "clip_prompts": [
                "a photo of a straight road or highway",
                "a photo of a road intersection with traffic lights"
            ],
            "positive_index": 1,
            "cv_features": ["edge_density"],
            "description": "Road intersections and crossroads"
        }
    }
    
    return template


def interactive_scene_creator():
    """Interactive prompt to create custom scenes"""
    
    print("\n" + "="*70)
    print("CUSTOM SCENE CREATOR")
    print("="*70)
    
    scenes = {}
    
    while True:
        print("\n" + "-"*70)
        scene_id = input("\nEnter scene ID (lowercase, no spaces, e.g., 'parking_lot'): ").strip()
        
        if not scene_id:
            break
        
        if not scene_id.replace('_', '').isalnum():
            print("❌ Scene ID must be alphanumeric with underscores only")
            continue
        
        scene_name = input("Enter scene display name (e.g., 'Parking Lots'): ").strip()
        description = input("Enter scene description: ").strip()
        
        print("\nDefine CLIP prompts (for semantic detection):")
        print("  Prompt 1 should describe what this scene is NOT")
        print("  Prompt 2 should describe what this scene IS")
        
        prompt1 = input("  Prompt 1 (negative): ").strip()
        prompt2 = input("  Prompt 2 (positive): ").strip()
        
        scenes[scene_id] = {
            "name": scene_name,
            "clip_prompts": [prompt1, prompt2],
            "positive_index": 1,
            "description": description
        }
        
        add_cv = input("\nAdd computer vision features? (y/n): ").lower()
        if add_cv == 'y':
            print("\nAvailable CV features:")
            print("  green_dominance, tree_detection, edge_density, vertical_lines,")
            print("  horizontal_lines, perspective, sky_detection, water_blue, etc.")
            
            cv_features = input("Enter features (comma-separated): ").strip()
            if cv_features:
                scenes[scene_id]["cv_features"] = [f.strip() for f in cv_features.split(',')]
        
        print(f"\n✅ Added scene: {scene_name}")
        
        another = input("\nAdd another scene? (y/n): ").lower()
        if another != 'y':
            break
    
    return scenes


def main():
    parser = argparse.ArgumentParser(
        description="Create custom scene definitions"
    )
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='Interactive scene creation')
    parser.add_argument('--template', '-t', action='store_true',
                       help='Generate template file')
    parser.add_argument('--output', '-o', default='custom_scenes.json',
                       help='Output JSON file')
    
    args = parser.parse_args()
    
    if args.template:
        scenes = create_scene_template()
        print(f"\n📄 Creating template file: {args.output}")
    elif args.interactive:
        scenes = interactive_scene_creator()
    else:
        parser.print_help()
        return
    
    if scenes:
        with open(args.output, 'w') as f:
            json.dump(scenes, f, indent=2)
        
        print(f"\n✅ Saved {len(scenes)} scene(s) to: {args.output}")
        print(f"\nUsage:")
        print(f"  python scene_classifier.py /path/to/images \\")
        print(f"    --custom-scenes {args.output} \\")
        print(f"    --scenes {' '.join(scenes.keys())}")


if __name__ == "__main__":
    main()