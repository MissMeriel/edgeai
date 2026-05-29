# save as: preview_tags.py

# Continuing preview_tags.py

"""
Preview and validate auto-generated tags before importing

Usage:
    python preview_tags.py tags.json --images-dir /path/to/images
"""

import json
import argparse
from pathlib import Path
from collections import Counter
import cv2
import numpy as np
from typing import Dict, List

def preview_tags(tags_file: str, images_dir: str = None,
                show_samples: int = 5):
    """Preview tags and show sample images"""
    
    print("="*70)
    print("TAG PREVIEW AND VALIDATION")
    print("="*70)
    
    # Load tags
    with open(tags_file) as f:
        tags_data = json.load(f)
    
    print(f"\nLoaded tags for {len(tags_data)} images")
    
    # Analyze tag distribution
    print("\n" + "="*70)
    print("TAG DISTRIBUTION")
    print("="*70)
    
    categories = {}
    
    for filename, tags in tags_data.items():
        for category, tag in tags.items():
            if category not in categories:
                categories[category] = Counter()
            categories[category][tag] += 1
    
    # Print distribution
    for category, counts in sorted(categories.items()):
        print(f"\n{category.upper()}:")
        total = sum(counts.values())
        for tag, count in counts.most_common():
            percentage = count / total * 100
            print(f"  {tag}: {count} ({percentage:.1f}%)")
    
    # Find untagged images
    print("\n" + "="*70)
    print("COVERAGE ANALYSIS")
    print("="*70)
    
    if images_dir:
        images_path = Path(images_dir)
        extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        
        all_images = set()
        for ext in extensions:
            all_images.update([f.name for f in images_path.glob(f"*{ext}")])
            all_images.update([f.name for f in images_path.glob(f"*{ext.upper()}")])
        
        tagged_images = set(tags_data.keys())
        untagged = all_images - tagged_images
        
        print(f"\nTotal images in directory: {len(all_images)}")
        print(f"Tagged images: {len(tagged_images)}")
        print(f"Untagged images: {len(untagged)}")
        print(f"Coverage: {len(tagged_images)/len(all_images)*100:.1f}%")
        
        if untagged and len(untagged) <= 20:
            print(f"\nUntagged images:")
            for img in sorted(untagged):
                print(f"  - {img}")
    
    # Show sample images for each category
    if images_dir and show_samples > 0:
        print("\n" + "="*70)
        print("SAMPLE VISUALIZATION")
        print("="*70)
        
        visualize_samples(tags_data, images_dir, show_samples)


def visualize_samples(tags_data: Dict, images_dir: str, samples_per_tag: int):
    """Visualize sample images for each tag"""
    
    images_path = Path(images_dir)
    
    # Group by tags
    tag_groups = {}
    
    for filename, tags in tags_data.items():
        for category, tag in tags.items():
            key = f"{category}:{tag}"
            if key not in tag_groups:
                tag_groups[key] = []
            tag_groups[key].append(filename)
    
    # Show samples for each tag
    for tag_key, filenames in sorted(tag_groups.items()):
        category, tag = tag_key.split(':', 1)
        
        print(f"\n{category.upper()}: {tag} ({len(filenames)} images)")
        
        # Select sample images
        import random
        samples = random.sample(filenames, min(samples_per_tag, len(filenames)))
        
        # Create visualization
        images = []
        for filename in samples:
            img_path = images_path / filename
            if img_path.exists():
                img = cv2.imread(str(img_path))
                if img is not None:
                    # Resize for display
                    h, w = img.shape[:2]
                    scale = 200 / max(h, w)
                    img = cv2.resize(img, None, fx=scale, fy=scale)
                    
                    # Add label
                    cv2.putText(img, filename[:20], (5, 15),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    
                    images.append(img)
        
        if images:
            # Concatenate horizontally
            if len(images) > 1:
                # Make all images same height
                max_h = max(img.shape[0] for img in images)
                images_padded = []
                for img in images:
                    if img.shape[0] < max_h:
                        pad = max_h - img.shape[0]
                        img = cv2.copyMakeBorder(img, 0, pad, 0, 0, 
                                                cv2.BORDER_CONSTANT, value=(0, 0, 0))
                    images_padded.append(img)
                
                combined = np.hstack(images_padded)
            else:
                combined = images[0]
            
            # Display
            window_name = f"{category}: {tag}"
            cv2.imshow(window_name, combined)
            
            print(f"  Samples: {', '.join([s[:20] for s in samples])}")
            print(f"  Press any key to continue...")
            cv2.waitKey(0)
            cv2.destroyWindow(window_name)


def validate_tags(tags_file: str, expected_categories: List[str] = None):
    """Validate tag file format and content"""
    
    print("\n" + "="*70)
    print("TAG VALIDATION")
    print("="*70)
    
    # Load tags
    try:
        with open(tags_file) as f:
            tags_data = json.load(f)
    except Exception as e:
        print(f"❌ Error loading tags file: {e}")
        return False
    
    if not isinstance(tags_data, dict):
        print(f"❌ Tags file must be a dictionary")
        return False
    
    print(f"✅ Valid JSON format")
    print(f"✅ {len(tags_data)} images tagged")
    
    # Check structure
    issues = []
    
    for filename, tags in tags_data.items():
        if not isinstance(tags, dict):
            issues.append(f"Invalid tags format for {filename}")
            continue
        
        for category, tag in tags.items():
            if not isinstance(tag, str):
                issues.append(f"Non-string tag value in {filename}: {category}")
    
    if issues:
        print(f"\n⚠️  Found {len(issues)} issues:")
        for issue in issues[:10]:  # Show first 10
            print(f"  - {issue}")
        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more")
        return False
    
    print(f"✅ All tags are properly formatted")
    
    # Check expected categories
    if expected_categories:
        found_categories = set()
        for tags in tags_data.values():
            found_categories.update(tags.keys())
        
        missing = set(expected_categories) - found_categories
        extra = found_categories - set(expected_categories)
        
        if missing:
            print(f"\n⚠️  Missing expected categories: {missing}")
        
        if extra:
            print(f"\n📝 Additional categories found: {extra}")
        
        if not missing and not extra:
            print(f"✅ All expected categories present")
    
    return True


def export_tag_summary(tags_file: str, output_file: str):
    """Export tag summary as CSV"""
    
    import csv
    
    with open(tags_file) as f:
        tags_data = json.load(f)
    
    # Get all categories
    all_categories = set()
    for tags in tags_data.values():
        all_categories.update(tags.keys())
    
    all_categories = sorted(all_categories)
    
    # Write CSV
    with open(output_file, 'w', newline='') as f:
        fieldnames = ['filename'] + all_categories
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        writer.writeheader()
        for filename, tags in sorted(tags_data.items()):
            row = {'filename': filename}
            row.update(tags)
            writer.writerow(row)
    
    print(f"\n✅ Exported tag summary to: {output_file}")


def compare_tags(tags_file1: str, tags_file2: str):
    """Compare two tag files"""
    
    print("\n" + "="*70)
    print("COMPARING TAG FILES")
    print("="*70)
    
    with open(tags_file1) as f:
        tags1 = json.load(f)
    
    with open(tags_file2) as f:
        tags2 = json.load(f)
    
    files1 = set(tags1.keys())
    files2 = set(tags2.keys())
    
    only_in_1 = files1 - files2
    only_in_2 = files2 - files1
    common = files1 & files2
    
    print(f"\nFile 1: {len(files1)} images")
    print(f"File 2: {len(files2)} images")
    print(f"Common: {len(common)} images")
    print(f"Only in file 1: {len(only_in_1)}")
    print(f"Only in file 2: {len(only_in_2)}")
    
    # Compare tags for common images
    differences = []
    
    for filename in common:
        t1 = tags1[filename]
        t2 = tags2[filename]
        
        if t1 != t2:
            differences.append(filename)
    
    print(f"\nImages with different tags: {len(differences)}")
    
    if differences and len(differences) <= 10:
        print("\nDifferences:")
        for filename in differences:
            print(f"\n  {filename}:")
            print(f"    File 1: {tags1[filename]}")
            print(f"    File 2: {tags2[filename]}")


def main():
    parser = argparse.ArgumentParser(
        description="Preview and validate auto-generated tags",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preview tags with statistics
  python preview_tags.py tags.json
  
  # Preview with image visualization
  python preview_tags.py tags.json --images-dir /path/to/images --show-samples 5
  
  # Validate tag format
  python preview_tags.py tags.json --validate
  
  # Export tag summary as CSV
  python preview_tags.py tags.json --export-csv tags_summary.csv
  
  # Compare two tag files
  python preview_tags.py tags1.json --compare tags2.json
        """
    )
    
    parser.add_argument('tags_file', help='Tags JSON file')
    parser.add_argument('--images-dir', help='Directory containing images')
    parser.add_argument('--show-samples', type=int, default=0,
                       help='Number of sample images to show per tag')
    parser.add_argument('--validate', action='store_true',
                       help='Validate tag file format')
    parser.add_argument('--expected-categories', nargs='+',
                       help='Expected tag categories')
    parser.add_argument('--export-csv', help='Export tag summary as CSV')
    parser.add_argument('--compare', help='Compare with another tags file')
    
    args = parser.parse_args()
    
    # Validate if requested
    if args.validate:
        valid = validate_tags(args.tags_file, args.expected_categories)
        if not valid:
            return
    
    # Export CSV if requested
    if args.export_csv:
        export_tag_summary(args.tags_file, args.export_csv)
    
    # Compare if requested
    if args.compare:
        compare_tags(args.tags_file, args.compare)
        return
    
    # Preview tags
    preview_tags(args.tags_file, args.images_dir, args.show_samples)


if __name__ == "__main__":
    main()