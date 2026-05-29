# save as: tag_manager.py

"""
Interactive tag management tool for FiftyOne datasets

Usage:
    python tag_manager.py annotation_dataset
"""

import fiftyone as fo
from fiftyone import ViewField as F
import argparse
from pathlib import Path

def interactive_tag_manager(dataset_name: str):
    """Interactive CLI for managing tags"""
    
    # Load dataset
    dataset = fo.load_dataset(dataset_name)
    
    print(f"\n{'='*70}")
    print(f"TAG MANAGER - Dataset: {dataset_name}")
    print(f"{'='*70}")
    print(f"Total samples: {len(dataset)}")
    
    # Get existing tag fields
    schema = dataset.get_field_schema()
    tag_fields = [f for f in schema.keys() if f.startswith('tag_')]
    
    if tag_fields:
        print(f"\nExisting tag categories:")
        for field in tag_fields:
            category = field.replace('tag_', '')
            tagged_count = len(dataset.match(F(field).exists() == True))
            print(f"  {category}: {tagged_count} samples tagged")
    else:
        print("\nNo tag fields found")
    
    while True:
        print(f"\n{'='*70}")
        print("OPTIONS:")
        print("  1. View tag statistics")
        print("  2. Add tag to filtered samples")
        print("  3. Remove tag category from all samples")
        print("  4. Export tags to JSON")
        print("  5. Import tags from JSON")
        print("  6. Auto-tag by filename patterns")
        print("  7. Create balanced splits")
        print("  8. View samples by tag")
        print("  9. Exit")
        print("="*70)
        
        choice = input("\nEnter choice (1-9): ").strip()
        
        if choice == '1':
            view_tag_statistics(dataset)
        
        elif choice == '2':
            add_tags_interactive(dataset)
        
        elif choice == '3':
            remove_tag_category(dataset)
        
        elif choice == '4':
            export_tags_interactive(dataset)
        
        elif choice == '5':
            import_tags_interactive(dataset)
        
        elif choice == '6':
            auto_tag_interactive(dataset)
        
        elif choice == '7':
            create_splits_interactive(dataset)
        
        elif choice == '8':
            view_by_tag(dataset)
        
        elif choice == '9':
            print("\n👋 Exiting...")
            break
        
        else:
            print("Invalid choice")


def view_tag_statistics(dataset):
    """View detailed tag statistics"""
    print(f"\n{'='*70}")
    print("TAG STATISTICS")
    print(f"{'='*70}")
    
    schema = dataset.get_field_schema()
    tag_fields = [f for f in schema.keys() if f.startswith('tag_')]
    
    if not tag_fields:
        print("No tag fields found")
        return
    
    for field in tag_fields:
        category = field.replace('tag_', '')
        tagged_view = dataset.match(F(field).exists() == True)
        total_tagged = len(tagged_view)
        
        if total_tagged > 0:
            print(f"\n{category.upper()} ({total_tagged} samples):")
            counts = dataset.count_values(f"{field}.label")
            
            for tag, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
                if tag:
                    percentage = count / len(dataset) * 100
                    print(f"  {tag}: {count} ({percentage:.1f}%)")


def add_tags_interactive(dataset):
    """Interactively add tags to samples"""
    print("\n📝 Add tags to samples")
    
    # Show available filters
    print("\nFilter options:")
    print("  1. All samples")
    print("  2. Samples with predictions")
    print("  3. Samples with ground truth")
    print("  4. Custom filter (expert)")
    
    filter_choice = input("Select filter (1-4): ").strip()
    
    if filter_choice == '1':
        view = dataset
    elif filter_choice == '2':
        view = dataset.match(F("predictions").exists() == True)
    elif filter_choice == '3':
        view = dataset.match(F("ground_truth").exists() == True)
    elif filter_choice == '4':
        filter_expr = input("Enter filter expression (e.g., F('filepath').contains('city')): ")
        try:
            view = eval(f"dataset.match({filter_expr})")
        except:
            print("Invalid filter expression")
            return
    else:
        print("Invalid choice")
        return
    
    print(f"\nFiltered to {len(view)} samples")
    
    if len(view) == 0:
        print("No samples match filter")
        return
    
    # Get tag details
    category = input("Enter tag category (e.g., scene, time, weather): ").strip()
    tag = input("Enter tag value: ").strip()
    
    if not category or not tag:
        print("Category and tag cannot be empty")
        return
    
    # Confirm
    confirm = input(f"\nAdd tag '{category}:{tag}' to {len(view)} samples? (y/n): ")
    
    if confirm.lower() == 'y':
        field_name = f"tag_{category}"
        
        for sample in view.iter_samples(progress=True):
            sample[field_name] = fo.Classification(label=tag)
            sample.save()
        
        print(f"✅ Added tag to {len(view)} samples")


def remove_tag_category(dataset):
    """Remove a tag category from all samples"""
    print("\n🗑️  Remove tag category")
    
    schema = dataset.get_field_schema()
    tag_fields = [f.replace('tag_', '') for f in schema.keys() if f.startswith('tag_')]
    
    if not tag_fields:
        print("No tag fields found")
        return
    
    print(f"\nAvailable categories: {', '.join(tag_fields)}")
    category = input("Enter category to remove: ").strip()
    
    if category not in tag_fields:
        print(f"Category '{category}' not found")
        return
    
    field_name = f"tag_{category}"
    confirm = input(f"Remove '{field_name}' from all samples? (y/n): ")
    
    if confirm.lower() == 'y':
        for sample in dataset.iter_samples(progress=True):
            if field_name in sample:
                sample[field_name] = None
                sample.save()
        
        print(f"✅ Removed {field_name} from all samples")


def export_tags_interactive(dataset):
    """Export tags to JSON"""
    output_file = input("\nEnter output filename (e.g., tags.json): ").strip()
    
    if not output_file:
        print("Filename cannot be empty")
        return
    
    tags_data = {}
    
    schema = dataset.get_field_schema()
    tag_fields = [f for f in schema.keys() if f.startswith('tag_')]
    
    for sample in dataset.iter_samples(progress=True):
        filename = Path(sample.filepath).name
        tags = {}
        
        for field in tag_fields:
            if field in sample and sample[field] is not None:
                category = field.replace('tag_', '')
                tags[category] = sample[field].label
        
        if tags:
            tags_data[filename] = tags
    
    import json
    with open(output_file, 'w') as f:
        json.dump(tags_data, f, indent=2)
    
    print(f"✅ Exported tags for {len(tags_data)} samples to {output_file}")


def import_tags_interactive(dataset):
    """Import tags from JSON"""
    input_file = input("\nEnter input filename (e.g., tags.json): ").strip()
    
    if not Path(input_file).exists():
        print(f"File not found: {input_file}")
        return
    
    import json
    with open(input_file, 'r') as f:
        tags_data = json.load(f)
    
    imported = 0
    
    for filename, tags in tags_data.items():
        samples = dataset.match(F("filepath").ends_with(filename))
        
        if len(samples) == 0:
            continue
        
        sample = samples.first()
        
        for category, tag in tags.items():
            field_name = f"tag_{category}"
            sample[field_name] = fo.Classification(label=tag)
            sample.save()
            imported += 1
    
    print(f"✅ Imported {imported} tags from {input_file}")


def auto_tag_interactive(dataset):
    """Auto-tag based on filename patterns"""
    print("\n🤖 Auto-tag by filename patterns")
    print("\nExample patterns:")
    print("  'city' in filename -> tag_scene: city_street")
    print("  'night' in filename -> tag_time: night")
    
    # Implementation would be similar to the app's auto_tag_by_filename
    print("\nThis feature creates tags based on keywords in filenames")
    print("Implement custom patterns as needed")


def create_splits_interactive(dataset):
    """Create train/val/test splits"""
    print("\n📊 Create train/val/test splits")
    
    train_ratio = float(input("Train ratio (default 0.7): ") or "0.7")
    val_ratio = float(input("Val ratio (default 0.15): ") or "0.15")
    
    import random
    samples = list(dataset)
    random.shuffle(samples)
    
    total = len(samples)
    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))
    
    for i, sample in enumerate(samples):
        if i < train_end:
            tag = 'train'
        elif i < val_end:
            tag = 'val'
        else:
            tag = 'test'
        
        sample["tag_distribution"] = fo.Classification(label=tag)
        sample.save()
    
    print(f"✅ Created splits:")
    print(f"   Train: {train_end}")
    print(f"   Val: {val_end - train_end}")
    print(f"   Test: {total - val_end}")


def view_by_tag(dataset):
    """Launch FiftyOne to view samples filtered by tag"""
    print("\n👁️  View samples by tag")
    
    schema = dataset.get_field_schema()
    tag_fields = [f for f in schema.keys() if f.startswith('tag_')]
    
    if not tag_fields:
        print("No tag fields found")
        return
    
    print("\nAvailable tag categories:")
    for i, field in enumerate(tag_fields, 1):
        category = field.replace('tag_', '')
        print(f"  {i}. {category}")
    
    choice = input("\nSelect category: ").strip()
    
    try:
        idx = int(choice) - 1
        field = tag_fields[idx]
        category = field.replace('tag_', '')
    except:
        print("Invalid choice")
        return
    
    # Get unique tags
    tags = dataset.distinct(f"{field}.label")
    tags = [t for t in tags if t]
    
    if not tags:
        print(f"No tags found in category '{category}'")
        return
    
    print(f"\nAvailable {category} tags:")
    for i, tag in enumerate(tags, 1):
        count = len(dataset.match(F(f"{field}.label") == tag))
        print(f"  {i}. {tag} ({count} samples)")
    
    tag_choice = input("\nSelect tag: ").strip()
    
    try:
        tag_idx = int(tag_choice) - 1
        selected_tag = tags[tag_idx]
    except:
        print("Invalid choice")
        return
    
    # Create view and launch
    view = dataset.match(F(f"{field}.label") == selected_tag)
    
    print(f"\n🚀 Launching FiftyOne for {len(view)} samples with {category}:{selected_tag}")
    session = fo.launch_app(view)
    session.wait()


# Continuing tag_manager.py from where it was cut off...

def main():
    parser = argparse.ArgumentParser(
        description="Interactive Tag Manager for FiftyOne Datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Launch interactive tag manager
  python tag_manager.py annotation_dataset
  
  # Batch add tags via command line
  python tag_manager.py annotation_dataset --add-tag scene city_street --filter "predictions"
  
  # Export tags
  python tag_manager.py annotation_dataset --export-tags tags.json
  
  # Import tags
  python tag_manager.py annotation_dataset --import-tags tags.json
  
  # View statistics
  python tag_manager.py annotation_dataset --stats
        """
    )
    
    parser.add_argument('dataset_name', help='Name of FiftyOne dataset')
    
    # Non-interactive options
    parser.add_argument('--add-tag', nargs=2, metavar=('CATEGORY', 'TAG'),
                       help='Add tag to samples (requires --filter)')
    parser.add_argument('--filter', type=str,
                       help='Filter for batch operations (e.g., "predictions", "ground_truth", "all")')
    parser.add_argument('--export-tags', type=str,
                       help='Export tags to JSON file')
    parser.add_argument('--import-tags', type=str,
                       help='Import tags from JSON file')
    parser.add_argument('--remove-category', type=str,
                       help='Remove tag category from all samples')
    parser.add_argument('--stats', action='store_true',
                       help='Print tag statistics')
    parser.add_argument('--list-tags', action='store_true',
                       help='List all tag categories and values')
    
    args = parser.parse_args()
    
    # Check if dataset exists
    if args.dataset_name not in fo.list_datasets():
        print(f"❌ Dataset '{args.dataset_name}' not found")
        print(f"\nAvailable datasets:")
        for name in fo.list_datasets():
            print(f"  - {name}")
        return
    
    # Load dataset
    dataset = fo.load_dataset(args.dataset_name)
    
    # Handle non-interactive commands
    if args.stats:
        view_tag_statistics(dataset)
        return
    
    if args.list_tags:
        list_all_tags(dataset)
        return
    
    if args.export_tags:
        export_tags_to_file(dataset, args.export_tags)
        return
    
    if args.import_tags:
        import_tags_from_file(dataset, args.import_tags)
        return
    
    if args.remove_category:
        remove_tag_category_batch(dataset, args.remove_category)
        return
    
    if args.add_tag:
        if not args.filter:
            print("❌ --filter is required when using --add-tag")
            return
        add_tag_batch(dataset, args.add_tag[0], args.add_tag[1], args.filter)
        return
    
    # Launch interactive mode
    interactive_tag_manager(args.dataset_name)


def list_all_tags(dataset):
    """List all tag categories and their values"""
    print(f"\n{'='*70}")
    print(f"ALL TAGS - Dataset: {dataset.name}")
    print(f"{'='*70}")
    
    schema = dataset.get_field_schema()
    tag_fields = [f for f in schema.keys() if f.startswith('tag_')]
    
    if not tag_fields:
        print("No tag fields found")
        return
    
    for field in sorted(tag_fields):
        category = field.replace('tag_', '')
        tagged_view = dataset.match(F(field).exists() == True)
        
        print(f"\n{category.upper()}:")
        
        if len(tagged_view) == 0:
            print("  (no samples tagged)")
            continue
        
        tags = dataset.distinct(f"{field}.label")
        tags = sorted([t for t in tags if t])
        
        for tag in tags:
            count = len(dataset.match(F(f"{field}.label") == tag))
            percentage = count / len(dataset) * 100
            print(f"  - {tag}: {count} samples ({percentage:.1f}%)")


def add_tag_batch(dataset, category: str, tag: str, filter_type: str):
    """Add tag to filtered samples (non-interactive)"""
    print(f"\n📝 Adding tag '{category}:{tag}' to filtered samples...")
    
    # Apply filter
    if filter_type.lower() == 'all':
        view = dataset
    elif filter_type.lower() == 'predictions':
        view = dataset.match(F("predictions").exists() == True)
    elif filter_type.lower() == 'ground_truth':
        view = dataset.match(F("ground_truth").exists() == True)
    elif filter_type.lower() == 'untagged':
        field_name = f"tag_{category}"
        view = dataset.match(F(field_name).exists() == False)
    else:
        # Try to evaluate as expression
        try:
            view = eval(f"dataset.match({filter_type})")
        except:
            print(f"❌ Invalid filter: {filter_type}")
            return
    
    print(f"Filtered to {len(view)} samples")
    
    if len(view) == 0:
        print("No samples match filter")
        return
    
    # Add tags
    field_name = f"tag_{category}"
    
    for sample in view.iter_samples(progress=True):
        sample[field_name] = fo.Classification(label=tag)
        sample.save()
    
    print(f"✅ Added tag to {len(view)} samples")


def remove_tag_category_batch(dataset, category: str):
    """Remove tag category from all samples (non-interactive)"""
    print(f"\n🗑️  Removing tag category '{category}'...")
    
    field_name = f"tag_{category}"
    
    # Check if field exists
    schema = dataset.get_field_schema()
    if field_name not in schema:
        print(f"❌ Tag category '{category}' not found")
        return
    
    # Count affected samples
    tagged_count = len(dataset.match(F(field_name).exists() == True))
    
    print(f"This will remove tags from {tagged_count} samples")
    
    # Remove tags
    for sample in dataset.iter_samples(progress=True):
        if field_name in sample:
            sample[field_name] = None
            sample.save()
    
    print(f"✅ Removed '{field_name}' from all samples")


def export_tags_to_file(dataset, output_file: str):
    """Export tags to JSON file (non-interactive)"""
    print(f"\n📤 Exporting tags to {output_file}...")
    
    tags_data = {}
    
    schema = dataset.get_field_schema()
    tag_fields = [f for f in schema.keys() if f.startswith('tag_')]
    
    if not tag_fields:
        print("No tag fields found")
        return
    
    for sample in dataset.iter_samples(progress=True):
        filename = Path(sample.filepath).name
        tags = {}
        
        for field in tag_fields:
            if field in sample and sample[field] is not None:
                category = field.replace('tag_', '')
                tags[category] = sample[field].label
        
        if tags:
            tags_data[filename] = tags
    
    # Save to file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    import json
    with open(output_path, 'w') as f:
        json.dump(tags_data, f, indent=2)
    
    print(f"✅ Exported tags for {len(tags_data)} samples to {output_file}")


def import_tags_from_file(dataset, input_file: str):
    """Import tags from JSON file (non-interactive)"""
    print(f"\n📥 Importing tags from {input_file}...")
    
    input_path = Path(input_file)
    if not input_path.exists():
        print(f"❌ File not found: {input_file}")
        return
    
    import json
    with open(input_path, 'r') as f:
        tags_data = json.load(f)
    
    print(f"Found tags for {len(tags_data)} images")
    
    imported = 0
    not_found = 0
    
    for filename, tags in tags_data.items():
        # Find sample by filename
        samples = dataset.match(F("filepath").ends_with(filename))
        
        if len(samples) == 0:
            not_found += 1
            continue
        
        sample = samples.first()
        
        # Add tags
        for category, tag in tags.items():
            field_name = f"tag_{category}"
            sample[field_name] = fo.Classification(label=tag)
            sample.save()
            imported += 1
    
    print(f"✅ Imported {imported} tags")
    if not_found > 0:
        print(f"⚠️  {not_found} images not found in dataset")


if __name__ == "__main__":
    main()