# save as: export_complete_dataset.py

"""
Export COMPLETE FiftyOne dataset for transfer to another computer
"""

import fiftyone as fo
from fiftyone import ViewField as F
from pathlib import Path
import json
import shutil
import argparse
from datetime import datetime
from tqdm import tqdm

def export_complete_dataset(dataset_name: str, output_dir: str):
    """Export complete dataset with all data"""
    
    dataset = fo.load_dataset(dataset_name)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("EXPORTING COMPLETE DATASET FOR TRANSFER")
    print("="*70)
    print(f"Dataset: {dataset_name}")
    print(f"Output: {output_path}")
    print(f"Samples: {len(dataset)}")
    
    images_dir = output_path / "images"
    images_dir.mkdir(exist_ok=True)
    
    complete_data = {
        'dataset_name': dataset_name,
        'export_date': datetime.now().isoformat(),
        'total_samples': len(dataset),
        'schema': {},
        'samples': []
    }
    
    schema = dataset.get_field_schema()
    for field_name, field in schema.items():
        complete_data['schema'][field_name] = str(type(field))
    
    print("\n📦 Exporting samples with all data...")
    
    for sample in tqdm(dataset, desc="Exporting"):
        img_filename = Path(sample.filepath).name
        img_source = Path(sample.filepath)
        img_dest = images_dir / img_filename
        
        try:
            if img_source.exists():
                shutil.copy(img_source, img_dest)
            else:
                print(f"   ⚠️  Image not found: {img_source}")
                continue
        except Exception as e:
            print(f"   ❌ Error copying {img_filename}: {e}")
            continue
        
        sample_data = {
            'filename': img_filename,
            'original_filepath': str(sample.filepath),
            'metadata': {},
            'tags': sample.tags if hasattr(sample, 'tags') else [],
            'ground_truth': [],
            'predictions': [],
            'classifications': {}
        }
        
        if hasattr(sample, 'metadata') and sample.metadata:
            metadata = sample.metadata
            sample_data['metadata'] = {
                'width': metadata.width if hasattr(metadata, 'width') else None,
                'height': metadata.height if hasattr(metadata, 'height') else None,
                'size_bytes': metadata.size_bytes if hasattr(metadata, 'size_bytes') else None
            }
        
        if sample.ground_truth:
            for det in sample.ground_truth.detections:
                det_data = {
                    'label': det.label,
                    'bounding_box': list(det.bounding_box),
                }
                if hasattr(det, 'confidence') and det.confidence:
                    det_data['confidence'] = float(det.confidence)
                if hasattr(det, 'attributes') and det.attributes:
                    det_data['attributes'] = det.attributes
                sample_data['ground_truth'].append(det_data)
        
        if sample.predictions:
            for det in sample.predictions.detections:
                det_data = {
                    'label': det.label,
                    'bounding_box': list(det.bounding_box),
                }
                if hasattr(det, 'confidence'):
                    det_data['confidence'] = float(det.confidence)
                if hasattr(det, 'model'):
                    det_data['model'] = det.model
                sample_data['predictions'].append(det_data)
        
        for field_name in sample.field_names:
            if field_name.startswith('tag_'):
                tag_obj = getattr(sample, field_name, None)
                if tag_obj and hasattr(tag_obj, 'label'):
                    category = field_name.replace('tag_', '')
                    sample_data['classifications'][category] = tag_obj.label
        
        complete_data['samples'].append(sample_data)
    
    data_file = output_path / "complete_dataset.json"
    print(f"\n💾 Saving dataset metadata...")
    with open(data_file, 'w') as f:
        json.dump(complete_data, f, indent=2)
    
    print(f"   ✅ Saved: {data_file}")
    print(f"   Size: {data_file.stat().st_size / 1024 / 1024:.2f} MB")
    
    create_import_script(output_path, dataset_name)
    create_readme(output_path, dataset_name, complete_data, images_dir)
    
    print(f"\n{'='*70}")
    print("EXPORT COMPLETE")
    print(f"{'='*70}")
    
    total_images = len(list(images_dir.glob("*")))
    total_with_gt = sum(1 for s in complete_data['samples'] if s['ground_truth'])
    total_with_pred = sum(1 for s in complete_data['samples'] if s['predictions'])
    
    print(f"\n📊 Export Summary:")
    print(f"   Images: {total_images}")
    print(f"   With ground_truth: {total_with_gt}")
    print(f"   With predictions: {total_with_pred}")
    
    total_size = sum(f.stat().st_size for f in output_path.rglob('*') if f.is_file())
    print(f"   Total size: {total_size / 1024 / 1024:.0f} MB")
    
    print(f"\n📦 Package: tar -czf {dataset_name}.tar.gz {output_path.name}/")
    
    return output_path



def export_tags_only(dataset_name: str, output_dir: str):
    """Export complete dataset with all data"""
    
    dataset = fo.load_dataset(dataset_name)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("EXPORTING COMPLETE DATASET FOR TRANSFER")
    print("="*70)
    print(f"Dataset: {dataset_name}")
    print(f"Output: {output_path}")
    print(f"Samples: {len(dataset)}")
    
    complete_data = {
        'dataset_name': dataset_name,
        'export_date': datetime.now().isoformat(),
        'total_samples': len(dataset),
        'schema': {},
        'samples': []
    }
    
    schema = dataset.get_field_schema()
    for field_name, field in schema.items():
        complete_data['schema'][field_name] = str(type(field))
    
    print("\n📦 Exporting samples with all data...")
    
    for sample in tqdm(dataset, desc="Exporting"):
        img_filename = Path(sample.filepath).name
        img_source = Path(sample.filepath)
        
        sample_data = {
            'filename': img_filename,
            'original_filepath': str(sample.filepath),
            'metadata': {},
            'tags': sample.tags if hasattr(sample, 'tags') else [],
            'ground_truth': [],
            'predictions': [],
            'classifications': {}
        }
        
        if hasattr(sample, 'metadata') and sample.metadata:
            metadata = sample.metadata
            sample_data['metadata'] = {
                'width': metadata.width if hasattr(metadata, 'width') else None,
                'height': metadata.height if hasattr(metadata, 'height') else None,
                'size_bytes': metadata.size_bytes if hasattr(metadata, 'size_bytes') else None
            }
        
        if sample.ground_truth:
            for det in sample.ground_truth.detections:
                det_data = {
                    'label': det.label,
                    'bounding_box': list(det.bounding_box),
                }
                if hasattr(det, 'confidence') and det.confidence:
                    det_data['confidence'] = float(det.confidence)
                if hasattr(det, 'attributes') and det.attributes:
                    det_data['attributes'] = det.attributes
                sample_data['ground_truth'].append(det_data)
        
        if sample.predictions:
            for det in sample.predictions.detections:
                det_data = {
                    'label': det.label,
                    'bounding_box': list(det.bounding_box),
                }
                if hasattr(det, 'confidence'):
                    det_data['confidence'] = float(det.confidence)
                if hasattr(det, 'model'):
                    det_data['model'] = det.model
                sample_data['predictions'].append(det_data)
        
        for field_name in sample.field_names:
            if field_name.startswith('tag_'):
                tag_obj = getattr(sample, field_name, None)
                if tag_obj and hasattr(tag_obj, 'label'):
                    category = field_name.replace('tag_', '')
                    sample_data['classifications'][category] = tag_obj.label
        
        complete_data['samples'].append(sample_data)
    
    data_file = output_path / "complete_dataset.json"
    print(f"\n💾 Saving dataset metadata...")
    with open(data_file, 'w') as f:
        json.dump(complete_data, f, indent=2)
    
    print(f"   ✅ Saved: {data_file}")
    print(f"   Size: {data_file.stat().st_size / 1024 / 1024:.2f} MB")
    
    print(f"\n{'='*70}")
    print("EXPORT COMPLETE")
    print(f"{'='*70}")
    
    total_images = len(list(images_dir.glob("*")))
    total_with_gt = sum(1 for s in complete_data['samples'] if s['ground_truth'])
    total_with_pred = sum(1 for s in complete_data['samples'] if s['predictions'])
    
    print(f"\n📊 Export Summary:")
    print(f"   Images: {total_images}")
    print(f"   With ground_truth: {total_with_gt}")
    print(f"   With predictions: {total_with_pred}")
    
    total_size = sum(f.stat().st_size for f in output_path.rglob('*') if f.is_file())
    print(f"   Total size: {total_size / 1024 / 1024:.0f} MB")
        
    return output_path



def create_import_script(output_path: Path, dataset_name: str):
    """Create import script"""
    
    script = f'''#!/usr/bin/env python
import fiftyone as fo
from fiftyone import ViewField as F
import json
from pathlib import Path
import argparse

def import_dataset(dataset_name="{dataset_name}"):
    print("="*70)
    print("IMPORTING FIFTYONE DATASET")
    print("="*70)
    
    if dataset_name in fo.list_datasets():
        response = input(f"Dataset '{{dataset_name}}' exists. Delete? (y/n): ")
        if response.lower() == 'y':
            fo.delete_dataset(dataset_name)
        else:
            dataset_name = f"{{dataset_name}}_{{datetime.now().strftime('%Y%m%d_%H%M%S')}}"
            print(f"Using: {{dataset_name}}")
    
    with open("complete_dataset.json") as f:
        data = json.load(f)
    
    dataset = fo.Dataset(dataset_name)
    dataset.persistent = True
    
    imported = 0
    skipped = 0
    
    for sample_data in data['samples']:
        try:
            img_path = Path("images") / sample_data['filename']
            
            if not img_path.exists():
                skipped += 1
                continue
            
            sample = fo.Sample(filepath=str(img_path.absolute()))
            sample.tags = sample_data.get('tags', [])
            
            if sample_data.get('ground_truth'):
                dets = []
                for d in sample_data['ground_truth']:
                    det = fo.Detection(label=d['label'], bounding_box=d['bounding_box'])
                    if 'confidence' in d and d['confidence']:
                        det.confidence = d['confidence']
                    dets.append(det)
                if dets:
                    sample["ground_truth"] = fo.Detections(detections=dets)
            
            if sample_data.get('predictions'):
                dets = []
                for d in sample_data['predictions']:
                    det = fo.Detection(label=d['label'], bounding_box=d['bounding_box'])
                    if 'confidence' in d:
                        det.confidence = d['confidence']
                    dets.append(det)
                if dets:
                    sample["predictions"] = fo.Detections(detections=dets)
            
            for category, tag in sample_data.get('classifications', {{}}).items():
                sample[f"tag_{{category}}"] = fo.Classification(label=tag)
            
            dataset.add_sample(sample)
            imported += 1
            
        except Exception as e:
            print(f"Error: {{e}}")
            skipped += 1
    
    print(f"\\n✅ Imported: {{imported}}, Skipped: {{skipped}}")
    print(f"Dataset: {{dataset_name}}")
    return dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-name', help='Custom name')
    args = parser.parse_args()
    
    from datetime import datetime
    import_dataset(args.dataset_name if args.dataset_name else "{dataset_name}")
'''
    
    import_file = output_path / "import_dataset.py"
    with open(import_file, 'w') as f:
        f.write(script)
    import_file.chmod(0o755)
    
    print(f"   ✅ Created: {import_file}")
    return import_file


def create_readme(output_path: Path, dataset_name: str, complete_data: dict, images_dir: Path):
    """Create README"""
    
    total_images = len(list(images_dir.glob("*")))
    total_with_gt = sum(1 for s in complete_data['samples'] if s['ground_truth'])
    total_with_pred = sum(1 for s in complete_data['samples'] if s['predictions'])
    
    readme = f"""# Portable FiftyOne Dataset: {dataset_name}

Exported: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Samples: {len(complete_data['samples'])}
With ground truth: {total_with_gt}
With predictions: {total_with_pred}

## Transfer Instructions

1. Extract: tar -xzf archive.tar.gz
2. Import: python import_dataset.py
3. Launch: python -c "import fiftyone as fo; fo.launch_app(fo.load_dataset('{dataset_name}'))"
"""
    
    readme_file = output_path / "README.md"
    with open(readme_file, 'w') as f:
        f.write(readme)
    
    print(f"   ✅ Created: {readme_file}")
    return readme_file


def main():
    parser = argparse.ArgumentParser(description="Export complete FiftyOne dataset")
    parser.add_argument('dataset_name', help='FiftyOne dataset name')
    parser.add_argument('--output', '-o', required=True, help='Output directory')
    parser.add_argument('--tags-only', '-t', action="store_true", help='export json tags and labells only')
    
    args = parser.parse_args()
    
    if args.dataset_name not in fo.list_datasets():
        print(f"❌ Dataset '{args.dataset_name}' not found")
        print("\nAvailable:")
        for name in fo.list_datasets():
            print(f"  - {name}")
        return
    if args.tags_only:
        export_tags_only(args.dataset_name, args.output)
    else:
        export_complete_dataset(args.dataset_name, args.output)


if __name__ == "__main__":
    main()