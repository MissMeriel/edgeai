# save as: save_fiftyone_config.py

"""
Save FiftyOne configuration settings to disk

This includes:
- Dataset configuration
- Annotation settings
- App preferences
- Custom views
- Workspace layouts

Usage:
    python save_fiftyone_config.py annotation_dataset --save-all
"""

import fiftyone as fo
from fiftyone import ViewField as F
from pathlib import Path
import json
import yaml
import argparse
from datetime import datetime

class FiftyOneConfigManager:
    """Manage FiftyOne configuration persistence"""
    
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.dataset = fo.load_dataset(dataset_name)
        self.config_dir = Path("fiftyone_configs")
        self.config_dir.mkdir(exist_ok=True)
    
    def save_all_configs(self):
        """Save all configuration types"""
        
        print("\n" + "="*70)
        print("SAVING ALL FIFTYONE CONFIGURATIONS")
        print("="*70)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{self.dataset_name}_{timestamp}"
        
        # 1. Dataset configuration
        dataset_config = self.save_dataset_config(base_name)
        
        # 2. Annotation configuration
        annotation_config = self.save_annotation_config(base_name)
        
        # 3. Saved views
        views_config = self.save_saved_views(base_name)
        
        # 4. App settings
        app_config = self.save_app_settings(base_name)
        
        # 5. Master config file
        master_config = {
            'dataset_name': self.dataset_name,
            'timestamp': timestamp,
            'files': {
                'dataset_config': str(dataset_config),
                'annotation_config': str(annotation_config),
                'views_config': str(views_config),
                'app_config': str(app_config)
            }
        }
        
        master_file = self.config_dir / f"{base_name}_master.json"
        with open(master_file, 'w') as f:
            json.dump(master_config, f, indent=2)
        
        print(f"\n{'='*70}")
        print("✅ ALL CONFIGURATIONS SAVED")
        print(f"{'='*70}")
        print(f"\n📁 Configuration directory: {self.config_dir}")
        print(f"📄 Master config: {master_file}")
        
        return master_file
    
    def save_dataset_config(self, base_name: str):
        """Save dataset configuration"""
        
        print("\n1️⃣  Saving dataset configuration...")
        
        config = {
            'name': self.dataset.name,
            'persistent': self.dataset.persistent,
            'media_type': self.dataset.media_type,
            'total_samples': len(self.dataset),
            'fields': {},
            'info': self.dataset.info,
            'classes': {}
        }
        
        # Save field schema
        schema = self.dataset.get_field_schema()
        for field_name, field in schema.items():
            config['fields'][field_name] = {
                'type': str(type(field)),
                'description': field.description if hasattr(field, 'description') else None
            }
        
        # Save classes from ground_truth
        if len(self.dataset.match(F("ground_truth").exists() == True)) > 0:
            classes = self.dataset.distinct("ground_truth.detections.label")
            config['classes']['ground_truth'] = sorted([c for c in classes if c])
        
        # Save classes from predictions
        if len(self.dataset.match(F("predictions").exists() == True)) > 0:
            classes = self.dataset.distinct("predictions.detections.label")
            config['classes']['predictions'] = sorted([c for c in classes if c])
        
        # Save
        config_file = self.config_dir / f"{base_name}_dataset.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"   ✅ Saved: {config_file}")
        return config_file
    
    def save_annotation_config(self, base_name: str):
        """Save annotation configuration"""
        
        print("\n2️⃣  Saving annotation configuration...")
        
        config = {
            'default_label_field': 'ground_truth',
            'label_fields': [],
            'annotation_instructions': {
                'ground_truth': {
                    'editable': True,
                    'description': 'Manual annotations and corrected predictions',
                    'workflow': [
                        '1. Press "a" to annotate',
                        '2. Select "ground_truth" field',
                        '3. Edit/add/delete boxes',
                        '4. Press Escape to save'
                    ]
                },
                'predictions': {
                    'editable': False,
                    'description': 'Auto-generated predictions from model'
                }
            }
        }
        
        # Get all Detection/Detections fields
        schema = self.dataset.get_field_schema()
        for field_name, field in schema.items():
            if 'Detection' in str(type(field)):
                config['label_fields'].append(field_name)
        
        config_file = self.config_dir / f"{base_name}_annotation.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"   ✅ Saved: {config_file}")
        return config_file
    
    def save_saved_views(self, base_name: str):
        """Save all saved views"""
        
        print("\n3️⃣  Saving saved views...")
        
        views_config = {
            'saved_views': {}
        }
        
        for view_name in self.dataset.list_saved_views():
            # Get the view
            view = self.dataset.load_saved_view(view_name)
            
            # We can't serialize the actual view object easily,
            # but we can save the view name and sample count
            views_config['saved_views'][view_name] = {
                'sample_count': len(view),
                'description': f"View with {len(view)} samples"
            }
        
        config_file = self.config_dir / f"{base_name}_views.json"
        with open(config_file, 'w') as f:
            json.dump(views_config, f, indent=2)
        
        print(f"   ✅ Saved: {config_file}")
        print(f"   Saved views: {list(views_config['saved_views'].keys())}")
        return config_file
    
    def save_app_settings(self, base_name: str):
        """Save app-level settings"""
        
        print("\n4️⃣  Saving app settings...")
        
        # Get FiftyOne config
        import fiftyone.core.config as foc
        
        config = {
            'database_uri': fo.config.database_uri,
            'database_name': fo.config.database_name,
            'default_ml_backend': fo.config.default_ml_backend,
            'show_progress_bars': fo.config.show_progress_bars,
            'annotation_defaults': {
                'label_field': 'ground_truth',
                'classes': sorted(self.dataset.distinct("ground_truth.detections.label") or []),
                'workflow': 'review_and_fix'
            }
        }
        
        config_file = self.config_dir / f"{base_name}_app.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"   ✅ Saved: {config_file}")
        return config_file
    
    def create_dataset_restore_script(self, base_name: str):
        """Create a script to restore this configuration"""
        
        print("\n5️⃣  Creating restore script...")
        
        script_content = f'''#!/usr/bin/env python
"""
Auto-generated script to restore dataset configuration
Generated: {datetime.now().isoformat()}
"""

import fiftyone as fo
from pathlib import Path
import json

def restore_configuration():
    """Restore dataset configuration"""
    
    print("Restoring configuration for: {self.dataset_name}")
    
    # Load dataset
    if "{self.dataset_name}" not in fo.list_datasets():
        print("❌ Dataset not found. Create it first with:")
        print("   python fiftyone_annotation_app_fixed_tags.py /path/to/images \\\\")
        print("          --dataset-name {self.dataset_name}")
        return
    
    dataset = fo.load_dataset("{self.dataset_name}")
    
    # Ensure ground_truth field exists on all samples
    print("Ensuring ground_truth field exists...")
    for sample in dataset.iter_samples(progress=True):
        if not sample.ground_truth:
            if sample.predictions:
                # Copy from predictions
                new_dets = []
                for pred in sample.predictions.detections:
                    new_dets.append(fo.Detection(
                        label=pred.label,
                        bounding_box=pred.bounding_box
                    ))
                sample["ground_truth"] = fo.Detections(detections=new_dets)
            else:
                sample["ground_truth"] = fo.Detections(detections=[])
            sample.save()
    
    print("\\n✅ Configuration restored!")
    print("\\nTo annotate:")
    print("  1. Launch: python annotation_workflow.py {self.dataset_name}")
    print("  2. Press 'a' key in FiftyOne")
    print("  3. Select 'ground_truth' field")
    print("  4. Edit annotations")
    
    return dataset

if __name__ == "__main__":
    restore_configuration()
'''
        
        script_file = self.config_dir / f"{base_name}_restore.py"
        with open(script_file, 'w') as f:
            f.write(script_content)
        
        script_file.chmod(0o755)  # Make executable
        
        print(f"   ✅ Saved: {script_file}")
        return script_file


def save_annotation_session_config(dataset_name: str):
    """Save a ready-to-use annotation configuration"""
    
    dataset = fo.load_dataset(dataset_name)
    
    config_dir = Path("fiftyone_configs")
    config_dir.mkdir(exist_ok=True)
    
    config = {
        'dataset_name': dataset_name,
        'annotation_field': 'ground_truth',
        'prediction_field': 'predictions',
        'tag_categories': {
            'scene': ['city_street', 'highway', 'parking_lot', 'residential', 'rural', 'indoor'],
            'time': ['day', 'night', 'dawn', 'dusk'],
            'weather': ['clear', 'cloudy', 'rainy', 'snowy', 'foggy'],
            'quality': ['high_quality', 'low_quality', 'blurry', 'dark', 'overexposed'],
            'distribution': ['train', 'val', 'test'],
            'review_status': ['todo', 'fixed', 'needs_work', 'reviewed', 'skip']
        },
        'annotation_workflow': {
            'step1': 'Open review_status_todo view',
            'step2': 'Click sample, press "a" key',
            'step3': 'Select "ground_truth" field',
            'step4': 'Edit annotations',
            'step5': 'Press "t" key, type "fixed"',
            'step6': 'Press "n" for next sample'
        },
        'export_command': f'python fix_export.py {dataset_name} output_dir --format yolo',
        'classes': sorted(dataset.distinct("ground_truth.detections.label") or 
                         dataset.distinct("predictions.detections.label") or [])
    }
    
    # Save as JSON
    config_file = config_dir / f"{dataset_name}_config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n✅ Saved annotation config: {config_file}")
    
    # Also save as YAML (easier to read/edit)
    yaml_file = config_dir / f"{dataset_name}_config.yaml"
    with open(yaml_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"✅ Saved annotation config: {yaml_file}")
    
    return config_file


def load_and_apply_config(config_file: str):
    """Load and apply a saved configuration"""
    
    print(f"\n📥 Loading configuration from {config_file}...")
    
    with open(config_file) as f:
        if config_file.endswith('.json'):
            config = json.load(f)
        else:
            config = yaml.safe_load(f)
    
    dataset_name = config['dataset_name']
    
    if dataset_name not in fo.list_datasets():
        print(f"❌ Dataset '{dataset_name}' not found")
        return
    
    dataset = fo.load_dataset(dataset_name)
    
    print(f"✅ Loaded config for: {dataset_name}")
    print(f"\n📋 Configuration:")
    print(f"   Annotation field: {config['annotation_field']}")
    print(f"   Classes: {len(config.get('classes', []))}")
    print(f"   Tag categories: {list(config['tag_categories'].keys())}")
    
    return config


def create_workspace_config(dataset_name: str):
    """Create a workspace configuration file"""
    
    dataset = fo.load_dataset(dataset_name)
    
    workspace = {
        'name': f'{dataset_name}_workspace',
        'dataset': dataset_name,
        'color_scheme': {
            'ground_truth': '#00FF00',  # Green
            'predictions': '#FFA500',    # Orange
            'todo': '#FF0000',           # Red
            'fixed': '#00FF00',          # Green
            'needs_work': '#FFFF00'      # Yellow
        },
        'sidebar_config': {
            'expanded_fields': [
                'tag_review_status',
                'tag_scene',
                'tag_time',
                'ground_truth',
                'predictions'
            ],
            'collapsed_fields': [
                'metadata'
            ]
        },
        'default_view': 'review_status_todo',
        'keyboard_shortcuts': {
            'mark_fixed': 't → type "fixed" → Enter',
            'next_sample': 'n',
            'previous_sample': 'p',
            'annotate': 'a → select "ground_truth"',
            'tag_samples': 't'
        }
    }
    
    workspace_file = Path("fiftyone_configs") / f"{dataset_name}_workspace.json"
    with open(workspace_file, 'w') as f:
        json.dump(workspace, f, indent=2)
    
    print(f"\n✅ Saved workspace config: {workspace_file}")
    return workspace_file


def save_readme(dataset_name: str):
    """Create a README with instructions"""
    
    readme_content = f"""# FiftyOne Dataset: {dataset_name}

## Configuration Files

This directory contains saved FiftyOne configurations for the `{dataset_name}` dataset.

## Files

- `{dataset_name}_config.json` - Main configuration
- `{dataset_name}_config.yaml` - Human-readable configuration
- `{dataset_name}_workspace.json` - UI workspace settings
- `{dataset_name}_restore.py` - Restoration script
- `README.md` - This file

## Quick Start

### Restore Configuration

```bash
# Load configuration
python -c "import json; print(json.load(open('{dataset_name}_config.json')))"

# Or restore dataset setup
python {dataset_name}_restore.py