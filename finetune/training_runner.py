# save as: training_runner.py

"""
Training runner that can be called from Streamlit UI or run standalone

This bridges the gap between the UI and actual training execution.
"""

import torch
from ultralytics import YOLO
import yaml
from pathlib import Path
import json
from datetime import datetime
import sys
import subprocess
import threading


class TrainingRunner:
    """Run training in background thread with progress updates"""
    
    def __init__(self, config_file: str = "training_config.yaml"):
        self.config_file = Path(config_file)
        self.config = self.load_config()
        self.model = None
        self.results = None
        self.is_training = False
        
    def load_config(self) -> dict:
        """Load training configuration"""
        if not self.config_file.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_file}")
        
        with open(self.config_file, 'r') as f:
            return yaml.safe_load(f)
    
    def prepare_training(self):
        """Prepare model and data for training"""
        print("Preparing training...")
        
        # Get model config
        model_config = self.config.get('model', {})
        model_name = model_config.get('name', 'yolov8m')
        
        if not model_name.endswith('.pt'):
            model_name = f"{model_name}.pt"
        
        # Load model
        print(f"Loading model: {model_name}")
        self.model = YOLO(model_name)
        
        print("✓ Model loaded")
    
    def train(self, callback=None):
        """
        Run training
        
        Args:
            callback: Optional callback function called after each epoch
        """
        self.is_training = True
        
        # Get training config
        train_config = self.config.get('training', {})
        dataset_config = self.config.get('dataset', {})
        
        # Prepare training args
        data_yaml = Path(dataset_config.get('output_dir', 'training_data')) / 'dataset.yaml'
        
        train_args = {
            'data': str(data_yaml),
            'epochs': train_config.get('epochs', 50),
            'batch': train_config.get('batch_size', 16),
            'lr0': train_config.get('learning_rate', 0.001),
            'optimizer': train_config.get('optimizer', 'Adam'),
            'patience': train_config.get('patience', 50),
            'save_period': train_config.get('save_period', 10),
            'device': torch.cuda.current_device() if torch.cuda.is_available() else 'cpu',
            'workers': 8,
            'project': 'trained_models',
            'name': f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'exist_ok': True,
            'pretrained': True,
            'plots': True,
            'verbose': True
        }
        
        print(f"\n{'='*70}")
        print("Starting Training")
        print(f"{'='*70}")
        for key, value in train_args.items():
            print(f"  {key}: {value}")
        print(f"{'='*70}\n")
        
        # Train
        try:
            self.results = self.model.train(**train_args)
            
            print(f"\n{'='*70}")
            print("✅ Training Complete!")
            print(f"{'='*70}")
            print(f"Best model saved to: {self.results.save_dir}/weights/best.pt")
            
            # Save results summary
            self.save_results()
            
            self.is_training = False
            return True
            
        except Exception as e:
            print(f"\n❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
            self.is_training = False
            return False
    
    def save_results(self):
        """Save training results to JSON"""
        if not self.results:
            return
        
        results_dict = {
            'config': self.config,
            'metrics': {
                'map50': float(self.results.results_dict.get('metrics/mAP50(B)', 0)),
                'map50_95': float(self.results.results_dict.get('metrics/mAP50-95(B)', 0)),
                'precision': float(self.results.results_dict.get('metrics/precision(B)', 0)),
                'recall': float(self.results.results_dict.get('metrics/recall(B)', 0)),
            },
            'best_epoch': getattr(self.results, 'best_epoch', None),
            'save_dir': str(self.results.save_dir),
            'completed_at': datetime.now().isoformat()
        }
        
        # Save to multiple locations
        results_file = Path(self.results.save_dir) / 'results.json'
        with open(results_file, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        # Copy best model to trained_models directory
        best_model_src = Path(self.results.save_dir) / 'weights' / 'best.pt'
        if best_model_src.exists():
            models_dir = Path('trained_models')
            models_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_name = f"model_{timestamp}_map{results_dict['metrics']['map50']:.3f}.pt"
            best_model_dst = models_dir / model_name
            
            import shutil
            shutil.copy(best_model_src, best_model_dst)
            
            # Also copy results
            results_dst = models_dir / f"{Path(model_name).stem}_results.json"
            with open(results_dst, 'w') as f:
                json.dump(results_dict, f, indent=2)
            
            print(f"✓ Model copied to: {best_model_dst}")
            print(f"✓ Results saved to: {results_dst}")


def run_training_background(config_file: str):
    """Run training in background (for use with Streamlit)"""
    runner = TrainingRunner(config_file)
    runner.prepare_training()
    runner.train()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--from-config':
        # Run from config file (called by Streamlit)
        config_file = sys.argv[2] if len(sys.argv) > 2 else 'training_config.yaml'
        run_training_background(config_file)
    else:
        # Run CLI mode
        from train_cli import main
        sys.exit(main())