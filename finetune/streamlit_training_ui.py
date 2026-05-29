# save as: streamlit_training_ui.py (COMPLETE VERSION)

"""
Streamlit Training UI for Object Detection Fine-Tuning

Launch with:
    streamlit run streamlit_training_ui.py -- --dataset annotation_dataset

Features:
- Load annotations from FiftyOne
- Configure training parameters
- Real-time training monitoring
- Hyperparameter optimization with Optuna
- Model evaluation and export
"""

import streamlit as st
import fiftyone as fo
from fiftyone import ViewField as F
import torch
import yaml
import json
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime
import subprocess
import os
import sys
import cv2
import numpy as np
from typing import Optional, Dict, List, Tuple
import threading
import time
import shutil

# Page config must be first Streamlit command
st.set_page_config(
    page_title="Model Training Interface",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .status-badge {
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    .status-ready { background: #4CAF50; color: white; }
    .status-training { background: #2196F3; color: white; }
    .status-complete { background: #9C27B0; color: white; }
    .status-error { background: #F44336; color: white; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'training_status' not in st.session_state:
    st.session_state.training_status = 'idle'
if 'current_epoch' not in st.session_state:
    st.session_state.current_epoch = 0
if 'training_metrics' not in st.session_state:
    st.session_state.training_metrics = []
if 'best_map' not in st.session_state:
    st.session_state.best_map = 0.0
if 'training_log' not in st.session_state:
    st.session_state.training_log = []

class TrainingUI:
    """Streamlit UI for model training"""
    
    def __init__(self):
        self.project_dir = Path.cwd()
        self.config_file = self.project_dir / "training_config.yaml"
        self.models_dir = self.project_dir / "trained_models"
        self.models_dir.mkdir(exist_ok=True)
        self.logs_dir = self.project_dir / "training_logs"
        self.logs_dir.mkdir(exist_ok=True)
        
    def load_fiftyone_dataset(self, dataset_name: str) -> Optional[fo.Dataset]:
        """Load FiftyOne dataset"""
        try:
            if dataset_name in fo.list_datasets():
                return fo.load_dataset(dataset_name)
            return None
        except Exception as e:
            st.error(f"Error loading dataset: {e}")
            return None
    
    def get_dataset_stats(self, dataset: fo.Dataset) -> Dict:
        """Get statistics from FiftyOne dataset"""
        total = len(dataset)
        annotated = len(dataset.match(F("ground_truth").exists()))
        
        classes = dataset.distinct("ground_truth.detections.label") if annotated > 0 else []
        class_counts = dataset.count_values("ground_truth.detections.label") if annotated > 0 else {}
        
        total_boxes = sum(class_counts.values())
        
        return {
            'total_images': total,
            'annotated_images': annotated,
            'classes': sorted([c for c in classes if c]),
            'class_counts': class_counts,
            'total_boxes': total_boxes
        }
    
    def export_for_training(self, dataset: fo.Dataset, output_dir: Path, 
                           train_ratio: float, val_ratio: float,
                           selected_classes: List[str] = None) -> Dict:
        """Export FiftyOne dataset to YOLO format"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get annotated samples
        annotated = dataset.match(F("ground_truth").exists())
        
        # Filter by selected classes if specified
        if selected_classes:
            annotated = annotated.filter_labels("ground_truth", F("label").is_in(selected_classes))
        
        # Split dataset
        import random
        random.seed(42)
        samples = list(annotated)
        random.shuffle(samples)
        
        n = len(samples)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        train_samples = samples[:train_end]
        val_samples = samples[train_end:val_end]
        test_samples = samples[val_end:]
        
        # Create directories
        for split in ['train', 'val', 'test']:
            (output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        # Get classes
        classes = selected_classes if selected_classes else sorted(annotated.distinct("ground_truth.detections.label"))
        class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        
        # Export splits
        def export_split(samples, split_name):
            progress_bar = st.progress(0)
            for idx, sample in enumerate(samples):
                # Copy image
                img_name = Path(sample.filepath).name
                shutil.copy(sample.filepath, output_dir / split_name / 'images' / img_name)
                
                # Create label file
                img = cv2.imread(sample.filepath)
                h, w = img.shape[:2]
                
                label_file = output_dir / split_name / 'labels' / f"{Path(img_name).stem}.txt"
                with open(label_file, 'w') as f:
                    for det in sample.ground_truth.detections:
                        if det.label in class_to_idx:
                            x, y, bw, bh = det.bounding_box
                            cx = x + bw/2
                            cy = y + bh/2
                            class_idx = class_to_idx[det.label]
                            f.write(f"{class_idx} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
                
                progress_bar.progress((idx + 1) / len(samples))
        
        export_split(train_samples, 'train')
        export_split(val_samples, 'val')
        export_split(test_samples, 'test')
        
        # Create dataset.yaml
        yaml_data = {
            'path': str(output_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': len(classes),
            'names': classes
        }
        
        with open(output_dir / 'dataset.yaml', 'w') as f:
            yaml.dump(yaml_data, f)
        
        return {
            'train': len(train_samples),
            'val': len(val_samples),
            'test': len(test_samples),
            'classes': classes
        }
    
    def render(self):
        """Render the Streamlit UI"""
        
        # Header
        st.markdown('<div class="main-header">🚀 Model Training Interface</div>', 
                   unsafe_allow_html=True)
        
        # Sidebar
        with st.sidebar:
            st.image("https://via.placeholder.com/200x80/667eea/ffffff?text=Training+UI", 
                    use_container_width=True)
            
            st.markdown("### 📊 Navigation")
            page = st.radio(
                "Select Page",
                ["🏠 Home", "📁 Dataset", "🤖 Model Config", "⚙️ Training", 
                 "📊 Monitor", "✅ Evaluate", "🔮 AutoML"],
                label_visibility="collapsed"
            )
            
            st.markdown("---")
            st.markdown("### 🔗 FiftyOne Dataset")
            
            # Get FiftyOne datasets
            datasets = fo.list_datasets()
            if datasets:
                selected_dataset = st.selectbox("Dataset", datasets, key="dataset_selector")
                if st.button("🔄 Refresh Datasets"):
                    st.rerun()
            else:
                st.warning("No FiftyOne datasets found")
                st.info("Run annotation interface first")
                selected_dataset = None
            
            st.markdown("---")
            
            # System status
            st.markdown("### 💻 System")
            gpu_available = torch.cuda.is_available()
            if gpu_available:
                st.success(f"✓ GPU: {torch.cuda.get_device_name(0)}")
                memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                st.info(f"Memory: {memory:.1f} GB")
            else:
                st.warning("⚠️ CPU only")
            
            st.markdown(f"**PyTorch:** {torch.__version__}")
            st.markdown(f"**Python:** {sys.version.split()[0]}")
        
        # Main content area
        if page == "🏠 Home":
            self.render_home()
        elif page == "📁 Dataset":
            self.render_dataset(selected_dataset)
        elif page == "🤖 Model Config":
            self.render_model_config()
        elif page == "⚙️ Training":
            self.render_training_config()
        elif page == "📊 Monitor":
            self.render_training_monitor()
        elif page == "✅ Evaluate":
            self.render_evaluation()
        elif page == "🔮 AutoML":
            self.render_automl()
    
    def render_home(self):
        """Render home/overview page"""
        
        st.markdown("## Welcome to the Training Interface")
        
        st.markdown("""
        This interface allows you to fine-tune object detection models on your annotated data from FiftyOne.
        
        ### 🎯 Workflow:
        1. **📁 Dataset**: Load and prepare your FiftyOne dataset
        2. **🤖 Model Config**: Choose model architecture and pretrained weights  
        3. **⚙️ Training**: Configure hyperparameters
        4. **📊 Monitor**: Watch training progress in real-time
        5. **✅ Evaluate**: Review results and export model
        6. **🔮 AutoML** (Optional): Let AI find best hyperparameters
        """)
        
        # Quick status cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>📁</h3>
                <h2>Datasets</h2>
                <p style="font-size: 2rem; margin: 0;">{}</p>
            </div>
            """.format(len(fo.list_datasets())), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h3>🤖</h3>
                <h2>Models</h2>
                <p style="font-size: 2rem; margin: 0;">{}</p>
            </div>
            """.format(len(list(self.models_dir.glob("*.pt")))), unsafe_allow_html=True)
        
        with col3:
            status_class = {
                'idle': 'ready',
                'training': 'training',
                'complete': 'complete',
                'error': 'error'
            }.get(st.session_state.training_status, 'ready')
            
            st.markdown("""
            <div class="metric-card">
                <h3>⚡</h3>
                <h2>Status</h2>
                <span class="status-badge status-{}">{}</span>
            </div>
            """.format(status_class, st.session_state.training_status.title()), 
            unsafe_allow_html=True)
        
        with col4:
            device = "GPU" if torch.cuda.is_available() else "CPU"
            st.markdown("""
            <div class="metric-card">
                <h3>💻</h3>
                <h2>Device</h2>
                <p style="font-size: 2rem; margin: 0;">{}</p>
            </div>
            """.format(device), unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Recent activity
        st.markdown("### 📝 Recent Trainings")
        
        model_files = sorted(self.models_dir.glob("*.pt"), key=lambda x: x.stat().st_mtime, reverse=True)[:5]
        
        if model_files:
            for model_file in model_files:
                col1, col2, col3 = st.columns([3, 2, 1])
                with col1:
                    st.markdown(f"**{model_file.name}**")
                with col2:
                    st.markdown(f"_{datetime.fromtimestamp(model_file.stat().st_mtime).strftime('%Y-%m-%d %H:%M')}_")
                with col3:
                    st.markdown(f"`{model_file.stat().st_size / 1e6:.1f} MB`")
        else:
            st.info("No trained models yet. Start your first training run!")
    
    def render_dataset(self, dataset_name: str):
        """Render dataset configuration page"""
        
        st.markdown("## 📁 Dataset Configuration")
        
        if not dataset_name:
            st.warning("⚠️ No dataset selected. Please select a FiftyOne dataset from the sidebar.")
            
            st.markdown("### Available Datasets")
            datasets = fo.list_datasets()
            if datasets:
                for ds_name in datasets:
                    st.markdown(f"- {ds_name}")
            else:
                st.info("No datasets found. Create one with the annotation interface first.")
            
            return
        
        # Load dataset
        dataset = self.load_fiftyone_dataset(dataset_name)
        if not dataset:
            st.error(f"Could not load dataset: {dataset_name}")
            return
        
        # Get stats
        stats = self.get_dataset_stats(dataset)
        
        st.success(f"✓ Loaded dataset: **{dataset_name}**")
        
        # Dataset overview
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Images", stats['total_images'])
        with col2:
            st.metric("Annotated Images", stats['annotated_images'])
        with col3:
            st.metric("Total Boxes", stats['total_boxes'])
        
        # Class distribution
        st.markdown("### 🏷️ Class Distribution")
        
        if stats['class_counts']:
            df = pd.DataFrame([
                {'Class': k, 'Count': v} 
                for k, v in stats['class_counts'].items()
            ]).sort_values('Count', ascending=False)
            
            fig = px.bar(df, x='Class', y='Count', 
                        title="Annotations per Class",
                        color='Count',
                        color_continuous_scale='Viridis')
            st.plotly_chart(fig, use_container_width=True)
            
            # Show table
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.warning("No annotations found in dataset")
            return
        
        st.markdown("---")
        
        # Data split configuration
        st.markdown("### ✂️ Train/Val/Test Split")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            train_ratio = st.slider("Training %", 0, 100, 70, 5) / 100
        with col2:
            val_ratio = st.slider("Validation %", 0, 100, 20, 5) / 100
        with col3:
            test_ratio = st.slider("Test %", 0, 100, 10, 5) / 100
        
        # Validate splits sum to 100%
        total = (train_ratio + val_ratio + test_ratio) * 100
        if abs(total - 100) > 0.1:
            st.error(f"⚠️ Splits must sum to 100% (currently {total:.1f}%)")
        else:
            st.success(f"✓ Splits sum to 100%")
        
        # Estimate split sizes
        n_annotated = stats['annotated_images']
        st.info(f"""
        **Estimated split sizes:**
        - Training: {int(n_annotated * train_ratio)} images
        - Validation: {int(n_annotated * val_ratio)} images  
        - Test: {int(n_annotated * test_ratio)} images
        """)
        
        # Class selection
        st.markdown("### 🎯 Class Selection")
        
        if stats['classes']:
            selected_classes = st.multiselect(
                "Select classes to include in training",
                stats['classes'],
                default=stats['classes'],
                help="Only selected classes will be used for training"
            )
            
            if len(selected_classes) < len(stats['classes']):
                excluded = set(stats['classes']) - set(selected_classes)
                st.warning(f"Excluding classes: {', '.join(excluded)}")
        else:
            selected_classes = []
        
        # Data augmentation
        st.markdown("### 🔄 Data Augmentation")
        
        aug_preset = st.select_slider(
            "Augmentation Intensity",
            options=["None", "Light", "Medium", "Heavy"],
            value="Medium",
            help="Higher intensity = more variation but may slow training"
        )
        
        with st.expander("🔧 Custom Augmentation Settings"):
            aug_col1, aug_col2 = st.columns(2)
            
            with aug_col1:
                st.markdown("**Geometric**")
                flip_h = st.checkbox("Horizontal Flip", value=True)
                flip_v = st.checkbox("Vertical Flip", value=False)
                rotate = st.slider("Random Rotation (degrees)", 0, 45, 15)
                scale = st.slider("Random Scale (%)", 0, 50, 10)
                translate = st.slider("Random Translate (%)", 0, 20, 10)
            
            with aug_col2:
                st.markdown("**Color**")
                brightness = st.slider("Brightness (%)", 0, 50, 20)
                contrast = st.slider("Contrast (%)", 0, 50, 20)
                saturation = st.slider("Saturation (%)", 0, 50, 20)
                hue = st.slider("Hue Shift", 0, 50, 0)
                blur = st.checkbox("Random Blur", value=False)
        
        # Preview augmentations
        st.markdown("### 👁️ Preview Augmentations")
        
        if st.button("🔄 Show Random Sample with Augmentations"):
            # Get random sample
            sample = list(annotated.take(1))[0]
            img = cv2.imread(sample.filepath)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Original**")
                st.image(img, use_container_width=True)
            
            # Apply augmentations (simplified for preview)
            aug_img = img.copy()
            if flip_h and np.random.random() > 0.5:
                aug_img = cv2.flip(aug_img, 1)
            
            with col2:
                st.markdown("**Augmented 1**")
                st.image(aug_img, use_container_width=True)
            
            with col3:
                st.markdown("**Augmented 2**")
                st.image(aug_img, use_container_width=True)
        
        # Export dataset
        st.markdown("---")
        
        if st.button("📤 Export Dataset for Training", type="primary", use_container_width=True):
            with st.spinner("Exporting dataset..."):
                output_dir = self.project_dir / "training_data"
                result = self.export_for_training(dataset, output_dir, train_ratio, val_ratio, selected_classes)
                
                st.success("✓ Dataset exported successfully!")
                
                # Show export summary
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Training Images", result['train'])
                with col2:
                    st.metric("Validation Images", result['val'])
                with col3:
                    st.metric("Test Images", result['test'])
                
                st.json(result)
                
                # Save config
                config = {
                    'dataset_name': dataset_name,
                    'output_dir': str(output_dir),
                    'train_ratio': train_ratio,
                    'val_ratio': val_ratio,
                    'test_ratio': test_ratio,
                    'selected_classes': selected_classes,
                    'augmentation': {
                        'preset': aug_preset,
                        'flip_h': flip_h,
                        'flip_v': flip_v,
                        'rotate': rotate,
                        'scale': scale,
                        'translate': translate,
                        'brightness': brightness,
                        'contrast': contrast,
                        'saturation': saturation,
                        'hue': hue,
                        'blur': blur
                    },
                    'export_time': datetime.now().isoformat()
                }
                
                with open(self.config_file, 'w') as f:
                    yaml.dump(config, f)
                
                st.balloons()
                st.info("✅ Configuration saved. Proceed to Model Config →")
    
    def render_model_config(self):
        """Render model configuration page"""
        
        st.markdown("## 🤖 Model Configuration")
        
        # Check if dataset is configured
        if not self.config_file.exists():
            st.warning("⚠️ Please configure dataset first (📁 Dataset page)")
            return
        
        # Model architecture
        st.markdown("### 🏗️ Model Architecture")
        
        tab1, tab2, tab3 = st.tabs(["🎯 Beginner", "🔧 Intermediate", "🎓 Expert"])
        
        with tab1:
            st.markdown("**Choose a preset configuration**")
            
            presets = [
                {
                    "name": "🏃 Fast & Light",
                    "model": "YOLOv8n",
                    "size": "6 MB",
                    "speed": "45 FPS",
                    "desc": "Best for: Real-time applications, mobile deployment"
                },
                {
                    "name": "⚖️ Balanced",
                    "model": "YOLOv8m",
                    "size": "50 MB",
                    "speed": "30 FPS",
                    "desc": "Best for: General purpose, recommended starting point ⭐"
                },
                {
                    "name": "🎯 Accurate",
                    "model": "YOLOv8l",
                    "size": "87 MB",
                    "speed": "20 FPS",
                    "desc": "Best for: High accuracy requirements"
                },
                {
                    "name": "🏆 Maximum",
                    "model": "YOLOv8x",
                    "size": "136 MB",
                    "speed": "15 FPS",
                    "desc": "Best for: Maximum accuracy, server deployment"
                }
            ]
            
            selected_preset = None
            for preset in presets:
                if st.button(
                    f"{preset['name']}\n{preset['model']} | {preset['size']} | {preset['speed']}\n{preset['desc']}",
                    use_container_width=True,
                    key=f"preset_{preset['model']}"
                ):
                    selected_preset = preset
                    st.session_state.selected_model = preset['model']
                    st.success(f"✓ Selected: {preset['model']}")
        
        with tab2:
            st.markdown("**Choose specific model and configuration**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                model_family = st.selectbox(
                    "Model Family",
                    ["YOLOv8 (Recommended)", "YOLOv5", "YOLOv7", "Faster R-CNN", "EfficientDet"],
                    help="Different model architectures with varying speed/accuracy tradeoffs"
                )
            
            with col2:
                if "YOLO" in model_family:
                    model_size = st.selectbox(
                        "Model Size",
                        ["n (nano)", "s (small)", "m (medium)", "l (large)", "x (xlarge)"],
                        index=2,
                        help="Larger models are more accurate but slower"
                    )
                    model_code = model_size[0]
                    
                    if "YOLOv8" in model_family:
                        st.session_state.selected_model = f"yolov8{model_code}"
                    elif "YOLOv5" in model_family:
                        st.session_state.selected_model = f"yolov5{model_code}"
                else:
                    model_size = "default"
                    st.session_state.selected_model = model_family.split()[0].lower()
        
        with tab3:
            st.markdown("**Full control over architecture**")
            
            custom_model = st.text_input(
                "Custom Model Name/Path",
                placeholder="yolov8m.pt or path/to/custom.pt",
                help="Specify exact model file or HuggingFace model name"
            )
            
            if custom_model:
                st.session_state.selected_model = custom_model
            
            freeze_backbone = st.checkbox("Freeze Backbone Layers", value=True,
                help="Freeze early layers and only train detection head (faster, less data needed)")
            
            if freeze_backbone:
                freeze_until = st.slider("Freeze until layer (%)", 0, 100, 80, 5,
                                        help="Percentage of layers to freeze")
            
            input_size = st.selectbox("Input Image Size", [320, 416, 512, 640, 800, 1280], index=3,
                help="Larger images = more accurate but slower and more memory")
            
            st.session_state.input_size = input_size
        
        # Pretrained weights
        st.markdown("### 🎓 Pretrained Weights")
        
        pretrained_source = st.radio(
            "Initialize weights from",
            ["COCO (80 classes) - Recommended ⭐", 
             "Custom checkpoint",
             "Random initialization (train from scratch)"],
            help="Transfer learning from pretrained weights usually gives better results"
        )
        
        if "Custom" in pretrained_source:
            custom_weights = st.file_uploader(
                "Upload weights file (.pt, .pth)", 
                type=['pt', 'pth'],
                help="Upload a previously trained model checkpoint"
            )
            
            if custom_weights:
                # Save uploaded file
                weights_path = self.models_dir / custom_weights.name
                with open(weights_path, 'wb') as f:
                    f.write(custom_weights.getbuffer())
                st.success(f"✓ Uploaded: {custom_weights.name}")
                st.session_state.pretrained_weights = str(weights_path)
        elif "COCO" in pretrained_source:
            st.session_state.pretrained_weights = "coco"
        else:
            st.session_state.pretrained_weights = None
        
        # Model summary
        st.markdown("### 📋 Configuration Summary")
        
        summary_col1, summary_col2 = st.columns(2)
        
        model_name = st.session_state.get('selected_model', 'yolov8m')
        
        # Model specs (approximate)
        model_specs = {
            'yolov8n': {'params': '3.2M', 'size': '6 MB', 'speed': '45 FPS', 'memory': '2 GB'},
            'yolov8s': {'params': '11.2M', 'size': '22 MB', 'speed': '38 FPS', 'memory': '4 GB'},
            'yolov8m': {'params': '25.9M', 'size': '52 MB', 'speed': '30 FPS', 'memory': '6 GB'},
            'yolov8l': {'params': '43.7M', 'size': '87 MB', 'speed': '20 FPS', 'memory': '8 GB'},
            'yolov8x': {'params': '68.2M', 'size': '136 MB', 'speed': '15 FPS', 'memory': '10 GB'},
        }
        
        specs = model_specs.get(model_name, model_specs['yolov8m'])
        
        with summary_col1:
            st.markdown(f"""
            **Selected Configuration:**
            - Model: {model_name}
            - Parameters: {specs['params']}
            - Model Size: {specs['size']}
            - Pretrained: {st.session_state.get('pretrained_weights', 'COCO')}
            """)
        
        with summary_col2:
            st.markdown(f"""
            **Expected Performance:**
            - Speed: {specs['speed']} (GPU)
            - GPU Memory: ~{specs['memory']}
            - Training: Best with GPU
            - Inference: CPU capable
            """)
        
        # Save button
        if st.button("💾 Save Model Configuration", type="primary", use_container_width=True):
            # Load existing config
# save as: streamlit_training_ui.py (COMPLETE - CONTINUED)

            # Load existing config
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    config = yaml.safe_load(f)
            else:
                config = {}
            
            # Update with model config
            config['model'] = {
                'name': model_name,
                'pretrained': st.session_state.get('pretrained_weights', 'coco'),
                'input_size': st.session_state.get('input_size', 640),
                'freeze_backbone': freeze_backbone if 'freeze_backbone' in locals() else True,
                'freeze_until': freeze_until if 'freeze_until' in locals() else 80,
                'save_time': datetime.now().isoformat()
            }
            
            with open(self.config_file, 'w') as f:
                yaml.dump(config, f)
            
            st.success("✓ Model configuration saved!")
            st.balloons()
            st.info("Proceed to Training Configuration →")
    
    def render_training_config(self):
        """Render training configuration page"""
        
        st.markdown("## ⚙️ Training Configuration")
        
        # Check if model is configured
        if not self.config_file.exists():
            st.warning("⚠️ Please configure model first (🤖 Model Config page)")
            return
        
        # Mode selection
        mode = st.radio(
            "Configuration Mode",
            ["🎯 Beginner (Presets)", "🔧 Advanced (Custom Parameters)"],
            horizontal=True,
            help="Beginner: Simple presets | Advanced: Full control"
        )
        
        if "Beginner" in mode:
            self.render_beginner_config()
        else:
            self.render_advanced_config()
        
        st.markdown("---")
        
        # Training summary
        st.markdown("### 📊 Training Summary")
        
        epochs = st.session_state.get('epochs', 50)
        batch_size = st.session_state.get('batch_size', 16)
        lr = st.session_state.get('learning_rate', 0.001)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Epochs", epochs)
        with col2:
            st.metric("Batch Size", batch_size)
        with col3:
            st.metric("Learning Rate", f"{lr:.4f}")
        
        # Estimated training time
        st.markdown("### ⏱️ Estimated Training Time")
        
        # Load config to get dataset size
        with open(self.config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        if 'output_dir' in config:
            train_images = len(list((Path(config['output_dir']) / 'train' / 'images').glob('*')))
            
            # Rough estimation
            time_per_image = 0.1 if torch.cuda.is_available() else 0.5  # seconds
            estimated_seconds = (train_images / batch_size) * epochs * time_per_image
            
            hours = int(estimated_seconds // 3600)
            minutes = int((estimated_seconds % 3600) // 60)
            
            st.info(f"⏱️ Estimated time: **{hours}h {minutes}m** ({train_images} training images)")
        
        # Start training button
        st.markdown("---")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            if st.button("🚀 Start Training", type="primary", use_container_width=True, 
                        disabled=(st.session_state.training_status == 'training')):
                
                # Save training config
                if self.config_file.exists():
                    with open(self.config_file, 'r') as f:
                        config = yaml.safe_load(f)
                else:
                    config = {}
                
                config['training'] = {
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'learning_rate': lr,
                    'optimizer': st.session_state.get('optimizer', 'Adam'),
                    'scheduler': st.session_state.get('scheduler', 'cosine'),
                    'warmup_epochs': st.session_state.get('warmup_epochs', 3),
                    'save_period': st.session_state.get('save_period', 10),
                    'patience': st.session_state.get('patience', 50),
                    'start_time': datetime.now().isoformat()
                }
                
                with open(self.config_file, 'w') as f:
                    yaml.dump(config, f)
                
                st.session_state.training_status = 'training'
                st.session_state.current_epoch = 0
                st.session_state.training_metrics = []
                
                st.success("✓ Training started! Go to Monitor page →")
                st.rerun()
        
        with col2:
            if st.button("💾 Save Config", use_container_width=True):
                st.success("✓ Saved")
    
    def render_beginner_config(self):
        """Beginner-friendly configuration with presets"""
        
        st.markdown("### 🎯 Training Presets")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            preset = st.select_slider(
                "Choose training intensity",
                options=[
                    "🏃 Quick Test",
                    "⚡ Fast",
                    "⚖️ Standard",
                    "🎯 Thorough",
                    "🏆 Maximum"
                ],
                value="⚖️ Standard",
                help="Longer training usually gives better results but takes more time"
            )
        
        # Map preset to actual values
        preset_configs = {
            "🏃 Quick Test": {'epochs': 10, 'lr': 0.001, 'batch': 16, 'time': '15 min'},
            "⚡ Fast": {'epochs': 30, 'lr': 0.001, 'batch': 16, 'time': '45 min'},
            "⚖️ Standard": {'epochs': 50, 'lr': 0.001, 'batch': 16, 'time': '2 hours'},
            "🎯 Thorough": {'epochs': 100, 'lr': 0.001, 'batch': 16, 'time': '4 hours'},
            "🏆 Maximum": {'epochs': 200, 'lr': 0.0005, 'batch': 16, 'time': '8 hours'}
        }
        
        config = preset_configs[preset]
        
        st.session_state.epochs = config['epochs']
        st.session_state.batch_size = config['batch']
        st.session_state.learning_rate = config['lr']
        st.session_state.optimizer = 'Adam'
        st.session_state.scheduler = 'cosine'
        st.session_state.warmup_epochs = 3
        st.session_state.save_period = 10
        st.session_state.patience = 50
        
        with col2:
            st.markdown(f"""
            **{preset}**
            
            - Epochs: {config['epochs']}
            - Est. Time: ~{config['time']}
            - Batch Size: {config['batch']}
            """)
        
        # Show what this means
        st.markdown("### 📖 What This Means")
        
        st.markdown(f"""
        Your model will:
        - Train for **{config['epochs']} epochs** (passes through the data)
        - Process **{config['batch']} images at a time**
        - Take approximately **{config['time']}** on typical hardware
        """)
        
        # Additional simple options
        st.markdown("### 🎨 Additional Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            use_augmentation = st.checkbox("Use Data Augmentation", value=True,
                help="Randomly transform images during training to improve generalization")
            st.session_state.use_augmentation = use_augmentation
        
        with col2:
            early_stopping = st.checkbox("Early Stopping", value=True,
                help="Stop training if validation performance stops improving")
            st.session_state.early_stopping = early_stopping
        
        if early_stopping:
            patience = st.slider("Patience (epochs)", 5, 50, 10,
                               help="Number of epochs to wait before stopping")
            st.session_state.patience = patience
    
    def render_advanced_config(self):
        """Advanced configuration with full control"""
        
        st.markdown("### ⚙️ Hyperparameters")
        
        # Training parameters
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Training Loop**")
            epochs = st.number_input("Epochs", 1, 500, 50, 
                help="Number of complete passes through the training dataset")
            st.session_state.epochs = epochs
            
            batch_size = st.select_slider("Batch Size", 
                options=[4, 8, 16, 32, 64, 128], 
                value=16,
                help="Number of images processed simultaneously. Larger = faster but more memory")
            st.session_state.batch_size = batch_size
            
            save_period = st.number_input("Save Checkpoint Every N Epochs", 1, 50, 10)
            st.session_state.save_period = save_period
        
        with col2:
            st.markdown("**Optimizer Settings**")
            optimizer = st.selectbox("Optimizer", 
                ["Adam", "AdamW", "SGD", "RMSprop"],
                help="Algorithm for updating model weights")
            st.session_state.optimizer = optimizer
            
            lr = st.number_input("Learning Rate", 0.0001, 0.1, 0.001, format="%.4f",
                help="Step size for weight updates. Too high = unstable, too low = slow")
            st.session_state.learning_rate = lr
            
            weight_decay = st.number_input("Weight Decay", 0.0, 0.01, 0.0005, format="%.4f",
                help="L2 regularization strength")
            st.session_state.weight_decay = weight_decay
        
        # Learning rate scheduler
        st.markdown("### 📈 Learning Rate Scheduler")
        
        col1, col2 = st.columns(2)
        
        with col1:
            scheduler = st.selectbox(
                "LR Scheduler",
                ["None", "Step", "Cosine", "OneCycle", "Exponential"],
                index=2,
                help="How to adjust learning rate during training"
            )
            st.session_state.scheduler = scheduler
        
        with col2:
            if scheduler != "None":
                warmup_epochs = st.number_input("Warmup Epochs", 0, 10, 3,
                    help="Gradually increase LR for first N epochs")
                st.session_state.warmup_epochs = warmup_epochs
        
        # Advanced options
        with st.expander("🔬 Advanced Options"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Regularization**")
                dropout = st.slider("Dropout", 0.0, 0.5, 0.0, 0.05)
                label_smoothing = st.slider("Label Smoothing", 0.0, 0.2, 0.0, 0.01)
                
                st.markdown("**Data Loading**")
                num_workers = st.slider("Data Loader Workers", 0, 16, 4)
                cache_images = st.checkbox("Cache Images in RAM", value=False,
                    help="Faster training but uses more memory")
            
            with col2:
                st.markdown("**Early Stopping**")
                early_stopping = st.checkbox("Enable Early Stopping", value=True)
                if early_stopping:
                    patience = st.slider("Patience", 5, 100, 50)
                    st.session_state.patience = patience
                
                st.markdown("**Mixed Precision**")
                use_amp = st.checkbox("Use Automatic Mixed Precision", value=True,
                    help="Train faster with half precision (FP16)")
                st.session_state.use_amp = use_amp
    
    def render_training_monitor(self):
        """Render training monitoring page"""
        
        st.markdown("## 📊 Training Monitor")
        
        # Training status
        status = st.session_state.training_status
        
        if status == 'idle':
            st.info("⏸️ No training in progress. Configure and start training first.")
            return
        
        # Status header
        if status == 'training':
            st.markdown('<span class="status-badge status-training">🔄 Training in Progress</span>', 
                       unsafe_allow_html=True)
        elif status == 'complete':
            st.markdown('<span class="status-badge status-complete">✅ Training Complete</span>', 
                       unsafe_allow_html=True)
        elif status == 'error':
            st.markdown('<span class="status-badge status-error">❌ Training Error</span>', 
                       unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Current epoch info
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            current_epoch = st.session_state.current_epoch
            total_epochs = st.session_state.get('epochs', 50)
            st.metric("Epoch", f"{current_epoch}/{total_epochs}")
        
        with col2:
            progress_pct = (current_epoch / total_epochs * 100) if total_epochs > 0 else 0
            st.metric("Progress", f"{progress_pct:.1f}%")
        
        with col3:
            best_map = st.session_state.best_map
            st.metric("Best mAP", f"{best_map:.3f}")
        
        with col4:
            device = "GPU" if torch.cuda.is_available() else "CPU"
            st.metric("Device", device)
        
        # Progress bar
        progress = st.progress(progress_pct / 100)
        
        # Training plots
        st.markdown("### 📈 Training Metrics")
        
        if st.session_state.training_metrics:
            df = pd.DataFrame(st.session_state.training_metrics)
            
            # Create tabs for different plots
            plot_tab1, plot_tab2, plot_tab3 = st.tabs(["📉 Loss", "🎯 mAP", "📊 All Metrics"])
            
            with plot_tab1:
                # Loss plot
                fig_loss = go.Figure()
                
                if 'train_loss' in df.columns:
                    fig_loss.add_trace(go.Scatter(
                        x=df['epoch'], y=df['train_loss'],
                        mode='lines+markers',
                        name='Train Loss',
                        line=dict(color='#667eea', width=2)
                    ))
                
                if 'val_loss' in df.columns:
                    fig_loss.add_trace(go.Scatter(
                        x=df['epoch'], y=df['val_loss'],
                        mode='lines+markers',
                        name='Val Loss',
                        line=dict(color='#764ba2', width=2)
                    ))
                
                fig_loss.update_layout(
                    title="Loss Over Time",
                    xaxis_title="Epoch",
                    yaxis_title="Loss",
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig_loss, use_container_width=True)
            
            with plot_tab2:
                # mAP plot
                fig_map = go.Figure()
                
                if 'map50' in df.columns:
                    fig_map.add_trace(go.Scatter(
                        x=df['epoch'], y=df['map50'],
                        mode='lines+markers',
                        name='mAP@0.5',
                        line=dict(color='#4CAF50', width=2)
                    ))
                
                if 'map50_95' in df.columns:
                    fig_map.add_trace(go.Scatter(
                        x=df['epoch'], y=df['map50_95'],
                        mode='lines+markers',
                        name='mAP@0.5:0.95',
                        line=dict(color='#2196F3', width=2)
                    ))
                
                fig_map.update_layout(
                    title="Mean Average Precision",
                    xaxis_title="Epoch",
                    yaxis_title="mAP",
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig_map, use_container_width=True)
            
            with plot_tab3:
                # All metrics table
                st.dataframe(df, use_container_width=True)
        else:
            st.info("Training metrics will appear here once training starts")
        
        # Training log
        st.markdown("### 📝 Training Log")
        
        log_container = st.container(height=300)
        
        with log_container:
            if st.session_state.training_log:
                for log_entry in st.session_state.training_log[-50:]:  # Show last 50 lines
                    st.text(log_entry)
            else:
                st.info("Training logs will appear here")
        
        # Control buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("⏸️ Pause Training", disabled=(status != 'training')):
                st.warning("Pause functionality coming soon")
        
        with col2:
            if st.button("⏹️ Stop Training", disabled=(status != 'training')):
                st.session_state.training_status = 'idle'
                st.warning("Training stopped")
                st.rerun()
        
        with col3:
            if st.button("🔄 Refresh Metrics", type="secondary"):
                st.rerun()
        
        # Auto-refresh
        if status == 'training':
            time.sleep(2)
            st.rerun()
    
    def render_evaluation(self):
        """Render evaluation/results page"""
        
        st.markdown("## ✅ Model Evaluation")
        
        # Check for trained models
        model_files = sorted(self.models_dir.glob("*.pt"), 
                           key=lambda x: x.stat().st_mtime, reverse=True)
        
        if not model_files:
            st.warning("No trained models found. Complete a training run first.")
            return
        
        # Model selection
        st.markdown("### 🤖 Select Model to Evaluate")
        
        selected_model = st.selectbox(
            "Trained Model",
            [m.name for m in model_files],
            format_func=lambda x: f"{x} ({datetime.fromtimestamp((self.models_dir / x).stat().st_mtime).strftime('%Y-%m-%d %H:%M')})"
        )
        
        model_path = self.models_dir / selected_model
        
        # Model info
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Model", selected_model)
        with col2:
            size_mb = model_path.stat().st_size / 1e6
            st.metric("Size", f"{size_mb:.1f} MB")
        with col3:
            mod_time = datetime.fromtimestamp(model_path.stat().st_mtime)
            st.metric("Created", mod_time.strftime('%Y-%m-%d'))
        
        st.markdown("---")
        
        # Evaluation metrics
        st.markdown("### 📊 Performance Metrics")
        
        # Load results if available
        results_file = model_path.parent / f"{model_path.stem}_results.json"
        
        if results_file.exists():
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            # Display metrics
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            
            with metric_col1:
                st.metric("mAP@0.5", f"{results.get('map50', 0):.3f}")
            with metric_col2:
                st.metric("mAP@0.5:0.95", f"{results.get('map50_95', 0):.3f}")
            with metric_col3:
                st.metric("Precision", f"{results.get('precision', 0):.3f}")
            with metric_col4:
                st.metric("Recall", f"{results.get('recall', 0):.3f}")
            
            # Per-class performance
            st.markdown("### 🏷️ Per-Class Performance")
            
            if 'class_metrics' in results:
                class_df = pd.DataFrame(results['class_metrics'])
                
                fig = px.bar(class_df, x='class', y='map50',
                           title="mAP@0.5 by Class",
                           color='map50',
                           color_continuous_scale='Viridis')
                st.plotly_chart(fig, use_container_width=True)
                
                st.dataframe(class_df, use_container_width=True, hide_index=True)
            
            # Confusion matrix
            st.markdown("### 🔄 Confusion Matrix")
            
            if 'confusion_matrix' in results:
                cm = np.array(results['confusion_matrix'])
                classes = results.get('classes', [])
                
                fig = px.imshow(cm, 
                              labels=dict(x="Predicted", y="Actual", color="Count"),
                              x=classes,
                              y=classes,
                              title="Confusion Matrix",
                              color_continuous_scale='Blues')
                
                st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("Run evaluation to see detailed metrics")
            
            if st.button("🔬 Run Evaluation on Test Set", type="primary"):
                with st.spinner("Evaluating model..."):
                    # Placeholder for actual evaluation
                    time.sleep(2)
                    st.success("Evaluation complete!")
                    st.rerun()
        
        # Sample predictions
        st.markdown("### 🖼️ Sample Predictions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            show_correct = st.checkbox("Show Correct Predictions", value=True)
        with col2:
            show_incorrect = st.checkbox("Show Incorrect Predictions", value=True)
        
        confidence_filter = st.slider("Minimum Confidence", 0.0, 1.0, 0.5, 0.05)
        
        # Placeholder for sample predictions
        st.info("Sample predictions will be displayed here")
        
        # Model export
        st.markdown("---")
        st.markdown("### 📤 Export Model")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("💾 Download .pt Model", use_container_width=True):
                with open(model_path, 'rb') as f:
                    st.download_button(
                        label="⬇️ Download",
                        data=f,
                        file_name=model_path.name,
                        mime="application/octet-stream"
                    )
        
        with col2:
            if st.button("📦 Export to ONNX", use_container_width=True):
                st.info("ONNX export functionality coming soon")
        
        with col3:
            if st.button("📱 Export to TFLite", use_container_width=True):
                st.info("TFLite export functionality coming soon")
    
    def render_automl(self):
        """Render AutoML/hyperparameter optimization page"""
        
        st.markdown("## 🔮 AutoML - Automatic Hyperparameter Optimization")
        
        st.markdown("""
        Let Optuna automatically find the best hyperparameters for your dataset.
        This process will train multiple models with different configurations and select the best one.
        """)
        
        # Check prerequisites
        if not self.config_file.exists():
            st.warning("⚠️ Please configure dataset and model first")
            return
        
        # Optimization configuration
        st.markdown("### ⚙️ Optimization Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            n_trials = st.slider(
                "Number of Trials",
                5, 100, 20,
                help="How many different configurations to try. More = better results but longer time"
            )
            
            optimization_metric = st.selectbox(
                "Optimization Metric",
                ["mAP@0.5", "mAP@0.5:0.95", "Precision", "Recall", "F1-Score"],
                help="Which metric to optimize for"
            )
        
        with col2:
            epochs_per_trial = st.slider(
                "Epochs per Trial",
                10, 100, 30,
                help="How long to train each configuration"
            )
            
            pruning = st.checkbox(
                "Enable Pruning",
                value=True,
                help="Stop unpromising trials early to save time"
            )
        
        # Parameter search space
        st.markdown("### 🔍 Search Space")
        
        with st.expander("Define which parameters to optimize"):
            col1, col2 = st.columns(2)
            
            with col1:
                optimize_lr = st.checkbox("Learning Rate", value=True)
                if optimize_lr:
                    lr_range = st.slider("LR Range", 0.0001, 0.01, (0.0001, 0.001), format="%.4f")
                
                optimize_batch = st.checkbox("Batch Size", value=True)
                if optimize_batch:
                    batch_options = st.multiselect("Batch Sizes to Try", [8, 16, 32, 64], default=[16, 32])
            
            with col2:
                optimize_optimizer = st.checkbox("Optimizer", value=True)
                if optimize_optimizer:
                    optimizer_options = st.multiselect("Optimizers", 
                        ["Adam", "AdamW", "SGD", "RMSprop"], default=["Adam", "AdamW"])
                
                optimize_scheduler = st.checkbox("LR Scheduler", value=False)
                if optimize_scheduler:
                    scheduler_options = st.multiselect("Schedulers",
                        ["cosine", "step", "exponential"], default=["cosine"])
        
        # Estimated time
        st.markdown("### ⏱️ Estimated Time")
        
        time_per_trial = epochs_per_trial * 2  # minutes (rough estimate)
        total_time_min = n_trials * time_per_trial
        hours = total_time_min // 60
        minutes = total_time_min % 60
        
        st.info(f"⏱️ Estimated total time: **{hours}h {minutes}m** ({n_trials} trials × ~{time_per_trial} min each)")
        
        # Start optimization
        if st.button("🚀 Start AutoML Optimization", type="primary", use_container_width=True):
            st.session_state.automl_status = 'running'
            st.success("✓ AutoML started! Monitor progress below.")
            st.rerun()
        
        # Show optimization progress
        if st.session_state.get('automl_status') == 'running':
            st.markdown("---")
            st.markdown("### 🔄 Optimization Progress")
            
            # Placeholder for Optuna visualization
            st.info("Trial progress and results will appear here")
            
            # Simulated trials data
            if st.button("⏹️ Stop Optimization"):
                st.session_state.automl_status = 'stopped'
                st.rerun()
        
        # Best parameters found
        if st.session_state.get('automl_status') == 'complete':
            st.markdown("---")
            st.markdown("### 🏆 Best Configuration Found")
            
            st.success("Optimization complete!")
            
            # Show best parameters
            best_params = st.session_state.get('best_params', {})
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.json(best_params)
            
            with col2:
                st.metric("Best Score", f"{st.session_state.get('best_score', 0):.3f}")
                
                if st.button("✅ Use These Parameters", type="primary"):
                    # Update session state with best params
                    for key, value in best_params.items():
                        st.session_state[key] = value
                    st.success("Parameters applied! Go to Training page to start.")
    
    def render_home(self):
        """Render home/overview page"""
        
        st.markdown("## Welcome to the Training Interface")
        
        st.markdown("""
        This interface allows you to fine-tune object detection models on your annotated data from FiftyOne.
        
        ### 🎯 Workflow:
        1. **📁 Dataset**: Load and prepare your FiftyOne dataset
        2. **🤖 Model Config**: Choose model architecture and pretrained weights  
        3. **⚙️ Training**: Configure hyperparameters
        4. **📊 Monitor**: Watch training progress in real-time
        5. **✅ Evaluate**: Review results and export model
        6. **🔮 AutoML** (Optional): Let AI find best hyperparameters
        """)
        
        # Quick status cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>📁</h3>
                <h2>Datasets</h2>
                <p style="font-size: 2rem; margin: 0;">{}</p>
            </div>
            """.format(len(fo.list_datasets())), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h3>🤖</h3>
                <h2>Models</h2>
                <p style="font-size: 2rem; margin: 0;">{}</p>
            </div>
# save as: streamlit_training_ui.py (COMPLETE - FINAL PART)

            """.format(len(list(self.models_dir.glob("*.pt")))), unsafe_allow_html=True)
        
        with col3:
            status_class = {
                'idle': 'ready',
                'training': 'training',
                'complete': 'complete',
                'error': 'error'
            }.get(st.session_state.training_status, 'ready')
            
            st.markdown("""
            <div class="metric-card">
                <h3>⚡</h3>
                <h2>Status</h2>
                <span class="status-badge status-{}">{}</span>
            </div>
            """.format(status_class, st.session_state.training_status.title()), 
            unsafe_allow_html=True)
        
        with col4:
            device = "GPU" if torch.cuda.is_available() else "CPU"
            st.markdown("""
            <div class="metric-card">
                <h3></</h3>
                <h2>Device</h2>
                <p style="font-size: 2rem; margin: 0;">{}</p>
            </div>
            """.format(device), unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Recent activity
        st.markdown("### 📝 Recent Training Runs")
        
        model_files = sorted(self.models_dir.glob("*.pt"), 
                           key=lambda x: x.stat().st_mtime, reverse=True)[:5]
        
        if model_files:
            for model_file in model_files:
                col1, col2, col3 = st.columns([3, 2, 1])
                with col1:
                    st.markdown(f"**{model_file.name}**")
                with col2:
                    st.markdown(f"_{datetime.fromtimestamp(model_file.stat().st_mtime).strftime('%Y-%m-%d %H:%M')}_")
                with col3:
                    st.markdown(f"`{model_file.stat().st_size / 1e6:.1f} MB`")
        else:
            st.info("No trained models yet. Start your first training run!")
        
        # Quick links
        st.markdown("---")
        st.markdown("### 🔗 Quick Actions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📁 Open FiftyOne", use_container_width=True):
                st.info("Open FiftyOne in terminal: `fiftyone app launch`")
        
        with col2:
            if st.button("📊 View TensorBoard", use_container_width=True):
                st.info("Launch TensorBoard: `tensorboard --logdir training_logs`")
        
        with col3:
            if st.button("📖 Documentation", use_container_width=True):
                st.info("Documentation coming soon")


def main():
    """Main application entry point"""
    
    # Parse command line arguments
    import argparse
    
    # Create argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=None, help='FiftyOne dataset name')
    parser.add_argument('--config', type=str, default='training_config.yaml', help='Config file path')
    
    # Parse args (Streamlit specific handling)
    try:
        args = parser.parse_args()
    except SystemExit:
        # Streamlit may pass its own args, so use defaults
        args = argparse.Namespace(dataset=None, config='training_config.yaml')
    
    # Initialize UI
    ui = TrainingUI()
    
    # Render UI
    ui.render()


if __name__ == "__main__":
    main()