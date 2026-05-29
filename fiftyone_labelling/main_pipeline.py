import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.brain as fob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import os

# ============================================================================
# 1. SETUP AND DATASET PREPARATION
# ============================================================================

class FiftyOneClassificationDataset(Dataset):
    """Custom Dataset wrapper for FiftyOne samples"""
    
    def __init__(self, fo_dataset, transform=None):
        self.samples = list(fo_dataset)
        self.transform = transform
        self.classes = fo_dataset.distinct("ground_truth.label")
        self.class_to_idx = {cls: idx for idx, cls in enumerate(sorted(self.classes))}
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample.filepath).convert('RGB')
        label = sample.ground_truth.label
        label_idx = self.class_to_idx[label]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label_idx

# ============================================================================
# 2. MODEL PREPARATION
# ============================================================================

def prepare_vgg_model(num_classes, freeze_features=True, pretrained=True):
    """
    Prepare VGG16 model for fine-tuning
    
    Args:
        num_classes: Number of classes in target dataset
        freeze_features: Whether to freeze feature extraction layers
        pretrained: Load ImageNet pre-trained weights
    """
    model = models.vgg16(pretrained=pretrained)
    
    # Freeze feature extraction layers if specified
    if freeze_features:
        for param in model.features.parameters():
            param.requires_grad = False
    
    # Replace the classifier for our custom classes
    num_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_features, num_classes)
    
    return model

# ============================================================================
# 3. TRAINING FUNCTIONS
# ============================================================================

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': running_loss/total, 'acc': 100.*correct/total})
    
    return running_loss / len(dataloader), 100. * correct / total

def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc='Validation'):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return running_loss / len(dataloader), 100. * correct / total

def add_predictions_to_fiftyone(model, fo_dataset, class_names, transform, device, field_name="predictions"):
    """Add model predictions to FiftyOne dataset"""
    model.eval()
    
    with fo.ProgressBar() as pb:
        for sample in pb(fo_dataset):
            image = Image.open(sample.filepath).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = model(image_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = probs.max(1)
            
            # Create classification object
            label = class_names[predicted.item()]
            confidence_val = confidence.item()
            
            sample[field_name] = fo.Classification(
                label=label,
                confidence=confidence_val
            )
            sample.save()

# ============================================================================
# 4. MAIN PIPELINE
# ============================================================================

def main():
    # Configuration
    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    LEARNING_RATE = 0.001
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {DEVICE}")
    
    # ========================================================================
    # STEP 1: Load or Create Vehicle Dataset
    # ========================================================================
    
    print("\n" + "="*70)
    print("STEP 1: Preparing Vehicle Dataset")
    print("="*70)
    
    # Option A: Create a custom dataset (you'll need to provide your own data)
    # For this example, I'll show the structure
    
    dataset_name = "vehicle_classification"
    
    # Check if dataset already exists
    if dataset_name in fo.list_datasets():
        print(f"Loading existing dataset: {dataset_name}")
        dataset = fo.load_dataset(dataset_name)
    else:
        print(f"Creating new dataset: {dataset_name}")
        dataset = fo.Dataset(dataset_name)
        dataset.persistent = True
        
        # Add your samples here
        # Example structure for adding samples:
        """
        samples = []
        for image_path in your_image_paths:
            sample = fo.Sample(
                filepath=image_path,
                ground_truth=fo.Classification(label=label)  # "humvee" or "ruggedized_vehicle"
            )
            samples.append(sample)
        dataset.add_samples(samples)
        """
    
    # Option B: Use a sample dataset for demonstration
    # Let's create a synthetic example with two classes
    if len(dataset) == 0:
        print("NOTE: Creating sample dataset structure.")
        print("You need to replace this with your actual vehicle images!")
        
        # This is just to show the structure - replace with real data
        print("\nTo use this pipeline with real data:")
        print("1. Organize images in folders: /path/to/data/humvee/ and /path/to/data/ruggedized_vehicle/")
        print("2. Use the code below to load them:\n")
        
        print("""
# Example code to load your data:
dataset = fo.Dataset("vehicle_classification")

# Load from directory structure
dataset_dir = "/path/to/vehicle/dataset"
for class_name in ["humvee", "ruggedized_vehicle"]:
    class_dir = os.path.join(dataset_dir, class_name)
    for img_file in os.listdir(class_dir):
        if img_file.endswith(('.jpg', '.png', '.jpeg')):
            sample = fo.Sample(
                filepath=os.path.join(class_dir, img_file),
                ground_truth=fo.Classification(label=class_name)
            )
            dataset.add_sample(sample)

dataset.persistent = True
        """)
        
        return
    
    # Split dataset into train/val/test
    print(f"\nDataset loaded: {len(dataset)} samples")
    print(f"Classes: {dataset.distinct('ground_truth.label')}")
    
    # Create train/val/test splits if not already done
    if "split" not in dataset.get_field_schema():
        print("\nCreating train/val/test splits...")
        
        # Shuffle and split
        np.random.seed(42)
        indices = np.random.permutation(len(dataset))
        train_size = int(0.7 * len(dataset))
        val_size = int(0.15 * len(dataset))
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        # Add split tags
        for idx in train_indices:
            dataset[idx]["split"] = "train"
            dataset[idx].save()
        for idx in val_indices:
            dataset[idx]["split"] = "val"
            dataset[idx].save()
        for idx in test_indices:
            dataset[idx]["split"] = "test"
            dataset[idx].save()
    
    # Create views for each split
    train_view = dataset.match_tags("train") if "train" in dataset.values("split") else dataset.match(fo.ViewField("split") == "train")
    val_view = dataset.match_tags("val") if "val" in dataset.values("split") else dataset.match(fo.ViewField("split") == "val")
    test_view = dataset.match_tags("test") if "test" in dataset.values("split") else dataset.match(fo.ViewField("split") == "test")
    
    print(f"Train samples: {len(train_view)}")
    print(f"Validation samples: {len(val_view)}")
    print(f"Test samples: {len(test_view)}")
    
    # ========================================================================
    # STEP 2: Prepare Data Transforms and Loaders
    # ========================================================================
    
    print("\n" + "="*70)
    print("STEP 2: Preparing Data Loaders")
    print("="*70)
    
    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = FiftyOneClassificationDataset(train_view, transform=train_transform)
    val_dataset = FiftyOneClassificationDataset(val_view, transform=val_transform)
    test_dataset = FiftyOneClassificationDataset(test_view, transform=val_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                             shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, 
                           shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, 
                            shuffle=False, num_workers=4, pin_memory=True)
    
    # Get class names
    class_names = sorted(dataset.distinct("ground_truth.label"))
    num_classes = len(class_names)
    print(f"\nNumber of classes: {num_classes}")
    print(f"Classes: {class_names}")
    
    # ========================================================================
    # STEP 3: Initialize Model (Pre-trained on ImageNet)
    # ========================================================================
    
    print("\n" + "="*70)
    print("STEP 3: Initializing VGG16 Model (Pre-trained on ImageNet)")
    print("="*70)
    
    model = prepare_vgg_model(
        num_classes=num_classes,
        freeze_features=True,  # Freeze feature layers, only train classifier
        pretrained=True  # Use ImageNet weights
    )
    model = model.to(DEVICE)
    
    print(f"\nModel architecture:")
    print(f"Feature extraction layers: FROZEN (pre-trained on ImageNet)")
    print(f"Classifier layers: TRAINABLE (fine-tuning for {num_classes} classes)")
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    # ========================================================================
    # STEP 4: Fine-tune the Model
    # ========================================================================
    
    print("\n" + "="*70)
    print("STEP 4: Fine-tuning Model on Vehicle Dataset")
    print("="*70)
    
    best_val_acc = 0.0
    best_model_path = "best_vgg_vehicle_classifier.pth"
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print("-" * 50)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, DEVICE
        )
        
        # Validate
        val_loss, val_acc = validate(
            model, val_loader, criterion, DEVICE
        )
        
        # Update scheduler
        scheduler.step()
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"✓ Saved best model (Val Acc: {val_acc:.2f}%)")
    
    # Load best model
    print(f"\nLoading best model (Val Acc: {best_val_acc:.2f}%)")
    model.load_state_dict(torch.load(best_model_path))
    
    # ========================================================================
    # STEP 5: Evaluate on Test Set
    # ========================================================================
    
    print("\n" + "="*70)
    print("STEP 5: Evaluating on Test Set")
    print("="*70)
    
    test_loss, test_acc = validate(model, test_loader, criterion, DEVICE)
    print(f"\nTest Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
    
    # ========================================================================
    # STEP 6: Add Predictions to FiftyOne Dataset
    # ========================================================================
    
    print("\n" + "="*70)
    print("STEP 6: Adding Predictions to FiftyOne Dataset")
    print("="*70)
    
    # Add predictions to all samples
    print("\nAdding predictions to dataset...")
    add_predictions_to_fiftyone(
        model, dataset, class_names, val_transform, DEVICE, 
        field_name="predictions"
    )
    
    # ========================================================================
    # STEP 7: Evaluate with FiftyOne
    # ========================================================================
    
    print("\n" + "="*70)
    print("STEP 7: Evaluating with FiftyOne Metrics")
    print("="*70)
    
    # Evaluate predictions
    results = dataset.evaluate_classifications(
        "predictions",
        gt_field="ground_truth",
        eval_key="eval",
        classes=class_names
    )
    
    # Print evaluation metrics
    print("\nClassification Report:")
    results.print_report()
    
    # Get confusion matrix
    print("\nGenerating confusion matrix...")
    plot = results.plot_confusion_matrix()
    plot.show()
    
    # ========================================================================
    # STEP 8: Analyze Results with FiftyOne
    # ========================================================================
    
    print("\n" + "="*70)
    print("STEP 8: Analyzing Results")
    print("="*70)
    
    # Find misclassified samples
    misclassified_view = dataset.match(
        fo.ViewField("predictions.label") != fo.ViewField("ground_truth.label")
    )
    print(f"\nMisclassified samples: {len(misclassified_view)}")
    
    # Find low confidence predictions
    low_confidence_view = dataset.match(
        fo.ViewField("predictions.confidence") < 0.7
    )
    print(f"Low confidence predictions (<0.7): {len(low_confidence_view)}")
    
    # Compute uniqueness (find edge cases)
    print("\nComputing sample uniqueness...")
    fob.compute_uniqueness(dataset)
    
    # ========================================================================
    # STEP 9: Launch FiftyOne App for Visualization
    # ========================================================================
    
    print("\n" + "="*70)
    print("STEP 9: Launching FiftyOne App")
    print("="*70)
    
    session = fo.launch_app(dataset)
    
    print("\nFiftyOne App launched! You can:")
    print("- View all samples and predictions")
    print("- Filter by misclassified samples")
    print("- Sort by confidence score")
    print("- Analyze confusion matrix")
    print("- Find edge cases using uniqueness scores")
    
    # Create useful views
    print("\n" + "="*70)
    print("Useful Views in FiftyOne App:")
    print("="*70)
    
    # Save views for easy access
    dataset.save_view("misclassified", misclassified_view)
    dataset.save_view("low_confidence", low_confidence_view)
    
    # View hardest examples
    hardest_view = dataset.sort_by("uniqueness", reverse=True).limit(50)
    dataset.save_view("hardest_samples", hardest_view)
    
    print("\nSaved views:")
    print("1. 'misclassified' - Incorrectly classified samples")
    print("2. 'low_confidence' - Low confidence predictions")
    print("3. 'hardest_samples' - Most unique/difficult samples")
    
    return model, dataset, history, session

# ============================================================================
# 5. HELPER FUNCTION: CREATE SAMPLE DATASET
# ============================================================================

def create_sample_vehicle_dataset(dataset_dir, output_name="vehicle_classification"):
    """
    Helper function to create a FiftyOne dataset from directory structure
    
    Expected directory structure:
    dataset_dir/
        humvee/
            image1.jpg
            image2.jpg
            ...
        ruggedized_vehicle/
            image1.jpg
            image2.jpg
            ...
    """
    
    if output_name in fo.list_datasets():
        print(f"Dataset '{output_name}' already exists. Loading...")
        return fo.load_dataset(output_name)
    
    dataset = fo.Dataset(output_name)
    dataset.persistent = True
    
    samples = []
    
    for class_name in os.listdir(dataset_dir):
        class_path = os.path.join(dataset_dir, class_name)
        
        if not os.path.isdir(class_path):
            continue
        
        print(f"Loading class: {class_name}")
        
        for img_file in os.listdir(class_path):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                filepath = os.path.join(class_path, img_file)
                
                sample = fo.Sample(
                    filepath=filepath,
                    ground_truth=fo.Classification(label=class_name)
                )
                samples.append(sample)
    
    dataset.add_samples(samples)
    print(f"\nCreated dataset with {len(dataset)} samples")
    print(f"Classes: {dataset.distinct('ground_truth.label')}")
    
    return dataset

# ============================================================================
# 6. ADVANCED: UNFREEZE AND FINE-TUNE MORE LAYERS
# ============================================================================

def advanced_fine_tuning(model, train_loader, val_loader, device, 
                        num_epochs=5, learning_rate=0.0001):
    """
    Advanced fine-tuning: Unfreeze some feature layers for better adaptation
    """
    
    print("\n" + "="*70)
    print("ADVANCED FINE-TUNING: Unfreezing Feature Layers")
    print("="*70)
    
    # Unfreeze last few convolutional blocks
    for param in model.features[20:].parameters():  # Unfreeze last layers
        param.requires_grad = True
    
    # Use lower learning rate for feature layers
    optimizer = optim.Adam([
        {'params': model.features[20:].parameters(), 'lr': learning_rate * 0.1},
        {'params': model.classifier.parameters(), 'lr': learning_rate}
    ])
    
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f"\nAdvanced Fine-tuning Epoch {epoch+1}/{num_epochs}")
        print("-" * 50)
        
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(
            model, val_loader, criterion, device
        )
        
        scheduler.step()
        
        print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_vgg_vehicle_advanced.pth")
    
    return model

# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    
    # Example 1: Create dataset from directory
    print("="*70)
    print("VGG FINE-TUNING PIPELINE WITH FIFTYONE")
    print("="*70)
    
    # Uncomment and modify this to load your actual data
    """
    dataset = create_sample_vehicle_dataset(
        dataset_dir="/path/to/your/vehicle/images",
        output_name="vehicle_classification"
    )
    """
    
    # Run the main pipeline
    try:
        model, dataset, history, session = main()
        
        # Optional: Advanced fine-tuning
        response = input("\nPerform advanced fine-tuning? (y/n): ")
        if response.lower() == 'y':
            # Recreate dataloaders for advanced training
            # (you would need to extract these from main() or restructure)
            print("Advanced fine-tuning would happen here...")
        
        # Keep session open
        print("\n" + "="*70)
        print("Pipeline complete! FiftyOne session is active.")
        print("Close the browser or press Ctrl+C to exit.")
        print("="*70)
        
        session.wait()
        
    except Exception as e:
        print(f"\nError: {e}")
        print("\nPlease ensure you have a valid dataset loaded.")
        print("See the instructions above for creating a dataset.")