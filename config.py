# save as: config.py

class Config:
    """Configuration for vehicle classification pipeline"""
    
    # Dataset settings
    DATASET_NAME = "vehicle_classification"
    DATA_DIR = "/path/to/vehicle/dataset"  # Change this!
    
    # Class names (modify based on your dataset)
    CLASSES = ["humvee", "ruggedized_vehicle"]
    
    # Split ratios
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    
    # Training hyperparameters
    BATCH_SIZE = 32
    NUM_EPOCHS = 15
    LEARNING_RATE = 0.001
    
    # Advanced fine-tuning
    ADVANCED_EPOCHS = 5
    ADVANCED_LR = 0.0001
    
    # Model settings
    MODEL_NAME = "vgg16"
    FREEZE_FEATURES = True  # Initially freeze backbone
    PRETRAINED = True  # Use ImageNet weights
    
    # Image preprocessing
    IMAGE_SIZE = 224
    NORMALIZE_MEAN = [0.485, 0.456, 0.406]
    NORMALIZE_STD = [0.229, 0.224, 0.225]
    
    # Paths
    BEST_MODEL_PATH = "models/best_vgg_vehicle.pth"
    ADVANCED_MODEL_PATH = "models/best_vgg_vehicle_advanced.pth"
    
    # FiftyOne settings
    PREDICTION_FIELD = "predictions"
    EVAL_KEY = "eval"