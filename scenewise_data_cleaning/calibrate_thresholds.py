# save as: calibrate_thresholds.py

"""
Calibration tool to find optimal thresholds for scene detection

Usage:
    python calibrate_thresholds.py /path/to/labeled/images --scene mountain
    python calibrate_thresholds.py /path/to/images --scene park --interactive
"""

import argparse
from pathlib import Path
from scene_classifier_fixed import ImprovedSceneClassifier, SCENE_DEFINITIONS
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def calibrate_scene_threshold(images_dir: str, scene_type: str, 
                              positive_subdir: str = None,
                              negative_subdir: str = None,
                              method: str = 'hybrid'):
    """
    Calibrate threshold for a scene type
    
    Directory structure:
        images_dir/
            positive/  (or specify with --positive)
                image1.jpg  # Images that ARE this scene
                image2.jpg
            negative/  (or specify with --negative)
                image1.jpg  # Images that are NOT this scene
                image2.jpg
    """
    
    images_path = Path(images_dir)
    
    # Find positive and negative samples
    if positive_subdir:
        pos_dir = images_path / positive_subdir
    else:
        pos_dir = images_path / "positive"
    
    if negative_subdir:
        neg_dir = images_path / negative_subdir
    else:
        neg_dir = images_path / "negative"
    
    if not pos_dir.exists() or not neg_dir.exists():
        print(f"❌ Expected directory structure:")
        print(f"   {images_dir}/")
        print(f"     positive/  (images that ARE {scene_type})")
        print(f"     negative/  (images that are NOT {scene_type})")
        return
    
    # Load images
    extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    pos_images = []
    neg_images = []
    
    for ext in extensions:
        pos_images.extend(pos_dir.glob(f"*{ext}"))
        neg_images.extend(neg_dir.glob(f"*{ext}"))
    
    if not pos_images or not neg_images:
        print(f"❌ No images found")
        print(f"   Positive: {len(pos_images)}")
        print(f"   Negative: {len(neg_images)}")
        return
    
    print(f"\n{'='*70}")
    print(f"CALIBRATING THRESHOLD FOR: {scene_type}")
    print(f"{'='*70}")
    print(f"Positive samples: {len(pos_images)}")
    print(f"Negative samples: {len(neg_images)}")
    print(f"Method: {method}")
    print(f"{'='*70}\n")
    
    # Create classifier
    classifier = ImprovedSceneClassifier(
        scenes=[scene_type],
        method=method,
        debug=False
    )
    
    # Get scores for all images
    pos_scores = []
    neg_scores = []
    
    print("Scoring positive samples...")
    for img_path in tqdm(pos_images):
        results = classifier.classify_image(str(img_path))
        _, score = results[scene_type]
        pos_scores.append(score)
    
    print("Scoring negative samples...")
    for img_path in tqdm(neg_images):
        results = classifier.classify_image(str(img_path))
        _, score = results[scene_type]
        neg_scores.append(score)
    
    # Calculate statistics
    pos_scores = np.array(pos_scores)
    neg_scores = np.array(neg_scores)
    
    print(f"\n{'='*70}")
    print("SCORE STATISTICS")
    print(f"{'='*70}")
    print(f"\nPositive samples ({scene_type}):")
    print(f"  Mean: {np.mean(pos_scores):.3f}")
    print(f"  Std:  {np.std(pos_scores):.3f}")
    print(f"  Min:  {np.min(pos_scores):.3f}")
    print(f"  Max:  {np.max(pos_scores):.3f}")
    
    print(f"\nNegative samples (not {scene_type}):")
    print(f"  Mean: {np.mean(neg_scores):.3f}")
    print(f"  Std:  {np.std(neg_scores):.3f}")
    print(f"  Min:  {np.min(neg_scores):.3f}")
    print(f"  Max:  {np.max(neg_scores):.3f}")
    
    # Find optimal threshold
    thresholds = np.linspace(0, 1, 100)
    best_threshold = 0.5
    best_accuracy = 0
    
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    
    for threshold in thresholds:
        # True positives, false positives, etc.
        tp = np.sum(pos_scores >= threshold)
        fn = np.sum(pos_scores < threshold)
        tn = np.sum(neg_scores < threshold)
        fp = np.sum(neg_scores >= threshold)
        
        accuracy = (tp + tn) / (len(pos_scores) + len(neg_scores))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold
    
    # Find threshold with best F1 score
    best_f1_idx = np.argmax(f1_scores)
    best_f1_threshold = thresholds[best_f1_idx]
    best_f1 = f1_scores[best_f1_idx]
    
    print(f"\n{'='*70}")
    print("RECOMMENDED THRESHOLDS")
    print(f"{'='*70}")
    print(f"\nBest Accuracy: {best_threshold:.3f} (accuracy: {best_accuracy:.1%})")
    print(f"Best F1 Score: {best_f1_threshold:.3f} (F1: {best_f1:.1%})")
    print(f"Current in code: {SCENE_DEFINITIONS[scene_type].get('threshold', 0.5):.3f}")
    
    # Conservative threshold (minimize false positives)
    conservative_idx = np.where(np.array(precisions) > 0.9)[0]
    if len(conservative_idx) > 0:
        conservative_threshold = thresholds[conservative_idx[0]]
        print(f"Conservative (90%+ precision): {conservative_threshold:.3f}")
    
    # Aggressive threshold (minimize false negatives)
    aggressive_idx = np.where(np.array(recalls) > 0.9)[0]
    if len(aggressive_idx) > 0:
        aggressive_threshold = thresholds[aggressive_idx[-1]]
        print(f"Aggressive (90%+ recall): {aggressive_threshold:.3f}")
    
    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Score distributions
    ax = axes[0, 0]
    ax.hist(pos_scores, bins=30, alpha=0.5, label='Positive', color='green')
    ax.hist(neg_scores, bins=30, alpha=0.5, label='Negative', color='red')
    ax.axvline(best_f1_threshold, color='blue', linestyle='--', label=f'Best F1 ({best_f1_threshold:.3f})')
    ax.axvline(SCENE_DEFINITIONS[scene_type].get('threshold', 0.5), 
               color='orange', linestyle='--', label='Current')
    ax.set_xlabel('Score')
    ax.set_ylabel('Count')
    ax.set_title(f'Score Distribution - {scene_type}')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 2. Accuracy vs Threshold
    ax = axes[0, 1]
    ax.plot(thresholds, accuracies, label='Accuracy', linewidth=2)
    ax.plot(thresholds, f1_scores, label='F1 Score', linewidth=2)
    ax.axvline(best_f1_threshold, color='blue', linestyle='--', alpha=0.5)
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Score')
    ax.set_title('Accuracy and F1 vs Threshold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 3. Precision vs Recall
    ax = axes[1, 0]
    ax.plot(thresholds, precisions, label='Precision', linewidth=2)
    ax.plot(thresholds, recalls, label='Recall', linewidth=2)
    ax.axvline(best_f1_threshold, color='blue', linestyle='--', alpha=0.5)
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Score')
    ax.set_title('Precision and Recall vs Threshold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 4. PR Curve
    ax = axes[1, 1]
    ax.plot(recalls, precisions, linewidth=2)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_file = f"calibration_{scene_type}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n📊 Visualization saved: {output_file}")
    
    plt.show()
    
    # Show confusion matrix at recommended threshold
    print(f"\n{'='*70}")
    print(f"CONFUSION MATRIX AT THRESHOLD {best_f1_threshold:.3f}")
    print(f"{'='*70}")
    
    tp = np.sum(pos_scores >= best_f1_threshold)
    fn = np.sum(pos_scores < best_f1_threshold)
    tn = np.sum(neg_scores < best_f1_threshold)
    fp = np.sum(neg_scores >= best_f1_threshold)
    
    print(f"\n                Predicted {scene_type}    Predicted NOT {scene_type}")
    print(f"Actual {scene_type:15s}        {tp:4d}                {fn:4d}")
    print(f"Actual NOT {scene_type:10s}        {fp:4d}                {tn:4d}")
    
    print(f"\nMetrics:")
    print(f"  Accuracy:  {(tp+tn)/(tp+tn+fp+fn):.1%}")
    print(f"  Precision: {tp/(tp+fp) if (tp+fp) > 0 else 0:.1%}")
    print(f"  Recall:    {tp/(tp+fn) if (tp+fn) > 0 else 0:.1%}")
    print(f"  F1 Score:  {best_f1:.1%}")
    
    print(f"\n{'='*70}\n")


def interactive_calibration(images_dir: str, scene_type: str, method: str = 'hybrid'):
    """Interactive calibration by showing images and asking for labels"""
    
    images_path = Path(images_dir)
    
    # Get all images
    extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    images = []
    for ext in extensions:
        images.extend(images_path.glob(f"*{ext}"))
    
    if not images:
        print("No images found")
        return
    
    # Sample random images
    import random
    random.shuffle(images)
    sample_size = min(50, len(images))
    sample_images = images[:sample_size]
    
    print(f"\n{'='*70}")
    print(f"INTERACTIVE CALIBRATION FOR: {scene_type}")
    print(f"{'='*70}")
    print(f"Will show {sample_size} images")
    print(f"For each, indicate if it IS or is NOT a {scene_type}")
    print(f"{'='*70}\n")
    
    # Create classifier
    classifier = ImprovedSceneClassifier(
        scenes=[scene_type],
        method=method,
        debug=False
    )
    
    labeled_data = []
    
    for idx, img_path in enumerate(sample_images, 1):
        # Get score
        results = classifier.classify_image(str(img_path))
        _, score = results[scene_type]
        
        # Show image
        img = Image.open(img_path)
        plt.figure(figsize=(10, 8))
        plt.imshow(img)
        plt.title(f"[{idx}/{sample_size}] {img_path.name}\nCurrent score: {score:.3f}")
        plt.axis('off')
        plt.show(block=False)
        plt.pause(0.1)
        
        # Ask for label
        while True:
            response = input(f"Is this a {scene_type}? (y/n/s=skip/q=quit): ").lower()
            if response in ['y', 'n', 's', 'q']:
                break
        
        plt.close()
        
        if response == 'q':
            break
        elif response == 's':
            continue
        
        is_positive = (response == 'y')
        labeled_data.append((score, is_positive))
    
    if len(labeled_data) < 5:
        print("Not enough labeled data")
        return
    
    # Analyze
    pos_scores = [s for s, is_pos in labeled_data if is_pos]
    neg_scores = [s for s, is_pos in labeled_data if not is_pos]
    
    if not pos_scores or not neg_scores:
        print("Need both positive and negative examples")
        return
    
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"Positive samples: {len(pos_scores)}")
    print(f"Negative samples: {len(neg_scores)}")
    
    pos_mean = np.mean(pos_scores)
    neg_mean = np.mean(neg_scores)
    
    print(f"\nAverage scores:")
    print(f"  Positive: {pos_mean:.3f}")
    print(f"  Negative: {neg_mean:.3f}")
    
    # Recommend threshold
    recommended = (pos_mean + neg_mean) / 2
    print(f"\nRecommended threshold: {recommended:.3f}")
    print(f"Current threshold: {SCENE_DEFINITIONS[scene_type].get('threshold', 0.5):.3f}")
    
    # Simple visualization
    plt.figure(figsize=(10, 6))
    plt.hist(pos_scores, bins=10, alpha=0.5, label='Positive', color='green')
    plt.hist(neg_scores, bins=10, alpha=0.5, label='Negative', color='red')
    plt.axvline(recommended, color='blue', linestyle='--', label=f'Recommended ({recommended:.3f})')
    plt.axvline(SCENE_DEFINITIONS[scene_type].get('threshold', 0.5), 
                color='orange', linestyle='--', label='Current')
    plt.xlabel('Score')
    plt.ylabel('Count')
    plt.title(f'Score Distribution - {scene_type}')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(f"interactive_calibration_{scene_type}.png", dpi=150)
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate scene detection thresholds"
    )
    
    parser.add_argument('images_dir', help='Directory with labeled images')
    parser.add_argument('--scene', required=True, help='Scene type to calibrate')
    parser.add_argument('--method', default='hybrid', choices=['clip', 'cv', 'hybrid'])
    parser.add_argument('--positive', help='Subdirectory with positive examples')
    parser.add_argument('--negative', help='Subdirectory with negative examples')
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='Interactive calibration mode')
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_calibration(args.images_dir, args.scene, args.method)
    else:
        calibrate_scene_threshold(
            args.images_dir, 
            args.scene,
            args.positive,
            args.negative,
            args.method
        )


if __name__ == "__main__":
    main()