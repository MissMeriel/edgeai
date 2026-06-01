# save as: visualize_scenes.py

"""
Visualize scene classification statistics

Usage:
    python visualize_scenes.py /path/to/classified/folder
    python visualize_scenes.py classification_report.json
"""

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def visualize_from_directory(classified_dir: str):
    """Visualize scene distribution from classified directory"""
    
    classified_path = Path(classified_dir)
    
    # Count images in each subdirectory
    scene_counts = {}
    
    for subdir in classified_path.iterdir():
        if not subdir.is_dir():
            continue
        
        # Count images
        extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        count = 0
        for ext in extensions:
            count += len(list(subdir.glob(f"*{ext}")))
            count += len(list(subdir.glob(f"*{ext.upper()}")))
        
        if count > 0:
            scene_counts[subdir.name] = count
    
    if not scene_counts:
        print("No classified images found")
        return
    
    # Create visualizations
    create_visualizations(scene_counts, classified_dir)


def visualize_from_report(report_file: str):
    """Visualize from JSON report"""
    
    with open(report_file) as f:
        data = json.load(f)
    
    # Extract scene counts
    scene_counts = {}
    
    # Handle different report formats
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, list) and key != 'unclassified' and key != 'errors':
                scene_counts[key] = len(value)
    
    create_visualizations(scene_counts, Path(report_file).parent)


def create_visualizations(scene_counts: dict, output_dir: Path):
    """Create and save visualization plots"""
    
    output_dir = Path(output_dir)
    
    # Sort by count
    sorted_scenes = sorted(scene_counts.items(), key=lambda x: x[1], reverse=True)
    scenes = [s[0] for s in sorted_scenes]
    counts = [s[1] for s in sorted_scenes]
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Bar chart
    ax1 = plt.subplot(2, 2, 1)
    bars = ax1.bar(range(len(scenes)), counts, color='steelblue', alpha=0.8)
    ax1.set_xticks(range(len(scenes)))
    ax1.set_xticklabels(scenes, rotation=45, ha='right')
    ax1.set_ylabel('Number of Images')
    ax1.set_title('Scene Distribution - Bar Chart')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=9)
    
    # 2. Pie chart
    ax2 = plt.subplot(2, 2, 2)
    colors = plt.cm.Set3(np.linspace(0, 1, len(scenes)))
    wedges, texts, autotexts = ax2.pie(counts, labels=scenes, autopct='%1.1f%%',
                                        colors=colors, startangle=90)
    ax2.set_title('Scene Distribution - Pie Chart')
    
    # 3. Horizontal bar chart
    ax3 = plt.subplot(2, 2, 3)
    y_pos = np.arange(len(scenes))
    ax3.barh(y_pos, counts, color='coral', alpha=0.8)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(scenes)
    ax3.invert_yaxis()
    ax3.set_xlabel('Number of Images')
    ax3.set_title('Scene Distribution - Horizontal')
    ax3.grid(axis='x', alpha=0.3)
    
    # 4. Statistics text
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('off')
    
    total = sum(counts)
    stats_text = f"Scene Classification Statistics\n\n"
    stats_text += f"Total Images: {total}\n"
    stats_text += f"Scene Types: {len(scenes)}\n\n"
    stats_text += "Distribution:\n"
    stats_text += "-" * 40 + "\n"
    
    for scene, count in sorted_scenes[:10]:  # Top 10
        pct = count / total * 100
        stats_text += f"{scene:20s}: {count:5d} ({pct:5.1f}%)\n"
    
    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    
    # Save
    output_file = output_dir / f"scene_visualization_{Path(output_dir).name}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✅ Visualization saved: {output_file}")
    
    # Show
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize scene classification results"
    )
    parser.add_argument('input', help='Classified directory or JSON report file')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if not input_path.exists():
        print(f"❌ Not found: {args.input}")
        return
    
    if input_path.is_dir():
        visualize_from_directory(args.input)
    elif input_path.suffix == '.json':
        visualize_from_report(args.input)
    else:
        print("❌ Input must be a directory or JSON file")


if __name__ == "__main__":
    main()