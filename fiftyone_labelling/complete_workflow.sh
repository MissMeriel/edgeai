# save as: complete_workflow.sh

#!/bin/bash

# Complete Auto-Tagging and Annotation Workflow
# Usage: ./complete_workflow.sh /path/to/images

set -e

IMAGES_DIR=$1

if [ -z "$IMAGES_DIR" ]; then
    echo "Usage: $0 /path/to/images"
    exit 1
fi

echo "============================================================================"
echo "COMPLETE AUTO-TAGGING AND ANNOTATION WORKFLOW"
echo "============================================================================"
echo "Images directory: $IMAGES_DIR"
echo ""

# Step 1: Auto-tag images
echo "============================================================================"
echo "STEP 1: Auto-tagging images with Places365"
echo "============================================================================"
python scene_classifier_places365.py "$IMAGES_DIR" \
    --output auto_tags_raw.json \
    --confidence 0.3 \
    --arch resnet18

# Step 2: Preview tags
echo ""
echo "============================================================================"
echo "STEP 2: Previewing generated tags"
echo "============================================================================"
python preview_tags.py auto_tags_raw.json \
    --images-dir "$IMAGES_DIR" \
    --validate \
    --expected-categories scene time weather quality

# Step 3: Add balanced splits
echo ""
echo "============================================================================"
echo "STEP 3: Creating balanced train/val/test splits"
echo "============================================================================"
python scene_classifier_places365.py \
    --balanced-split auto_tags_raw.json \
    --stratify scene \
    --output auto_tags_with_splits.json

# Step 4: Export CSV summary
echo ""
echo "============================================================================"
echo "STEP 4: Exporting tag summary"
echo "============================================================================"
python preview_tags.py auto_tags_with_splits.json \
    --export-csv tags_summary.csv

# Step 5: Launch FiftyOne annotation app
echo ""
echo "============================================================================"
echo "STEP 5: Launching FiftyOne annotation app"
echo "============================================================================"
echo "Importing tags and running object detection..."
python fiftyone_annotation_app_fixed.py "$IMAGES_DIR" \
    --model yolov5 \
    --import-tags auto_tags_with_splits.json \
    --dataset-name vehicle_annotation

echo ""
echo "============================================================================"
echo "WORKFLOW COMPLETE!"
echo "============================================================================"
echo "Generated files:"
echo "  - auto_tags_raw.json          : Raw auto-generated tags"
echo "  - auto_tags_with_splits.json  : Tags with train/val/test splits"
echo "  - tags_summary.csv            : CSV summary of all tags"
echo ""
echo "FiftyOne dataset created: vehicle_annotation"
echo ""
echo "Next steps:"
echo "  1. Review and correct annotations in FiftyOne"
echo "  2. Add/modify tags using tag_manager.py"
echo "  3. Export: python fiftyone_annotation_app_fixed.py $IMAGES_DIR --export output --format yolo"