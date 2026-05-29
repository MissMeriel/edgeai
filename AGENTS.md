# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

EdgeAI is a computer vision annotation and dataset curation pipeline for edge AI applications. It supports human-in-the-loop labeling with semi-automated detection predictions, designed primarily for object detection and scene classification tasks on drone/surveillance video datasets (VisDrone, VIRAT, etc.).

## Environment Setup

Requires Python 3.12.8. The virtualenv is `.venv-eai` at the repo root.

```bash
# First-time setup (creates .venv-eai and datasets/ directory)
./install_mac.sh   # macOS
./install.sh       # Linux
install.bat        # Windows

# Activate environment
source .venv-eai/bin/activate
```

## Running the Primary Annotation UI (FiftyOne)

```bash
cd fiftyone_labelling/

# New dataset — import tags and export in COCO format
python fiftyone_human_annotation.py /path/to/images/ \
  --dataset-name annotation_dataset \
  --import-tags tags.json \
  --export annotation_dataset \
  --export-tags annotation_dataset.json \
  --format coco

# Existing dataset — export new bboxes only (skip auto-predict)
python fiftyone_human_annotation.py /path/to/images/ \
  --dataset-name annotation_dataset \
  --no-auto-predict \
  --import-tags tags.json \
  --export annotation_dataset \
  --format yolo

# Export dataset to transferable zip
python fix_export.py annotation_dataset
```

## Running the Gradio Annotation UI (lightweight alternative)

```bash
# Extract frames from video first
python3 video_frame_extractor_fixed.py path/to/videos

# Launch Gradio annotation UI on extracted frames (open in Firefox)
python3 gradio_labelling/gradio_annotation_ui_synced_rectangles.py ./test_data/Explosion004_x264_24_20260327_104203_5324ec27/
```

## Running the Video Analysis Pipeline

```bash
# Full pipeline: extract → detect → annotate
python run_video_analysis.py input_video.mp4 --fps 2 --model yolov8

# Extract frames only
python run_video_analysis.py input_video.mp4 --extract-only

# Detect on existing frames
python run_video_analysis.py --detect-only --frames-dir extracted_frames/my_video_dir

# Specific time range
python run_video_analysis.py input_video.mp4 --start-time 10 --end-time 60 --fps 1
```

## Scene-wise Dataset Cleaning

```bash
python scenewise_data_cleaning/scene_classifier.py /path/to/dataset --scenes all
python scenewise_data_cleaning/scene_classifier.py /path/to/dataset --scenes night park highway --preview
```

## Tests

```bash
cd tests/
python test_annotator.py

# Diagnose image loading issues
python fix_image_loading.py ./test_data/Explosion004_x264_24_20260327_104203_5324ec27/
```

## Transferring the Annotation Dataset Between Machines

```bash
# Export with images + metadata
cd fiftyone_labelling/
python export_complete_dataset.py

# On destination machine: extract and import
tar -xzf archive.tar.gz
python import_dataset.py
python -c "import fiftyone as fo; fo.launch_app(fo.load_dataset('annotation_dataset'))"
```

## Architecture

The codebase has five main modules:

### `fiftyone_labelling/` — Primary Annotation Interface
- **`fiftyone_human_annotation.py`**: Main entry point. `FiftyOneAnnotationApp` class wraps FiftyOne's MongoDB-backed dataset with auto-detection predictions (YOLOv5/v8, Faster R-CNN) and a tag system for scene type, weather, time-of-day, quality, distribution split, and review status (`todo/fixed/needs_work/reviewed/skip`). Exports to COCO, YOLO, or FiftyOne native format.
- **`export_complete_dataset.py`**: Packages annotated datasets (images + metadata) for transfer between machines.
- **`fix_export.py`**: Repairs or finalizes dataset exports from MongoDB.
- **`batch_annotate.py`**: Bulk annotation operations.
- **`copy_all_to_ground_truth.py`**: Promotes model predictions to ground truth labels.

### `gradio_labelling/` — Lightweight Web Annotation UI
- **`gradio_annotation_ui_synced_rectangles.py`**: Gradio-based bounding box annotation with rectangle syncing across video frames. Use this for quick/local annotation without FiftyOne's full stack.
- **`gradio_annotation_with_predictions.py`**: Variant with integrated model predictions.

### `video_extraction/` — Frame Extraction Pipeline
- **`video_frame_extractor.py`**: Core engine using OpenCV + FFmpeg. Handles codec compatibility, configurable FPS/time ranges, outputs organized directories with JSON manifests (hash, timestamp, frame sequence).
- **`run_video_analysis.py`**: CLI runner orchestrating the extract → detect → annotate workflow. Supports YOLOv5, YOLOv8, Faster R-CNN, CLIP.

### `scenewise_data_cleaning/` — Automatic Scene Classification
- **`scene_classifier.py`**: Uses CLIP text prompts + OpenCV feature detection (edge density, green dominance, vertical lines) to classify scenes (night/day, parks, city streets, highways, indoor/outdoor). Used to filter/organize datasets by scene type.
- **`scene_classifier_places365.py`**: Alternative classifier using the Places365 model.
- **`calibrate_thresholds.py`**: Utility for tuning classification thresholds.

### `scrape_and_classify/` — Web Scraping and Vehicle Classification
- **`image_scraper.py`**: Multi-source scraper (Bing, Google, etc.) with hash-based deduplication, creates FiftyOne datasets from downloads.
- **`full_pipeline_with_scraper.py`**: End-to-end scrape → create FiftyOne dataset → clean → train pipeline, integrated with `config.py` (VGG16-based vehicle classifier hyperparameters).

## Data Flow

```
Video Files / Images / Web Scraping
          ↓
  video_extraction/ or scrape_and_classify/
          ↓
  FiftyOne Dataset (MongoDB backend)
          ↓
  ┌───────────────────┐
  │  Model Predictions │  (YOLOv5/v8, Faster R-CNN, CLIP)
  └────────┬──────────┘
           ↓
  Human Annotation (FiftyOne UI or Gradio UI)
           ↓
  Scene Classification (scenewise_data_cleaning/)
           ↓
  Export (COCO / YOLO / FiftyOne native / zip transfer)
```

## Key Notes

- **FiftyOne uses MongoDB** as its backend. The dataset `annotation_dataset` contains ~6,471 samples (6,447 with ground truth). FiftyOne manages its own database; use `fo.load_dataset()` / `fo.launch_app()` to interact.
- **`version_graveyard/`** contains deprecated Gradio v1–v3 and early FiftyOne iterations — do not modify these.
- **Datasets go in `datasets/`** — this directory is created by the install scripts but is not tracked by git.
- **Docker deployment** is supported via `docker_setup/` for distributing the Gradio UI to annotators without local Python setup. See `docker_setup/README.md`.
- The first 11 images in the annotation dataset are canonical examples of correctly labeled images — preserve them as reference samples.
