# Fine-Tuning Pipeline

Fine-tune YOLO and Faster R-CNN models on annotated drone/surveillance data.
The pipeline has two entry points:

- **`train_by_distribution.py`** — splits a dataset by scene/time/weather tags and trains one model per group (standalone, no sequence structure required)
- **`../finetune_sequence/train_sequence.py`** — trains per group across pre-organised video sequences (see [finetune_sequence/README.md](../finetune_sequence/README.md))

Both share the model definitions in `train_faster_rcnn.py` and training utilities.

---

## Files

### `export_for_training.py`

Exports annotated samples from the FiftyOne `annotation_dataset` into a format the training scripts consume.

Two output formats:

- **`yolo`** (default) — YOLO `.txt` labels + `dataset.yaml`; consumed by `train_yolo.py` and `train_faster_rcnn.py`
- **`fiftyone`** — full portable FiftyOne dataset (JSON + media); re-importable on any machine with `fo.Dataset.from_dir()`

Only samples with `tag_review_status == "fixed"` are exported by default; pass `--status` to include others.

```bash
# Standard export for training (YOLO format, 15% val split):
python finetune/export_for_training.py --output-dir runs/finetune_data

# Include "reviewed" samples alongside "fixed":
python finetune/export_for_training.py \
    --output-dir runs/finetune_data \
    --status fixed reviewed

# Larger val split (25%):
python finetune/export_for_training.py \
    --output-dir runs/finetune_data \
    --val-ratio 0.25

# Different FiftyOne dataset name:
python finetune/export_for_training.py \
    --output-dir runs/finetune_data \
    --dataset-name my_dataset

# Portable full-fidelity backup (FiftyOne format):
python finetune/export_for_training.py \
    --output-dir runs/finetune_fo \
    --format fiftyone
```

Output structure (YOLO format):

```text
runs/finetune_data/
    images/train/   images/val/
    labels/train/   labels/val/
    dataset.yaml
```

---

### `train_yolo.py`

Fine-tunes YOLOv8/v11/RT-DETR from COCO pretrained weights on the exported dataset.

Augmentation defaults are tuned for drone/surveillance footage: mosaic + mixup, multi-scale training, cosine LR with warmup, and early stopping on val mAP50-95.

Available models: `yolov8n`, `yolov8s`, `yolov8m`, `yolov8l`, `yolo11n`, `yolo11s`, `yolo11m`, `rtdetr-l`

```bash
# Fastest edge-deployable model (YOLOv8n):
python finetune/train_yolo.py \
    --data runs/finetune_data/dataset.yaml

# Stronger model, more epochs:
python finetune/train_yolo.py \
    --data runs/finetune_data/dataset.yaml \
    --model yolo11m \
    --epochs 150

# Tiny-object mode (1280px input, lower conf threshold):
python finetune/train_yolo.py \
    --data runs/finetune_data/dataset.yaml \
    --model yolov8s \
    --tiny-objects

# Freeze first 10 backbone layers (good when data < 500 samples):
python finetune/train_yolo.py \
    --data runs/finetune_data/dataset.yaml \
    --model yolov8n \
    --freeze-backbone 10

# Specify GPU and batch size:
python finetune/train_yolo.py \
    --data runs/finetune_data/dataset.yaml \
    --model yolov8m \
    --batch 8 \
    --device 0

# Resume an interrupted run:
python finetune/train_yolo.py \
    --resume runs/yolo_finetune/yolov8n/weights/last.pt
```

Outputs to `runs/yolo_finetune/<model>/weights/best.pt`.

---

### `train_faster_rcnn.py`

Fine-tunes torchvision two-stage and one-stage detection models from COCO pretrained weights.

Anchors are tuned for drone imagery (8–128px instead of the COCO default 32–512px). The dataset class reads YOLO-format labels and wraps them as `tv_tensors.BoundingBoxes` for torchvision v2 transforms.

Available models:

| Key | Architecture | Notes |
| --- | --- | --- |
| `fasterrcnn_resnet50_v2` | Faster R-CNN ResNet-50 FPN v2 | Default, strongest two-stage |
| `fasterrcnn_resnet50` | Faster R-CNN ResNet-50 FPN v1 | Classic baseline |
| `fasterrcnn_mobilenet` | Faster R-CNN MobileNetV3 | Lightweight, edge-deployable |
| `retinanet` | RetinaNet ResNet-50 FPN v2 | One-stage, focal loss |
| `fcos` | FCOS ResNet-50 FPN | Anchor-free, good for small/irregular objects |
| `ssdlite` | SSDLite MobileNetV3 320 | Fastest inference, lowest memory |

```bash
# Recommended default (Faster R-CNN v2, progressive fine-tune):
python finetune/train_faster_rcnn.py \
    --data-dir runs/finetune_data

# Lightweight edge model:
python finetune/train_faster_rcnn.py \
    --data-dir runs/finetune_data \
    --model fasterrcnn_mobilenet

# Head-only training (very small dataset < 200 samples):
python finetune/train_faster_rcnn.py \
    --data-dir runs/finetune_data \
    --mode head_only

# Full fine-tune (large dataset, differential LR):
python finetune/train_faster_rcnn.py \
    --data-dir runs/finetune_data \
    --mode full \
    --epochs 26

# Anchor-free, good for irregular/small objects:
python finetune/train_faster_rcnn.py \
    --data-dir runs/finetune_data \
    --model fcos

# One-stage with focal loss (handles class imbalance):
python finetune/train_faster_rcnn.py \
    --data-dir runs/finetune_data \
    --model retinanet

# Resume from checkpoint:
python finetune/train_faster_rcnn.py \
    --data-dir runs/finetune_data \
    --resume runs/frcnn_finetune/checkpoint_epoch10.pth
```

Fine-tune modes (`--mode`):

- `head_only` — freeze backbone + FPN, train only detection heads (best for < 200 images)
- `full` — fine-tune entire network with differential LR (backbone @ 0.1× head LR)
- `progressive` — head-only for first half, then unfreeze all (default, good general choice)

---

### `train_by_distribution.py`

Trains per-distribution-tag specialist models. Splits the dataset by scene/time/weather/quality tags and trains a separate model for each group — e.g. one model for `city_street`, another for `highway`.

This is the standalone alternative to `finetune_sequence/train_sequence.py`. Use this when your data comes from a flat image directory or FiftyOne dataset rather than organised video sequences.

Data sources (pick one):

- `--visdrone` — native VisDrone `annotations/` + `images/` directory (all 6471 images, no FiftyOne required)
- `--fiftyone` — load from FiftyOne MongoDB (only `fixed` samples)
- `--data-dir` — pre-exported YOLO directory

```bash
# VisDrone native — split by scene, YOLO two-stage:
python finetune/train_by_distribution.py \
    --visdrone datasets/VisDrone2019-DET-train \
    --annotation-json datasets/VisDrone2019-DET-train/annotation_dataset.json \
    --split-by scene \
    --family yolo \
    --technique two_stage

# All images combined (no split), YOLO:
python finetune/train_by_distribution.py \
    --visdrone datasets/VisDrone2019-DET-train \
    --split-by all \
    --family yolo

# Split by weather, FRCNN with LoRA adapters:
python finetune/train_by_distribution.py \
    --visdrone datasets/VisDrone2019-DET-train \
    --annotation-json datasets/VisDrone2019-DET-train/annotation_dataset.json \
    --split-by weather \
    --family frcnn \
    --technique lora

# FiftyOne source, split by time of day, both families:
python finetune/train_by_distribution.py \
    --split-by time \
    --family both \
    --technique two_stage

# Split by scene, progressive fine-tune, freeze+LoRA combined:
python finetune/train_by_distribution.py \
    --visdrone datasets/VisDrone2019-DET-train \
    --annotation-json datasets/VisDrone2019-DET-train/annotation_dataset.json \
    --split-by scene \
    --family frcnn \
    --technique freeze lora
```

Split options: `scene`, `time`, `weather`, `quality`, `all`

Fine-tuning techniques: `freeze`, `two_stage`, `full`, `lora`, `cosine` (combinable)

---

### `validate.py`

Validates a fine-tuned YOLO or Faster R-CNN model against the VisDrone test-dev set.

Loads ground-truth from native VisDrone `annotations/` and computes per-class and overall mAP@0.5, mAP@0.5:0.95, precision, recall, and F1.

```bash
# Validate a YOLO model:
python finetune/validate.py \
    --dataset datasets/VisDrone2019-DET-test-dev \
    --weights runs/dist_finetune/models/city_street/yolo/train/weights/best.pt \
    --model-type yolo

# Validate a Faster R-CNN checkpoint:
python finetune/validate.py \
    --dataset datasets/VisDrone2019-DET-test-dev \
    --weights runs/dist_finetune/models/city_street/frcnn/best.pth \
    --model-type frcnn \
    --frcnn-arch fasterrcnn_resnet50_v2

# Custom IoU thresholds and confidence cutoff:
python finetune/validate.py \
    --dataset datasets/VisDrone2019-DET-test-dev \
    --weights runs/dist_finetune/models/city_street/yolo/train/weights/best.pt \
    --model-type yolo \
    --iou-thresholds 0.5 0.75 \
    --conf 0.25

# Save per-image results to JSON:
python finetune/validate.py \
    --dataset datasets/VisDrone2019-DET-test-dev \
    --weights runs/dist_finetune/models/city_street/yolo/train/weights/best.pt \
    --model-type yolo \
    --save-json
```

---

### `training_dashboard.py` / `training_dashboard_squelch.py`

Streamlit dashboard that reads all completed runs from `finetune/runs/` and displays:

- Per-run summary metrics (mAP50, mAP50-95, precision, recall)
- Cross-run comparison charts
- Per-scene per-epoch training curves (loss, mAP)
- Confusion matrices and YOLO plot images

`training_dashboard_squelch.py` is a variant that suppresses verbose Streamlit cache warnings.

```bash
# Launch dashboard:
streamlit run finetune/training_dashboard.py

# Auto-reload on save (development):
streamlit run --server.runOnSave true finetune/training_dashboard.py

# If charts look stale after a new run, clear the cache first:
streamlit cache clear
streamlit run finetune/training_dashboard.py

# Squelch variant (fewer console warnings):
streamlit run finetune/training_dashboard_squelch.py
```

Dashboard is available at `http://localhost:8501`.

---

### `training_runner.py`

Programmatic training runner used by the Streamlit UI or called directly from Python. Reads `training_config.yaml`, runs YOLO training in a background thread, saves a results JSON, and copies the best model to `trained_models/`.

```bash
# Run standalone (reads training_config.yaml):
python finetune/training_runner.py
```

Or from Python:

```python
from finetune.training_runner import TrainingRunner
runner = TrainingRunner("finetune/training_config.yaml")
runner.prepare_training()
runner.train()
```

---

### `training_config.yaml`

YAML configuration file consumed by `training_runner.py` and the Streamlit UI. Defines model name, dataset path, epochs, batch size, learning rate, optimizer, and patience. Edit this file to change training defaults without modifying the scripts.

---

### `docker-compose-complete.yml`

Docker Compose stack that runs the full pipeline as services:

| Service | Port | Purpose |
| --- | --- | --- |
| FiftyOne | 5151 | Annotate images |
| Streamlit | 8501 | Configure and monitor training |
| TensorBoard | 6006 | Epoch-by-epoch metrics |
| Optuna | 8080 | Hyperparameter optimisation |
| Nginx | 80 | Unified access point |

```bash
# Start all services:
cd finetune/
./start_complete_stack.sh

# Stop all services:
./stop_stack.sh

# Start manually via Docker Compose:
docker compose -f finetune/docker-compose-complete.yml up -d

# View logs for a specific service:
docker compose -f finetune/docker-compose-complete.yml logs -f streamlit
```

---

## Typical Workflow

```text
1. Annotate in FiftyOne, mark samples as "fixed"
         ↓
2. export_for_training.py   →  runs/finetune_data/
         ↓
3a. train_yolo.py           →  runs/yolo_finetune/<model>/weights/best.pt
3b. train_faster_rcnn.py    →  runs/frcnn_finetune/best.pth
         ↓
4. validate.py              →  mAP / precision / recall vs. test-dev
         ↓
5. training_dashboard.py    →  compare runs, inspect curves
```

For sequence-structured datasets (VisDrone VID), use `finetune_sequence/train_sequence.py` instead of steps 2–3 — it handles export and training in one command.
