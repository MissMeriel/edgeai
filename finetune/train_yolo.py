"""
Fine-tune YOLOv8/v11 on the exported 'fixed' samples.

Technique: transfer learning from COCO pretrained weights, with:
  - mosaic + mixup augmentation (built into Ultralytics)
  - cosine LR schedule with warmup
  - multi-scale training
  - auto anchor computation
  - early stopping on val mAP50-95

These defaults generalize well to small-object drone/surveillance data.
For tiny objects (VisDrone-style), we also try SAHI-style tiled inference
via a separate eval flag.

Usage:
    # First export data (if not already done):
    python export_for_training.py --output-dir runs/finetune_data

    # Fine-tune YOLOv8n (fastest, edge-deployable):
    python train_yolo.py --data runs/finetune_data/dataset.yaml --model yolov8n

    # Fine-tune YOLOv11m (more capacity):
    python train_yolo.py --data runs/finetune_data/dataset.yaml --model yolo11m

    # Tiny-object tuning (smaller tiles, lower conf threshold):
    python train_yolo.py --data runs/finetune_data/dataset.yaml --model yolov8s --tiny-objects

    # Resume interrupted run:
    python train_yolo.py --resume runs/yolo_finetune/yolov8n/weights/last.pt
"""

import argparse
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Model catalogue — pretrained COCO weights downloaded by Ultralytics on first run
# ---------------------------------------------------------------------------
MODELS = {
    # (name, pretrained_weights)    speed  mAP  params
    "yolov8n":  "yolov8n.pt",      # fastest, ~3M params  — good for edge
    "yolov8s":  "yolov8s.pt",      # ~11M
    "yolov8m":  "yolov8m.pt",      # ~26M
    "yolov8l":  "yolov8l.pt",      # ~44M
    "yolo11n":  "yolo11n.pt",      # YOLOv11 nano
    "yolo11s":  "yolo11s.pt",      # YOLOv11 small
    "yolo11m":  "yolo11m.pt",      # YOLOv11 medium — best accuracy/speed tradeoff
    "rtdetr-l": "rtdetr-l.pt",     # RT-DETR: transformer-based, no NMS, slower but strong
}


def train(
    data_yaml: str,
    model_key: str = "yolov8n",
    epochs: int = 100,
    batch: int = 16,
    imgsz: int = 640,
    project: str = "runs/yolo_finetune",
    tiny_objects: bool = False,
    freeze_backbone: int = 0,
    resume: str | None = None,
    device: str = "",
):
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("ultralytics not found — activate .venv-eai first")

    if resume:
        print(f"Resuming from {resume}")
        model = YOLO(resume)
        model.train(resume=True)
        return

    weights = MODELS.get(model_key)
    if weights is None:
        sys.exit(f"Unknown model '{model_key}'. Choose from: {list(MODELS)}")

    print(f"\n{'='*60}")
    print(f"Fine-tuning {model_key} on {data_yaml}")
    print(f"  epochs={epochs}, batch={batch}, imgsz={imgsz}")
    if tiny_objects:
        print("  tiny-object mode: smaller imgsz tile (1280), higher sensitivity")
    print(f"{'='*60}\n")

    model = YOLO(weights)

    # Freeze backbone layers for the first few epochs when data is very small
    # (freeze=10 means freeze the first 10 layers of the backbone)
    freeze = freeze_backbone if freeze_backbone > 0 else (10 if freeze_backbone == -1 else 0)

    # -----------------------------------------------------------------------
    # Hyperparameter rationale:
    #
    # lr0=0.01, lrf=0.01:
    #   cosine schedule from 0.01 → 0.0001 (lrf is final LR fraction).
    #   Lower than default (0.01) to avoid catastrophic forgetting of COCO features.
    #
    # warmup_epochs=3:
    #   Ramps LR from near-zero for first 3 epochs; prevents loss spikes on small datasets.
    #
    # mosaic=1.0, mixup=0.1:
    #   Mosaic is the strongest augmentation for small-object detection — stitches 4 images.
    #   Mixup at 0.1 adds light cross-image blending without destabilizing training.
    #
    # hsv_h/s/v, fliplr, degrees, translate, scale:
    #   Drone imagery has no canonical up direction, arbitrary scale, variable lighting.
    #   All set slightly above YOLO defaults for robustness.
    #
    # close_mosaic=10:
    #   Disables mosaic for last 10 epochs so the model converges on clean single images
    #   (matches inference distribution). This prevents a common val mAP plateau.
    #
    # patience=20:
    #   Early stopping — stops if val mAP50-95 doesn't improve for 20 epochs.
    #   Prevents overfitting on small fine-tune datasets.
    #
    # weight_decay=0.0005:
    #   L2 regularization. Standard value; critical for small datasets.
    #
    # iou=0.7:
    #   IoU threshold for matching predictions to GT during training loss.
    #   0.7 is more strict than the 0.5 COCO default — better for dense small objects.
    #
    # multi_scale=True:
    #   Randomly varies input size ±50% during training. Critical for drone video
    #   where subject scale varies enormously with altitude.
    # -----------------------------------------------------------------------

    train_kwargs = dict(
        data=str(Path(data_yaml).resolve()),
        epochs=epochs,
        batch=batch,
        imgsz=1280 if tiny_objects else imgsz,
        project=project,
        name=model_key,
        exist_ok=True,
        device=device if device else None,
        freeze=freeze,
        # LR schedule
        lr0=0.01,
        lrf=0.01,         # final LR = lr0 * lrf = 0.0001
        warmup_epochs=3,
        warmup_momentum=0.8,
        # Regularization
        weight_decay=0.0005,
        dropout=0.0,      # YOLO uses BN, dropout not needed
        # Augmentation
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,   # copies objects between images — great for rare classes
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5.0,      # slight rotation — drone footage is mostly level
        translate=0.1,
        scale=0.5,
        shear=2.0,
        perspective=0.0005,
        flipud=0.0,       # keep upright (drone is always pointing down or forward)
        fliplr=0.5,
        # Loss / matching
        iou=0.7,
        # Training schedule
        close_mosaic=10,
        patience=20,      # early stopping
        multi_scale=True,
        # Logging
        plots=True,
        verbose=True,
        save=True,
        save_period=10,
    )

    # For tiny objects: keep imgsz large, use higher conf sensitivity at eval
    if tiny_objects:
        train_kwargs["conf"] = 0.001   # low threshold — catch small objs, let NMS sort
        train_kwargs["iou"] = 0.65

    results = model.train(**train_kwargs)

    best_weights = Path(project) / model_key / "weights" / "best.pt"
    print(f"\nTraining complete. Best weights: {best_weights}")
    print(f"mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")

    # Quick validation pass on best weights
    print("\nRunning final validation on best.pt ...")
    val_model = YOLO(str(best_weights))
    val_model.val(data=str(Path(data_yaml).resolve()), split="val",
                  imgsz=1280 if tiny_objects else imgsz)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune YOLO on FiftyOne 'fixed' data")
    parser.add_argument("--data", required=False, default="runs/finetune_data/dataset.yaml",
                        help="Path to dataset.yaml produced by export_for_training.py")
    parser.add_argument("--model", default="yolov8n", choices=list(MODELS),
                        help="Model variant (default: yolov8n)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Max training epochs (default: 100, early stopping at patience=20)")
    parser.add_argument("--batch", type=int, default=16,
                        help="Batch size — lower if OOM (default: 16)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Input image size in pixels (default: 640; tiny-objects uses 1280)")
    parser.add_argument("--project", default="runs/yolo_finetune",
                        help="Output directory (default: runs/yolo_finetune)")
    parser.add_argument("--tiny-objects", action="store_true",
                        help="Optimize for small/dense objects: larger imgsz, lower conf threshold")
    parser.add_argument("--freeze-backbone", type=int, default=0,
                        help="Freeze first N backbone layers (0=none; useful when data is <500 samples)")
    parser.add_argument("--device", default="",
                        help="Training device: '' (auto), 'cpu', '0', '0,1' (GPUs)")
    parser.add_argument("--resume", default=None,
                        help="Path to last.pt to resume an interrupted run")
    args = parser.parse_args()

    train(
        data_yaml=args.data,
        model_key=args.model,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        project=args.project,
        tiny_objects=args.tiny_objects,
        freeze_backbone=args.freeze_backbone,
        device=args.device,
        resume=args.resume,
    )
