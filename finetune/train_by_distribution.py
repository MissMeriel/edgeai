"""
Fine-tune YOLO and/or Faster R-CNN on per-distribution-tag splits of the dataset.

Distribution tags come from scene_classifier_places365_autotagger.py and are stored
in annotation_dataset.json (keys: scene, time, weather, quality, review_status).

Splitting modes (--split-by):
  scene    — one model per scene type  (city_street, highway, parking_lot, …)
  time     — one model per time of day (day, night, dawn, dusk)
  weather  — one model per condition   (clear, cloudy, foggy)
  quality  — one model per quality tag (high_quality, blurry, dark, overexposed)
  all      — one combined model across all samples (no split)

Fine-tuning techniques (--technique):
  freeze        — freeze backbone, train heads only
  two_stage     — head-only for first half, unfreeze all for second half (progressive)
  full          — full network with differential LR (backbone @ 0.1× head LR)
  lora          — LoRA adapters on backbone Linear layers + frozen backbone
  cosine        — full fine-tune with cosine LR annealing (FRCNN only; YOLO always uses cosine)

Can be combined: --technique freeze lora

Model families (--family):
  yolo      — YOLOv8 / YOLOv11 via Ultralytics
  frcnn     — torchvision two-stage models (Faster R-CNN, RetinaNet, FCOS, SSDLite)
  both      — train both families on each split

Data sources (choose one):
  --visdrone <dataset_dir>           use native VisDrone annotations/ + images/ directly
                                     (all 6471 images, no FiftyOne required)
  --fiftyone <dataset_name>          load from FiftyOne MongoDB (only "fixed" samples)
  --data-dir <yolo_export_dir>       use pre-exported YOLO directory + annotation JSON tags

  For --visdrone and --fiftyone, pass --annotation-json to enable tag-based splitting.
  Without it, --split-by all is forced.

Custom classes (--extra-classes):
  Names appended to the VisDrone class list. Must already exist in the annotations.

Usage examples:
    # VisDrone native — split by scene, YOLO two-stage (uses all 6471 images):
    python train_by_distribution.py \\
        --visdrone ../datasets/VisDrone2019-DET-train \\
        --annotation-json ../datasets/VisDrone2019-DET-train/annotation_dataset.json \\
        --split-by scene --family yolo --technique two_stage

    # VisDrone — all images as one combined model, skip untagged scenes:
    python train_by_distribution.py \\
        --visdrone ../datasets/VisDrone2019-DET-train \\
        --split-by all --family yolo

    # FiftyOne path (only reviewed/fixed samples):
    python train_by_distribution.py --split-by scene --family yolo --technique two_stage

    # Split by weather, FRCNN with LoRA:
    python train_by_distribution.py \\
        --visdrone ../datasets/VisDrone2019-DET-train \\
        --annotation-json ../datasets/VisDrone2019-DET-train/annotation_dataset.json \\
        --split-by weather --family frcnn --technique lora
"""

import argparse
import hashlib
import json
import random
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import yaml

import traceback

# ---------------------------------------------------------------------------
# LoRA implementation
# ---------------------------------------------------------------------------

class LoRALinear(torch.nn.Module):
    """Low-rank adaptation of a frozen nn.Linear layer.

    Adds trainable A (in→r) and B (r→out) matrices whose product approximates
    the weight update ΔW ≈ B·A. The original weight is frozen.

    rank=4 is a good starting point for backbone adaptation; increase to 8 or 16
    if the dataset is large and underfitting is observed.
    """

    def __init__(self, linear: torch.nn.Linear, rank: int = 4, alpha: float = 1.0):
        super().__init__()
        in_f, out_f = linear.in_features, linear.out_features
        self.in_features = in_f
        self.out_features = out_f
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Frozen original weight
        self.weight = torch.nn.Parameter(linear.weight.data.clone(), requires_grad=False)
        self.bias = (torch.nn.Parameter(linear.bias.data.clone(), requires_grad=False)
                     if linear.bias is not None else None)

        # Trainable low-rank matrices
        self.lora_A = torch.nn.Parameter(torch.empty(rank, in_f))
        self.lora_B = torch.nn.Parameter(torch.zeros(out_f, rank))
        torch.nn.init.kaiming_uniform_(self.lora_A)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = torch.nn.functional.linear(x, self.weight, self.bias)
        lora = torch.nn.functional.linear(
            torch.nn.functional.linear(x, self.lora_A), self.lora_B
        ) * self.scaling
        return base + lora


def apply_lora(model: torch.nn.Module, rank: int = 4, alpha: float = 1.0,
               target_modules: tuple[str, ...] = ("backbone", "fpn")) -> int:
    """Replace Linear layers in target_modules with LoRALinear; return count replaced."""
    replaced = 0
    for module_name, module in model.named_modules():
        if not any(module_name.startswith(t) for t in target_modules):
            continue
        for child_name, child in list(module.named_children()):
            if isinstance(child, torch.nn.Linear) and child.out_features > rank:
                setattr(module, child_name, LoRALinear(child, rank=rank, alpha=alpha))
                replaced += 1
    return replaced


def freeze_non_lora(model: torch.nn.Module):
    """Freeze everything except LoRA parameters and detection heads."""
    for name, param in model.named_parameters():
        is_lora = "lora_A" in name or "lora_B" in name
        is_head = any(name.startswith(h) for h in
                      ("roi_heads", "head", "cls_logits", "bbox_pred", "rpn"))
        param.requires_grad = is_lora or is_head


# ---------------------------------------------------------------------------
# Dataset split utilities
# ---------------------------------------------------------------------------

SPLIT_TAG_KEY = {
    "scene": "scene",
    "time": "time",
    "weather": "weather",
    "quality": "quality",
    "all": None,
}

UNLABELED_VALUE = "__untagged__"


def load_annotation_json(json_path: Path) -> dict[str, dict]:
    """Load annotation_dataset.json → {filename: {scene, time, weather, ...}}"""
    with json_path.open() as f:
        return json.load(f)


def group_by_tag(annotations: dict[str, dict], tag_key: str | None,
                 status_filter: set[str] | None = None) -> dict[str, list[str]]:
    """
    Returns {tag_value: [filename, ...]} groups.
    Images missing the tag are placed under UNLABELED_VALUE.
    If tag_key is None, all samples go into a single 'all' group.
    """
    groups: dict[str, list[str]] = defaultdict(list)
    for fname, meta in annotations.items():
        if status_filter and meta.get("review_status") not in status_filter:
            continue
        if tag_key is None:
            groups["all"].append(fname)
        else:
            val = meta.get(tag_key, UNLABELED_VALUE)
            groups[val].append(fname)
    return dict(groups)


# ---------------------------------------------------------------------------
# VisDrone native reader
# ---------------------------------------------------------------------------

# VisDrone category IDs (1-indexed in the spec; 0 = ignored region, skip it)
VISDRONE_CLASSES = {
    1:  "pedestrian",
    2:  "people",
    3:  "bicycle",
    4:  "car",
    5:  "van",
    6:  "truck",
    7:  "tricycle",
    8:  "awning-tricycle",
    9:  "bus",
    10: "motor",
}


def parse_visdrone_annotation(txt_path: Path, img_w: int, img_h: int,
                               class_to_idx: dict[str, int]) -> list[str]:
    """
    Parse one VisDrone DET .txt annotation file into YOLO-format label lines.

    DET format per row: x_left, y_top, w, h, score, category_id, truncation, occlusion
      - All coordinates are absolute pixels.
      - score=0 means "ignored region" — skip these.
      - category_id=0 means "ignored region", category_id=11 means "others" — skip both.
    """
    lines = []
    for row in txt_path.read_text().splitlines():
        row = row.strip()
        if not row:
            continue
        parts = row.split(",")
        if len(parts) < 6:
            continue
        x, y, w, h = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
        score = int(parts[4])
        cat_id = int(parts[5])
        if score == 0 or cat_id == 0 or cat_id == 11:  # ignored region or "others"
            continue
        label = VISDRONE_CLASSES.get(cat_id)
        if label is None or label not in class_to_idx:
            continue
        # Convert absolute x,y,w,h → normalised cx,cy,w,h
        cx = (x + w / 2) / img_w
        cy = (y + h / 2) / img_h
        nw = w / img_w
        nh = h / img_h
        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        nw = max(0.0, min(1.0, nw))
        nh = max(0.0, min(1.0, nh))
        if nw < 1e-4 or nh < 1e-4:
            continue
        lines.append(f"{class_to_idx[label]} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    return lines


def parse_visdrone_vid_annotation(
    txt_path: Path,
    img_w: int,
    img_h: int,
    class_to_idx: dict[str, int],
) -> dict[int, list[str]]:
    """
    Parse one VisDrone VID sequence annotation file into a dict of
    {frame_id (1-based int): [YOLO label lines]}.

    VID format per row:
        frame_id, object_id, x_left, y_top, w, h, score, category_id, truncation, occlusion
      - frame_id is 1-based; image filenames are %07d.jpg with the same numbering.
      - score=0 and category_id=0 or 11 are ignored regions — skip them.
    """
    frames: dict[int, list[str]] = {}
    for row in txt_path.read_text().splitlines():
        row = row.strip()
        if not row:
            continue
        parts = row.split(",")
        if len(parts) < 8:
            continue
        frame_id = int(parts[0])
        x, y, w, h = int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5])
        score = int(parts[6])
        cat_id = int(parts[7])
        if score == 0 or cat_id == 0 or cat_id == 11:
            continue
        label = VISDRONE_CLASSES.get(cat_id)
        if label is None or label not in class_to_idx:
            continue
        cx = max(0.0, min(1.0, (x + w / 2) / img_w))
        cy = max(0.0, min(1.0, (y + h / 2) / img_h))
        nw = max(0.0, min(1.0, w / img_w))
        nh = max(0.0, min(1.0, h / img_h))
        if nw < 1e-4 or nh < 1e-4:
            continue
        frames.setdefault(frame_id, []).append(
            f"{class_to_idx[label]} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"
        )
    return frames


def export_visdrone_vid_split(
    train_sequences: list[tuple[Path, Path]],
    val_sequences: list[tuple[Path, Path]],
    out_dir: Path,
    extra_classes: list[str],
) -> Path | None:
    """
    Export VisDrone VID sequences to a YOLO dataset directory.

    Each entry in train_sequences / val_sequences is a (images_dir, annotation_txt) pair.
    Images from all train sequences are written to out_dir/images/train/ (prefixed with
    the sequence name to avoid filename collisions); likewise for val.

    Returns out_dir, or None if no valid images were found.
    """
    from PIL import Image as _PIL

    classes = list(VISDRONE_CLASSES.values()) + [
        c for c in extra_classes if c not in VISDRONE_CLASSES.values()
    ]
    class_to_idx = {c: i for i, c in enumerate(classes)}

    total_written = {"train": 0, "val": 0}

    for split_name, seqs in (("train", train_sequences), ("val", val_sequences)):
        img_out = out_dir / "images" / split_name
        lbl_out = out_dir / "labels" / split_name
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for seq_images_dir, ann_txt in seqs:
            seq_name = seq_images_dir.name  # e.g. uav0000009_03358_v

            # Discover frames and read image dimensions from the first one
            frame_files = sorted(
                p for p in seq_images_dir.iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
            )
            if not frame_files:
                print(f"  WARNING: no images found in {seq_images_dir}, skipping")
                continue

            with _PIL.open(frame_files[0]) as im:
                img_w, img_h = im.size

            # Parse all annotations for this sequence
            if ann_txt.exists():
                frame_labels = parse_visdrone_vid_annotation(ann_txt, img_w, img_h, class_to_idx)
            else:
                print(f"  WARNING: annotation file not found: {ann_txt}")
                frame_labels = {}

            for frame_path in frame_files:
                # Frame ID is the numeric stem (e.g. "0000001" → 1)
                try:
                    frame_id = int(frame_path.stem)
                except ValueError:
                    continue

                # Prefix with sequence name so frames from different sequences never clash
                out_name = f"{seq_name}__{frame_path.name}"
                shutil.copy2(frame_path, img_out / out_name)

                label_lines = frame_labels.get(frame_id, [])
                (lbl_out / f"{seq_name}__{frame_path.stem}.txt").write_text(
                    "\n".join(label_lines)
                )
                total_written[split_name] += 1

    n_train = total_written["train"]
    n_val = total_written["val"]
    if n_train + n_val == 0:
        print("  WARNING: no frames exported")
        return None

    print(f"  VID export: {n_train} train frames + {n_val} val frames  ({len(classes)} classes)")

    yaml_data = {
        "path": str(out_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": len(classes),
        "names": classes,
    }
    (out_dir / "dataset.yaml").write_text(yaml.dump(yaml_data, sort_keys=False))
    return out_dir


def export_visdrone_split(
    visdrone_dir: Path,
    filenames: list[str],
    out_dir: Path,
    val_ratio: float,
    seed: int,
    extra_classes: list[str],
) -> Path | None:
    """
    Export a subset of the VisDrone dataset as YOLO-format labels.

    Reads images from <visdrone_dir>/images/ and annotations from
    <visdrone_dir>/annotations/. No FiftyOne dependency.
    """
    images_dir = visdrone_dir / "images"
    ann_dir = visdrone_dir / "annotations"
    if not images_dir.exists() or not ann_dir.exists():
        sys.exit(f"Expected {images_dir} and {ann_dir} to exist. "
                 "Check --visdrone points to the VisDrone dataset root.")

    classes = list(VISDRONE_CLASSES.values()) + [c for c in extra_classes
                                                  if c not in VISDRONE_CLASSES.values()]
    class_to_idx = {c: i for i, c in enumerate(classes)}

    # Filter to files that actually exist on disk
    valid = [f for f in filenames if (images_dir / f).exists()]
    if not valid:
        print("  WARNING: no matching images found in VisDrone images/, skipping")
        return None

    random.seed(seed)
    shuffled = valid[:]
    random.shuffle(shuffled)
    n_val = max(1, round(len(shuffled) * val_ratio))
    splits = {"val": shuffled[:n_val], "train": shuffled[n_val:]}

    out_dir.mkdir(parents=True, exist_ok=True)
    skipped_ann = 0

    for split_name, split_files in splits.items():
        img_out = out_dir / "images" / split_name
        lbl_out = out_dir / "labels" / split_name
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for fname in split_files:
            src_img = images_dir / fname
            shutil.copy2(src_img, img_out / fname)

            ann_file = ann_dir / (Path(fname).stem + ".txt")
            if ann_file.exists():
                from PIL import Image as _PIL
                with _PIL.open(src_img) as im:
                    img_w, img_h = im.size
                label_lines = parse_visdrone_annotation(ann_file, img_w, img_h, class_to_idx)
            else:
                label_lines = []
                skipped_ann += 1

            (lbl_out / (Path(fname).stem + ".txt")).write_text("\n".join(label_lines))

    if skipped_ann:
        print(f"  WARNING: {skipped_ann} images had no annotation file")

    n_train = len(splits["train"])
    n_val_w = len(splits["val"])
    print(f"  Exported {n_train} train + {n_val_w} val  ({len(classes)} classes)")

    yaml_data = {
        "path": str(out_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": len(classes),
        "names": classes,
    }
    (out_dir / "dataset.yaml").write_text(yaml.dump(yaml_data, sort_keys=False))
    return out_dir


# ---------------------------------------------------------------------------
# YOLO export helpers (independent of FiftyOne)
# ---------------------------------------------------------------------------

def copy_yolo_split(
    filenames: list[str],
    src_images_dir: Path,
    src_labels_dir: Path,
    out_dir: Path,
    val_ratio: float,
    seed: int,
    classes: list[str],
) -> Path:
    """Copy a subset of a pre-exported YOLO dataset into a new train/val split."""
    random.seed(seed)
    shuffled = filenames[:]
    random.shuffle(shuffled)
    n_val = max(1, round(len(shuffled) * val_ratio))
    splits = {"val": shuffled[:n_val], "train": shuffled[n_val:]}

    for split_name, names in splits.items():
        (out_dir / "images" / split_name).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split_name).mkdir(parents=True, exist_ok=True)
        for fname in names:
            stem = Path(fname).stem
            img_src = src_images_dir / fname
            lbl_src = src_labels_dir / (stem + ".txt")
            if img_src.exists():
                shutil.copy2(img_src, out_dir / "images" / split_name / fname)
            if lbl_src.exists():
                shutil.copy2(lbl_src, out_dir / "labels" / split_name / (stem + ".txt"))
            else:
                (out_dir / "labels" / split_name / (stem + ".txt")).write_text("")

    yaml_data = {
        "path": str(out_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": len(classes),
        "names": classes,
    }
    (out_dir / "dataset.yaml").write_text(yaml.dump(yaml_data, sort_keys=False))
    return out_dir


# ---------------------------------------------------------------------------
# FiftyOne export helpers
# ---------------------------------------------------------------------------

def export_fiftyone_split(
    dataset_name: str,
    sample_ids: list[str] | None,
    out_dir: Path,
    val_ratio: float,
    seed: int,
    extra_classes: list[str],
) -> Path | None:
    """Export a subset from FiftyOne as YOLO format. sample_ids=None means all fixed samples."""
    try:
        import fiftyone as fo
        from fiftyone import ViewField as F
    except ImportError:
        sys.exit("fiftyone not found — activate .venv-eai: source ../.venv-eai/bin/activate")

    try:
        dataset = fo.load_dataset(dataset_name)
    except Exception as e:
        sys.exit(f"Cannot load FiftyOne dataset '{dataset_name}': {e}")

    if sample_ids is not None:
        # Select exactly these samples; no re-lookup by filename needed
        view = dataset.select(sample_ids).match(F("ground_truth").exists())
    else:
        view = dataset.match(F("ground_truth").exists())

    if len(view) == 0:
        print(f"  WARNING: no samples found for this split, skipping")
        return None

    classes = sorted(set(view.distinct("ground_truth.detections.label") + extra_classes) - {None})
    if not classes:
        print(f"  WARNING: no class labels found, skipping")
        return None
    class_to_idx = {c: i for i, c in enumerate(classes)}

    random.seed(seed)
    samples = list(view.select_fields(["filepath", "ground_truth"]))
    random.shuffle(samples)
    n_val = max(1, round(len(samples) * val_ratio))
    splits = {"val": samples[:n_val], "train": samples[n_val:]}

    out_dir.mkdir(parents=True, exist_ok=True)
    skipped = 0
    for split_name, split_samples in splits.items():
        img_dir = out_dir / "images" / split_name
        lbl_dir = out_dir / "labels" / split_name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for sample in split_samples:
            src = Path(sample.filepath)
            if not src.exists():
                skipped += 1
                continue
            shutil.copy2(src, img_dir / src.name)
            lines = []
            if sample.ground_truth and sample.ground_truth.detections:
                for det in sample.ground_truth.detections:
                    if det.label not in class_to_idx:
                        continue
                    x, y, bw, bh = det.bounding_box
                    cx = max(0.0, min(1.0, x + bw / 2))
                    cy = max(0.0, min(1.0, y + bh / 2))
                    bw = max(0.0, min(1.0, bw))
                    bh = max(0.0, min(1.0, bh))
                    lines.append(f"{class_to_idx[det.label]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            (lbl_dir / (src.stem + ".txt")).write_text("\n".join(lines))

    if skipped:
        print(f"  WARNING: {skipped} samples skipped (missing image files)")

    n_train = len(splits["train"])
    n_val_actual = len(splits["val"])
    print(f"  Exported {n_train} train + {n_val_actual} val samples, {len(classes)} classes")

    yaml_data = {
        "path": str(out_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": len(classes),
        "names": classes,
    }
    (out_dir / "dataset.yaml").write_text(yaml.dump(yaml_data, sort_keys=False))
    return out_dir


# ---------------------------------------------------------------------------
# YOLO training
# ---------------------------------------------------------------------------

YOLO_MODELS = {
    "yolov8n": "yolov8n.pt",
    "yolov8s": "yolov8s.pt",
    "yolov8m": "yolov8m.pt",
    "yolo11n": "yolo11n.pt",
    "yolo11s": "yolo11s.pt",
    "yolo11m": "yolo11m.pt",
    "rtdetr-l": "rtdetr-l.pt",
}


def train_yolo(
    data_yaml: Path,
    model_key: str,
    techniques: set[str],
    epochs: int,
    batch: int,
    imgsz: int,
    out_dir: Path,
    device: str,
) -> dict:
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("ultralytics not found")

    t_start = time.time()

    weights = YOLO_MODELS.get(model_key, model_key)

    # Freeze: freeze first 10 backbone layers when freeze or lora requested
    # (LoRA is not natively supported by Ultralytics; we approximate with layer freezing
    # and a lower LR to preserve pretrained features — true LoRA applies to FRCNN only)
    freeze_layers = 0
    if "freeze" in techniques or "lora" in techniques:
        freeze_layers = 10  # freeze backbone stem + first CSP block

    # Two-stage: run head-only phase first, then reload for phase 2
    if "two_stage" in techniques:
        print("  [two_stage] Phase 1: head-only (freeze=10) ...")
        phase1_epochs = max(5, epochs // 3)
        YOLO(weights).train(
            data=str(data_yaml.resolve()),
            epochs=phase1_epochs,
            batch=batch,
            imgsz=imgsz,
            project=str(out_dir.resolve()),
            name="phase1",
            exist_ok=True,
            device=device or None,
            freeze=10,
            lr0=0.001,
            lrf=0.01,
            warmup_epochs=2,
            weight_decay=0.0005,
            mosaic=1.0,
            mixup=0.1,
            copy_paste=0.1,
            close_mosaic=5,
            patience=10,
            plots=True,
            verbose=True,
            save=True,
        )
        phase1_weights = out_dir / "phase1" / "weights" / "best.pt"
        if phase1_weights.exists():
            weights = str(phase1_weights)
        print(f"  [two_stage] Phase 2: full fine-tune ...")

    lr0 = 0.01
    if "full" in techniques:
        lr0 = 0.005
    elif "freeze" in techniques or "lora" in techniques:
        lr0 = 0.001

    # Always construct a fresh YOLO instance — reusing an instance across .train()
    # calls drops self.overrides["model"] and raises KeyError in newer Ultralytics.
    # Ultralytics silently ignores a relative project path and falls back to
    # runs/detect/train — resolve to absolute to ensure weights land where expected.
    results = YOLO(weights).train(
        data=str(data_yaml.resolve()),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        project=str(out_dir.resolve()),
        name="train",
        exist_ok=True,
        device=device or None,
        freeze=freeze_layers if "freeze" in techniques else 0,
        lr0=lr0,
        lrf=0.01,
        warmup_epochs=3,
        warmup_momentum=0.8,
        weight_decay=0.0005,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        fliplr=0.5,
        iou=0.7,
        close_mosaic=10,
        patience=20,
        plots=True,
        verbose=True,
        save=True,
        save_period=10,
    )

    elapsed = time.time() - t_start
    rd = results.results_dict
    best_weights = out_dir / "train" / "weights" / "best.pt"
    data_root = data_yaml.parent
    n_train = sum(1 for _ in (data_root / "images" / "train").iterdir()
                  if _.suffix.lower() in {".jpg", ".jpeg", ".png"})
    n_val = sum(1 for _ in (data_root / "images" / "val").iterdir()
                if _.suffix.lower() in {".jpg", ".jpeg", ".png"})
    map50 = rd.get("metrics/mAP50(B)")
    map50_95 = rd.get("metrics/mAP50-95(B)")
    print(f"  YOLO done. mAP50={map50}  mAP50-95={map50_95}  "
          f"time={elapsed:.0f}s  best={best_weights}")
    return {
        "best_weights": str(best_weights),
        "train_time_seconds": round(elapsed, 1),
        "n_train": n_train,
        "n_val": n_val,
        "mAP50": map50,
        "mAP50_95": map50_95,
        "precision": rd.get("metrics/precision(B)"),
        "recall": rd.get("metrics/recall(B)"),
        "fitness": rd.get("fitness"),
    }


# ---------------------------------------------------------------------------
# FRCNN training
# ---------------------------------------------------------------------------

# Reuse model catalogue and helpers from train_faster_rcnn
def _get_frcnn_registry():
    import torchvision.models.detection as det
    return {
        "fasterrcnn_resnet50_v2": (
            det.fasterrcnn_resnet50_fpn_v2,
            det.FasterRCNN_ResNet50_FPN_V2_Weights, "fasterrcnn"),
        "fasterrcnn_resnet50": (
            det.fasterrcnn_resnet50_fpn,
            det.FasterRCNN_ResNet50_FPN_Weights, "fasterrcnn"),
        "fasterrcnn_mobilenet": (
            det.fasterrcnn_mobilenet_v3_large_fpn,
            det.FasterRCNN_MobileNet_V3_Large_FPN_Weights, "fasterrcnn"),
        "retinanet": (
            det.retinanet_resnet50_fpn_v2,
            det.RetinaNet_ResNet50_FPN_V2_Weights, "retinanet"),
        "fcos": (
            det.fcos_resnet50_fpn,
            det.FCOS_ResNet50_FPN_Weights, "fcos"),
        "ssdlite": (
            det.ssdlite320_mobilenet_v3_large,
            det.SSDLite320_MobileNet_V3_Large_Weights, "ssd"),
    }


def _build_frcnn(num_classes: int, model_key: str) -> torch.nn.Module:
    import torchvision.models.detection as det
    from torchvision.models.detection.anchor_utils import AnchorGenerator
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    reg = _get_frcnn_registry()
    if model_key not in reg:
        sys.exit(f"Unknown FRCNN model '{model_key}'. Choose from: {list(reg)}")
    builder_fn, weights_cls, head_type = reg[model_key]
    model = builder_fn(weights=weights_cls.DEFAULT)

    if head_type == "fasterrcnn":
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        if hasattr(model, "rpn"):
            model.rpn.anchor_generator = AnchorGenerator(
                sizes=((8,), (16,), (32,), (64,), (128,)),
                aspect_ratios=((0.5, 1.0, 2.0),) * 5,
            )
    elif head_type == "retinanet":
        from torchvision.models.detection.retinanet import RetinaNetClassificationHead
        model.head.classification_head = RetinaNetClassificationHead(
            in_channels=model.head.classification_head.conv[0][0].in_channels,
            num_anchors=model.head.classification_head.num_anchors,
            num_classes=num_classes,
        )
        model.anchor_generator = AnchorGenerator(
            sizes=tuple((s,) for s in (8, 16, 32, 64, 128)),
            aspect_ratios=((0.5, 1.0, 2.0),) * 5,
        )
    elif head_type == "fcos":
        from torchvision.models.detection.fcos import FCOSClassificationHead
        model.head.classification_head = FCOSClassificationHead(
            in_channels=model.head.classification_head.conv[0][0].in_channels,
            num_anchors=model.head.classification_head.num_anchors,
            num_classes=num_classes,
        )
    elif head_type == "ssd":
        model_fresh = builder_fn(weights=None, num_classes=num_classes)
        model_fresh.backbone.load_state_dict(model.backbone.state_dict())
        model = model_fresh

    return model


def _frcnn_dataset_loader(data_dir: Path):
    """Import YOLODetectionDataset from train_faster_rcnn.py in the same directory."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "train_faster_rcnn",
        Path(__file__).parent / "train_faster_rcnn.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.YOLODetectionDataset, mod.get_train_transforms, mod.get_val_transforms, mod.collate_fn


def train_frcnn(
    data_dir: Path,
    model_key: str,
    techniques: set[str],
    epochs: int,
    batch: int,
    lr: float,
    out_dir: Path,
    device_str: str,
    lora_rank: int,
) -> dict:
    from torch.utils.data import DataLoader

    t_start = time.time()

    yaml_path = data_dir / "dataset.yaml"
    with yaml_path.open() as f:
        meta = yaml.safe_load(f)
    classes = meta["names"]
    num_classes = len(classes) + 1

    YOLODetectionDataset, get_train_tf, get_val_tf, collate_fn = _frcnn_dataset_loader(data_dir)

    if device_str:
        device = torch.device(device_str)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_ds = YOLODetectionDataset(
        data_dir / "images" / "train", data_dir / "labels" / "train",
        classes, transforms=get_train_tf())
    val_ds = YOLODetectionDataset(
        data_dir / "images" / "val", data_dir / "labels" / "val",
        classes, transforms=get_val_tf())

    nw = 0 if str(device) == "mps" else 4
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True,
                              num_workers=nw, collate_fn=collate_fn,
                              pin_memory=(str(device) == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False,
                            num_workers=nw, collate_fn=collate_fn,
                            pin_memory=(str(device) == "cuda"))

    model = _build_frcnn(num_classes, model_key)

    # Apply techniques to model before moving to device
    if "lora" in techniques:
        n_replaced = apply_lora(model, rank=lora_rank, alpha=float(lora_rank))
        print(f"  [lora] Replaced {n_replaced} Linear layers with LoRA(rank={lora_rank})")
        freeze_non_lora(model)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"  [lora] Trainable params: {trainable:,} / {total:,} "
              f"({100*trainable/total:.1f}%)")

    elif "freeze" in techniques:
        for name, param in model.named_parameters():
            if name.startswith("backbone") or name.startswith("fpn"):
                param.requires_grad = False
        print("  [freeze] Backbone frozen")

    model.to(device)

    # Optimizer param groups
    def _is_backbone(name):
        return name.startswith("backbone") or name.startswith("fpn")

    if "full" in techniques and "lora" not in techniques and "freeze" not in techniques:
        # Differential LR: backbone at 0.1×
        param_groups = [
            {"params": [p for n, p in model.named_parameters()
                        if _is_backbone(n) and p.requires_grad], "lr": lr * 0.1},
            {"params": [p for n, p in model.named_parameters()
                        if not _is_backbone(n) and p.requires_grad], "lr": lr},
        ]
    elif "two_stage" in techniques and "lora" not in techniques:
        # Head-only first — freeze backbone, will unfreeze at midpoint
        for name, param in model.named_parameters():
            if _is_backbone(name):
                param.requires_grad = False
        param_groups = [{"params": [p for p in model.parameters() if p.requires_grad]}]
        print("  [two_stage] Phase 1: head-only")
    else:
        param_groups = [{"params": [p for p in model.parameters() if p.requires_grad]}]

    optimizer = torch.optim.SGD(param_groups, lr=lr, momentum=0.9,
                                weight_decay=0.0005, nesterov=True)

    if "cosine" in techniques:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    else:
        milestones = [max(1, int(epochs * 0.6)), max(2, int(epochs * 0.8))]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    out_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    patience_counter = 0
    patience = 10
    history = []

    for epoch in range(epochs):
        # Two-stage: unfreeze backbone at midpoint, add param group at lower LR
        if "two_stage" in techniques and "lora" not in techniques and epoch == epochs // 2:
            print(f"  [two_stage] Phase 2: unfreezing backbone at epoch {epoch}")
            for name, param in model.named_parameters():
                param.requires_grad = True
            optimizer.add_param_group({
                "params": [p for n, p in model.named_parameters() if _is_backbone(n)],
                "lr": lr * 0.01,
            })

        model.train()
        total_loss = 0.0
        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss_dict.values())
            optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            total_loss += losses.item()

        train_loss = total_loss / max(1, len(train_loader))

        model.train()
        val_total = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                val_total += sum(model(images, targets).values()).item()
        val_loss = val_total / max(1, len(val_loader))

        scheduler.step()
        print(f"  Epoch {epoch:3d}/{epochs} | train={train_loss:.4f} | val={val_loss:.4f}")
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if (epoch + 1) % 5 == 0:
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "optimizer": optimizer.state_dict()},
                       out_dir / f"checkpoint_epoch{epoch}.pth")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_path = out_dir / "best.pth"
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "classes": classes, "num_classes": num_classes,
                        "model_key": model_key}, best_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    elapsed = time.time() - t_start
    n_train = sum(1 for _ in (data_dir / "images" / "train").iterdir()
                  if _.suffix.lower() in {".jpg", ".jpeg", ".png"})
    n_val = sum(1 for _ in (data_dir / "images" / "val").iterdir()
                if _.suffix.lower() in {".jpg", ".jpeg", ".png"})
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"  FRCNN done. best_val_loss={best_val_loss:.4f}  "
          f"time={elapsed:.0f}s  best={out_dir / 'best.pth'}")
    return {
        "best_weights": str(out_dir / "best.pth"),
        "train_time_seconds": round(elapsed, 1),
        "n_train": n_train,
        "n_val": n_val,
        "best_val_loss": best_val_loss,
    }


# ---------------------------------------------------------------------------
# Run-name generation
# ---------------------------------------------------------------------------

def make_run_name(
    split_by: str,
    family: str,
    techniques: list[str],
    yolo_model: str,
    frcnn_model: str,
    epochs_yolo: int,
    epochs_frcnn: int,
    batch_yolo: int,
    batch_frcnn: int,
    imgsz: int,
    lr: float,
    lora_rank: int,
    val_ratio: float,
    seed: int,
    extra_classes: list[str],
    skip_untagged: bool,
) -> str:
    """
    Build a human-readable directory name that captures the training params
    that determine the outcome, plus a 6-char hash to prevent silent overwrites
    when two runs produce the same readable slug.

    Example:
        dist_finetune-split-by-scene-yolo-yolov8n-epochs-50-two-stage-imgsz-640-af3c12
    """
    # Readable slug parts
    parts = ["dist_finetune"]
    parts.append(f"split-by-{split_by.replace('_', '-')}")
    parts.append(family)
    if family in ("yolo", "both"):
        parts.append(yolo_model)
        parts.append(f"epochs-{epochs_yolo}")
    if family in ("frcnn", "both"):
        parts.append(frcnn_model.replace("_", "-"))
        parts.append(f"frcnn-epochs-{epochs_frcnn}")
    technique_slug = "-".join(sorted(t.replace("_", "-") for t in techniques))
    parts.append(technique_slug)
    parts.append(f"imgsz-{imgsz}")
    if lora_rank != 4 and "lora" in techniques:
        parts.append(f"lora-r{lora_rank}")
    if skip_untagged:
        parts.append("skip-untagged")
    if extra_classes:
        parts.append("extra-" + "-".join(sorted(extra_classes)))

    slug = "-".join(parts)

    # Hash covers every param that affects the result (including seed, lr, batch, val_ratio)
    # so that two runs differing only in a non-slug param still get different directories.
    hash_payload = json.dumps({
        "split_by": split_by,
        "family": family,
        "techniques": sorted(techniques),
        "yolo_model": yolo_model,
        "frcnn_model": frcnn_model,
        "epochs_yolo": epochs_yolo,
        "epochs_frcnn": epochs_frcnn,
        "batch_yolo": batch_yolo,
        "batch_frcnn": batch_frcnn,
        "imgsz": imgsz,
        "lr": lr,
        "lora_rank": lora_rank,
        "val_ratio": val_ratio,
        "seed": seed,
        "extra_classes": sorted(extra_classes),
        "skip_untagged": skip_untagged,
    }, sort_keys=True)
    short_hash = hashlib.sha256(hash_payload.encode()).hexdigest()[:6]

    return f"{slug}-{short_hash}"


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run(
    split_by: str,
    family: str,
    techniques: list[str],
    visdrone_dir: str | None,
    fiftyone_name: str | None,
    data_dir: str | None,
    annotation_json: str | None,
    vid_train: list[str] | None,
    vid_val: list[str] | None,
    yolo_model: str,
    frcnn_model: str,
    epochs_yolo: int,
    epochs_frcnn: int,
    batch_yolo: int,
    batch_frcnn: int,
    imgsz: int,
    lr: float,
    lora_rank: int,
    val_ratio: float,
    seed: int,
    project: str,
    device: str,
    extra_classes: list[str],
    status_filter: list[str],
    skip_untagged: bool,
    min_samples: int,
):
    techniques_set = set(techniques)
    tag_key = SPLIT_TAG_KEY.get(split_by)
    out_root = Path(project)

    # ---- VID sequences mode (explicit train/val sequence dirs + annotation files) ----
    # Takes priority over all other data sources. No tag-based splitting is done;
    # the caller decides which sequences are train and which are val.
    if vid_train:
        train_pairs = [(Path(d), Path(a)) for d, a in vid_train]
        val_pairs   = [(Path(d), Path(a)) for d, a in (vid_val or [])]

        # Validate paths up front so failures are obvious before any work starts
        for role, pairs in (("train", train_pairs), ("val", val_pairs)):
            for seq_dir, ann_txt in pairs:
                if not seq_dir.exists():
                    sys.exit(f"VID {role} sequence directory not found: {seq_dir}")
                if not ann_txt.exists():
                    sys.exit(f"VID {role} annotation file not found: {ann_txt}")

        if not val_pairs:
            print("WARNING: no --vid-val sequences provided; validation set will be empty")

        vid_data_dir = out_root / "data" / "all"
        exported = export_visdrone_vid_split(
            train_sequences=train_pairs,
            val_sequences=val_pairs,
            out_dir=vid_data_dir,
            extra_classes=extra_classes,
        )
        if exported is None:
            sys.exit("VID export produced no frames — check sequence directories and annotations")

        data_yaml = exported / "dataset.yaml"
        group_results: dict = {}

        if family in ("yolo", "both"):
            yolo_out = out_root / "models" / "all" / "yolo"
            print(f"\n[YOLO] techniques={sorted(techniques_set)}")
            try:
                group_results["yolo"] = train_yolo(
                    data_yaml=data_yaml,
                    model_key=yolo_model,
                    techniques=techniques_set,
                    epochs=epochs_yolo,
                    batch=batch_yolo,
                    imgsz=imgsz,
                    out_dir=yolo_out,
                    device=device,
                )
            except Exception as e:
                print(f"  ERROR during YOLO training: {e}")
                print(traceback.format_exc())
                group_results["yolo"] = {"error": str(e)}

        if family in ("frcnn", "both"):
            frcnn_out = out_root / "models" / "all" / "frcnn"
            print(f"\n[FRCNN] techniques={sorted(techniques_set)}")
            try:
                group_results["frcnn"] = train_frcnn(
                    data_dir=exported,
                    model_key=frcnn_model,
                    techniques=techniques_set,
                    epochs=epochs_frcnn,
                    batch=batch_frcnn,
                    lr=lr,
                    out_dir=frcnn_out,
                    device_str=device,
                    lora_rank=lora_rank,
                )
            except Exception as e:
                print(f"  ERROR during FRCNN training: {e}")
                group_results["frcnn"] = {"error": str(e)}

        all_results = {"all": group_results}

        config_snapshot = {
            "vid_train": [[str(d), str(a)] for d, a in train_pairs],
            "vid_val":   [[str(d), str(a)] for d, a in val_pairs],
            "family": family,
            "yolo_model": yolo_model,
            "frcnn_model": frcnn_model,
            "techniques": list(techniques_set),
            "lora_rank": lora_rank,
            "epochs_yolo": epochs_yolo,
            "epochs_frcnn": epochs_frcnn,
            "batch_yolo": batch_yolo,
            "batch_frcnn": batch_frcnn,
            "imgsz": imgsz,
            "lr": lr,
            "seed": seed,
            "extra_classes": extra_classes,
            "project": project,
            "device": device,
        }
        config_path = out_root / "training_config.json"
        config_path.write_text(json.dumps(config_snapshot, indent=2, default=str))
        summary_path = out_root / "training_summary.json"
        summary_path.write_text(json.dumps(all_results, indent=2, default=str))
        print(f"\n{'='*60}")
        print(f"VID training complete. Summary → {summary_path}")
        print(f"Config → {config_path}")
        print(f"{'='*60}")
        for fam, r in group_results.items():
            best = r.get("best_weights", r.get("error", "?"))
            metric = r.get("mAP50_95") or r.get("best_val_loss", "")
            metric_str = f"  metric={metric:.4f}" if isinstance(metric, float) else ""
            print(f"  all [{fam}]  {best}{metric_str}")
        return

    # ---- Determine groups (tag_val → list of filenames or FiftyOne IDs) ----
    if visdrone_dir:
        # All images from the VisDrone images/ directory; tag lookup from annotation JSON
        vd_path = Path(visdrone_dir)
        images_dir = vd_path / "images"
        all_fnames = [p.name for p in sorted(images_dir.iterdir())
                      if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]

        ann_path = Path(annotation_json) if annotation_json else None
        if ann_path and ann_path.exists():
            annotations = load_annotation_json(ann_path)
            # group_by_tag uses status_filter only when annotation_json has review_status;
            # VisDrone images don't require review — pass no status filter here
            groups = group_by_tag(annotations, tag_key, status_filter=None)
            # Images not present in annotation_json have no tag — add them to untagged
            tagged = set(fname for names in groups.values() for fname in names)
            for fname in all_fnames:
                if fname not in tagged:
                    groups.setdefault(UNLABELED_VALUE, []).append(fname)
        else:
            if tag_key is not None:
                print("WARNING: --annotation-json not provided; forcing --split-by all")
            groups = {"all": all_fnames}

    elif fiftyone_name:
        try:
            import fiftyone as fo
            from fiftyone import ViewField as F
        except ImportError:
            sys.exit("fiftyone not found")

        dataset = fo.load_dataset(fiftyone_name)
        fixed_view = dataset.match(
            F("ground_truth").exists() &
            (F("tag_review_status.label").is_in(status_filter) if status_filter
             else F("ground_truth").exists())
        )

        if tag_key is None:
            groups: dict[str, list[str] | None] = {"all": None}
        else:
            # Store FiftyOne sample IDs directly — no re-lookup by filename later
            groups = defaultdict(list)
            for sample in fixed_view.select_fields([f"tag_{split_by}"]):
                tag_field = getattr(sample, f"tag_{split_by}", None)
                val = tag_field.label if tag_field else UNLABELED_VALUE
                groups[val].append(sample.id)

    else:
        # Pre-exported YOLO dir; tag lookup from annotation JSON
        ann_path = Path(annotation_json) if annotation_json else None
        if ann_path is None or not ann_path.exists():
            sys.exit("Provide --visdrone, --fiftyone, or --data-dir with --annotation-json")
        annotations = load_annotation_json(ann_path)
        groups = group_by_tag(annotations, tag_key,
                              status_filter=set(status_filter) if status_filter else None)

    if skip_untagged and UNLABELED_VALUE in groups:
        removed = len(groups.pop(UNLABELED_VALUE))
        print(f"Skipping {removed} untagged samples (--skip-untagged)")

    print(f"\nDistribution split '{split_by}' → {len(groups)} groups:")
    for k, v in sorted(groups.items(), key=lambda x: -(len(x[1]) if x[1] else 0)):
        print(f"  {k}: {len(v) if v is not None else 'all'} samples")

    all_results = {}

    for tag_val, sample_ids in groups.items():
        n = len(sample_ids) if sample_ids is not None else "all"
        safe_tag = tag_val.replace(" ", "_").replace("/", "_")

        if sample_ids is not None and len(sample_ids) < min_samples:
            print(f"\nSKIPPING {split_by}={tag_val}: only {len(sample_ids)} samples "
                  f"(--min-samples={min_samples})")
            continue

        print(f"\n{'='*60}")
        print(f"GROUP: {split_by}={tag_val}  ({n} samples)")
        print(f"{'='*60}")

        split_data_dir = out_root / "data" / safe_tag
        split_data_dir.mkdir(parents=True, exist_ok=True)

        # ---- Export data for this split ----
        if visdrone_dir:
            exported = export_visdrone_split(
                Path(visdrone_dir),
                sample_ids,  # list of image filenames
                split_data_dir,
                val_ratio, seed, extra_classes,
            )
        elif fiftyone_name:
            exported = export_fiftyone_split(
                fiftyone_name,
                sample_ids,  # list of FiftyOne IDs, or None for "all"
                split_data_dir,
                val_ratio, seed, extra_classes,
            )
        else:
            # data_dir is a pre-exported YOLO directory; re-split subset
            src = Path(data_dir)
            yaml_path = src / "dataset.yaml"
            with yaml_path.open() as f:
                meta = yaml.safe_load(f)
            classes = sorted(set(meta["names"] + extra_classes))

            # Images may be in train or val subdirs; search both
            src_images = {}
            src_labels = {}
            for split_name in ("train", "val"):
                img_d = src / "images" / split_name
                lbl_d = src / "labels" / split_name
                if img_d.exists():
                    for p in img_d.iterdir():
                        src_images[p.name] = p
                if lbl_d.exists():
                    for p in lbl_d.iterdir():
                        src_labels[p.name] = p

            # Filter to filenames in this group (sample_ids are filenames in the data_dir path)
            group_fnames = [f for f in sample_ids if f in src_images] if sample_ids else list(src_images)
            if not group_fnames:
                print(f"  WARNING: no matching images found in {src}, skipping")
                continue

            # Write to split_data_dir with a flat images/labels structure first
            tmp_img = split_data_dir / "_src_images"
            tmp_lbl = split_data_dir / "_src_labels"
            tmp_img.mkdir(exist_ok=True)
            tmp_lbl.mkdir(exist_ok=True)
            for fname in group_fnames:
                if fname in src_images:
                    shutil.copy2(src_images[fname], tmp_img / fname)
                stem = Path(fname).stem + ".txt"
                if stem in src_labels:
                    shutil.copy2(src_labels[stem], tmp_lbl / (Path(fname).stem + ".txt"))

            exported = copy_yolo_split(
                group_fnames, tmp_img, tmp_lbl,
                split_data_dir, val_ratio, seed, classes,
            )
            shutil.rmtree(tmp_img, ignore_errors=True)
            shutil.rmtree(tmp_lbl, ignore_errors=True)

        if exported is None:
            continue

        data_yaml = exported / "dataset.yaml"
        group_results = {}

        # ---- Train YOLO ----
        if family in ("yolo", "both"):
            yolo_out = out_root / "models" / safe_tag / "yolo"
            print(f"\n[YOLO] techniques={sorted(techniques_set)}")
            try:
                r = train_yolo(
                    data_yaml=data_yaml,
                    model_key=yolo_model,
                    techniques=techniques_set,
                    epochs=epochs_yolo,
                    batch=batch_yolo,
                    imgsz=imgsz,
                    out_dir=yolo_out,
                    device=device,
                )
                group_results["yolo"] = r
            except Exception as e:
                print(f"  ERROR during YOLO training: {e}")
                print(traceback.format_exc())
                group_results["yolo"] = {"error": str(e)}

        # ---- Train FRCNN ----
        if family in ("frcnn", "both"):
            frcnn_out = out_root / "models" / safe_tag / "frcnn"
            print(f"\n[FRCNN] techniques={sorted(techniques_set)}")
            try:
                r = train_frcnn(
                    data_dir=exported,
                    model_key=frcnn_model,
                    techniques=techniques_set,
                    epochs=epochs_frcnn,
                    batch=batch_frcnn,
                    lr=lr,
                    out_dir=frcnn_out,
                    device_str=device,
                    lora_rank=lora_rank,
                )
                group_results["frcnn"] = r
            except Exception as e:
                print(f"  ERROR during FRCNN training: {e}")
                group_results["frcnn"] = {"error": str(e)}

        all_results[tag_val] = group_results

    # Config snapshot — maps 1-to-1 to CLI flags so --config can replay this run
    config_snapshot = {
        "visdrone": visdrone_dir,
        "fiftyone": fiftyone_name,
        "data_dir": data_dir,
        "annotation_json": annotation_json,
        "split_by": split_by,
        "skip_untagged": skip_untagged,
        "status_filter": status_filter,
        "family": family,
        "yolo_model": yolo_model,
        "frcnn_model": frcnn_model,
        "techniques": list(techniques_set),
        "lora_rank": lora_rank,
        "epochs_yolo": epochs_yolo,
        "epochs_frcnn": epochs_frcnn,
        "batch_yolo": batch_yolo,
        "batch_frcnn": batch_frcnn,
        "imgsz": imgsz,
        "lr": lr,
        "val_ratio": val_ratio,
        "seed": seed,
        "extra_classes": extra_classes,
        "project": project,
        "device": device,
        "min_samples": min_samples,
    }
    config_path = out_root / "training_config.json"
    config_path.write_text(json.dumps(config_snapshot, indent=2, default=str))

    # Summary
    summary_path = out_root / "training_summary.json"
    summary_path.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\n{'='*60}")
    print(f"All training complete. Summary written to {summary_path}")
    print(f"Config saved to {config_path}  (rerun with --config {config_path})")
    print(f"{'='*60}")
    for tag_val, res in all_results.items():
        for fam, r in res.items():
            best = r.get("best_weights", r.get("error", "?"))
            metric = r.get("mAP50_95") or r.get("best_val_loss", "")
            metric_str = f"  metric={metric:.4f}" if isinstance(metric, float) else ""
            print(f"  {split_by}={tag_val}  [{fam}]  {best}{metric_str}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLO/FRCNN on per-distribution-tag dataset splits",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Config replay
    parser.add_argument("--config", default=None, metavar="CONFIG_JSON",
                        help="Path to a training_config.json produced by a previous run; "
                             "sets defaults for all flags (explicit flags still override)")

    # Data source
    src = parser.add_argument_group("Data source (choose one)")
    src.add_argument("--visdrone", default=None, metavar="DATASET_DIR",
                     help="VisDrone DET dataset root containing images/ and annotations/ "
                          "(uses all images; no FiftyOne required)")
    src.add_argument("--fiftyone", default=None, metavar="DATASET_NAME",
                     help="FiftyOne dataset name (default when neither --visdrone nor "
                          "--data-dir is given: annotation_dataset)")
    src.add_argument("--data-dir", default=None,
                     help="Pre-exported YOLO directory")
    src.add_argument("--annotation-json", default=None,
                     help="Path to annotation_dataset.json for tag-based splitting "
                          "(required for --split-by scene/time/weather/quality with "
                          "--visdrone or --data-dir)")

    vid = parser.add_argument_group(
        "VisDrone VID source (overrides all other data sources)",
        "Pass one or more sequence dirs for train and val separately. "
        "Each --vid-train / --vid-val entry is IMAGES_DIR:ANNOTATION_TXT. "
        "Multiple sequences can be passed; they are merged into a single dataset. "
        "Example: --vid-train sequences/uav0000009_03358_v:annotations/uav0000009_03358_v.txt",
    )
    vid.add_argument(
        "--vid-train", nargs="+", default=None, metavar="IMAGES_DIR:ANNOTATION_TXT",
        help="Training sequence(s) as dir:annotation pairs",
    )
    vid.add_argument(
        "--vid-val", nargs="+", default=None, metavar="IMAGES_DIR:ANNOTATION_TXT",
        help="Validation sequence(s) as dir:annotation pairs",
    )

    # Split
    parser.add_argument("--split-by", default="scene",
                        choices=list(SPLIT_TAG_KEY),
                        help="Tag dimension to split on (default: scene)")
    parser.add_argument("--skip-untagged", action="store_true",
                        help="Skip images that have no value for the split tag")
    parser.add_argument("--status", nargs="+", default=["fixed"],
                        dest="status_filter",
                        help="Review statuses to include (default: fixed)")

    # Model family
    parser.add_argument("--family", default="yolo",
                        choices=["yolo", "frcnn", "both"],
                        help="Model family to train (default: yolo)")
    parser.add_argument("--yolo-model", default="yolov8n",
                        choices=list(YOLO_MODELS),
                        help="YOLO model variant (default: yolov8n)")
    parser.add_argument("--frcnn-model", default="fasterrcnn_resnet50_v2",
                        choices=list(_get_frcnn_registry()),
                        help="Torchvision model (default: fasterrcnn_resnet50_v2)")

    # Fine-tuning technique
    parser.add_argument("--technique", nargs="+",
                        default=["two_stage"],
                        dest="techniques",
                        choices=["freeze", "two_stage", "full", "lora", "cosine"],
                        help="Fine-tuning technique(s); combinable (default: two_stage)")
    parser.add_argument("--lora-rank", type=int, default=4,
                        help="LoRA rank for backbone Linear layers (default: 4)")

    # Training hyperparams
    parser.add_argument("--epochs-yolo", type=int, default=100,
                        help="YOLO max epochs (default: 100, early stopping at patience=20)")
    parser.add_argument("--epochs-frcnn", type=int, default=20,
                        help="FRCNN max epochs (default: 20, early stopping at patience=10)")
    parser.add_argument("--batch-yolo", type=int, default=16)
    parser.add_argument("--batch-frcnn", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--lr", type=float, default=0.005,
                        help="Base LR for FRCNN (default: 0.005)")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)

    # Classes
    parser.add_argument("--extra-classes", nargs="*", default=[],
                        help="Additional class names to include beyond dataset defaults")

    # Output
    parser.add_argument("--project", default=None,
                        help="Output root directory. Defaults to an auto-generated name under "
                             "runs/ that encodes key training params + a short hash, e.g. "
                             "runs/dist_finetune-split-by-scene-yolo-yolov8n-epochs-50-two-stage-imgsz-640-af3c12")
    parser.add_argument("--device", default="",
                        help="Device: '' (auto), 'cpu', 'cuda', 'cuda:0', 'mps'")
    parser.add_argument("--min-samples", type=int, default=20,
                        help="Skip any distribution group with fewer than this many samples "
                             "(default: 20 — too small to produce a valid train/val split)")

    # Two-pass parse: load config defaults first, then let explicit flags override.
    # Exclude "project" from config replay — a replayed run should always get a
    # fresh output directory, not silently reuse the original one.
    _pre = parser.parse_known_args(sys.argv[1:])[0]
    if _pre.config:
        import json as _json
        _cfg = _json.loads(Path(_pre.config).read_text())
        parser.set_defaults(**{k: v for k, v in _cfg.items()
                               if v is not None and k != "project"})
    args = parser.parse_args()

    # Parse VID sequence pairs (IMAGES_DIR:ANNOTATION_TXT)
    def _parse_vid_pairs(entries: list[str] | None) -> list[list[str]] | None:
        if not entries:
            return None
        pairs = []
        for entry in entries:
            if ":" not in entry:
                parser.error(
                    f"--vid-train/--vid-val entries must be IMAGES_DIR:ANNOTATION_TXT, got: {entry!r}"
                )
            img_dir, ann_txt = entry.split(":", 1)
            pairs.append([img_dir, ann_txt])
        return pairs

    vid_train = _parse_vid_pairs(args.vid_train)
    vid_val   = _parse_vid_pairs(args.vid_val)

    # Priority: --vid-train > --visdrone > --data-dir > --fiftyone (default)
    visdrone_dir = args.visdrone if not vid_train else None
    data_dir = args.data_dir if not (vid_train or visdrone_dir) else None
    fiftyone_name = (None if (vid_train or visdrone_dir or data_dir)
                     else (args.fiftyone or "annotation_dataset"))

    # Auto-generate project path when the user did not supply --project explicitly
    if args.project is None:
        run_name = make_run_name(
            split_by=args.split_by,
            family=args.family,
            techniques=args.techniques,
            yolo_model=args.yolo_model,
            frcnn_model=args.frcnn_model,
            epochs_yolo=args.epochs_yolo,
            epochs_frcnn=args.epochs_frcnn,
            batch_yolo=args.batch_yolo,
            batch_frcnn=args.batch_frcnn,
            imgsz=args.imgsz,
            lr=args.lr,
            lora_rank=args.lora_rank,
            val_ratio=args.val_ratio,
            seed=args.seed,
            extra_classes=args.extra_classes,
            skip_untagged=args.skip_untagged,
        )
        project = str(Path("runs") / run_name)
        print(f"Auto-generated run directory: {project}")
    else:
        project = args.project

    run(
        split_by=args.split_by,
        family=args.family,
        techniques=args.techniques,
        visdrone_dir=visdrone_dir,
        fiftyone_name=fiftyone_name,
        data_dir=data_dir,
        annotation_json=args.annotation_json,
        vid_train=vid_train,
        vid_val=vid_val,
        yolo_model=args.yolo_model,
        frcnn_model=args.frcnn_model,
        epochs_yolo=args.epochs_yolo,
        epochs_frcnn=args.epochs_frcnn,
        batch_yolo=args.batch_yolo,
        batch_frcnn=args.batch_frcnn,
        imgsz=args.imgsz,
        lr=args.lr,
        lora_rank=args.lora_rank,
        val_ratio=args.val_ratio,
        seed=args.seed,
        project=project,
        device=args.device,
        extra_classes=args.extra_classes,
        status_filter=args.status_filter,
        skip_untagged=args.skip_untagged,
        min_samples=args.min_samples,
    )

# https://docs.ultralytics.com/guides/yolo-performance-metrics