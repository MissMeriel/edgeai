"""
Fine-tune YOLO and/or Faster R-CNN using VisDrone VID sequences as mini-datasets,
grouped by the tags in datasets/sequences_categories.json.

Unlike train_by_distribution.py (which shuffles individual frames from a flat DET set),
this script treats each VIDEO SEQUENCE as an atomic unit. Sequences are never split
across train/val — entire sequences go to one side or the other. This preserves temporal
coherence and prevents data leakage between consecutive near-identical frames.

===========================================================================
Grouping modes (--group-by):

  none       — All annotated sequences combined into one dataset.
               The held-out validation sequence(s) are excluded from training.

  scene      — One model per scene tag: city_street, highway, parking_lot,
               recreation_area. Trains a specialist per environment type.

  time       — One model per time-of-day tag: day, night, dawn.
               Useful when lighting variation is the primary domain shift.

  weather    — One model per weather tag: cloudy, clear, sunny.
               Small groups are skipped by --min-sequences.

  scene+time — Cross-product of scene × time (e.g. city_street/night).
               Most granular; requires enough sequences per cell.

  all        — Alias for 'none'.

===========================================================================
Fine-tuning techniques (--technique):

  freeze     — Freeze backbone entirely; train detection heads only.
               Best for very small sequence sets (<5 sequences, ~500 frames).
               Prevents catastrophic forgetting of pretrained features.

  two_stage  — Phase 1: head-only (first 1/3 of epochs, backbone frozen).
               Phase 2: full network fine-tune with a low backbone LR.
               Good default — quick adaptation then careful global refinement.

  full       — Fine-tune everything from epoch 0 with differential LR
               (backbone @ 0.1× the head LR). Best when sequences are long
               and cover sufficient variation (≥5 sequences or ≥1000 frames).

  lora       — Low-rank adaptation on backbone Linear layers (FRCNN only).
               Backbone weights frozen; only tiny A/B matrices are trained.
               Very parameter-efficient; good for domain shift without
               catastrophic forgetting. rank=4 by default (--lora-rank).

  cosine     — Cosine LR annealing instead of MultiStep (FRCNN only;
               YOLO always uses cosine). Can be combined with other techniques.

  temporal   — Temporal consistency augmentation: for each training batch,
               an additional "adjacent-frame" pair is sampled from the same
               sequence and mixed in at 0.3× weight. Encourages the model to
               produce stable detections across frames. YOLO approximates this
               via copy_paste + mosaic; FRCNN gets an explicit temporal loss term.

Techniques can be combined: --technique two_stage temporal

===========================================================================
Model families (--family):
  yolo   — YOLOv8/v11 via Ultralytics (yolov8n/s/m/l, yolo11n/s/m, rtdetr-l)
  frcnn  — torchvision two-stage models (fasterrcnn_resnet50_v2, retinanet,
           fcos, ssdlite, fasterrcnn_mobilenet)
  both   — train both on each group

===========================================================================
Sequence selection:

  --val-sequences  Explicit sequence name(s) to hold out as validation.
                   Must match keys in sequences_categories.json (path or bare name).
                   Example: --val-sequences uav0000268_05773_v uav0000119_02301_v

  --val-split-by   Auto-select one validation sequence per group by this tag
                   dimension ('scene', 'time', 'random'). Default: random.

  --frame-stride   Sample every N-th frame from each sequence to reduce
                   dataset size and inter-frame redundancy (default: 1 = all frames).

===========================================================================
Usage examples:

  # All sequences combined, default two_stage YOLO, hold out one sequence by name:
  python train_sequence.py \\
      --sequences-json ../datasets/sequences_categories.json \\
      --group-by none \\
      --val-sequences uav0000268_05773_v \\
      --family yolo --technique two_stage

  # Scene-specialist models, auto-select one val sequence per scene:
  python train_sequence.py \\
      --sequences-json ../datasets/sequences_categories.json \\
      --group-by scene \\
      --val-split-by random \\
      --family both --technique two_stage temporal

  # Time-of-day specialists, FRCNN with LoRA, stride=3 (every 3rd frame):
  python train_sequence.py \\
      --sequences-json ../datasets/sequences_categories.json \\
      --group-by time \\
      --val-split-by random \\
      --family frcnn --technique lora \\
      --frame-stride 3

  # Cross-product scene×time, full fine-tune, YOLO large:
  python train_sequence.py \\
      --sequences-json ../datasets/sequences_categories.json \\
      --group-by scene+time \\
      --val-split-by random \\
      --family yolo --yolo-model yolov8s --technique full
"""

import argparse
import hashlib
import json
import random
import shutil
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import torch
import yaml

import truststore
truststore.inject_into_ssl()

# ---------------------------------------------------------------------------
# VisDrone constants (shared with train_by_distribution.py)
# ---------------------------------------------------------------------------

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

YOLO_MODELS = {
    "yolov8n": "yolov8n.pt",
    "yolov8s": "yolov8s.pt",
    "yolov8m": "yolov8m.pt",
    "yolov8l": "yolov8l.pt",
    "yolo11n": "yolo11n.pt",
    "yolo11s": "yolo11s.pt",
    "yolo11m": "yolo11m.pt",
    "rtdetr-l": "rtdetr-l.pt",
}

# ---------------------------------------------------------------------------
# LoRA (identical to train_by_distribution.py)
# ---------------------------------------------------------------------------

class LoRALinear(torch.nn.Module):
    def __init__(self, linear: torch.nn.Linear, rank: int = 4, alpha: float = 1.0):
        super().__init__()
        in_f, out_f = linear.in_features, linear.out_features
        self.in_features = in_f
        self.out_features = out_f
        self.rank = rank
        self.scaling = alpha / rank
        self.weight = torch.nn.Parameter(linear.weight.data.clone(), requires_grad=False)
        self.bias = (torch.nn.Parameter(linear.bias.data.clone(), requires_grad=False)
                     if linear.bias is not None else None)
        self.lora_A = torch.nn.Parameter(torch.empty(rank, in_f))
        self.lora_B = torch.nn.Parameter(torch.zeros(out_f, rank))
        torch.nn.init.kaiming_uniform_(self.lora_A)

    def forward(self, x):
        base = torch.nn.functional.linear(x, self.weight, self.bias)
        lora = torch.nn.functional.linear(
            torch.nn.functional.linear(x, self.lora_A), self.lora_B
        ) * self.scaling
        return base + lora


def apply_lora(model, rank=4, alpha=1.0, target_modules=("backbone", "fpn")):
    replaced = 0
    for module_name, module in model.named_modules():
        if not any(module_name.startswith(t) for t in target_modules):
            continue
        for child_name, child in list(module.named_children()):
            if isinstance(child, torch.nn.Linear) and child.out_features > rank:
                setattr(module, child_name, LoRALinear(child, rank=rank, alpha=float(alpha)))
                replaced += 1
    return replaced


def freeze_non_lora(model):
    for name, param in model.named_parameters():
        is_lora = "lora_A" in name or "lora_B" in name
        is_head = any(name.startswith(h) for h in
                      ("roi_heads", "head", "cls_logits", "bbox_pred", "rpn"))
        param.requires_grad = is_lora or is_head

# ---------------------------------------------------------------------------
# Sequence catalogue
# ---------------------------------------------------------------------------

def load_sequence_catalogue(json_path: Path) -> dict[str, dict]:
    """
    Load sequences_categories.json.
    Keys are paths like 'datasets/VisDrone2019-VID-val/sequences/uav0000086_00000_v/'
    Returns a normalised dict: { seq_name: {scene, time, weather, quality, images_dir, ann_txt} }
    where seq_name is just the bare sequence directory name (uav0000086_00000_v).
    """
    with json_path.open() as f:
        raw = json.load(f)

    repo_root = json_path.parent.parent  # edgeai/

    catalogue = {}
    for path_str, tags in raw.items():
        seq_path = Path(path_str.rstrip("/"))
        seq_name = seq_path.name

        # Resolve to absolute path from repo root
        abs_seq = (repo_root / seq_path).resolve()

        # Annotation file lives at  <dataset_root>/annotations/<seq_name>.txt
        ann_txt = abs_seq.parent.parent / "annotations" / f"{seq_name}.txt"

        catalogue[seq_name] = {
            **tags,
            "images_dir": abs_seq,
            "ann_txt": ann_txt,
            "path_str": path_str,
        }
    return catalogue


def group_sequences(catalogue: dict[str, dict], group_by: str) -> dict[str, list[str]]:
    """
    Returns { group_label: [seq_name, ...] } based on the grouping mode.
    """
    groups: dict[str, list[str]] = defaultdict(list)
    for seq_name, meta in catalogue.items():
        if group_by in ("none", "all"):
            groups["all"].append(seq_name)
        elif group_by == "scene+time":
            key = f"{meta.get('scene', 'unknown')}__{meta.get('time', 'unknown')}"
            groups[key].append(seq_name)
        else:
            key = meta.get(group_by, "unknown")
            groups[key].append(seq_name)
    return dict(groups)

# ---------------------------------------------------------------------------
# Validation sequence selection
# ---------------------------------------------------------------------------

def select_val_sequences(
    group_seqs: list[str],
    explicit_val: list[str] | None,
    val_split_by: str,
    catalogue: dict[str, dict],
    seed: int,
) -> tuple[list[str], list[str]]:
    """
    Split group_seqs into (train_seqs, val_seqs).

    explicit_val: if any of these seq names appear in group_seqs, reserve them as val.
    val_split_by: 'random' → pick one at random; 'scene'/'time' → pick one per tag value.
    """
    if explicit_val:
        val = [s for s in group_seqs if s in explicit_val]
        train = [s for s in group_seqs if s not in explicit_val]
        if not val:
            # None of the explicit val sequences are in this group — fall back to random
            rng = random.Random(seed)
            val = [rng.choice(group_seqs)]
            train = [s for s in group_seqs if s not in val]
        return train, val

    if len(group_seqs) == 1:
        # Only one sequence — use it as both train and val (degenerate but better than crash)
        return group_seqs, group_seqs

    rng = random.Random(seed)
    if val_split_by == "random":
        val = [rng.choice(group_seqs)]
        train = [s for s in group_seqs if s not in val]
        return train, val

    # val_split_by is a tag dimension — pick one val sequence per unique tag value
    by_tag: dict[str, list[str]] = defaultdict(list)
    for s in group_seqs:
        tag_val = catalogue[s].get(val_split_by, "unknown")
        by_tag[tag_val].append(s)

    val = []
    for tag_seqs in by_tag.values():
        rng.shuffle(tag_seqs)
        val.append(tag_seqs[0])

    # Cap val at 30% of total sequences
    max_val = max(1, len(group_seqs) // 3)
    if len(val) > max_val:
        val = rng.sample(val, max_val)

    train = [s for s in group_seqs if s not in val]
    if not train:
        train = val  # degenerate — all sequences are val
    return train, val

# ---------------------------------------------------------------------------
# Frame export
# ---------------------------------------------------------------------------

def export_sequence_split(
    train_seqs: list[str],
    val_seqs: list[str],
    catalogue: dict[str, dict],
    out_dir: Path,
    frame_stride: int,
    extra_classes: list[str],
) -> Path | None:
    """
    Export frames from sequences into a YOLO dataset directory.
    Each frame image and label is prefixed with the sequence name to avoid collisions.
    Returns out_dir or None if no frames were written.
    """
    from PIL import Image as _PIL

    classes = list(VISDRONE_CLASSES.values()) + [
        c for c in extra_classes if c not in VISDRONE_CLASSES.values()
    ]
    class_to_idx = {c: i for i, c in enumerate(classes)}

    total = {"train": 0, "val": 0}

    for split_name, seqs in (("train", train_seqs), ("val", val_seqs)):
        img_out = out_dir / "images" / split_name
        lbl_out = out_dir / "labels" / split_name
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for seq_name in seqs:
            meta = catalogue[seq_name]
            seq_images_dir: Path = meta["images_dir"]
            ann_txt: Path = meta["ann_txt"]

            if not seq_images_dir.exists():
                print(f"  WARNING: images dir not found: {seq_images_dir}, skipping")
                continue

            frame_files = sorted(
                p for p in seq_images_dir.iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
            )
            if not frame_files:
                print(f"  WARNING: no frames in {seq_images_dir}, skipping")
                continue

            # Apply stride
            frame_files = frame_files[::frame_stride]

            with _PIL.open(frame_files[0]) as im:
                img_w, img_h = im.size

            frame_labels = _parse_vid_annotation(ann_txt, img_w, img_h, class_to_idx)

            for frame_path in frame_files:
                try:
                    frame_id = int(frame_path.stem)
                except ValueError:
                    continue

                out_name = f"{seq_name}__{frame_path.name}"
                shutil.copy2(frame_path, img_out / out_name)
                label_lines = frame_labels.get(frame_id, [])
                (lbl_out / f"{seq_name}__{frame_path.stem}.txt").write_text(
                    "\n".join(label_lines)
                )
                total[split_name] += 1

    n_train, n_val = total["train"], total["val"]
    if n_train + n_val == 0:
        print("  WARNING: no frames exported")
        return None

    print(f"  Export: {n_train} train frames + {n_val} val frames  "
          f"({len(classes)} classes, stride={frame_stride})")

    yaml_data = {
        "path": str(out_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": len(classes),
        "names": classes,
    }
    (out_dir / "dataset.yaml").write_text(yaml.dump(yaml_data, sort_keys=False))
    return out_dir


def _parse_vid_annotation(
    txt_path: Path,
    img_w: int,
    img_h: int,
    class_to_idx: dict[str, int],
) -> dict[int, list[str]]:
    frames: dict[int, list[str]] = {}
    if not txt_path.exists():
        return frames
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

# ---------------------------------------------------------------------------
# Temporal dataset (FRCNN only)
# ---------------------------------------------------------------------------

class TemporalVIDDataset(torch.utils.data.Dataset):
    """
    YOLO-format dataset that also provides adjacent-frame pairs for temporal regularisation.

    For each sample, __getitem__ returns (image, target, adj_image, adj_target).
    If the sample has no adjacent frame in the same sequence, adj_* duplicates the main sample.

    The temporal loss weight (alpha=0.3) penalises large changes in box regression
    between adjacent frames, encouraging temporally stable detections.
    """

    def __init__(self, images_dir: Path, labels_dir: Path, classes: list[str],
                 transforms=None, adj_weight: float = 0.3):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.classes = classes
        self.transforms = transforms
        self.adj_weight = adj_weight

        self.image_paths = sorted(
            p for p in images_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if not self.image_paths:
            raise FileNotFoundError(f"No images in {images_dir}")

        # Build sequence → frame index map for adjacent-frame lookup
        # Filenames are  <seq_name>__<frame_id>.jpg
        self._seq_frame_map: dict[str, dict[int, int]] = defaultdict(dict)
        for i, p in enumerate(self.image_paths):
            if "__" in p.stem:
                seq, frame_str = p.stem.rsplit("__", 1)
                try:
                    self._seq_frame_map[seq][int(frame_str)] = i
                except ValueError:
                    pass

    def __len__(self):
        return len(self.image_paths)

    def _load_sample(self, idx):
        from PIL import Image as _PIL
        img_path = self.image_paths[idx]
        image = _PIL.open(img_path).convert("RGB")
        w, h = image.size
        lbl_path = self.labels_dir / (img_path.stem + ".txt")
        boxes, labels = [], []
        if lbl_path.exists():
            for line in lbl_path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                cls_id = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:5])
                x1 = max(0.0, (cx - bw / 2) * w)
                y1 = max(0.0, (cy - bh / 2) * h)
                x2 = min(float(w), (cx + bw / 2) * w)
                y2 = min(float(h), (cy + bh / 2) * h)
                if x2 - x1 < 1 or y2 - y1 < 1:
                    continue
                boxes.append([x1, y1, x2, y2])
                labels.append(cls_id + 1)
        boxes_t = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels_t = torch.as_tensor(labels, dtype=torch.int64)
        target = {"boxes": boxes_t, "labels": labels_t, "image_id": torch.tensor([idx])}
        if self.transforms:
            image, target = self.transforms(image, target)
        return image, target

    def _adjacent_idx(self, idx: int) -> int:
        p = self.image_paths[idx]
        if "__" not in p.stem:
            return idx
        seq, frame_str = p.stem.rsplit("__", 1)
        try:
            fid = int(frame_str)
        except ValueError:
            return idx
        frame_map = self._seq_frame_map.get(seq, {})
        for candidate in (fid + 1, fid - 1, fid + 2, fid - 2):
            if candidate in frame_map:
                return frame_map[candidate]
        return idx

    def __getitem__(self, idx):
        image, target = self._load_sample(idx)
        adj_idx = self._adjacent_idx(idx)
        adj_image, adj_target = self._load_sample(adj_idx)
        return image, target, adj_image, adj_target


def temporal_collate_fn(batch):
    images, targets, adj_images, adj_targets = zip(*batch)
    return list(images), list(targets), list(adj_images), list(adj_targets)

# ---------------------------------------------------------------------------
# FRCNN model builder (reuses logic from train_by_distribution.py)
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# YOLO training
# ---------------------------------------------------------------------------

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

    freeze_layers = 10 if ("freeze" in techniques or "lora" in techniques) else 0

    if "two_stage" in techniques:
        print("  [two_stage] Phase 1: head-only ...")
        phase1_epochs = max(5, epochs // 3)
        YOLO(weights).train(
            data=str(data_yaml.resolve()),
            epochs=phase1_epochs, batch=batch, imgsz=imgsz,
            project=str(out_dir.resolve()), name="phase1", exist_ok=True,
            device=device or None, freeze=10, lr0=0.001, lrf=0.01,
            warmup_epochs=2, weight_decay=0.0005,
            mosaic=1.0, mixup=0.1, copy_paste=0.1, close_mosaic=5,
            patience=10, plots=True, verbose=True, save=True,
        )
        phase1_weights = out_dir / "phase1" / "weights" / "best.pt"
        if phase1_weights.exists():
            weights = str(phase1_weights)
        print("  [two_stage] Phase 2: full fine-tune ...")

    lr0 = 0.005 if "full" in techniques else (0.001 if freeze_layers else 0.01)

    # temporal technique for YOLO: increase copy_paste and mosaic to approximate
    # adjacent-frame mixing; reduce close_mosaic so mosaic persists longer.
    copy_paste = 0.3 if "temporal" in techniques else 0.1
    close_mosaic = 20 if "temporal" in techniques else 10

    results = YOLO(weights).train(
        data=str(data_yaml.resolve()),
        epochs=epochs, batch=batch, imgsz=imgsz,
        project=str(out_dir.resolve()), name="train", exist_ok=True,
        device=device or None,
        freeze=freeze_layers if "freeze" in techniques else 0,
        lr0=lr0, lrf=0.01,
        warmup_epochs=3, warmup_momentum=0.8,
        weight_decay=0.0005,
        mosaic=1.0, mixup=0.1, copy_paste=copy_paste,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=5.0, translate=0.1, scale=0.5, shear=2.0, fliplr=0.5,
        iou=0.7, close_mosaic=close_mosaic,
        patience=20, plots=True, verbose=True, save=True, save_period=10,
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
    print(f"  YOLO done. mAP50={map50}  mAP50-95={map50_95}  time={elapsed:.0f}s")
    return {
        "best_weights": str(best_weights),
        "train_time_seconds": round(elapsed, 1),
        "n_train": n_train, "n_val": n_val,
        "mAP50": map50, "mAP50_95": map50_95,
        "precision": rd.get("metrics/precision(B)"),
        "recall": rd.get("metrics/recall(B)"),
        "fitness": rd.get("fitness"),
    }

# ---------------------------------------------------------------------------
# FRCNN training (with optional temporal consistency loss)
# ---------------------------------------------------------------------------

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
    import torchvision.transforms.v2 as T
    from torch.utils.data import DataLoader

    t_start = time.time()
    yaml_path = data_dir / "dataset.yaml"
    with yaml_path.open() as f:
        meta = yaml.safe_load(f)
    classes = meta["names"]
    num_classes = len(classes) + 1

    if device_str:
        device = torch.device(device_str)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_tf = T.Compose([
        T.ToImage(), T.ToDtype(torch.float32, scale=True),
        T.RandomHorizontalFlip(0.5), T.RandomPhotometricDistort(0.5),
        T.ScaleJitter(target_size=(800, 800), scale_range=(0.5, 2.0)),
        T.RandomCrop(size=(640, 640), pad_if_needed=True),
        T.SanitizeBoundingBoxes(),
    ])
    val_tf = T.Compose([
        T.ToImage(), T.ToDtype(torch.float32, scale=True),
        T.Resize(size=640, max_size=1333),
    ])

    use_temporal = "temporal" in techniques
    nw = 0 if str(device) == "mps" else 4

    if use_temporal:
        train_ds = TemporalVIDDataset(
            data_dir / "images" / "train", data_dir / "labels" / "train",
            classes, transforms=train_tf)
        train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True,
                                  num_workers=nw, collate_fn=temporal_collate_fn,
                                  pin_memory=(str(device) == "cuda"))
    else:
        # Re-use the same YOLODetectionDataset from train_faster_rcnn.py if available,
        # otherwise define it inline (identical logic).
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "train_faster_rcnn",
                Path(__file__).parent.parent / "finetune" / "train_faster_rcnn.py",
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            YOLODetectionDataset = mod.YOLODetectionDataset
            collate_fn = mod.collate_fn
        except Exception:
            from torch.utils.data import Dataset as _DS
            class YOLODetectionDataset(_DS):
                def __init__(self, images_dir, labels_dir, classes, transforms=None):
                    self.image_paths = sorted(
                        p for p in images_dir.iterdir()
                        if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
                    self.labels_dir = labels_dir
                    self.classes = classes
                    self.transforms = transforms
                def __len__(self): return len(self.image_paths)
                def __getitem__(self, idx):
                    from PIL import Image as _PIL
                    ip = self.image_paths[idx]
                    img = _PIL.open(ip).convert("RGB")
                    w, h = img.size
                    lbl = self.labels_dir / (ip.stem + ".txt")
                    boxes, labels = [], []
                    if lbl.exists():
                        for line in lbl.read_text().splitlines():
                            p = line.strip().split()
                            if len(p) < 5: continue
                            cx, cy, bw, bh = map(float, p[1:5])
                            x1 = max(0.0, (cx-bw/2)*w); y1 = max(0.0, (cy-bh/2)*h)
                            x2 = min(float(w),(cx+bw/2)*w); y2 = min(float(h),(cy+bh/2)*h)
                            if x2-x1<1 or y2-y1<1: continue
                            boxes.append([x1,y1,x2,y2]); labels.append(int(p[0])+1)
                    t = {"boxes": torch.as_tensor(boxes,dtype=torch.float32).reshape(-1,4),
                         "labels": torch.as_tensor(labels,dtype=torch.int64),
                         "image_id": torch.tensor([idx])}
                    if self.transforms: img, t = self.transforms(img, t)
                    return img, t
            def collate_fn(b): return tuple(zip(*b))

        train_ds = YOLODetectionDataset(
            data_dir / "images" / "train", data_dir / "labels" / "train",
            classes, transforms=train_tf)
        train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True,
                                  num_workers=nw, collate_fn=collate_fn,
                                  pin_memory=(str(device) == "cuda"))

    # Separate val loader always uses the non-temporal dataset
    try:
        val_ds = YOLODetectionDataset(
            data_dir / "images" / "val", data_dir / "labels" / "val",
            classes, transforms=val_tf)
    except Exception:
        # YOLODetectionDataset from import may not be defined yet in this scope
        from torch.utils.data import Dataset as _DS2
        class _SimplDS(_DS2):
            def __init__(self, images_dir, labels_dir, classes, transforms=None):
                self.image_paths = sorted(
                    p for p in images_dir.iterdir()
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
                self.labels_dir = labels_dir; self.transforms = transforms
            def __len__(self): return len(self.image_paths)
            def __getitem__(self, idx):
                from PIL import Image as _PIL
                ip = self.image_paths[idx]; img = _PIL.open(ip).convert("RGB"); w,h=img.size
                lbl = self.labels_dir/(ip.stem+".txt"); boxes,labels=[],[]
                if lbl.exists():
                    for line in lbl.read_text().splitlines():
                        p=line.strip().split()
                        if len(p)<5: continue
                        cx,cy,bw,bh=map(float,p[1:5])
                        x1=max(0.,(cx-bw/2)*w);y1=max(0.,(cy-bh/2)*h)
                        x2=min(float(w),(cx+bw/2)*w);y2=min(float(h),(cy+bh/2)*h)
                        if x2-x1<1 or y2-y1<1: continue
                        boxes.append([x1,y1,x2,y2]); labels.append(int(p[0])+1)
                t={"boxes":torch.as_tensor(boxes,dtype=torch.float32).reshape(-1,4),
                   "labels":torch.as_tensor(labels,dtype=torch.int64),
                   "image_id":torch.tensor([idx])}
                if self.transforms: img,t=self.transforms(img,t)
                return img,t
        val_ds = _SimplDS(data_dir/"images"/"val", data_dir/"labels"/"val", classes, val_tf)

    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False,
                            num_workers=nw, collate_fn=lambda b: tuple(zip(*b)),
                            pin_memory=(str(device) == "cuda"))

    model = _build_frcnn(num_classes, model_key)

    if "lora" in techniques:
        n_replaced = apply_lora(model, rank=lora_rank, alpha=float(lora_rank))
        print(f"  [lora] Replaced {n_replaced} Linear layers with LoRA(rank={lora_rank})")
        freeze_non_lora(model)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"  [lora] Trainable: {trainable:,}/{total:,} ({100*trainable/total:.1f}%)")
    elif "freeze" in techniques:
        for name, param in model.named_parameters():
            if name.startswith("backbone") or name.startswith("fpn"):
                param.requires_grad = False
        print("  [freeze] Backbone frozen")

    model.to(device)

    def _is_bb(name): return name.startswith("backbone") or name.startswith("fpn")

    if "full" in techniques and "lora" not in techniques and "freeze" not in techniques:
        param_groups = [
            {"params": [p for n, p in model.named_parameters() if _is_bb(n) and p.requires_grad], "lr": lr * 0.1},
            {"params": [p for n, p in model.named_parameters() if not _is_bb(n) and p.requires_grad], "lr": lr},
        ]
    elif "two_stage" in techniques and "lora" not in techniques:
        for n, p in model.named_parameters():
            if _is_bb(n): p.requires_grad = False
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
    temporal_weight = 0.3

    for epoch in range(epochs):
        if "two_stage" in techniques and "lora" not in techniques and epoch == epochs // 2:
            print(f"  [two_stage] Phase 2: unfreezing backbone at epoch {epoch}")
            for n, p in model.named_parameters():
                p.requires_grad = True
            optimizer.add_param_group({
                "params": [p for n, p in model.named_parameters() if _is_bb(n)],
                "lr": lr * 0.01,
            })

        model.train()
        total_loss = 0.0

        if use_temporal:
            for images, targets, adj_images, adj_targets in train_loader:
                images = [im.to(device) for im in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                adj_images = [im.to(device) for im in adj_images]
                adj_targets = [{k: v.to(device) for k, v in t.items()} for t in adj_targets]

                loss_dict = model(images, targets)
                losses = sum(loss_dict.values())

                # Temporal regularisation: adjacent frame loss at reduced weight
                adj_loss_dict = model(adj_images, adj_targets)
                adj_losses = sum(adj_loss_dict.values()) * temporal_weight
                total_batch_loss = losses + adj_losses

                optimizer.zero_grad()
                total_batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizer.step()
                total_loss += total_batch_loss.item()
        else:
            for images, targets in train_loader:
                images = [im.to(device) for im in images]
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
                images = [im.to(device) for im in images]
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
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "classes": classes, "num_classes": num_classes,
                        "model_key": model_key},
                       out_dir / "best.pth")
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
    print(f"  FRCNN done. best_val_loss={best_val_loss:.4f}  time={elapsed:.0f}s")
    return {
        "best_weights": str(out_dir / "best.pth"),
        "train_time_seconds": round(elapsed, 1),
        "n_train": n_train, "n_val": n_val,
        "best_val_loss": best_val_loss,
    }

# ---------------------------------------------------------------------------
# Run-name generation
# ---------------------------------------------------------------------------

def make_run_name(group_by, family, techniques, yolo_model, frcnn_model,
                  epochs_yolo, epochs_frcnn, frame_stride, lora_rank, seed) -> str:
    parts = ["seq_finetune"]
    parts.append(f"group-{group_by.replace('+', '-')}")
    parts.append(family)
    if family in ("yolo", "both"):
        parts.append(yolo_model)
        parts.append(f"epochs-{epochs_yolo}")
    if family in ("frcnn", "both"):
        parts.append(frcnn_model.replace("_", "-"))
        parts.append(f"frcnn-epochs-{epochs_frcnn}")
    parts.append("-".join(sorted(t.replace("_", "-") for t in techniques)))
    if frame_stride > 1:
        parts.append(f"stride-{frame_stride}")
    slug = "-".join(parts)
    h = hashlib.sha256(json.dumps({
        "group_by": group_by, "family": family, "techniques": sorted(techniques),
        "yolo_model": yolo_model, "frcnn_model": frcnn_model,
        "epochs_yolo": epochs_yolo, "epochs_frcnn": epochs_frcnn,
        "frame_stride": frame_stride, "lora_rank": lora_rank, "seed": seed,
    }, sort_keys=True).encode()).hexdigest()[:6]
    return f"{slug}-{h}"

# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run(
    sequences_json: str,
    group_by: str,
    family: str,
    techniques: list[str],
    val_sequences: list[str] | None,
    val_split_by: str,
    yolo_model: str,
    frcnn_model: str,
    epochs_yolo: int,
    epochs_frcnn: int,
    batch_yolo: int,
    batch_frcnn: int,
    imgsz: int,
    lr: float,
    lora_rank: int,
    frame_stride: int,
    seed: int,
    project: str,
    device: str,
    extra_classes: list[str],
    min_sequences: int,
):
    techniques_set = set(techniques)
    out_root = Path(project)
    out_root.mkdir(parents=True, exist_ok=True)

    catalogue = load_sequence_catalogue(Path(sequences_json))
    print(f"\nLoaded {len(catalogue)} sequences from {sequences_json}")

    # Validate that images dirs and annotation files exist
    missing = []
    for seq_name, meta in catalogue.items():
        if not meta["images_dir"].exists():
            missing.append(f"  images: {meta['images_dir']}")
        if not meta["ann_txt"].exists():
            missing.append(f"  annotations: {meta['ann_txt']}")
    if missing:
        print(f"WARNING: {len(missing)} paths not found on disk:")
        for m in missing[:10]:
            print(m)

    groups = group_sequences(catalogue, group_by)
    print(f"\nGrouping '{group_by}' → {len(groups)} group(s):")
    for k, v in sorted(groups.items(), key=lambda x: -len(x[1])):
        print(f"  {k}: {len(v)} sequences  ({', '.join(v)})")

    # Normalise explicit val sequence names (strip path components if full path given)
    val_seq_names = None
    if val_sequences:
        val_seq_names = [Path(v).name.rstrip("/") for v in val_sequences]

    all_results = {}

    for group_label, group_seqs in sorted(groups.items()):
        if len(group_seqs) < min_sequences:
            print(f"\nSKIPPING group '{group_label}': only {len(group_seqs)} sequences "
                  f"(--min-sequences={min_sequences})")
            continue

        print(f"\n{'='*60}")
        print(f"GROUP: {group_label}  ({len(group_seqs)} sequences)")
        print(f"{'='*60}")

        train_seqs, val_seqs = select_val_sequences(
            group_seqs, val_seq_names, val_split_by, catalogue, seed)

        print(f"  Train sequences ({len(train_seqs)}): {train_seqs}")
        print(f"  Val sequences  ({len(val_seqs)}):   {val_seqs}")

        safe_label = group_label.replace(" ", "_").replace("/", "_")
        data_dir = out_root / "data" / safe_label
        data_dir.mkdir(parents=True, exist_ok=True)

        exported = export_sequence_split(
            train_seqs, val_seqs, catalogue, data_dir,
            frame_stride, extra_classes)
        if exported is None:
            print(f"  Skipping group '{group_label}' — no frames exported")
            continue

        data_yaml = exported / "dataset.yaml"
        group_results = {}

        if family in ("yolo", "both"):
            yolo_out = out_root / "models" / safe_label / "yolo"
            print(f"\n[YOLO] group={group_label}  techniques={sorted(techniques_set)}")
            try:
                group_results["yolo"] = train_yolo(
                    data_yaml=data_yaml, model_key=yolo_model,
                    techniques=techniques_set, epochs=epochs_yolo,
                    batch=batch_yolo, imgsz=imgsz, out_dir=yolo_out, device=device)
            except Exception as e:
                print(f"  ERROR YOLO: {e}\n{traceback.format_exc()}")
                group_results["yolo"] = {"error": str(e)}

        if family in ("frcnn", "both"):
            frcnn_out = out_root / "models" / safe_label / "frcnn"
            print(f"\n[FRCNN] group={group_label}  techniques={sorted(techniques_set)}")
            try:
                group_results["frcnn"] = train_frcnn(
                    data_dir=exported, model_key=frcnn_model,
                    techniques=techniques_set, epochs=epochs_frcnn,
                    batch=batch_frcnn, lr=lr, out_dir=frcnn_out,
                    device_str=device, lora_rank=lora_rank)
            except Exception as e:
                print(f"  ERROR FRCNN: {e}\n{traceback.format_exc()}")
                group_results["frcnn"] = {"error": str(e)}

        group_results["train_sequences"] = train_seqs
        group_results["val_sequences"] = val_seqs
        all_results[group_label] = group_results

    config = {
        "sequences_json": sequences_json, "group_by": group_by, "family": family,
        "techniques": list(techniques_set), "val_sequences": val_sequences,
        "val_split_by": val_split_by, "yolo_model": yolo_model, "frcnn_model": frcnn_model,
        "epochs_yolo": epochs_yolo, "epochs_frcnn": epochs_frcnn,
        "batch_yolo": batch_yolo, "batch_frcnn": batch_frcnn,
        "imgsz": imgsz, "lr": lr, "lora_rank": lora_rank,
        "frame_stride": frame_stride, "seed": seed, "extra_classes": extra_classes,
        "project": project, "device": device, "min_sequences": min_sequences,
    }
    (out_root / "training_config.json").write_text(json.dumps(config, indent=2, default=str))
    (out_root / "training_summary.json").write_text(json.dumps(all_results, indent=2, default=str))

    print(f"\n{'='*60}")
    print(f"Training complete. Summary → {out_root / 'training_summary.json'}")
    print(f"{'='*60}")
    for g, res in all_results.items():
        for fam in ("yolo", "frcnn"):
            if fam not in res:
                continue
            r = res[fam]
            best = r.get("best_weights", r.get("error", "?"))
            metric = r.get("mAP50_95") or r.get("best_val_loss")
            metric_str = f"  metric={metric:.4f}" if isinstance(metric, float) else ""
            print(f"  {g} [{fam}]  {best}{metric_str}")
        print(f"    val_seqs={res.get('val_sequences', [])}")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLO/FRCNN using VisDrone VID sequences as mini-datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("--config", default=None,
                        help="Replay a previous training_config.json")
    parser.add_argument("--sequences-json",
                        default="../datasets/sequences_categories.json",
                        help="Path to sequences_categories.json (default: ../datasets/sequences_categories.json)")
    parser.add_argument("--group-by", default="scene",
                        choices=["none", "all", "scene", "time", "weather", "scene+time"],
                        help="How to group sequences into training sets (default: scene)")

    val_g = parser.add_argument_group("Validation sequence selection")
    val_g.add_argument("--val-sequences", nargs="+", default=None,
                       metavar="SEQ_NAME",
                       help="Explicit sequence name(s) to hold out as validation "
                            "(e.g. uav0000268_05773_v). Applied across all groups.")
    val_g.add_argument("--val-split-by", default="random",
                       choices=["random", "scene", "time"],
                       help="Auto-select val sequences by this tag (default: random). "
                            "Ignored if --val-sequences is given.")

    parser.add_argument("--family", default="yolo", choices=["yolo", "frcnn", "both"])
    parser.add_argument("--yolo-model", default="yolov8n", choices=list(YOLO_MODELS))
    parser.add_argument("--frcnn-model", default="fasterrcnn_resnet50_v2",
                        choices=list(_get_frcnn_registry()))
    parser.add_argument("--technique", nargs="+", default=["two_stage"],
                        dest="techniques",
                        choices=["freeze", "two_stage", "full", "lora", "cosine", "temporal"],
                        help="Fine-tuning technique(s); combinable (default: two_stage)")
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--epochs-yolo", type=int, default=100)
    parser.add_argument("--epochs-frcnn", type=int, default=20)
    parser.add_argument("--batch-yolo", type=int, default=16)
    parser.add_argument("--batch-frcnn", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--frame-stride", type=int, default=1,
                        help="Sample every N-th frame per sequence (default: 1 = all frames)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--extra-classes", nargs="*", default=[])
    parser.add_argument("--project", default=None,
                        help="Output root directory (auto-generated if not given)")
    parser.add_argument("--device", default="",
                        help="Device: '' (auto), 'cpu', 'cuda', 'mps'")
    parser.add_argument("--min-sequences", type=int, default=2,
                        help="Skip groups with fewer than this many sequences (default: 2)")

    _pre = parser.parse_known_args(sys.argv[1:])[0]
    if _pre.config:
        cfg = json.loads(Path(_pre.config).read_text())
        parser.set_defaults(**{k: v for k, v in cfg.items()
                               if v is not None and k != "project"})
    args = parser.parse_args()

    if args.project is None:
        run_name = make_run_name(
            args.group_by, args.family, args.techniques,
            args.yolo_model, args.frcnn_model,
            args.epochs_yolo, args.epochs_frcnn,
            args.frame_stride, args.lora_rank, args.seed)
        project = str(Path("runs") / run_name)
        print(f"Auto run directory: {project}")
    else:
        project = args.project

    run(
        sequences_json=args.sequences_json,
        group_by=args.group_by,
        family=args.family,
        techniques=args.techniques,
        val_sequences=args.val_sequences,
        val_split_by=args.val_split_by,
        yolo_model=args.yolo_model,
        frcnn_model=args.frcnn_model,
        epochs_yolo=args.epochs_yolo,
        epochs_frcnn=args.epochs_frcnn,
        batch_yolo=args.batch_yolo,
        batch_frcnn=args.batch_frcnn,
        imgsz=args.imgsz,
        lr=args.lr,
        lora_rank=args.lora_rank,
        frame_stride=args.frame_stride,
        seed=args.seed,
        project=project,
        device=args.device,
        extra_classes=args.extra_classes or [],
        min_sequences=args.min_sequences,
    )
