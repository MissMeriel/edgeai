"""
Validate a fine-tuned YOLO or Faster R-CNN model against the VisDrone test-dev set.

Loads ground-truth bounding boxes from the native VisDrone annotations/ directory,
runs inference with the supplied model weights, then computes per-class and overall
detection metrics: mAP@0.5, mAP@0.5:0.95, precision, recall, F1.

Supported model types (--model-type):
  yolo    — Ultralytics YOLO .pt weights  (YOLOv8/v11/RT-DETR)
  frcnn   — torchvision checkpoint .pth produced by train_faster_rcnn.py or
              train_by_distribution.py  (contains 'model', 'classes', 'model_key')

Usage:
    # Validate a YOLO model:
    python validate.py \\
        --dataset ../datasets/VisDrone2019-DET-test-dev \\
        --weights runs/dist_finetune/models/city_street/yolo/train/weights/best.pt \\
        --model-type yolo

    # Validate a Faster R-CNN checkpoint:
    python validate.py \\
        --dataset ../datasets/VisDrone2019-DET-test-dev \\
        --weights runs/dist_finetune/models/city_street/frcnn/best.pth \\
        --model-type frcnn \\
        --frcnn-arch fasterrcnn_resnet50_v2

    # Adjust IoU thresholds and confidence cutoff:
    python validate.py --dataset ... --weights ... --model-type yolo \\
        --iou-thresholds 0.5 0.75 --conf 0.25

    # Save per-image result JSON:
    python validate.py --dataset ... --weights ... --model-type yolo --save-json

Output (always printed to stdout):
    Per-class AP@0.5, precision, recall, F1
    Overall mAP@0.5 and mAP@0.5:0.95
    Confusion matrix path (if --save-plots)
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import truststore
truststore.inject_into_ssl()
# ---------------------------------------------------------------------------
# VisDrone ground-truth reader  (mirrors train_by_distribution.py)
# ---------------------------------------------------------------------------

VISDRONE_CLASSES = {
    1: "pedestrian",
    2: "people",
    3: "bicycle",
    4: "car",
    5: "van",
    6: "truck",
    7: "tricycle",
    8: "awning-tricycle",
    9: "bus",
    10: "motor",
}
# Ordered list — index in this list == class index used throughout the script
CLASS_NAMES = [VISDRONE_CLASSES[i] for i in range(1, 11)]  # 10 classes


def load_ground_truth(ann_path: Path, img_w: int, img_h: int) -> dict:
    """
    Return {"boxes": np.array (N,4) xyxy absolute, "labels": np.array (N,) int 0-indexed}.
    Skips score=0 (ignored regions), cat_id=0, cat_id=11 (others).
    """
    boxes, labels = [], []
    for row in ann_path.read_text().splitlines():
        row = row.strip()
        if not row:
            continue
        parts = row.split(",")
        if len(parts) < 6:
            continue
        x, y, w, h = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
        score = int(parts[4])
        cat_id = int(parts[5])
        if score == 0 or cat_id == 0 or cat_id == 11:
            continue
        # Convert to xyxy absolute; clamp to image bounds
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(img_w, x + w), min(img_h, y + h)
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        boxes.append([x1, y1, x2, y2])
        labels.append(cat_id - 1)   # 0-indexed: cat_id 1→0, 10→9

    return {
        "boxes": np.array(boxes, dtype=np.float32).reshape(-1, 4),
        "labels": np.array(labels, dtype=np.int64),
    }


# ---------------------------------------------------------------------------
# IoU utilities
# ---------------------------------------------------------------------------

def box_iou(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU between (M,4) and (N,4) xyxy boxes → (M,N)."""
    if boxes_a.shape[0] == 0 or boxes_b.shape[0] == 0:
        return np.zeros((boxes_a.shape[0], boxes_b.shape[0]), dtype=np.float32)

    ax1, ay1, ax2, ay2 = boxes_a[:, 0], boxes_a[:, 1], boxes_a[:, 2], boxes_a[:, 3]
    bx1, by1, bx2, by2 = boxes_b[:, 0], boxes_b[:, 1], boxes_b[:, 2], boxes_b[:, 3]

    inter_x1 = np.maximum(ax1[:, None], bx1[None, :])
    inter_y1 = np.maximum(ay1[:, None], by1[None, :])
    inter_x2 = np.minimum(ax2[:, None], bx2[None, :])
    inter_y2 = np.minimum(ay2[:, None], by2[None, :])

    inter_w = np.maximum(0, inter_x2 - inter_x1)
    inter_h = np.maximum(0, inter_y2 - inter_y1)
    inter = inter_w * inter_h

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a[:, None] + area_b[None, :] - inter

    return np.where(union > 0, inter / union, 0.0).astype(np.float32)


# ---------------------------------------------------------------------------
# AP computation  (11-point and area-under-curve)
# ---------------------------------------------------------------------------

def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """Compute AP as area under the precision-recall curve (COCO-style)."""
    # Append sentinel values
    recalls = np.concatenate([[0.0], recalls, [1.0]])
    precisions = np.concatenate([[1.0], precisions, [0.0]])
    # Make precision monotonically decreasing
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    # Find recall change points
    idx = np.where(recalls[1:] != recalls[:-1])[0] + 1
    return float(np.sum((recalls[idx] - recalls[idx - 1]) * precisions[idx]))


def evaluate_class(
    pred_boxes: list[np.ndarray],   # per-image predicted boxes (xyxy)
    pred_scores: list[np.ndarray],  # per-image scores
    gt_boxes: list[np.ndarray],     # per-image GT boxes (xyxy)
    iou_threshold: float,
    n_images: int,
) -> tuple[float, float, float, float]:
    """
    Compute AP, precision, recall, F1 for one class across all images.
    Each list element corresponds to one image.
    """
    # Flatten all predictions, remembering which image they came from
    all_scores, all_tp, all_fp = [], [], []
    n_gt_total = 0

    for img_idx in range(n_images):
        pb = pred_boxes[img_idx]   # (M, 4)
        ps = pred_scores[img_idx]  # (M,)
        gb = gt_boxes[img_idx]     # (K, 4)

        n_gt_total += len(gb)

        if len(ps) == 0:
            continue

        # Sort predictions by descending score
        order = np.argsort(-ps)
        pb, ps = pb[order], ps[order]

        matched_gt = set()
        for m in range(len(ps)):
            all_scores.append(ps[m])
            if len(gb) == 0:
                all_tp.append(0)
                all_fp.append(1)
                continue
            ious = box_iou(pb[m : m + 1], gb)[0]   # (K,)
            best_j = int(np.argmax(ious))
            if ious[best_j] >= iou_threshold and best_j not in matched_gt:
                all_tp.append(1)
                all_fp.append(0)
                matched_gt.add(best_j)
            else:
                all_tp.append(0)
                all_fp.append(1)

    if n_gt_total == 0 and len(all_scores) == 0:
        return 0.0, 0.0, 0.0, 0.0

    if len(all_scores) == 0:
        return 0.0, 0.0, 0.0, 0.0

    # Sort by score descending
    order = np.argsort(-np.array(all_scores))
    tp_cum = np.cumsum(np.array(all_tp)[order])
    fp_cum = np.cumsum(np.array(all_fp)[order])

    recalls = tp_cum / max(n_gt_total, 1)
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1)

    ap = compute_ap(recalls, precisions)

    # Precision/recall/F1 at the confidence that maximises F1
    f1s = 2 * precisions * recalls / np.maximum(precisions + recalls, 1e-9)
    best = int(np.argmax(f1s))
    return ap, float(precisions[best]), float(recalls[best]), float(f1s[best])


# ---------------------------------------------------------------------------
# mAP@0.5:0.95 (COCO metric)
# ---------------------------------------------------------------------------

def compute_map_range(
    all_pred_boxes: dict[int, list],   # class_idx → per-image pred boxes
    all_pred_scores: dict[int, list],
    all_gt_boxes: dict[int, list],
    n_images: int,
    iou_thresholds: tuple[float, ...] = tuple(np.arange(0.5, 1.0, 0.05)),
) -> dict[str, float]:
    """Compute mAP over a range of IoU thresholds."""
    aps_per_iou = []
    for iou_t in iou_thresholds:
        aps = []
        for cls in range(len(CLASS_NAMES)):
            ap, _, _, _ = evaluate_class(
                all_pred_boxes.get(cls, [np.zeros((0, 4))] * n_images),
                all_pred_scores.get(cls, [np.zeros(0)] * n_images),
                all_gt_boxes.get(cls, [np.zeros((0, 4))] * n_images),
                iou_t, n_images,
            )
            aps.append(ap)
        aps_per_iou.append(np.mean(aps))
    return {
        "mAP_50": float(aps_per_iou[0]),
        "mAP_50_95": float(np.mean(aps_per_iou)),
    }


# ---------------------------------------------------------------------------
# YOLO inference
# ---------------------------------------------------------------------------

def run_yolo(weights: str, image_paths: list[Path], conf: float,
             device: str) -> list[dict]:
    """
    Run YOLO inference on a list of images.
    Returns list of {"boxes": (N,4) xyxy abs, "scores": (N,), "labels": (N,) int} per image.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("ultralytics not found — activate .venv-eai")

    model = YOLO(weights)
    results_out = []

    # Batch inference in chunks to avoid OOM on large test sets
    chunk_size = 32
    for i in tqdm(range(0, len(image_paths), chunk_size), desc="YOLO inference"):
        chunk = [str(p) for p in image_paths[i: i + chunk_size]]
        results = model.predict(
            chunk, conf=conf, device=device or None,
            verbose=False, save=False, stream=False,
        )
        for r in results:
            boxes = r.boxes
            if boxes is None or len(boxes) == 0:
                results_out.append({
                    "boxes": np.zeros((0, 4), dtype=np.float32),
                    "scores": np.zeros(0, dtype=np.float32),
                    "labels": np.zeros(0, dtype=np.int64),
                })
            else:
                results_out.append({
                    "boxes": boxes.xyxy.cpu().numpy().astype(np.float32),
                    "scores": boxes.conf.cpu().numpy().astype(np.float32),
                    "labels": boxes.cls.cpu().numpy().astype(np.int64),
                })

    return results_out


# ---------------------------------------------------------------------------
# Faster R-CNN inference
# ---------------------------------------------------------------------------

def run_frcnn(weights: str, frcnn_arch: str, image_paths: list[Path],
              conf: float, device_str: str) -> list[dict]:
    """
    Run torchvision FRCNN-family inference.
    Loads checkpoint saved by train_faster_rcnn.py / train_by_distribution.py.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "train_faster_rcnn",
        Path(__file__).parent / "train_faster_rcnn.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    if device_str:
        device = torch.device(device_str)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    ckpt = torch.load(weights, map_location=device)
    classes = ckpt.get("classes", CLASS_NAMES)
    num_classes = ckpt.get("num_classes", len(classes) + 1)
    arch = ckpt.get("model_key", frcnn_arch)

    model = mod.build_model(num_classes, model_key=arch, pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    import torchvision.transforms.v2 as T
    transform = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])

    results_out = []
    for img_path in tqdm(image_paths, desc="FRCNN inference"):
        img = Image.open(img_path).convert("RGB")
        img_t = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            preds = model(img_t)[0]

        keep = preds["scores"].cpu().numpy() >= conf
        results_out.append({
            "boxes": preds["boxes"].cpu().numpy()[keep].astype(np.float32),
            "scores": preds["scores"].cpu().numpy()[keep].astype(np.float32),
            # torchvision uses 1-indexed labels (0 = background); subtract 1
            "labels": (preds["labels"].cpu().numpy()[keep] - 1).astype(np.int64),
        })

    return results_out


# ---------------------------------------------------------------------------
# Tag-based grouping
# ---------------------------------------------------------------------------

SPLIT_KEYS = ("scene", "time", "weather", "quality")
UNTAGGED = "__untagged__"


def load_tags(tags_path: Path) -> dict[str, dict]:
    """Load tags.json → {filename: {scene, time, weather, quality, ...}}"""
    with tags_path.open() as f:
        return json.load(f)


def build_tag_groups(
    image_paths: list[Path],
    tags: dict[str, dict],
    split_by: str,
) -> dict[str, list[int]]:
    """
    Return {tag_value: [image_index, ...]} for the chosen split_by key.
    Images absent from tags.json or missing the key go into UNTAGGED.
    """
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, p in enumerate(image_paths):
        val = tags.get(p.name, {}).get(split_by, UNTAGGED)
        groups[val].append(idx)
    return dict(groups)


# ---------------------------------------------------------------------------
# Per-subset metric computation (operates on index subsets, not copies)
# ---------------------------------------------------------------------------

def compute_subset_metrics(
    indices: list[int],
    predictions: list[dict],
    gt_all: list[dict],
    iou_threshold: float,
) -> tuple[dict, float, float]:
    """
    Compute per-class metrics and mAP for a subset of images identified by index.
    Returns (class_metrics_dict, mAP_50, mAP_50_95).
    """
    n = len(indices)

    pred_boxes_by_class: dict[int, list[np.ndarray]] = {}
    pred_scores_by_class: dict[int, list[np.ndarray]] = {}
    gt_boxes_by_class: dict[int, list[np.ndarray]] = {}

    for cls in range(len(CLASS_NAMES)):
        pred_boxes_by_class[cls] = []
        pred_scores_by_class[cls] = []
        gt_boxes_by_class[cls] = []
        for idx in indices:
            pred = predictions[idx]
            gt = gt_all[idx]
            pmask = pred["labels"] == cls
            gmask = gt["labels"] == cls
            pred_boxes_by_class[cls].append(pred["boxes"][pmask])
            pred_scores_by_class[cls].append(pred["scores"][pmask])
            gt_boxes_by_class[cls].append(gt["boxes"][gmask])

    class_metrics = {}
    for cls, name in enumerate(CLASS_NAMES):
        ap, prec, rec, f1 = evaluate_class(
            pred_boxes_by_class[cls],
            pred_scores_by_class[cls],
            gt_boxes_by_class[cls],
            iou_threshold=iou_threshold,
            n_images=n,
        )
        n_gt = sum(len(b) for b in gt_boxes_by_class[cls])
        n_pred = sum(len(b) for b in pred_boxes_by_class[cls])
        class_metrics[name] = {
            "AP_50": round(ap, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "F1": round(f1, 4),
            "n_gt": int(n_gt),
            "n_pred": int(n_pred),
        }

    iou_range = tuple(float(f"{v:.2f}") for v in np.arange(0.5, 1.0, 0.05))
    map_result = compute_map_range(
        pred_boxes_by_class, pred_scores_by_class, gt_boxes_by_class,
        n_images=n, iou_thresholds=iou_range,
    )
    return class_metrics, map_result["mAP_50"], map_result["mAP_50_95"]


def _print_metrics_table(label: str, class_metrics: dict, map50: float, map50_95: float):
    print(f"\n{'='*72}")
    print(f"  {label}")
    print(f"{'='*72}")
    print(f"{'Class':22s} {'AP@50':>8} {'Prec':>8} {'Recall':>8} {'F1':>8} {'GT':>6} {'Pred':>6}")
    print(f"{'-'*72}")
    for name, m in class_metrics.items():
        print(f"{name:22s} {m['AP_50']:8.4f} {m['precision']:8.4f} "
              f"{m['recall']:8.4f} {m['F1']:8.4f} {m['n_gt']:6d} {m['n_pred']:6d}")
    print(f"{'-'*72}")
    print(f"{'mAP@0.50':22s} {map50:8.4f}")
    print(f"{'mAP@0.50:0.95':22s} {map50_95:8.4f}")
    print(f"{'='*72}")


# ---------------------------------------------------------------------------
# Main validation routine
# ---------------------------------------------------------------------------

def validate(
    dataset_dir: str,
    weights: str,
    model_type: str,
    frcnn_arch: str,
    conf: float,
    iou_threshold: float,
    device: str,
    save_json: bool,
    save_plots: bool,
    output_dir: str,
    max_images: int | None,
    tags_path: str | None,
    split_by: str | None,
):
    dataset_path = Path(dataset_dir)
    images_dir = dataset_path / "images"
    ann_dir = dataset_path / "annotations"

    if not images_dir.exists():
        sys.exit(f"images/ not found in {dataset_path}")
    if not ann_dir.exists():
        sys.exit(f"annotations/ not found in {dataset_path}")

    image_paths = sorted(
        p for p in images_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if max_images:
        image_paths = image_paths[:max_images]

    n_images = len(image_paths)
    print(f"\nDataset : {dataset_path}")
    print(f"Images  : {n_images}")
    print(f"Weights : {weights}")
    print(f"Model   : {model_type}  conf≥{conf}  IoU@{iou_threshold}")
    if tags_path and split_by:
        print(f"Split   : {split_by}  (tags: {tags_path})")
    print(f"Classes : {CLASS_NAMES}\n")

    # ---- Load tags (optional) --------------------------------------------
    tags: dict[str, dict] = {}
    if tags_path:
        tags = load_tags(Path(tags_path))
        print(f"Loaded tags for {len(tags)} images from {tags_path}")

    # ---- Load ground truth -----------------------------------------------
    print("Loading ground truth annotations...")
    gt_all: list[dict] = []
    for img_path in tqdm(image_paths, desc="GT"):
        ann_path = ann_dir / (img_path.stem + ".txt")
        with Image.open(img_path) as im:
            img_w, img_h = im.size
        if ann_path.exists():
            gt = load_ground_truth(ann_path, img_w, img_h)
        else:
            gt = {"boxes": np.zeros((0, 4), dtype=np.float32),
                  "labels": np.zeros(0, dtype=np.int64)}
        gt_all.append(gt)

    total_gt = sum(len(g["labels"]) for g in gt_all)
    print(f"Total GT boxes: {total_gt}")

    # ---- Run inference (once over all images) ----------------------------
    t0 = time.time()
    if model_type == "yolo":
        predictions = run_yolo(weights, image_paths, conf=conf, device=device)
    else:
        predictions = run_frcnn(weights, frcnn_arch, image_paths,
                                conf=conf, device_str=device)
    elapsed = time.time() - t0
    print(f"\nInference done in {elapsed:.1f}s  "
          f"({elapsed/n_images*1000:.1f} ms/image)")

    for pred in predictions:
        pred["labels"] = np.clip(pred["labels"], 0, len(CLASS_NAMES) - 1)

    # ---- Build groups ----------------------------------------------------
    # Always evaluate the full set; additionally evaluate per tag group if requested
    all_indices = list(range(n_images))
    groups: dict[str, list[int]] = {"ALL": all_indices}

    if tags and split_by:
        tag_groups = build_tag_groups(image_paths, tags, split_by)
        # Sort groups by size descending so the biggest ones print first
        for tag_val, idxs in sorted(tag_groups.items(), key=lambda x: -len(x[1])):
            groups[f"{split_by}={tag_val}"] = idxs

    # ---- Evaluate each group ---------------------------------------------
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    all_group_results = {}

    for group_label, indices in groups.items():
        n = len(indices)
        class_metrics, map50, map50_95 = compute_subset_metrics(
            indices, predictions, gt_all, iou_threshold,
        )
        _print_metrics_table(f"{group_label}  (n={n})", class_metrics, map50, map50_95)

        all_group_results[group_label] = {
            "n_images": n,
            "mAP_50": round(map50, 4),
            "mAP_50_95": round(map50_95, 4),
            "per_class": class_metrics,
        }

        if save_plots:
            safe = group_label.replace("=", "_").replace(" ", "_")
            _save_plots(class_metrics, map50, map50_95,
                        out_path, filename_prefix=safe)

    # ---- Grouped summary table (if split was used) -----------------------
    if tags and split_by and len(groups) > 1:
        print(f"\n{'='*72}")
        print(f"  SUMMARY BY {split_by.upper()}")
        print(f"{'='*72}")
        print(f"{'Group':35s} {'n':>5} {'mAP@50':>8} {'mAP@50:95':>10}")
        print(f"{'-'*72}")
        for label, res in all_group_results.items():
            print(f"{label:35s} {res['n_images']:5d} "
                  f"{res['mAP_50']:8.4f} {res['mAP_50_95']:10.4f}")
        print(f"{'='*72}\n")

    # ---- Per-image JSON (optional) ---------------------------------------
    if save_json:
        per_image = []
        for img_path, pred, gt in zip(image_paths, predictions, gt_all):
            img_tags = tags.get(img_path.name, {})
            per_image.append({
                "image": img_path.name,
                "tags": img_tags,
                "n_gt": int(len(gt["labels"])),
                "n_pred": int(len(pred["labels"])),
                "predictions": [
                    {"box": pred["boxes"][i].tolist(),
                     "score": float(pred["scores"][i]),
                     "label": CLASS_NAMES[int(pred["labels"][i])]}
                    for i in range(len(pred["labels"]))
                ],
                "ground_truth": [
                    {"box": gt["boxes"][i].tolist(),
                     "label": CLASS_NAMES[int(gt["labels"][i])]}
                    for i in range(len(gt["labels"]))
                ],
            })
        detail_path = out_path / "validation_per_image.json"
        detail_path.write_text(json.dumps(per_image, indent=2))
        print(f"Per-image results saved → {detail_path}")

    # ---- Save summary JSON -----------------------------------------------
    summary = {
        "weights": weights,
        "model_type": model_type,
        "dataset": str(dataset_path),
        "n_images": n_images,
        "conf_threshold": conf,
        "iou_threshold": iou_threshold,
        "tags_file": tags_path,
        "split_by": split_by,
        "inference_seconds": round(elapsed, 2),
        "groups": all_group_results,
    }
    summary_path = out_path / "validation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Summary saved → {summary_path}")

    return summary


# ---------------------------------------------------------------------------
# Optional plots
# ---------------------------------------------------------------------------

def _save_plots(class_metrics: dict, map50: float, map50_95: float, out_path: Path,
                filename_prefix: str = "validation"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plots")
        return

    names = list(class_metrics.keys())
    aps = [class_metrics[n]["AP_50"] for n in names]
    precs = [class_metrics[n]["precision"] for n in names]
    recs = [class_metrics[n]["recall"] for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, vals, title, color in zip(
        axes,
        [aps, precs, recs],
        ["AP@0.50", "Precision", "Recall"],
        ["steelblue", "seagreen", "tomato"],
    ):
        bars = ax.barh(names, vals, color=color, alpha=0.8)
        ax.set_xlim(0, 1)
        ax.set_xlabel(title)
        ax.set_title(title)
        for bar, v in zip(bars, vals):
            ax.text(min(v + 0.01, 0.95), bar.get_y() + bar.get_height() / 2,
                    f"{v:.3f}", va="center", fontsize=8)

    fig.suptitle(f"mAP@0.50={map50:.4f}  mAP@0.50:0.95={map50_95:.4f}", fontsize=12)
    plt.tight_layout()
    plot_path = out_path / f"{filename_prefix}_metrics.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Metrics plot saved → {plot_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate a fine-tuned model against the VisDrone test-dev set",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("--dataset",
                        default="../datasets/VisDrone2019-DET-test-dev",
                        help="VisDrone dataset root with images/ and annotations/ "
                             "(default: ../datasets/VisDrone2019-DET-test-dev)")
    parser.add_argument("--weights", required=True,
                        help="Path to model weights (.pt for YOLO, .pth for FRCNN)")
    parser.add_argument("--model-type", required=True, choices=["yolo", "frcnn"],
                        help="Model family: 'yolo' or 'frcnn'")
    parser.add_argument("--frcnn-arch", default="fasterrcnn_resnet50_v2",
                        help="Torchvision architecture key — only needed if the .pth "
                             "checkpoint does not store model_key "
                             "(default: fasterrcnn_resnet50_v2)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold for predictions (default: 0.25)")
    parser.add_argument("--iou", type=float, default=0.5, dest="iou_threshold",
                        help="IoU threshold for TP/FP matching (default: 0.5)")
    parser.add_argument("--device", default="",
                        help="Device: '' (auto), 'cpu', 'cuda', 'cuda:0', 'mps'")
    parser.add_argument("--output-dir", default="runs/validation",
                        help="Directory for output files (default: runs/validation)")
    parser.add_argument("--save-json", action="store_true",
                        help="Save per-image predictions and GT to JSON")
    parser.add_argument("--save-plots", action="store_true",
                        help="Save per-class metrics bar chart as PNG "
                             "(one chart per group when --split-by is used)")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Limit evaluation to first N images (useful for quick checks)")
    parser.add_argument("--tags", default=None, dest="tags_path",
                        metavar="TAGS_JSON",
                        help="Path to tags.json with per-image scene/time/weather/quality labels "
                             "(e.g. datasets/VisDrone2019-DET-test-dev/tags.json)")
    parser.add_argument("--split-by", default=None,
                        choices=list(SPLIT_KEYS),
                        help="Tag dimension to split results by — requires --tags "
                             "(choices: scene, time, weather, quality)")

    args = parser.parse_args()

    if args.split_by and not args.tags_path:
        parser.error("--split-by requires --tags")

    validate(
        dataset_dir=args.dataset,
        weights=args.weights,
        model_type=args.model_type,
        frcnn_arch=args.frcnn_arch,
        conf=args.conf,
        iou_threshold=args.iou_threshold,
        device=args.device,
        save_json=args.save_json,
        save_plots=args.save_plots,
        output_dir=args.output_dir,
        max_images=args.max_images,
        tags_path=args.tags_path,
        split_by=args.split_by,
    )
