"""
Validate a trained model against a held-out VisDrone VID sequence.

Computes per-class and overall mAP50, mAP50-95, precision, and recall
by running inference on every frame of the sequence (or a strided subset)
and comparing against the VID annotation file.

Supports YOLO (.pt via Ultralytics) and FRCNN (.pth via torchvision).
Both model types are auto-detected from the file extension / checkpoint keys.

Usage:

  # Validate a YOLO best.pt against a single sequence:
  python validate_sequence.py \\
      --weights runs/seq_finetune.../models/scene/yolo/train/weights/best.pt \\
      --sequence ../datasets/VisDrone2019-VID-val/sequences/uav0000268_05773_v \\
      --annotation ../datasets/VisDrone2019-VID-val/annotations/uav0000268_05773_v.txt

  # Validate FRCNN best.pth:
  python validate_sequence.py \\
      --weights runs/seq_finetune.../models/scene/frcnn/best.pth \\
      --sequence ../datasets/VisDrone2019-VID-val/sequences/uav0000268_05773_v \\
      --annotation ../datasets/VisDrone2019-VID-val/annotations/uav0000268_05773_v.txt

  # Validate all val sequences recorded in a training_summary.json:
  python validate_sequence.py \\
      --summary runs/seq_finetune.../training_summary.json \\
      --sequences-json ../datasets/sequences_categories.json

  # Save annotated frames to disk for visual inspection:
  python validate_sequence.py \\
      --weights best.pt \\
      --sequence .../uav0000268_05773_v \\
      --annotation .../uav0000268_05773_v.txt \\
      --save-frames

  # Limit to every 5th frame to save time:
  python validate_sequence.py \\
      --weights best.pt \\
      --sequence .../uav0000268_05773_v \\
      --annotation .../uav0000268_05773_v.txt \\
      --frame-stride 5
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

# VisDrone class map
VISDRONE_CLASSES = {
    1: "pedestrian", 2: "people", 3: "bicycle", 4: "car",
    5: "van", 6: "truck", 7: "tricycle", 8: "awning-tricycle",
    9: "bus", 10: "motor",
}
CLASS_NAMES = list(VISDRONE_CLASSES.values())

# ---------------------------------------------------------------------------
# Annotation parsing
# ---------------------------------------------------------------------------

def load_sequence_gt(ann_txt: Path, img_w: int, img_h: int) -> dict[int, list[dict]]:
    """
    Parse a VID annotation file into:
      { frame_id: [ {label, x1, y1, x2, y2}, ... ] }
    Ignores score==0, cat_id==0 or 11.
    """
    gt: dict[int, list[dict]] = {}
    if not ann_txt.exists():
        return gt
    for row in ann_txt.read_text().splitlines():
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
        if label is None:
            continue
        gt.setdefault(frame_id, []).append({
            "label": label,
            "x1": float(x), "y1": float(y),
            "x2": float(x + w), "y2": float(y + h),
        })
    return gt

# ---------------------------------------------------------------------------
# IoU and matching
# ---------------------------------------------------------------------------

def box_iou(a: dict, b: dict) -> float:
    ix1 = max(a["x1"], b["x1"]); iy1 = max(a["y1"], b["y1"])
    ix2 = min(a["x2"], b["x2"]); iy2 = min(a["y2"], b["y2"])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"])
    area_b = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
    return inter / (area_a + area_b - inter + 1e-9)


def match_detections(
    preds: list[dict],
    gts: list[dict],
    iou_threshold: float,
) -> tuple[list[bool], int]:
    """
    Match predictions to GT boxes for one frame at a given IoU threshold.
    Returns (tp_flags per pred, n_fn).
    preds: sorted descending by score, each has {label, score, x1, y1, x2, y2}
    gts:   each has {label, x1, y1, x2, y2}
    """
    matched_gt = set()
    tp_flags = []
    for pred in preds:
        best_iou = 0.0
        best_j = -1
        for j, gt in enumerate(gts):
            if j in matched_gt:
                continue
            if gt["label"] != pred["label"]:
                continue
            iou = box_iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_iou >= iou_threshold and best_j >= 0:
            tp_flags.append(True)
            matched_gt.add(best_j)
        else:
            tp_flags.append(False)
    n_fn = len(gts) - len(matched_gt)
    return tp_flags, n_fn


def compute_ap(tp_flags: list[bool], n_gt: int) -> float:
    """Compute average precision from a list of TP/FP flags (sorted by confidence) and GT count."""
    if n_gt == 0:
        return float("nan")
    precisions, recalls = [], []
    tp_cum = 0
    fp_cum = 0
    for tp in tp_flags:
        if tp:
            tp_cum += 1
        else:
            fp_cum += 1
        precisions.append(tp_cum / (tp_cum + fp_cum))
        recalls.append(tp_cum / n_gt)

    # 11-point interpolation (VOC-style)
    ap = 0.0
    for t in [i / 10 for i in range(11)]:
        ps = [p for p, r in zip(precisions, recalls) if r >= t]
        ap += (max(ps) if ps else 0.0) / 11
    return ap

# ---------------------------------------------------------------------------
# YOLO inference
# ---------------------------------------------------------------------------

def run_yolo_inference(weights: Path, frames: list[Path], imgsz: int, device: str,
                       conf_threshold: float) -> list[list[dict]]:
    """Returns list of prediction lists, one per frame. Each pred: {label, score, x1,y1,x2,y2}"""
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("ultralytics not found")

    model = YOLO(str(weights))
    if device:
        model.to(device)

    all_preds = []
    for frame_path in frames:
        results = model.predict(
            source=str(frame_path),
            imgsz=imgsz, conf=conf_threshold, verbose=False,
            device=device or None,
        )
        preds = []
        if results:
            r = results[0]
            for box in r.boxes:
                cls_id = int(box.cls[0].item())
                if cls_id < len(r.names):
                    label = r.names[cls_id]
                else:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                score = float(box.conf[0].item())
                preds.append({"label": label, "score": score,
                              "x1": x1, "y1": y1, "x2": x2, "y2": y2})
        preds.sort(key=lambda p: -p["score"])
        all_preds.append(preds)
    return all_preds

# ---------------------------------------------------------------------------
# FRCNN inference
# ---------------------------------------------------------------------------

def run_frcnn_inference(weights: Path, frames: list[Path], device_str: str,
                        conf_threshold: float) -> list[list[dict]]:
    """Returns list of prediction lists, one per frame."""
    import torchvision.transforms.v2 as T
    from PIL import Image as _PIL

    ckpt = torch.load(str(weights), map_location="cpu")
    model_key = ckpt.get("model_key", "fasterrcnn_resnet50_v2")
    classes = ckpt.get("classes", CLASS_NAMES)
    num_classes = ckpt.get("num_classes", len(classes) + 1)

    # Import builder from same directory or parent finetune/
    try:
        import importlib.util
        for candidate in [
            Path(__file__).parent / "train_sequence.py",
            Path(__file__).parent.parent / "finetune" / "train_by_distribution.py",
        ]:
            if candidate.exists():
                spec = importlib.util.spec_from_file_location("_seq", candidate)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                build_fn = mod._build_frcnn
                break
        else:
            raise ImportError("could not find _build_frcnn")
    except Exception as e:
        sys.exit(f"Cannot load FRCNN builder: {e}")

    model = build_fn(num_classes, model_key)
    model.load_state_dict(ckpt["model"])
    model.eval()

    if device_str:
        device = torch.device(device_str)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model.to(device)

    tf = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])

    all_preds = []
    with torch.no_grad():
        for frame_path in frames:
            img = _PIL.open(frame_path).convert("RGB")
            tensor = tf(img).unsqueeze(0).to(device)
            outputs = model(tensor)[0]

            preds = []
            boxes = outputs["boxes"].cpu().tolist()
            scores = outputs["scores"].cpu().tolist()
            labels_t = outputs["labels"].cpu().tolist()

            for box, score, lbl in zip(boxes, scores, labels_t):
                if score < conf_threshold:
                    continue
                # lbl is 1-indexed (background=0)
                cls_idx = lbl - 1
                if cls_idx < 0 or cls_idx >= len(classes):
                    continue
                label = classes[cls_idx]
                preds.append({"label": label, "score": score,
                              "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3]})
            preds.sort(key=lambda p: -p["score"])
            all_preds.append(preds)
    return all_preds

# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate_sequence(
    weights: Path,
    seq_dir: Path,
    ann_txt: Path,
    frame_stride: int,
    imgsz: int,
    device: str,
    conf_threshold: float,
    iou_thresholds: list[float],
    save_frames: bool,
    out_dir: Path | None,
) -> dict:
    from PIL import Image as _PIL

    frame_files = sorted(
        p for p in seq_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not frame_files:
        sys.exit(f"No images found in {seq_dir}")

    frame_files = frame_files[::frame_stride]

    with _PIL.open(frame_files[0]) as im:
        img_w, img_h = im.size

    gt_all = load_sequence_gt(ann_txt, img_w, img_h)
    frame_ids = []
    for p in frame_files:
        try:
            frame_ids.append(int(p.stem))
        except ValueError:
            frame_ids.append(-1)

    # Inference
    is_yolo = weights.suffix == ".pt"
    t0 = time.time()
    if is_yolo:
        all_preds = run_yolo_inference(weights, frame_files, imgsz, device, conf_threshold)
    else:
        all_preds = run_frcnn_inference(weights, frame_files, device, conf_threshold)
    inference_time = time.time() - t0

    # Accumulate TP/FP/FN per class per IoU threshold
    # Structure: {iou_t: {class_label: {"tp_flags": [], "n_gt": int}}}
    iou_class_data: dict[float, dict[str, dict]] = {
        t: defaultdict(lambda: {"tp_flags": [], "n_gt": 0})
        for t in iou_thresholds
    }

    total_preds = 0
    for fid, preds in zip(frame_ids, all_preds):
        gts = gt_all.get(fid, [])
        total_preds += len(preds)

        for iou_t in iou_thresholds:
            tp_flags, _ = match_detections(preds, gts, iou_t)
            # Per class
            for pred, tp in zip(preds, tp_flags):
                iou_class_data[iou_t][pred["label"]]["tp_flags"].append(tp)
            for gt in gts:
                iou_class_data[iou_t][gt["label"]]["n_gt"] += 1

    # Compute AP per class and mAP
    results_by_iou: dict[float, dict] = {}
    for iou_t in iou_thresholds:
        class_ap = {}
        for cls, data in iou_class_data[iou_t].items():
            ap = compute_ap(data["tp_flags"], data["n_gt"])
            class_ap[cls] = ap

        valid_aps = [v for v in class_ap.values() if not (isinstance(v, float) and v != v)]
        map_val = sum(valid_aps) / len(valid_aps) if valid_aps else float("nan")

        total_tp = sum(sum(1 for x in d["tp_flags"] if x)
                       for d in iou_class_data[iou_t].values())
        total_fp = sum(sum(1 for x in d["tp_flags"] if not x)
                       for d in iou_class_data[iou_t].values())
        total_gt = sum(d["n_gt"] for d in iou_class_data[iou_t].values())
        precision = total_tp / (total_tp + total_fp + 1e-9)
        recall = total_tp / (total_gt + 1e-9)

        results_by_iou[iou_t] = {
            "mAP": map_val,
            "precision": precision,
            "recall": recall,
            "per_class_AP": class_ap,
            "total_tp": total_tp, "total_fp": total_fp, "total_gt": total_gt,
        }

    # mAP50 and mAP50-95 summary
    map50 = results_by_iou.get(0.5, {}).get("mAP", float("nan"))
    map50_95 = sum(
        results_by_iou[t]["mAP"] for t in iou_thresholds
        if not (isinstance(results_by_iou[t]["mAP"], float)
                and results_by_iou[t]["mAP"] != results_by_iou[t]["mAP"])
    ) / max(1, len(iou_thresholds))

    n_gt_total = sum(len(v) for v in gt_all.values())
    n_frames = len(frame_files)

    result = {
        "sequence": seq_dir.name,
        "weights": str(weights),
        "n_frames_evaluated": n_frames,
        "n_gt_boxes": n_gt_total,
        "n_pred_boxes": total_preds,
        "inference_time_seconds": round(inference_time, 2),
        "fps": round(n_frames / max(inference_time, 1e-3), 1),
        "mAP50": round(map50, 4) if map50 == map50 else None,
        "mAP50_95": round(map50_95, 4),
        "precision_at_50": round(results_by_iou.get(0.5, {}).get("precision", 0), 4),
        "recall_at_50": round(results_by_iou.get(0.5, {}).get("recall", 0), 4),
        "per_iou": {
            str(t): {
                "mAP": round(r["mAP"], 4) if r["mAP"] == r["mAP"] else None,
                "precision": round(r["precision"], 4),
                "recall": round(r["recall"], 4),
                "per_class_AP": {
                    k: (round(v, 4) if v == v else None)
                    for k, v in r["per_class_AP"].items()
                },
            }
            for t, r in results_by_iou.items()
        },
    }

    # Optionally save annotated frames
    if save_frames and out_dir:
        _save_annotated_frames(
            frame_files, frame_ids, all_preds, gt_all, out_dir / seq_dir.name)

    return result


def _save_annotated_frames(
    frame_files: list[Path],
    frame_ids: list[int],
    all_preds: list[list[dict]],
    gt_all: dict[int, list[dict]],
    out_dir: Path,
):
    try:
        import cv2
        import numpy as np
    except ImportError:
        print("  WARNING: opencv-python not installed; skipping frame save")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    for frame_path, fid, preds in zip(frame_files, frame_ids, all_preds):
        img = cv2.imread(str(frame_path))
        if img is None:
            continue
        # Draw GT in green
        for gt in gt_all.get(fid, []):
            cv2.rectangle(img,
                          (int(gt["x1"]), int(gt["y1"])),
                          (int(gt["x2"]), int(gt["y2"])),
                          (0, 200, 0), 2)
            cv2.putText(img, gt["label"],
                        (int(gt["x1"]), max(0, int(gt["y1"]) - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 0), 1)
        # Draw predictions in red
        for pred in preds:
            cv2.rectangle(img,
                          (int(pred["x1"]), int(pred["y1"])),
                          (int(pred["x2"]), int(pred["y2"])),
                          (0, 0, 220), 2)
            label_str = f"{pred['label']} {pred['score']:.2f}"
            cv2.putText(img, label_str,
                        (int(pred["x1"]), min(img.shape[0] - 4, int(pred["y2"]) + 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 220), 1)
        out_path = out_dir / frame_path.name
        cv2.imwrite(str(out_path), img)
    print(f"  Saved annotated frames to {out_dir}")

# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

def print_result(result: dict):
    print(f"\n{'='*60}")
    print(f"Sequence : {result['sequence']}")
    print(f"Weights  : {result['weights']}")
    print(f"Frames   : {result['n_frames_evaluated']}  "
          f"GT boxes={result['n_gt_boxes']}  "
          f"Pred boxes={result['n_pred_boxes']}")
    print(f"Speed    : {result['fps']} fps  "
          f"({result['inference_time_seconds']}s total)")
    print(f"\nmAP50    : {result['mAP50']}")
    print(f"mAP50-95 : {result['mAP50_95']}")
    print(f"Precision: {result['precision_at_50']}  (IoU=0.5)")
    print(f"Recall   : {result['recall_at_50']}  (IoU=0.5)")

    if "0.5" in result["per_iou"]:
        per_cls = result["per_iou"]["0.5"]["per_class_AP"]
        if per_cls:
            print("\nPer-class AP (IoU=0.5):")
            for cls, ap in sorted(per_cls.items(), key=lambda x: -(x[1] or 0)):
                bar = ("█" * int((ap or 0) * 20)).ljust(20)
                print(f"  {cls:20s}  {bar}  {ap}")
    print(f"{'='*60}")

# ---------------------------------------------------------------------------
# Batch validation from summary.json
# ---------------------------------------------------------------------------

def validate_from_summary(
    summary_path: Path,
    sequences_json_path: Path,
    frame_stride: int,
    imgsz: int,
    device: str,
    conf_threshold: float,
    iou_thresholds: list[float],
    save_frames: bool,
):
    with summary_path.open() as f:
        summary = json.load(f)

    from train_sequence import load_sequence_catalogue
    catalogue = load_sequence_catalogue(sequences_json_path)

    all_results = []
    for group_label, group_data in summary.items():
        val_seqs = group_data.get("val_sequences", [])
        for fam in ("yolo", "frcnn"):
            if fam not in group_data:
                continue
            model_data = group_data[fam]
            if "error" in model_data:
                print(f"  SKIP {group_label}/{fam}: training error — {model_data['error']}")
                continue
            weights = Path(model_data["best_weights"])
            if not weights.exists():
                print(f"  SKIP {group_label}/{fam}: weights not found at {weights}")
                continue

            for seq_name in val_seqs:
                if seq_name not in catalogue:
                    print(f"  SKIP {seq_name}: not in catalogue")
                    continue
                meta = catalogue[seq_name]
                seq_dir = meta["images_dir"]
                ann_txt = meta["ann_txt"]
                if not seq_dir.exists():
                    print(f"  SKIP {seq_name}: images dir not found")
                    continue

                print(f"\nValidating {group_label}/{fam}  seq={seq_name}")
                out_dir = summary_path.parent / "val_frames" / group_label / fam if save_frames else None
                result = evaluate_sequence(
                    weights, seq_dir, ann_txt,
                    frame_stride, imgsz, device, conf_threshold,
                    iou_thresholds, save_frames, out_dir)
                result["group"] = group_label
                result["family"] = fam
                print_result(result)
                all_results.append(result)

    out_path = summary_path.parent / "validation_results.json"
    out_path.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\nAll validation results → {out_path}")
    return all_results

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    DEFAULT_IOU = [round(0.5 + i * 0.05, 2) for i in range(10)]  # 0.50 … 0.95

    parser = argparse.ArgumentParser(
        description="Validate a trained model on a held-out VisDrone VID sequence",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--weights", default=None,
                      help="Path to best.pt (YOLO) or best.pth (FRCNN)")
    mode.add_argument("--summary", default=None,
                      help="Path to training_summary.json — validates all recorded val sequences")

    parser.add_argument("--sequence", default=None,
                        help="Path to sequence directory (required with --weights)")
    parser.add_argument("--annotation", default=None,
                        help="Path to annotation .txt (required with --weights). "
                             "Auto-detected as <dataset_root>/annotations/<seq>.txt "
                             "if omitted.")
    parser.add_argument("--sequences-json", default="../datasets/sequences_categories.json",
                        help="sequences_categories.json (needed with --summary)")

    parser.add_argument("--frame-stride", type=int, default=1,
                        help="Evaluate every N-th frame (default: 1 = all frames)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Inference image size for YOLO (default: 640)")
    parser.add_argument("--device", default="",
                        help="Device: '' (auto), 'cpu', 'cuda', 'mps'")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold for predictions (default: 0.25)")
    parser.add_argument("--iou-thresholds", nargs="+", type=float, default=DEFAULT_IOU,
                        help="IoU threshold(s) for AP calculation "
                             "(default: 0.50 0.55 … 0.95 for COCO-style mAP)")
    parser.add_argument("--save-frames", action="store_true",
                        help="Save annotated frames with GT (green) and predictions (red)")
    parser.add_argument("--out-dir", default=None,
                        help="Directory for annotated frames and results JSON. "
                             "Defaults to the weights directory.")

    args = parser.parse_args()

    if args.summary:
        validate_from_summary(
            summary_path=Path(args.summary),
            sequences_json_path=Path(args.sequences_json),
            frame_stride=args.frame_stride,
            imgsz=args.imgsz,
            device=args.device,
            conf_threshold=args.conf,
            iou_thresholds=args.iou_thresholds,
            save_frames=args.save_frames,
        )
    else:
        if not args.sequence:
            parser.error("--sequence is required with --weights")

        weights = Path(args.weights)
        seq_dir = Path(args.sequence)

        if args.annotation:
            ann_txt = Path(args.annotation)
        else:
            # Auto-detect: annotations/ sits next to sequences/ in the dataset root
            ann_txt = seq_dir.parent.parent / "annotations" / f"{seq_dir.name}.txt"
            if not ann_txt.exists():
                parser.error(
                    f"Could not auto-detect annotation file at {ann_txt}. "
                    "Pass --annotation explicitly.")

        out_dir = Path(args.out_dir) if args.out_dir else weights.parent

        result = evaluate_sequence(
            weights=weights,
            seq_dir=seq_dir,
            ann_txt=ann_txt,
            frame_stride=args.frame_stride,
            imgsz=args.imgsz,
            device=args.device,
            conf_threshold=args.conf,
            iou_thresholds=args.iou_thresholds,
            save_frames=args.save_frames,
            out_dir=out_dir,
        )
        print_result(result)

        out_json = out_dir / f"val_{seq_dir.name}.json"
        out_json.write_text(json.dumps(result, indent=2, default=str))
        print(f"\nResults saved → {out_json}")
