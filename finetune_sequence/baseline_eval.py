"""
Evaluate pretrained (pre-fine-tuning) YOLO and FRCNN models on VisDrone VID sequences.

Produces the same annotated frames and JSON metrics as validate_sequence.py so you
can directly compare a baseline run against a fine-tuned run.

The pretrained models use COCO class names.  A mapping is applied to convert COCO
labels to VisDrone labels before evaluation.  COCO classes with no VisDrone
equivalent are discarded; COCO classes that map to the same VisDrone class are
merged (e.g. "person" and "bicycle" → "pedestrian" and "bicycle" respectively).

COCO → VisDrone mapping used:
  person       → pedestrian
  bicycle      → bicycle
  car          → car
  motorcycle   → motor
  bus          → bus
  truck        → truck

All other COCO detections are ignored.

FRCNN models are loaded with their COCO pretrained weights (no head replacement,
no anchor modification) so classes match the COCO label space.  The mapping is
applied at evaluation time.

Usage:

  # All default YOLO models, one sequence:
  python finetune_sequence/baseline_eval.py \\
      --sequence datasets/VisDrone2019-VID-val/sequences/uav0000268_05773_v

  # Specific models:
  python finetune_sequence/baseline_eval.py \\
      --sequence datasets/VisDrone2019-VID-val/sequences/uav0000268_05773_v \\
      --yolo-models yolov8n yolov8s \\
      --frcnn-models fasterrcnn_resnet50_v2 retinanet

  # All sequences in the catalogue, save annotated frames, stride 3:
  python finetune_sequence/baseline_eval.py \\
      --sequences-json datasets/sequences_categories.json \\
      --save-frames --frame-stride 3

  # Skip FRCNN (slow), just YOLO baselines:
  python finetune_sequence/baseline_eval.py \\
      --sequence datasets/VisDrone2019-VID-val/sequences/uav0000268_05773_v \\
      --frcnn-models none
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# VisDrone class names
# ---------------------------------------------------------------------------

VISDRONE_CLASSES = {
    1: "pedestrian", 2: "people", 3: "bicycle", 4: "car",
    5: "van", 6: "truck", 7: "tricycle", 8: "awning-tricycle",
    9: "bus", 10: "motor",
}
VISDRONE_NAMES = list(VISDRONE_CLASSES.values())

# COCO label → VisDrone label.  Unmapped COCO classes are discarded.
COCO_TO_VISDRONE = {
    "person":     "pedestrian",
    "bicycle":    "bicycle",
    "car":        "car",
    "motorcycle": "motor",
    "bus":        "bus",
    "truck":      "truck",
    # van, people, tricycle, awning-tricycle have no COCO equivalent
}

YOLO_ALL = ["yolov8n", "yolov8s", "yolov8m", "yolov8l",
            "yolo11n", "yolo11s", "yolo11m", "rtdetr-l"]
FRCNN_ALL = ["fasterrcnn_resnet50_v2", "fasterrcnn_resnet50",
             "fasterrcnn_mobilenet", "retinanet", "fcos", "ssdlite"]

YOLO_WEIGHTS = {
    "yolov8n": "yolov8n.pt", "yolov8s": "yolov8s.pt",
    "yolov8m": "yolov8m.pt", "yolov8l": "yolov8l.pt",
    "yolo11n": "yolo11n.pt", "yolo11s": "yolo11s.pt",
    "yolo11m": "yolo11m.pt", "rtdetr-l": "rtdetr-l.pt",
}

# ---------------------------------------------------------------------------
# Shared evaluation helpers (duplicated from validate_sequence.py to keep
# this script self-contained)
# ---------------------------------------------------------------------------

def load_sequence_gt(ann_txt: Path, img_w: int, img_h: int) -> dict[int, list[dict]]:
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
        gt.setdefault(frame_id, []).append(
            {"label": label, "x1": float(x), "y1": float(y),
             "x2": float(x + w), "y2": float(y + h)})
    return gt


def box_iou(a: dict, b: dict) -> float:
    ix1 = max(a["x1"], b["x1"]); iy1 = max(a["y1"], b["y1"])
    ix2 = min(a["x2"], b["x2"]); iy2 = min(a["y2"], b["y2"])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"])
    area_b = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
    return inter / (area_a + area_b - inter + 1e-9)


def match_detections(preds, gts, iou_threshold):
    matched_gt = set()
    tp_flags = []
    for pred in preds:
        best_iou, best_j = 0.0, -1
        for j, gt in enumerate(gts):
            if j in matched_gt or gt["label"] != pred["label"]:
                continue
            iou = box_iou(pred, gt)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= iou_threshold and best_j >= 0:
            tp_flags.append(True)
            matched_gt.add(best_j)
        else:
            tp_flags.append(False)
    return tp_flags, len(gts) - len(matched_gt)


def compute_ap(tp_flags, n_gt):
    if n_gt == 0:
        return float("nan")
    prec, rec = [], []
    tp_c = fp_c = 0
    for tp in tp_flags:
        if tp:
            tp_c += 1
        else:
            fp_c += 1
        prec.append(tp_c / (tp_c + fp_c))
        rec.append(tp_c / n_gt)
    ap = 0.0
    for t in [i / 10 for i in range(11)]:
        ps = [p for p, r in zip(prec, rec) if r >= t]
        ap += (max(ps) if ps else 0.0) / 11
    return ap


def compute_metrics(all_preds, frame_ids, gt_all, iou_thresholds):
    iou_data = {t: defaultdict(lambda: {"tp_flags": [], "n_gt": 0})
                for t in iou_thresholds}
    total_preds = 0
    for fid, preds in zip(frame_ids, all_preds):
        gts = gt_all.get(fid, [])
        total_preds += len(preds)
        for iou_t in iou_thresholds:
            tp_flags, _ = match_detections(preds, gts, iou_t)
            for pred, tp in zip(preds, tp_flags):
                iou_data[iou_t][pred["label"]]["tp_flags"].append(tp)
            for gt in gts:
                iou_data[iou_t][gt["label"]]["n_gt"] += 1

    results_by_iou = {}
    for iou_t in iou_thresholds:
        class_ap = {cls: compute_ap(d["tp_flags"], d["n_gt"])
                    for cls, d in iou_data[iou_t].items()}
        valid = [v for v in class_ap.values() if v == v]
        map_v = sum(valid) / len(valid) if valid else float("nan")
        ttp = sum(sum(1 for x in d["tp_flags"] if x) for d in iou_data[iou_t].values())
        tfp = sum(sum(1 for x in d["tp_flags"] if not x) for d in iou_data[iou_t].values())
        tgt = sum(d["n_gt"] for d in iou_data[iou_t].values())
        results_by_iou[iou_t] = {
            "mAP": map_v,
            "precision": ttp / (ttp + tfp + 1e-9),
            "recall": ttp / (tgt + 1e-9),
            "per_class_AP": class_ap,
            "total_tp": ttp, "total_fp": tfp, "total_gt": tgt,
        }
    return results_by_iou, total_preds

# ---------------------------------------------------------------------------
# YOLO inference (pretrained COCO weights, mapped to VisDrone)
# ---------------------------------------------------------------------------

def run_yolo_baseline(model_key: str, frames: list[Path], imgsz: int,
                      device: str, conf: float) -> list[list[dict]]:
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("ultralytics not found — activate .venv-eai first")
    weights = YOLO_WEIGHTS[model_key]
    model = YOLO(weights)
    if device:
        model.to(device)

    all_preds = []
    for frame_path in frames:
        results = model.predict(source=str(frame_path), imgsz=imgsz,
                                conf=conf, verbose=False, device=device or None)
        preds = []
        if results:
            r = results[0]
            for box in r.boxes:
                coco_label = r.names[int(box.cls[0].item())]
                vd_label = COCO_TO_VISDRONE.get(coco_label)
                if vd_label is None:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                preds.append({"label": vd_label, "score": float(box.conf[0]),
                              "x1": x1, "y1": y1, "x2": x2, "y2": y2})
        preds.sort(key=lambda p: -p["score"])
        all_preds.append(preds)
    return all_preds

# ---------------------------------------------------------------------------
# FRCNN inference (pretrained COCO weights, no head replacement, mapped)
# ---------------------------------------------------------------------------

def _build_frcnn_coco(model_key: str):
    import torchvision.models.detection as det

    registry = {
        "fasterrcnn_resnet50_v2": (det.fasterrcnn_resnet50_fpn_v2,
                                   det.FasterRCNN_ResNet50_FPN_V2_Weights),
        "fasterrcnn_resnet50":    (det.fasterrcnn_resnet50_fpn,
                                   det.FasterRCNN_ResNet50_FPN_Weights),
        "fasterrcnn_mobilenet":   (det.fasterrcnn_mobilenet_v3_large_fpn,
                                   det.FasterRCNN_MobileNet_V3_Large_FPN_Weights),
        "retinanet":              (det.retinanet_resnet50_fpn_v2,
                                   det.RetinaNet_ResNet50_FPN_V2_Weights),
        "fcos":                   (det.fcos_resnet50_fpn,
                                   det.FCOS_ResNet50_FPN_Weights),
        "ssdlite":                (det.ssdlite320_mobilenet_v3_large,
                                   det.SSDLite320_MobileNet_V3_Large_Weights),
    }
    if model_key not in registry:
        sys.exit(f"Unknown FRCNN model '{model_key}'. Choose from: {list(registry)}")
    builder_fn, weights_cls = registry[model_key]
    model = builder_fn(weights=weights_cls.DEFAULT)
    # Return both model and its COCO label list
    coco_names = list(weights_cls.DEFAULT.meta["categories"])
    return model, coco_names


def run_frcnn_baseline(model_key: str, frames: list[Path], device_str: str,
                       conf: float) -> list[list[dict]]:
    import torchvision.transforms.v2 as T
    from PIL import Image as _PIL

    model, coco_names = _build_frcnn_coco(model_key)
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
            out = model(tensor)[0]
            preds = []
            for box, score, lbl in zip(out["boxes"].cpu().tolist(),
                                        out["scores"].cpu().tolist(),
                                        out["labels"].cpu().tolist()):
                if score < conf:
                    continue
                # torchvision COCO models: label 0 = background, 1 = person, …
                # meta["categories"] list is 0-indexed but background is absent —
                # index into it with lbl-1
                idx = lbl - 1
                if idx < 0 or idx >= len(coco_names):
                    continue
                coco_label = coco_names[idx]
                vd_label = COCO_TO_VISDRONE.get(coco_label)
                if vd_label is None:
                    continue
                preds.append({"label": vd_label, "score": score,
                              "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3]})
            preds.sort(key=lambda p: -p["score"])
            all_preds.append(preds)
    return all_preds

# ---------------------------------------------------------------------------
# Annotated frame saving
# ---------------------------------------------------------------------------

def save_annotated_frames(frame_files, frame_ids, all_preds, gt_all, out_dir: Path):
    try:
        import cv2
    except ImportError:
        print("  WARNING: opencv-python not installed; skipping frame save")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    for frame_path, fid, preds in zip(frame_files, frame_ids, all_preds):
        img = cv2.imread(str(frame_path))
        if img is None:
            continue
        for gt in gt_all.get(fid, []):
            cv2.rectangle(img, (int(gt["x1"]), int(gt["y1"])),
                          (int(gt["x2"]), int(gt["y2"])), (0, 200, 0), 2)
            cv2.putText(img, gt["label"],
                        (int(gt["x1"]), max(0, int(gt["y1"]) - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 0), 1)
        for pred in preds:
            cv2.rectangle(img, (int(pred["x1"]), int(pred["y1"])),
                          (int(pred["x2"]), int(pred["y2"])), (0, 0, 220), 2)
            cv2.putText(img, f"{pred['label']} {pred['score']:.2f}",
                        (int(pred["x1"]), min(img.shape[0] - 4, int(pred["y2"]) + 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 220), 1)
        cv2.imwrite(str(out_dir / frame_path.name), img)
    print(f"  Saved annotated frames → {out_dir}")

# ---------------------------------------------------------------------------
# Core: evaluate one model on one sequence
# ---------------------------------------------------------------------------

def evaluate_one(
    family: str,
    model_key: str,
    seq_dir: Path,
    ann_txt: Path,
    frame_stride: int,
    imgsz: int,
    device: str,
    conf: float,
    iou_thresholds: list[float],
    save_frames: bool,
    out_root: Path,
) -> dict:
    from PIL import Image as _PIL

    frame_files = sorted(
        p for p in seq_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not frame_files:
        return {"error": f"no images in {seq_dir}"}

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

    t0 = time.time()
    if family == "yolo":
        all_preds = run_yolo_baseline(model_key, frame_files, imgsz, device, conf)
    else:
        all_preds = run_frcnn_baseline(model_key, frame_files, device, conf)
    elapsed = time.time() - t0

    results_by_iou, total_preds = compute_metrics(all_preds, frame_ids, gt_all, iou_thresholds)
    map50 = results_by_iou.get(0.5, {}).get("mAP", float("nan"))
    map50_95 = sum(
        r["mAP"] for r in results_by_iou.values() if r["mAP"] == r["mAP"]
    ) / max(1, len(iou_thresholds))

    result = {
        "family": family,
        "model": model_key,
        "sequence": seq_dir.name,
        "pretrained": True,
        "n_frames_evaluated": len(frame_files),
        "n_gt_boxes": sum(len(v) for v in gt_all.values()),
        "n_pred_boxes": total_preds,
        "inference_time_seconds": round(elapsed, 2),
        "fps": round(len(frame_files) / max(elapsed, 1e-3), 1),
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
                    for k, v in r["per_class_AP"].items()},
            }
            for t, r in results_by_iou.items()
        },
    }

    if save_frames:
        frames_out = out_root / "baseline_frames" / seq_dir.name / family / model_key
        save_annotated_frames(frame_files, frame_ids, all_preds, gt_all, frames_out)

    return result

# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------

def print_result(r: dict):
    print(f"\n{'='*60}")
    print(f"[{r['family'].upper()}] {r['model']}  —  {r['sequence']}")
    print(f"  Frames : {r['n_frames_evaluated']}  "
          f"GT={r['n_gt_boxes']}  Pred={r['n_pred_boxes']}  "
          f"{r['fps']} fps")
    print(f"  mAP50    : {r['mAP50']}")
    print(f"  mAP50-95 : {r['mAP50_95']}")
    print(f"  Precision: {r['precision_at_50']}  (IoU=0.5)")
    print(f"  Recall   : {r['recall_at_50']}  (IoU=0.5)")
    per_cls = r.get("per_iou", {}).get("0.5", {}).get("per_class_AP", {})
    if per_cls:
        print("  Per-class AP (IoU=0.5):")
        for cls, ap in sorted(per_cls.items(), key=lambda x: -(x[1] or 0)):
            bar = ("█" * int((ap or 0) * 20)).ljust(20)
            print(f"    {cls:20s}  {bar}  {ap}")
    print(f"{'='*60}")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    DEFAULT_IOU = [round(0.5 + i * 0.05, 2) for i in range(10)]

    parser = argparse.ArgumentParser(
        description="Evaluate pretrained YOLO/FRCNN baselines on VisDrone VID sequences",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    seq_g = parser.add_mutually_exclusive_group(required=True)
    seq_g.add_argument("--sequence", default=None, metavar="PATH",
                       help="Single sequence directory to evaluate")
    seq_g.add_argument("--sequences-json", default=None, metavar="PATH",
                       help="Evaluate all sequences in sequences_categories.json")

    parser.add_argument("--annotation", default=None,
                        help="Annotation .txt path (auto-detected if omitted)")
    parser.add_argument("--yolo-models", nargs="+", default=YOLO_ALL,
                        choices=YOLO_ALL + ["none"],
                        help="YOLO model keys to evaluate (default: all). "
                             "Pass 'none' to skip YOLO.")
    parser.add_argument("--frcnn-models", nargs="+", default=FRCNN_ALL,
                        choices=FRCNN_ALL + ["none"],
                        help="FRCNN model keys to evaluate (default: all). "
                             "Pass 'none' to skip FRCNN.")
    parser.add_argument("--frame-stride", type=int, default=1,
                        help="Evaluate every N-th frame (default: 1)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Inference image size for YOLO (default: 640)")
    parser.add_argument("--device", default="",
                        help="Device: '' (auto), 'cpu', 'cuda', 'mps'")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold (default: 0.25)")
    parser.add_argument("--iou-thresholds", nargs="+", type=float, default=DEFAULT_IOU)
    parser.add_argument("--save-frames", action="store_true",
                        help="Save annotated frames with GT (green) and predictions (red)")
    parser.add_argument("--out-dir", default="runs/baseline_eval",
                        help="Output directory for results JSON and annotated frames "
                             "(default: runs/baseline_eval)")

    args = parser.parse_args()

    yolo_models = [] if args.yolo_models == ["none"] else args.yolo_models
    frcnn_models = [] if args.frcnn_models == ["none"] else args.frcnn_models

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Build list of (seq_dir, ann_txt) pairs to evaluate
    sequences: list[tuple[Path, Path]] = []
    if args.sequence:
        seq_dir = Path(args.sequence).resolve()
        if args.annotation:
            ann_txt = Path(args.annotation)
        else:
            ann_txt = seq_dir.parent.parent / "annotations" / f"{seq_dir.name}.txt"
            if not ann_txt.exists():
                parser.error(f"Could not auto-detect annotation at {ann_txt}. "
                             "Pass --annotation explicitly.")
        sequences.append((seq_dir, ann_txt))
    else:
        # Load from catalogue
        sys.path.insert(0, str(Path(__file__).parent))
        try:
            from train_sequence import load_sequence_catalogue
        except ImportError:
            sys.exit("Could not import load_sequence_catalogue from train_sequence.py")
        catalogue = load_sequence_catalogue(Path(args.sequences_json))
        for seq_name, meta in catalogue.items():
            if meta["images_dir"].exists() and meta["ann_txt"].exists():
                sequences.append((meta["images_dir"], meta["ann_txt"]))
            else:
                print(f"  SKIP {seq_name}: path not found on disk")

    if not sequences:
        sys.exit("No valid sequences to evaluate.")

    all_results = []

    for seq_dir, ann_txt in sequences:
        print(f"\n>>> Sequence: {seq_dir.name}")

        for model_key in yolo_models:
            print(f"  [YOLO] {model_key} ...")
            try:
                r = evaluate_one("yolo", model_key, seq_dir, ann_txt,
                                 args.frame_stride, args.imgsz, args.device,
                                 args.conf, args.iou_thresholds,
                                 args.save_frames, out_root)
            except Exception as e:
                import traceback
                print(f"  ERROR: {e}\n{traceback.format_exc()}")
                r = {"family": "yolo", "model": model_key,
                     "sequence": seq_dir.name, "error": str(e)}
            print_result(r)
            all_results.append(r)

        for model_key in frcnn_models:
            print(f"  [FRCNN] {model_key} ...")
            try:
                r = evaluate_one("frcnn", model_key, seq_dir, ann_txt,
                                 args.frame_stride, args.imgsz, args.device,
                                 args.conf, args.iou_thresholds,
                                 args.save_frames, out_root)
            except Exception as e:
                import traceback
                print(f"  ERROR: {e}\n{traceback.format_exc()}")
                r = {"family": "frcnn", "model": model_key,
                     "sequence": seq_dir.name, "error": str(e)}
            print_result(r)
            all_results.append(r)

    out_json = out_root / "baseline_results.json"
    out_json.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\nAll results → {out_json}")

    # Print summary table
    print(f"\n{'Model':<30} {'Sequence':<25} {'mAP50':>7} {'mAP50-95':>9} {'P@50':>6} {'R@50':>6}")
    print("-" * 90)
    for r in all_results:
        if "error" in r:
            print(f"  {r['family']}/{r['model']:<26} {r['sequence']:<25}  ERROR: {r['error']}")
            continue
        name = f"{r['family']}/{r['model']}"
        print(f"  {name:<28} {r['sequence']:<25} "
              f"{str(r['mAP50']):>7} {str(r['mAP50_95']):>9} "
              f"{str(r['precision_at_50']):>6} {str(r['recall_at_50']):>6}")
