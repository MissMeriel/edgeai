"""
Generate GIFs from baseline_eval.py output, overlaying GT (green) and model
predictions (red) on the raw sequence frames.

The script re-runs inference on the requested time window rather than requiring
pre-saved annotated frames, so it works even if --save-frames was not passed to
baseline_eval.py.  The baseline_results.json is used to look up which models are
available and what their overall mAP50 is, and to filter which models are shown
based on --min-map.

Usage examples:

  # GIF for one model, first 5 seconds
  python finetune_sequence/make_eval_gif.py \\
      --sequence datasets/VisDrone2019-VID-val/sequences/uav0000086_00000_v \\
      --model yolo/yolov8n \\
      --start 0 --duration 5

  # All models that beat 0.3 mAP50, seconds 10-20, 10 fps output
  python finetune_sequence/make_eval_gif.py \\
      --sequence datasets/VisDrone2019-VID-val/sequences/uav0000086_00000_v \\
      --results runs/baseline_eval/baseline_results.json \\
      --min-map 0.30 \\
      --start 10 --duration 10 \\
      --output-fps 10

  # Compare two models side by side in one GIF
  python finetune_sequence/make_eval_gif.py \\
      --sequence datasets/VisDrone2019-VID-val/sequences/uav0000086_00000_v \\
      --model yolo/yolov8n yolo/yolov8m \\
      --start 5 --duration 8 \\
      --side-by-side

  # GT-only GIF (no model inference)
  python finetune_sequence/make_eval_gif.py \\
      --sequence datasets/VisDrone2019-VID-val/sequences/uav0000086_00000_v \\
      --gt-only --start 0 --duration 5

Arguments:
  --sequence PATH     Sequence directory (images/*.jpg, 30 fps assumed)
  --annotation PATH   Annotation .txt (auto-detected from sequence path)
  --results PATH      baseline_results.json from baseline_eval.py
                      (optional; used to filter by --min-map)
  --model FAMILY/KEY  One or more "family/model_key" strings, e.g. yolo/yolov8n
                      or frcnn/fasterrcnn_mobilenet.  If omitted, all models
                      that pass --min-map are included.
  --min-map FLOAT     Skip models whose mAP50 is below this threshold
                      (default 0.0 = show all).  Requires --results.
  --gt-only           Draw only ground-truth boxes; skip all inference.
  --start FLOAT       Start time in seconds (default 0)
  --duration FLOAT    Duration in seconds (default 5)
  --source-fps FLOAT  Frame rate of the source sequence (default 30)
  --output-fps FLOAT  Frame rate of the output GIF (default 8)
  --conf FLOAT        Confidence threshold for predictions (default 0.25)
  --imgsz INT         Inference image size for YOLO (default 640)
  --device STR        Device: '' (auto), 'cpu', 'cuda', 'mps'
  --scale FLOAT       Resize output frames by this factor (default 0.5)
  --side-by-side      Stitch multiple models horizontally per frame
  --out-dir PATH      Output directory (default runs/eval_gifs)
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Re-use helpers from baseline_eval.py
# ---------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).parent))
from baseline_eval import (
    COCO_TO_VISDRONE,
    VISDRONE_CLASSES,
    YOLO_WEIGHTS,
    load_sequence_gt,
    run_yolo_baseline,
    run_frcnn_baseline,
)

# ---------------------------------------------------------------------------
# Colours (BGR for OpenCV)
# ---------------------------------------------------------------------------

GT_COLOUR    = (30, 210, 30)    # green
PRED_COLOUR  = (30, 30, 220)    # red
LABEL_BG     = (20, 20, 20)     # near-black for text background

# ---------------------------------------------------------------------------
# Frame helpers
# ---------------------------------------------------------------------------

def frames_for_window(seq_dir: Path, start_s: float, duration_s: float,
                      source_fps: float) -> list[Path]:
    """Return the frame files covering [start_s, start_s+duration_s)."""
    all_frames = sorted(
        p for p in seq_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not all_frames:
        sys.exit(f"No image files found in {seq_dir}")

    # VisDrone frame stems are 1-indexed integers; use position if not numeric
    try:
        stem_ids = [int(p.stem) for p in all_frames]
        first_id = stem_ids[0]
    except ValueError:
        # Fall back to positional indexing at source_fps
        start_idx = int(start_s * source_fps)
        end_idx   = int((start_s + duration_s) * source_fps)
        return all_frames[start_idx:end_idx]

    start_id = first_id + int(start_s * source_fps)
    end_id   = first_id + int((start_s + duration_s) * source_fps)
    window = [p for p, fid in zip(all_frames, stem_ids)
              if start_id <= fid < end_id]
    return window


def frame_id(p: Path) -> int:
    try:
        return int(p.stem)
    except ValueError:
        return -1


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_frame(img, preds: list[dict], gts: list[dict], conf_threshold: float,
               show_gt: bool = True, show_pred: bool = True):
    """Draw GT (green) and predictions (red) on img in-place. Returns img."""
    import cv2

    if show_gt:
        for g in gts:
            x1, y1, x2, y2 = int(g["x1"]), int(g["y1"]), int(g["x2"]), int(g["y2"])
            cv2.rectangle(img, (x1, y1), (x2, y2), GT_COLOUR, 2)
            label = g["label"]
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
            cv2.rectangle(img, (x1, max(0, y1 - th - 4)), (x1 + tw + 2, y1), GT_COLOUR, -1)
            cv2.putText(img, label, (x1 + 1, max(th, y1 - 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

    if show_pred:
        for p in preds:
            if p["score"] < conf_threshold:
                continue
            x1, y1, x2, y2 = int(p["x1"]), int(p["y1"]), int(p["x2"]), int(p["y2"])
            cv2.rectangle(img, (x1, y1), (x2, y2), PRED_COLOUR, 2)
            label = f"{p['label']} {p['score']:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
            by = min(img.shape[0] - 2, y2 + th + 4)
            cv2.rectangle(img, (x1, by - th - 4), (x1 + tw + 2, by), PRED_COLOUR, -1)
            cv2.putText(img, label, (x1 + 1, by - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

    return img


def add_overlay(img, model_label: str, map50, frame_num: int):
    """Top-left overlay with model name, mAP50, frame number."""
    import cv2
    lines = [model_label]
    if map50 is not None:
        lines.append(f"mAP50={map50:.3f}")
    lines.append(f"frame {frame_num}")
    y = 18
    for line in lines:
        (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(img, (4, y - th - 3), (4 + tw + 4, y + 3), LABEL_BG, -1)
        cv2.putText(img, line, (6, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (220, 220, 220), 1, cv2.LINE_AA)
        y += th + 8
    return img


def legend_bar(width: int, height: int = 22) -> "np.ndarray":
    """Return a small legend image strip."""
    import cv2, numpy as np
    bar = np.zeros((height, width, 3), dtype=np.uint8)
    bar[:] = (40, 40, 40)
    cv2.rectangle(bar, (4, 4), (24, height - 4), GT_COLOUR, -1)
    cv2.putText(bar, "GT", (28, height - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.rectangle(bar, (70, 4), (90, height - 4), PRED_COLOUR, -1)
    cv2.putText(bar, "Prediction", (94, height - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
    return bar


# ---------------------------------------------------------------------------
# Model resolution
# ---------------------------------------------------------------------------

def parse_model_spec(spec: str) -> tuple[str, str]:
    """'yolo/yolov8n' → ('yolo', 'yolov8n').  Also accepts 'yolov8n' alone."""
    if "/" in spec:
        parts = spec.split("/", 1)
        return parts[0], parts[1]
    # Guess family from key prefix
    if spec.startswith("yolov") or spec.startswith("yolo11") or spec.startswith("rtdetr"):
        return "yolo", spec
    return "frcnn", spec


def models_from_results(results_path: Path, seq_name: str,
                        min_map: float) -> list[tuple[str, str, float | None]]:
    """Return [(family, model_key, map50)] filtered by min_map for seq_name."""
    data = json.loads(results_path.read_text())
    out = []
    for r in data:
        if "error" in r:
            continue
        if r.get("sequence") != seq_name:
            continue
        map50 = r.get("mAP50")
        if map50 is None or map50 < min_map:
            continue
        out.append((r["family"], r["model"], map50))
    return out


# ---------------------------------------------------------------------------
# GIF assembly
# ---------------------------------------------------------------------------

def frames_to_gif(pil_frames: list, output_path: Path, fps: float):
    duration_ms = int(1000 / max(fps, 0.1))
    pil_frames[0].save(
        str(output_path),
        format="GIF",
        save_all=True,
        append_images=pil_frames[1:],
        loop=0,
        duration=duration_ms,
        optimize=False,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_gif(
    seq_dir: Path,
    ann_txt: Path,
    models: list[tuple[str, str, float | None]],  # [(family, key, map50), ...]
    start_s: float,
    duration_s: float,
    source_fps: float,
    output_fps: float,
    conf: float,
    imgsz: int,
    device: str,
    scale: float,
    side_by_side: bool,
    gt_only: bool,
    out_dir: Path,
):
    import cv2
    import numpy as np
    from PIL import Image as PILImage

    window_frames = frames_for_window(seq_dir, start_s, duration_s, source_fps)
    if not window_frames:
        sys.exit(f"No frames found in [{start_s}s, {start_s + duration_s}s) — "
                 f"check --start / --duration against sequence length")

    print(f"  Sequence : {seq_dir.name}")
    print(f"  Window   : {start_s}s – {start_s + duration_s}s  "
          f"({len(window_frames)} frames at source fps={source_fps})")

    with PILImage.open(window_frames[0]) as im:
        img_w, img_h = im.size
    gt_all = load_sequence_gt(ann_txt, img_w, img_h)

    # Stride to hit target output_fps from source_fps
    frame_step = max(1, round(source_fps / output_fps))
    sampled = window_frames[::frame_step]
    sampled_ids = [frame_id(p) for p in sampled]

    # Run inference for each model
    model_preds: dict[tuple[str, str], list[list[dict]]] = {}
    if not gt_only:
        for family, key, _ in models:
            print(f"  Running inference: [{family}] {key} on {len(sampled)} frames ...")
            if family == "yolo":
                preds = run_yolo_baseline(key, sampled, imgsz, device, conf)
            else:
                preds = run_frcnn_baseline(key, sampled, device, conf)
            model_preds[(family, key)] = preds

    out_dir.mkdir(parents=True, exist_ok=True)
    ow = max(1, int(img_w * scale))
    oh = max(1, int(img_h * scale))

    if side_by_side and models and not gt_only:
        # All models in one GIF: frames stacked horizontally
        panels = len(models)
        legend = legend_bar(ow * panels)
        gif_frames = []

        for i, fid in enumerate(sampled_ids):
            raw = cv2.imread(str(sampled[i]))
            if raw is None:
                continue
            gts = gt_all.get(fid, [])
            strips = []
            for family, key, map50 in models:
                tile = raw.copy()
                tile = draw_frame(tile, model_preds[(family, key)][i], gts, conf)
                tile = add_overlay(tile, f"{family}/{key}", map50, fid)
                tile = cv2.resize(tile, (ow, oh))
                strips.append(tile)
            row = np.concatenate(strips, axis=1)
            row = np.concatenate([row, legend], axis=0)
            gif_frames.append(PILImage.fromarray(cv2.cvtColor(row, cv2.COLOR_BGR2RGB)))

        tag = "_".join(f"{k}" for _, k, _ in models)
        out_path = out_dir / f"{seq_dir.name}_{tag}_{start_s}s-{start_s+duration_s}s_sbs.gif"
        frames_to_gif(gif_frames, out_path, output_fps)
        print(f"  Saved → {out_path}")

    else:
        # One GIF per model (or one GT-only GIF)
        entries = [("gt", "gt-only", None)] if gt_only else [(f, k, m) for f, k, m in models]
        legend = legend_bar(ow)

        for family, key, map50 in entries:
            gif_frames = []
            for i, fid in enumerate(sampled_ids):
                raw = cv2.imread(str(sampled[i]))
                if raw is None:
                    continue
                gts = gt_all.get(fid, [])
                preds = model_preds.get((family, key), [[]]*len(sampled))[i] if not gt_only else []
                img = raw.copy()
                img = draw_frame(img, preds, gts, conf,
                                 show_gt=True, show_pred=not gt_only)
                lbl = "GT only" if gt_only else f"{family}/{key}"
                img = add_overlay(img, lbl, map50, fid)
                img = cv2.resize(img, (ow, oh))
                img = np.concatenate([img, legend], axis=0)
                gif_frames.append(PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))

            tag = "gt_only" if gt_only else key
            out_path = out_dir / f"{seq_dir.name}_{tag}_{start_s}s-{start_s+duration_s}s.gif"
            frames_to_gif(gif_frames, out_path, output_fps)
            print(f"  Saved → {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate evaluation GIFs from baseline_eval.py output",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("--sequence", required=True, metavar="PATH",
                        help="Sequence directory containing frames")
    parser.add_argument("--annotation", default=None, metavar="PATH",
                        help="Annotation .txt (auto-detected if omitted)")
    parser.add_argument("--results", default=None, metavar="PATH",
                        help="baseline_results.json from baseline_eval.py "
                             "(used to filter by --min-map)")
    parser.add_argument("--model", nargs="+", default=None, metavar="FAMILY/KEY",
                        help="Explicit model(s) to include, e.g. yolo/yolov8n "
                             "frcnn/fasterrcnn_mobilenet. "
                             "If omitted, all models in --results that pass "
                             "--min-map are included.")
    parser.add_argument("--min-map", type=float, default=0.0, metavar="FLOAT",
                        help="Skip models with mAP50 < this value (default 0.0). "
                             "Requires --results.")
    parser.add_argument("--gt-only", action="store_true",
                        help="Draw ground truth only; skip all model inference")
    parser.add_argument("--start", type=float, default=0.0, metavar="SECONDS",
                        help="Start time in seconds (default 0)")
    parser.add_argument("--duration", type=float, default=5.0, metavar="SECONDS",
                        help="Duration in seconds (default 5)")
    parser.add_argument("--source-fps", type=float, default=30.0,
                        help="Source video frame rate (default 30 for VisDrone)")
    parser.add_argument("--output-fps", type=float, default=8.0,
                        help="Output GIF frame rate (default 8)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold for displayed predictions (default 0.25)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Inference image size for YOLO (default 640)")
    parser.add_argument("--device", default="",
                        help="Device: '' (auto), 'cpu', 'cuda', 'mps'")
    parser.add_argument("--scale", type=float, default=0.5,
                        help="Output frame scale factor (default 0.5)")
    parser.add_argument("--side-by-side", action="store_true",
                        help="Stitch all selected models into a single wide GIF "
                             "rather than one GIF per model")
    parser.add_argument("--out-dir", default="runs/eval_gifs",
                        help="Output directory (default runs/eval_gifs)")

    args = parser.parse_args()

    seq_dir = Path(args.sequence).resolve()
    if not seq_dir.is_dir():
        sys.exit(f"Sequence directory not found: {seq_dir}")

    # Auto-detect annotation path
    if args.annotation:
        ann_txt = Path(args.annotation)
    else:
        ann_txt = seq_dir.parent.parent / "annotations" / f"{seq_dir.name}.txt"
    if not ann_txt.exists():
        sys.exit(f"Annotation file not found: {ann_txt}\n"
                 "Pass --annotation explicitly if the file is elsewhere.")

    # Resolve model list
    models: list[tuple[str, str, float | None]] = []

    if args.gt_only:
        models = []  # no inference needed
    elif args.model:
        # Explicit list; map50 unknown unless --results also provided
        results_map = {}
        if args.results:
            rp = Path(args.results)
            if rp.exists():
                for r in json.loads(rp.read_text()):
                    if r.get("sequence") == seq_dir.name and "error" not in r:
                        results_map[(r["family"], r["model"])] = r.get("mAP50")
        for spec in args.model:
            family, key = parse_model_spec(spec)
            map50 = results_map.get((family, key))
            models.append((family, key, map50))
    elif args.results:
        rp = Path(args.results)
        if not rp.exists():
            sys.exit(f"Results file not found: {rp}")
        models = models_from_results(rp, seq_dir.name, args.min_map)
        if not models:
            sys.exit(f"No models in {rp} passed --min-map={args.min_map} "
                     f"for sequence '{seq_dir.name}'")
        print(f"  Selected {len(models)} model(s) with mAP50 ≥ {args.min_map}:")
        for f, k, m in models:
            print(f"    [{f}] {k}  mAP50={m}")
    else:
        sys.exit("Provide --model, --results, or --gt-only")

    build_gif(
        seq_dir=seq_dir,
        ann_txt=ann_txt,
        models=models,
        start_s=args.start,
        duration_s=args.duration,
        source_fps=args.source_fps,
        output_fps=args.output_fps,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        scale=args.scale,
        side_by_side=args.side_by_side,
        gt_only=args.gt_only,
        out_dir=Path(args.out_dir),
    )
