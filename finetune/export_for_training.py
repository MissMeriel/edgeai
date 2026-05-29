"""
Export "fixed" samples from FiftyOne for fine-tuning.

Two export formats:
  yolo      (default) — YOLO txt labels + dataset.yaml; consumed by train_yolo.py
                        and train_faster_rcnn.py
  fiftyone  — full FiftyOne dataset on disk (JSON + media); preserves every field,
              all class labels (including newly added ones), tags, and metadata.
              Re-importable with fo.Dataset.from_dir() on any machine.

Usage:
    # YOLO format (for training scripts):
    python export_for_training.py --output-dir ./runs/finetune_data

    # FiftyOne format (portable, full-fidelity backup / transfer):
    python export_for_training.py --output-dir ./runs/finetune_fo --format fiftyone

    # Include "reviewed" samples alongside "fixed":
    python export_for_training.py --output-dir ./runs/finetune_data --status fixed reviewed

YOLO output structure:
    <output-dir>/
        images/train/   images/val/
        labels/train/   labels/val/
        dataset.yaml

FiftyOne output structure:
    <output-dir>/
        metadata.json   data.json   data/  (media files)
"""

import argparse
import random
import shutil
import sys
from pathlib import Path

import yaml


def _build_status_filter(statuses: list[str], F):
    status_filter = None
    for s in statuses:
        cond = F("tag_review_status.label") == s
        status_filter = cond if status_filter is None else (status_filter | cond)
    return status_filter


def export_fiftyone(view, output_dir: Path) -> Path:
    """Export view as a portable FiftyOne dataset (JSON + media)."""
    import fiftyone as fo

    out = output_dir
    out.mkdir(parents=True, exist_ok=True)

    print(f"Exporting {len(view)} samples as FiftyOne dataset to {out} ...")
    view.export(
        export_dir=str(out),
        dataset_type=fo.types.FiftyOneDataset,
        export_media=True,
        overwrite=True,
    )

    classes = sorted(c for c in view.distinct("ground_truth.detections.label") if c)
    print(f"Exported {len(view)} samples | {len(classes)} classes: {classes}")
    print(f"\nTo re-import on any machine:")
    print(f"  import fiftyone as fo")
    print(f"  ds = fo.Dataset.from_dir('{out}', dataset_type=fo.types.FiftyOneDataset)")
    return out


def export_yolo(view, output_dir: Path, val_ratio: float, seed: int) -> Path:
    """Export view as YOLO txt labels + dataset.yaml, split into train/val."""
    total = len(view)

    # Collect class names from ground truth (sorted for stable class IDs)
    classes = sorted(c for c in view.distinct("ground_truth.detections.label") if c)
    if not classes:
        sys.exit("ground_truth exists but no class labels found")
    class_to_idx = {c: i for i, c in enumerate(classes)}
    print(f"Found {total} samples | {len(classes)} classes: {classes}")

    # Reproducible train/val split
    random.seed(seed)
    samples = list(view.select_fields(["filepath", "ground_truth"]))
    random.shuffle(samples)
    n_val = max(1, round(total * val_ratio))
    val_samples = samples[:n_val]
    train_samples = samples[n_val:]
    print(f"Split → train: {len(train_samples)}, val: {len(val_samples)}")

    out = output_dir
    out.mkdir(parents=True, exist_ok=True)

    def write_split(split_samples, split_name: str) -> int:
        img_dir = out / "images" / split_name
        lbl_dir = out / "labels" / split_name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        skipped = 0
        for sample in split_samples:
            src = Path(sample.filepath)
            if not src.exists():
                skipped += 1
                continue

            shutil.copy2(src, img_dir / src.name)

            lbl_path = lbl_dir / (src.stem + ".txt")
            lines = []
            if sample.ground_truth and sample.ground_truth.detections:
                for det in sample.ground_truth.detections:
                    if det.label not in class_to_idx:
                        continue
                    cls_id = class_to_idx[det.label]
                    x, y, bw, bh = det.bounding_box  # FiftyOne: top-left + wh, normalized
                    cx = x + bw / 2
                    cy = y + bh / 2
                    # Clamp to [0, 1] — bad annotations can exceed bounds
                    cx = max(0.0, min(1.0, cx))
                    cy = max(0.0, min(1.0, cy))
                    bw = max(0.0, min(1.0, bw))
                    bh = max(0.0, min(1.0, bh))
                    lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            lbl_path.write_text("\n".join(lines))

        if skipped:
            print(f"  WARNING: skipped {skipped} samples with missing image files")
        return len(split_samples) - skipped

    n_train = write_split(train_samples, "train")
    n_val_written = write_split(val_samples, "val")
    print(f"Wrote {n_train} train, {n_val_written} val images + labels")

    yaml_data = {
        "path": str(out.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": len(classes),
        "names": classes,
    }
    yaml_path = out / "dataset.yaml"
    yaml_path.write_text(yaml.dump(yaml_data, sort_keys=False, allow_unicode=True))
    print(f"Wrote {yaml_path}")
    return out


def export(output_dir: str, fmt: str = "yolo", val_ratio: float = 0.15, seed: int = 42,
           dataset_name: str = "annotation_dataset",
           statuses: list[str] | None = None) -> Path:
    try:
        import fiftyone as fo
        from fiftyone import ViewField as F
    except ImportError:
        sys.exit("fiftyone not found — activate .venv-eai first")

    if statuses is None:
        statuses = ["fixed"]

    print(f"Loading dataset '{dataset_name}'...")
    try:
        dataset = fo.load_dataset(dataset_name)
    except Exception as e:
        sys.exit(f"Could not load dataset '{dataset_name}': {e}")

    status_filter = _build_status_filter(statuses, F)
    view = dataset.match(F("ground_truth").exists() & status_filter)

    if len(view) == 0:
        sys.exit(
            f"No samples with ground_truth and status in {statuses}. "
            "Have you annotated and marked samples as 'fixed' yet?"
        )

    out = Path(output_dir)

    if fmt == "fiftyone":
        result = export_fiftyone(view, out)
    elif fmt == "yolo":
        result = export_yolo(view, out, val_ratio=val_ratio, seed=seed)
    else:
        sys.exit(f"Unknown format '{fmt}'. Choose 'yolo' or 'fiftyone'.")

    print(f"\nDataset ready at: {result.resolve()}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export FiftyOne 'fixed' samples for fine-tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--output-dir", default="runs/finetune_data",
                        help="Output directory (default: runs/finetune_data)")
    parser.add_argument("--format", default="yolo", choices=["yolo", "fiftyone"],
                        dest="fmt",
                        help="Export format: 'yolo' (default) for training scripts, "
                             "'fiftyone' for portable full-fidelity export")
    parser.add_argument("--dataset-name", default="annotation_dataset",
                        help="FiftyOne dataset name (default: annotation_dataset)")
    parser.add_argument("--val-ratio", type=float, default=0.15,
                        help="Fraction for validation set, YOLO format only (default: 0.15)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible split (default: 42)")
    parser.add_argument("--status", nargs="+", default=["fixed"],
                        dest="statuses",
                        help="Review status(es) to include (default: fixed)")
    args = parser.parse_args()

    export(
        output_dir=args.output_dir,
        fmt=args.fmt,
        val_ratio=args.val_ratio,
        seed=args.seed,
        dataset_name=args.dataset_name,
        statuses=args.statuses,
    )
