"""
Import annotation_dataset from complete_dataset.json.

Images are read directly from their original location (datasets/VisDrone2019-DET-train/images/).
The annotation_dataset/images/ copy is not needed and can be deleted to save disk space.

Usage:
    python import_dataset.py
    python import_dataset.py --json annotation_dataset/complete_dataset.json
    python import_dataset.py --images /custom/path/to/images
    python import_dataset.py --dataset-name my_name
"""

import argparse
import json
from pathlib import Path

import fiftyone as fo


def import_dataset(
    json_path: str = "annotation_dataset/complete_dataset.json",
    images_dir: str = None,
    dataset_name: str = None,
):
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON not found: {json_path}")

    with open(json_path) as f:
        data = json.load(f)

    dataset_name = dataset_name or data["dataset_name"]

    if dataset_name in fo.list_datasets():
        response = input(f"Dataset '{dataset_name}' already exists. Delete and reimport? (y/n): ")
        if response.lower() != "y":
            print("Aborted.")
            return fo.load_dataset(dataset_name)
        fo.delete_dataset(dataset_name)

    print(f"Importing {data['total_samples']} samples into '{dataset_name}'...")

    # --- Pass 1: build all Sample objects in memory, then add in one batch ---
    samples = []
    skipped = []

    for s in data["samples"]:
        # Resolve image path: prefer original_filepath, override with images_dir if given
        if images_dir:
            img_path = Path(images_dir) / s["filename"]
        else:
            img_path = Path(s["original_filepath"])

        if not img_path.exists():
            skipped.append(s["filename"])
            continue

        sample = fo.Sample(filepath=str(img_path))
        sample.tags = s.get("tags", [])

        if s.get("ground_truth"):
            sample["ground_truth"] = fo.Detections(detections=[
                fo.Detection(label=d["label"], bounding_box=d["bounding_box"],
                             **({"confidence": d["confidence"]} if d.get("confidence") else {}))
                for d in s["ground_truth"]
            ])

        if s.get("predictions"):
            sample["predictions"] = fo.Detections(detections=[
                fo.Detection(label=d["label"], bounding_box=d["bounding_box"],
                             confidence=d.get("confidence", 0.0))
                for d in s["predictions"]
            ])

        # Store classifications as temp attributes so we can bulk-set after add_samples
        sample._import_classifications = s.get("classifications", {})
        samples.append(sample)

    if skipped:
        print(f"  Warning: {len(skipped)} images not found, skipped.")

    # One bulk insert
    dataset = fo.Dataset(dataset_name)
    dataset.persistent = True
    dataset.add_samples(samples, progress=True)

    # --- Pass 2: bulk-set each classification field ---
    # Collect per-field lists aligned to sample order
    classification_fields: dict[str, list] = {}
    for sample in samples:
        for category, label in sample._import_classifications.items():
            field_name = f"tag_{category}"
            if field_name not in classification_fields:
                classification_fields[field_name] = [None] * len(samples)

    for i, sample in enumerate(samples):
        for category, label in sample._import_classifications.items():
            field_name = f"tag_{category}"
            classification_fields[field_name][i] = fo.Classification(label=label)

    for field_name, values in classification_fields.items():
        dataset.set_values(field_name, values)
        print(f"  Set {sum(v is not None for v in values)} values for {field_name}")

    print(f"\nImported {len(samples)} samples, skipped {len(skipped)}.")
    print(f"Launch with: python -c \"import fiftyone as fo; fo.launch_app(fo.load_dataset('{dataset_name}')); import time; time.sleep(9999)\"")
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Import FiftyOne dataset from complete_dataset.json")
    parser.add_argument("--json", default="annotation_dataset/complete_dataset.json",
                        help="Path to complete_dataset.json (default: annotation_dataset/complete_dataset.json)")
    parser.add_argument("--images", default=None,
                        help="Override image directory (default: uses original_filepath from JSON)")
    parser.add_argument("--dataset-name", default=None,
                        help="Override dataset name (default: uses name from JSON)")
    args = parser.parse_args()

    import_dataset(args.json, args.images, args.dataset_name)
