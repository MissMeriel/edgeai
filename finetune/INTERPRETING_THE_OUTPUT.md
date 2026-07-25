# Interpreting Training Output

This document explains every file written to a training run directory and how to read the graphs they contain.

---

## Run directory location and naming

Both `train_by_distribution.py` and `finetune_sequence/train_sequence.py` write runs under:

```
finetune/runs/        ← dist_finetune runs
runs/                 ← seq_finetune runs (repo root)
```

The directory name encodes the key parameters:

```
dist_finetune-split-by-scene-yolo-yolov8n-epochs-50-two-stage-<hash>/
seq_finetune-group-scene-frcnn-fasterrcnn-resnet50-v2-frcnn-epochs-20-two-stage-<hash>/
```

Pattern: `<script>-<grouping>-<family>-<model>-epochs-<N>-<technique>-<8-char hash>`

The hash is derived from the config so re-runs with identical settings overwrite the same directory instead of creating duplicates.

---

## Top-level files

```
<run_dir>/
    training_config.json
    training_summary.json
    data/
    models/
```

### `training_config.json`

All CLI arguments frozen at run time. Use this to reproduce a run exactly or to understand what settings produced a given result.

```json
{
  "family": "yolo",
  "model": "yolov8n",
  "epochs": 50,
  "technique": "two_stage",
  "group_by": "scene",
  "sequences_json": "datasets/sequences_categories.json",
  "val_sequences": ["uav0000268_05773_v"],
  ...
}
```

### `training_summary.json`

Final evaluation metrics for each group written at the end of the run. The top-level keys are the group names (scene values, `"all"`, etc.).

```json
{
  "city_street": {
    "mAP50": 0.412,
    "mAP50_95": 0.231,
    "precision": 0.558,
    "recall": 0.387,
    "model_path": "models/city_street/yolo/train/weights/best.pt"
  },
  "highway": {
    "error": "No training samples found for group highway"
  }
}
```

A group with an `"error"` key failed — the metrics keys will be absent. The most common causes are an empty training split, a data export problem, or the `BoundingBoxes` transform error (fixed in the current code). Load this file to compare runs programmatically or in the training dashboard.

---

## Data directory

```
data/<group>/
    dataset.yaml
    images/
        train/   *.jpg / *.png
        val/     *.jpg / *.png
    labels/
        train/   *.txt    (YOLO format: class cx cy w h, normalised)
        val/     *.txt
```

`dataset.yaml` declares the class names and the `train`/`val` paths. YOLO training reads this file directly.

Label files are plain text, one detection per line:

```
2 0.512 0.431 0.042 0.089
```

`class_index  cx  cy  width  height` — all values normalised 0–1 relative to image dimensions.

---

## YOLO model output

### Two-stage layout

When `--technique two_stage` is used, YOLO trains in two phases:

```
models/<group>/yolo/
    phase1/    ← head-only pass: backbone frozen, only detection heads trained
    train/     ← full fine-tune: all layers unlocked at lower LR
```

If `--technique` is anything other than `two_stage`, only the `train/` directory exists.

Always use `train/weights/best.pt` as your final model. `phase1/weights/best.pt` is an intermediate checkpoint used to initialise `train/`.

### Files inside each phase directory

```
phase1/ or train/
    args.yaml
    results.csv
    results.png
    weights/
        best.pt
        last.pt
        epoch0.pt, epoch10.pt, ...   (save_period=10, train/ only)
    BoxF1_curve.png
    BoxP_curve.png
    BoxR_curve.png
    BoxPR_curve.png
    confusion_matrix.png
    confusion_matrix_normalized.png
    labels.jpg
    train_batch0.jpg
    train_batch1.jpg
    train_batch2.jpg
    train_batch<N>.jpg         (last epoch)
    val_batch0_labels.jpg
    val_batch0_pred.jpg
    val_batch1_labels.jpg
    val_batch1_pred.jpg
    val_batch2_labels.jpg
    val_batch2_pred.jpg
```

#### `args.yaml`

All YOLO hyperparameters used for this phase. Includes `lr0`, `lrf`, `momentum`, `weight_decay`, `warmup_epochs`, `mosaic`, `mixup`, image size, and anchor settings. Useful for debugging or replicating training outside the pipeline.

#### `results.csv`

One row per epoch. Columns:

| Column | Meaning |
|---|---|
| `epoch` | Epoch index (0-based) |
| `train/box_loss` | Bounding-box regression loss (CIoU) on train set |
| `train/cls_loss` | Classification loss on train set |
| `train/dfl_loss` | Distribution Focal Loss (localisation fine detail) on train set |
| `metrics/precision(B)` | Precision at conf threshold maximising F1 on val set |
| `metrics/recall(B)` | Recall at same threshold |
| `metrics/mAP50(B)` | mAP @ IoU=0.50 on val set |
| `metrics/mAP50-95(B)` | mAP averaged over IoU=0.50:0.05:0.95 on val set |
| `val/box_loss` | Box regression loss on val set |
| `val/cls_loss` | Classification loss on val set |
| `val/dfl_loss` | DFL loss on val set |
| `lr/pg0 … lr/pg7` | Per-parameter-group learning rates at end of epoch |

`best.pt` is saved at the epoch with the highest `metrics/mAP50-95(B)`.

#### `results.png`

A 2-row × 5-column grid. Top row (training metrics):

```
train/box_loss | train/cls_loss | train/dfl_loss | precision | recall
```

Bottom row (validation metrics):

```
val/box_loss | val/cls_loss | val/dfl_loss | mAP50 | mAP50-95
```

Each panel shows raw values in blue and an exponentially-smoothed overlay in orange.

**What to look for:**

- **Losses (columns 1–3):** Should decrease and flatten. If training loss keeps falling but val loss flattens or rises after the midpoint, the model is overfitting — either add augmentation, reduce epochs, or collect more data.
- **Precision and recall (columns 4–5):** Should rise as training progresses. Noisy behaviour here usually means the val set is small (< 50 images) or has a different scene distribution than train.
- **mAP50 (column 9):** The primary metric for drone-imagery detection. A well-converged run on VisDrone sequences typically reaches 0.3–0.5 depending on scene complexity and the number of training images.
- **mAP50-95 (column 10):** Stricter — averages over multiple IoU thresholds. Expect it to be roughly 0.4–0.6× the mAP50 value. Low mAP50-95 with decent mAP50 means the model localises objects roughly but not precisely.
- **Flatline from epoch 0:** If mAP stays near zero for the first 10–15 epochs, the learning rate is too low, the data directory is misconfigured, or there are effectively no positive training examples in the split.

#### `BoxPR_curve.png`

Precision-Recall curve per class, with the overall mAP@0.5 shown in the legend for each class and the `all classes` mean as a thick blue line.

**How to read it:** A curve that extends far to the right before dropping (large area under the curve) is better. Pedestrian and bicycle classes on VisDrone often have near-zero area — this reflects how small these objects are relative to the image resolution; it is normal and does not indicate a bug.

#### `BoxP_curve.png` and `BoxR_curve.png`

Precision and recall as a function of the detection confidence threshold. Use these to choose an operating point:

- Higher confidence threshold → higher precision, lower recall (fewer but more reliable detections).
- Lower confidence threshold → higher recall, lower precision (catches more objects, more false positives).

For downstream annotation assistance, a threshold around 0.2–0.3 is typical.

#### `BoxF1_curve.png`

F1 score (harmonic mean of precision and recall) vs confidence threshold. The peak of this curve is the threshold that best balances precision and recall for each class. The vertical dashed line marks the threshold chosen at evaluation time.

#### `confusion_matrix.png`

Raw count confusion matrix. Each column is the true class; each row is the predicted class. The `background` row counts missed detections (false negatives); the `background` column counts false positives.

#### `confusion_matrix_normalized.png`

Same matrix normalised by the true-class count (each column sums to 1.0). This is more interpretable when class sizes are imbalanced. Diagonal values are per-class recall. Off-diagonal values show which classes are confused with each other.

**Common patterns on VisDrone:**

- High `background` column values (e.g. `pedestrian → background = 1.0`) mean the model almost never detects that class — the objects are too small or the class has too few training samples.
- `car → van` or `van → car` confusion is expected because these classes share visual features at drone altitude.
- `background → background = 1.0` is expected and not an error.

#### `labels.jpg`

Grid of all validation labels (ground-truth bounding boxes drawn on representative images). Use this to visually inspect whether the dataset was loaded and formatted correctly before reading the metrics. If boxes look wildly wrong (misaligned, all at image edges), there is a label coordinate format problem.

#### `train_batch0/1/2.jpg`

Mosaic-augmented training batches from early epochs. These show the augmentation pipeline (random crop, mosaic, colour jitter) applied to your training data. If images look completely distorted or empty, the data pipeline has a problem.

The three additional batch images written at the last epoch (named `train_batch<N>.jpg` with high N) show what the model was seeing at the end of training — useful for confirming that augmentation was consistent throughout.

#### `val_batch*_labels.jpg` and `val_batch*_pred.jpg`

Side-by-side ground truth vs model predictions on the validation set. The `_labels` image shows the true bounding boxes; the `_pred` image shows what the model predicted at evaluation confidence. Compare these to diagnose whether the model is:

- Missing detections (ground truth shows boxes, predictions do not)
- Producing false positives (predictions show boxes where no ground truth exists)
- Misclassifying correctly localised objects (box is in the right place but wrong label colour)

#### `weights/best.pt`

Checkpoint at the epoch with the highest `mAP50-95`. This is the model to use for inference and validation.

#### `weights/last.pt`

Checkpoint from the final epoch. Usually worse than `best.pt` unless training was cut short early. Useful for resuming training.

#### `weights/epoch0.pt, epoch10.pt, ...` (train/ only)

Periodic checkpoints saved every 10 epochs. Useful for diagnosing at what point training peaked or diverged. Only written in the `train/` phase, not in `phase1/`.

---

## Faster R-CNN model output

```
models/<group>/frcnn/
    best.pth
    checkpoint_epoch<N>.pth    (one per save_period, typically every 5 epochs)
```

FRCNN runs do not produce the YOLO diagnostic plots. Metrics are written only to `training_summary.json`. Use `validate.py` to generate a full evaluation report against a held-out test set.

`best.pth` is the checkpoint with the best validation mAP. `checkpoint_epoch<N>.pth` files are periodic saves for resumability.

---

## Typical signs of a healthy run

| Signal | Location |
|---|---|
| All three training losses decrease smoothly over the first 30–40 epochs | `results.png` top row |
| Validation losses track training losses without diverging | `results.png` bottom row |
| mAP50 reaches ≥ 0.3 on a reasonably-sized VisDrone split | `results.png` col 9 |
| Diagonal values in the normalised confusion matrix are ≥ 0.3 for car/van | `confusion_matrix_normalized.png` |
| PR curves extend to recall ≥ 0.3 before dropping | `BoxPR_curve.png` |
| Val batch predictions visually match the labels images | `val_batch*_pred.jpg` |

## Typical signs of a problem

| Symptom | Likely cause |
|---|---|
| mAP is 0.00 for all epochs | Empty val split, label format mismatch, or IoU threshold too strict |
| `"error"` key in `training_summary.json` | Training crashed — see the run log or re-run with `--verbose` |
| Val loss rises while train loss falls | Overfitting — reduce epochs or augment more aggressively |
| All predictions in `val_batch*_pred.jpg` are background | Confidence threshold too high at eval, or model has not converged |
| Confusion matrix shows all mass in `background` column | Val set has no positive examples, or sequence split left all of one class in train |
| Results plateau in `phase1` but `train/` never improves | Two-stage LR needs tuning; try `--technique full` instead |
