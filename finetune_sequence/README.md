# finetune_sequence

Fine-tune YOLO and Faster R-CNN on VisDrone VID video sequences, grouped by the scene/time/weather/quality tags in `datasets/sequences_categories.json`.

The key design difference from `finetune/train_by_distribution.py` (which shuffles individual frames from a flat DET set): **sequences are treated as atomic units**. Whole sequences go to train or val — never split across both. This prevents data leakage from temporally adjacent near-identical frames and makes validation meaningful.

---

## Files

| File | Purpose |
|---|---|
| `train_sequence.py` | Training pipeline — exports frames, builds datasets, trains models |
| `validate_sequence.py` | Evaluation — runs inference on a held-out sequence, reports mAP |
| `draw_architectures.py` | Generates architecture diagrams (see `diagrams/`) |

## Architecture diagrams

Pre-rendered SVGs showing which parts of each model each fine-tuning technique modifies:

| Diagram | Description |
|---|---|
| [diagrams/overview.svg](diagrams/overview.svg) | Layer × technique matrix for both YOLO and FRCNN |
| [diagrams/yolo_techniques.svg](diagrams/yolo_techniques.svg) | YOLO architecture per technique (one column per technique) |
| [diagrams/frcnn_techniques.svg](diagrams/frcnn_techniques.svg) | Faster R-CNN / RetinaNet / FCOS architecture per technique |

Regenerate with:

```bash
python finetune_sequence/draw_architectures.py          # SVG (default)
python finetune_sequence/draw_architectures.py --format png
```

**Colour key** (consistent across all diagrams):

| Colour | Meaning |
|---|---|
| Slate blue | Frozen — weights unchanged |
| Red | Trained — full learning rate |
| Teal | Low LR — trained at 0.01–0.1× backbone learning rate |
| Amber | LoRA A/B matrices — trainable low-rank injection |
| Green | New head — freshly initialised, always trained |
| Purple | Temporal loss path / adjacent-frame augmentation |

---

## Dataset coverage

24 annotated sequences across two VisDrone VID splits:

| Split | Sequences |
|---|---|
| `VisDrone2019-VID-val` | 7 sequences |
| `VisDrone2019-VID-test-dev` | 17 sequences |

Each sequence is tagged in `datasets/sequences_categories.json` with `scene`, `time`, `weather`, and `quality`.

---

## Grouping modes (`--group-by`)

Controls how sequences are combined into training datasets. One model is trained per group.

| Mode | Groups produced | Sequences per group |
|---|---|---|
| `none` / `all` | 1 combined model | all 24 |
| `scene` | city\_street (15), highway (4), parking\_lot (3), recreation\_area (2) | 2–15 |
| `time` | day (18), night (4), dawn (2) | 2–18 |
| `weather` | cloudy (14), clear (6), sunny (4) | 4–14 |
| `scene+time` | cross-product, e.g. city\_street\_\_night | 1–12 |

Groups with fewer sequences than `--min-sequences` (default: 2) are skipped.

---

## Fine-tuning techniques (`--technique`)

Techniques are combinable: `--technique two_stage temporal`.

### `freeze`
Backbone frozen entirely; only detection heads are trained.

**When to use:** very small sequence sets (<5 sequences, ~500 frames). Prevents catastrophic forgetting of pretrained COCO features. Fastest to train.

**Resource consumption:** Cheapest of all techniques. Gradients are computed only for the head, so VRAM usage and step time are minimal — roughly 2–3× lower than full fine-tuning. Optimizer state (Adam momentum buffers) is also limited to head parameters, keeping checkpoint sizes small. Good choice when GPU memory is the binding constraint or when you need many quick experiments.

### `lora` *(FRCNN only)*
Low-rank adaptation: backbone weights are frozen; tiny trainable A/B matrices (rank=4 by default, set with `--lora-rank`) are inserted into backbone Linear layers. The backbone update `ΔW ≈ B·A` is constrained to a low-dimensional subspace.

**When to use:** domain shift from COCO → drone footage where you want to preserve pretrained features as much as possible. Very parameter-efficient (~1–5% of parameters trainable). YOLO approximates this with `freeze + lower lr`.

**Resource consumption:** Marginally more expensive than `freeze` — the A/B matrices add a small number of trainable parameters (proportional to `lora_rank × layer_width`), but the backbone itself still requires no gradient storage. In practice the cost difference from `freeze` is negligible; the main overhead is a modest increase in checkpoint size from the extra adapter weights. Prefer this over `freeze` when you suspect the backbone needs light adaptation but cannot afford a full backward pass.

### `two_stage` *(default)*
**Phase 1** (first 1/3 of epochs): head-only with backbone frozen.
**Phase 2** (remaining epochs): full network fine-tune with a low backbone LR (0.01×).

**When to use:** general default. Gives a quick initial adaptation then careful global refinement. Works well across all group sizes.

**Resource consumption:** Wall-clock cost sits between `freeze` and `full`. Phase 1 is cheap (head only); Phase 2 pays the full fine-tuning cost but starts from a better initialisation, so fewer phase-2 epochs are typically needed before early stopping. There is also a one-time overhead of reloading the model between phases. On tight GPU budgets, reducing `--epochs-frcnn` shortens Phase 2 without sacrificing Phase 1 gains.

### `full`
Fine-tunes everything from epoch 0 with differential LR: backbone at 0.1× the head learning rate.

**When to use:** sequences are long and cover sufficient variation (≥5 sequences or ≥1000 frames). Highest capacity but risks catastrophic forgetting on small groups.

**Resource consumption:** Most expensive single-technique option in terms of per-step cost. Every layer accumulates gradients and carries optimizer state, roughly doubling VRAM versus `freeze` on ResNet-50. Step time increases proportionally with backbone depth. This is the right trade-off when the dataset is large enough that the model genuinely needs to adapt its feature extractor, but it should be paired with careful early stopping on small groups.

### `cosine` *(FRCNN only)*
Cosine LR annealing (`CosineAnnealingLR`) instead of the default MultiStep schedule. Combinable with all other techniques.

**When to use:** smoother convergence; useful when training duration is uncertain or early stopping kicks in at irregular epochs.

**Resource consumption:** Per-step cost is identical to whatever base technique it is combined with — `cosine` only changes the learning rate schedule, not which parameters are updated. The practical expense comes from run time: smooth cosine decay tends to keep loss decreasing for more epochs before plateauing, so training often runs longer before early stopping triggers compared to MultiStep. Budget ~10–20% more wall-clock time versus the same technique with MultiStep.

### `temporal`
Adds an adjacent-frame consistency loss during training. For each batch, the next (or previous) frame from the same sequence is run through the model at 0.3× loss weight, regularising the model to produce stable detections across frames.

- **FRCNN**: explicit temporal loss term — adjacent-frame forward pass added to each training step.
- **YOLO**: approximated by increasing `copy_paste` (0.3 vs 0.1) and `close_mosaic` (20 vs 10) so mosaic augmentation persists longer.

**When to use:** sequence data specifically, when frame-to-frame jitter in detections matters (e.g. tracking downstream consumers).

**Resource consumption:** Most expensive technique. Each training step runs two forward + backward passes (the primary frame and its temporal neighbour), increasing GPU memory and compute by roughly 30% versus the equivalent non-temporal run. Convergence is also slower because the temporal regularisation term can conflict with the detection loss early in training. Only enable this when downstream stability across frames is a hard requirement; for pure mAP optimisation it rarely outperforms `full` at equal epoch budget.

---

## Model families (`--family`)

### YOLO (`--yolo-model`)

| Key | Weights | Params | Notes |
|---|---|---|---|
| `yolov8n` *(default)* | yolov8n.pt | ~3M | Fastest, edge-deployable. CSP-DarkNet53 backbone with C2f bottleneck blocks; decoupled classification and regression heads (separate conv branches per task). Anchor-free with distribution focal loss (DFL) for sub-pixel box regression. |
| `yolov8s` | yolov8s.pt | ~11M | Better accuracy, still fast. Same C2f architecture as `n` but wider channels throughout backbone and neck; the additional capacity meaningfully improves recall on small/dense objects at modest latency cost. |
| `yolov8m` | yolov8m.pt | ~26M | Good accuracy/speed tradeoff. Deeper C2f stacks and a wider FPN/PAN neck than `s`; the extra depth helps multi-scale feature fusion, which matters for VisDrone's mix of distant pedestrians and nearby vehicles. |
| `yolov8l` | yolov8l.pt | ~44M | High accuracy. Largest standard YOLOv8 variant; same architecture as `m/s` but maximally wide — primarily relevant when GPU memory is not a constraint and per-frame throughput is secondary. |
| `yolo11n` | yolo11n.pt | ~3M | YOLOv11 nano. Replaces C2f with C3k2 blocks (cross-stage partial with two smaller kernels) and adds a C2PSA attention module in the backbone's final stage; better feature selectivity at the same parameter count as `yolov8n`. |
| `yolo11s` | yolo11s.pt | — | YOLOv11 small. C3k2 + C2PSA backbone scaled up from nano; the attention stage gives better recall on occluded objects compared to the equivalent YOLOv8s, at slightly higher latency. |
| `yolo11m` | yolo11m.pt | — | Best YOLOv11 accuracy/speed. Wider C3k2 stacks and a larger C2PSA window than `11s`; typically outperforms `yolov8l` on small-object benchmarks while being faster due to the more efficient attention design. |
| `rtdetr-l` | rtdetr-l.pt | — | Transformer-based, no NMS, slower but strong. Uses a ResNet-50 + Hybrid Encoder (CNN feature extraction feeding a transformer encoder) with a set-prediction decoder; outputs a fixed set of non-duplicate boxes directly, eliminating NMS entirely. Slower per-frame but removes the NMS confidence/IoU threshold as a tuning variable and handles overlapping objects more robustly. |

**NMS (Non-Maximum Suppression)** is the post-processing step used by all YOLO variants to collapse duplicate box proposals: after the head produces hundreds of candidate boxes, NMS keeps only the highest-confidence box within each cluster of overlapping detections (controlled by the `iou` and `conf` thresholds). For dense VisDrone scenes — pedestrians and vehicles packed tightly together — NMS thresholds are a meaningful tuning knob; too aggressive and nearby-but-distinct objects are suppressed, too loose and duplicates inflate false positives. RT-DETR's set-prediction decoder sidesteps this entirely by design, which can help in the densest frames but removes a familiar lever for precision/recall trade-offs.

#### Pre-training datasets

| Key | ImageNet | COCO | Objects365 | OpenImages |
|---|---|---|---|---|
| `yolov8n` | ✓ | ✓ | | |
| `yolov8s` | ✓ | ✓ | | |
| `yolov8m` | ✓ | ✓ | | |
| `yolov8l` | ✓ | ✓ | | |
| `yolo11n` | ✓ | ✓ | ✓ | |
| `yolo11s` | ✓ | ✓ | ✓ | |
| `yolo11m` | ✓ | ✓ | ✓ | |
| `rtdetr-l` | ✓ | ✓ | ✓ | ✓ |

### FRCNN (`--frcnn-model`)

| Key | Architecture | Notes |
|---|---|---|
| `fasterrcnn_resnet50_v2` *(default)* | Faster R-CNN ResNet-50 FPN v2 | Strongest two-stage model |
| `fasterrcnn_resnet50` | Faster R-CNN ResNet-50 FPN v1 | Classic baseline |
| `fasterrcnn_mobilenet` | Faster R-CNN MobileNetV3 | Lightweight, edge-deployable |
| `retinanet` | RetinaNet ResNet-50 FPN v2 | One-stage, focal loss, good recall |
| `fcos` | FCOS ResNet-50 FPN | Anchor-free, good for irregular/small objects |
| `ssdlite` | SSDLite MobileNetV3 320 | Fastest inference, lowest memory |

All FRCNN models use small-object anchors tuned for drone footage: `(8, 16, 32, 64, 128)` px instead of the COCO default `(32–512)` px.

---

## Validation sequence selection

Validation uses entire held-out sequences — never individual frames — to avoid leakage.

**`--val-sequences SEQ_NAME [...]`**
Explicitly name one or more sequences to hold out (by bare directory name, e.g. `uav0000268_05773_v`). Applied across all groups; groups where none of the named sequences appear fall back to random selection.

**`--val-split-by {random | scene | time}`** *(used when `--val-sequences` is not given)*
- `random`: pick one sequence per group at random (seed-controlled).
- `scene` / `time`: pick one sequence per unique tag value within each group, capped at 30% of the group.

---

## Frame stride

`--frame-stride N` samples every Nth frame from each sequence before exporting. This reduces dataset size and inter-frame redundancy (consecutive frames are often near-identical).

| Stride | Effect |
|---|---|
| 1 *(default)* | All frames — full temporal density |
| 2 | Every other frame — ~50% dataset size |
| 3 | Every 3rd frame — ~33% — good default for long sequences |
| 5 | Every 5th frame — ~20% — fast experiments |

---

## Usage examples

### Training

```bash
# All sequences combined, two_stage YOLO, hold out one sequence:
python finetune_sequence/train_sequence.py \
    --sequences-json datasets/sequences_categories.json \
    --group-by none \
    --val-sequences uav0000268_05773_v \
    --family yolo --technique two_stage

# Scene-specialist models, auto-select val sequence per scene:
python finetune_sequence/train_sequence.py \
    --sequences-json datasets/sequences_categories.json \
    --group-by scene \
    --val-split-by random \
    --family both --technique two_stage temporal

# Time-of-day specialists, FRCNN + LoRA, sample every 3rd frame:
python finetune_sequence/train_sequence.py \
    --sequences-json datasets/sequences_categories.json \
    --group-by time \
    --val-split-by random \
    --family frcnn --technique lora \
    --frame-stride 3

# Scene×time cross-product, full fine-tune, YOLOv8s:
python finetune_sequence/train_sequence.py \
    --sequences-json datasets/sequences_categories.json \
    --group-by scene+time \
    --val-split-by random \
    --family yolo --yolo-model yolov8s --technique full

# Replay a previous run (all flags read from saved config, project gets a fresh directory):
python finetune_sequence/train_sequence.py \
    --config runs/seq_finetune-.../training_config.json
```

### Validation

```bash
# Validate a YOLO model against a single sequence (annotation auto-detected):
python finetune_sequence/validate_sequence.py \
    --weights runs/seq_finetune-.../models/highway/yolo/train/weights/best.pt \
    --sequence datasets/VisDrone2019-VID-val/sequences/uav0000268_05773_v

# Validate FRCNN with explicit annotation path:
python finetune_sequence/validate_sequence.py \
    --weights runs/seq_finetune-.../models/city_street/frcnn/best.pth \
    --sequence datasets/VisDrone2019-VID-test-dev/sequences/uav0000249_00001_v \
    --annotation datasets/VisDrone2019-VID-test-dev/annotations/uav0000249_00001_v.txt

# Save annotated frames (GT=green, predictions=red) for visual inspection:
python finetune_sequence/validate_sequence.py \
    --weights best.pt \
    --sequence datasets/VisDrone2019-VID-val/sequences/uav0000268_05773_v \
    --save-frames

# Batch-validate all recorded val sequences from a training run:
python finetune_sequence/validate_sequence.py \
    --summary runs/seq_finetune-.../training_summary.json \
    --sequences-json datasets/sequences_categories.json

# Faster evaluation: every 5th frame, lower confidence threshold:
python finetune_sequence/validate_sequence.py \
    --weights best.pt \
    --sequence datasets/VisDrone2019-VID-val/sequences/uav0000268_05773_v \
    --frame-stride 5 --conf 0.1
```

---

## Output structure

```
runs/seq_finetune-<slug>-<hash>/
  training_config.json        # all CLI flags — pass to --config to replay
  training_summary.json       # per-group per-family results (weights, mAP, loss)
  data/
    city_street/              # exported YOLO dataset for this group
      images/train/
      images/val/
      labels/train/
      labels/val/
      dataset.yaml
    highway/
    ...
  models/
    city_street/
      yolo/train/weights/best.pt
      frcnn/best.pth
      frcnn/history.json
    highway/
    ...
  val_frames/                 # only with --save-frames in validate_sequence.py
    city_street/yolo/
      uav0000117_02622_v/
        0000001.jpg           # annotated: GT green, predictions red
        ...
```

After validation, `validate_sequence.py` writes:
- `val_<seq_name>.json` (single sequence mode) or `validation_results.json` (batch mode) next to the weights file.

---

## Metrics reported

`validate_sequence.py` computes COCO-style metrics:

| Metric | Description |
|---|---|
| **mAP50** | Mean AP at IoU=0.5 across all classes |
| **mAP50-95** | Mean AP averaged over IoU 0.50:0.05:0.95 (primary COCO metric) |
| **Precision** | TP / (TP + FP) at IoU=0.5 |
| **Recall** | TP / (TP + FN) at IoU=0.5 |
| **Per-class AP** | AP at IoU=0.5 for each of the 10 VisDrone classes |
| **FPS** | Inference throughput on the evaluated hardware |

---

## Key parameters reference

| Flag | Default | Notes |
|---|---|---|
| `--sequences-json` | `../datasets/sequences_categories.json` | Tag catalogue |
| `--group-by` | `scene` | Grouping dimension |
| `--family` | `yolo` | `yolo`, `frcnn`, or `both` |
| `--technique` | `two_stage` | Combinable; see above |
| `--val-sequences` | *(auto)* | Explicit hold-out sequence names |
| `--val-split-by` | `random` | Auto val selection strategy |
| `--frame-stride` | `1` | Sample every Nth frame |
| `--epochs-yolo` | `100` | Early stopping at patience=20 |
| `--epochs-frcnn` | `20` | Early stopping at patience=10 |
| `--batch-yolo` | `16` | Reduce if OOM |
| `--batch-frcnn` | `4` | Two-stage models are memory-intensive |
| `--imgsz` | `640` | Input resolution |
| `--lr` | `0.005` | Base LR for FRCNN (YOLO uses its own schedule) |
| `--lora-rank` | `4` | LoRA rank; increase to 8–16 for larger datasets |
| `--min-sequences` | `2` | Skip groups with fewer sequences than this |
| `--seed` | `42` | Controls val sequence random selection |
| `--device` | *(auto)* | `cpu`, `cuda`, `cuda:0`, `mps` |
