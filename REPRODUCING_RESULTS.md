### Baseline assessment

Why validate_sequence.py can't do this directly: it expects a weights file for a fine-tuned model that already outputs VisDrone class names. Pretrained COCO models output COCO names — most VisDrone classes (van, tricycle, awning-tricycle, people) have no COCO equivalent and would silently score zero if passed through unchanged. baseline_eval.py applies the mapping at inference time.

#### Pre-training datasets

All base models are trained on datasets in the general-photography / street-level domain; none have seen drone or aerial imagery before fine-tuning.

##### YOLO models (Ultralytics weights)

| Model | ImageNet | COCO | Objects365 | OpenImages | Notes |
| --- | --- | --- | --- | --- | --- |
| `yolov8n/s/m/l` | ✓ | ✓ | | | Backbone pre-trained on ImageNet, detection head on COCO 2017 (118k images, 80 classes) |
| `yolo11n/s/m` | ✓ | ✓ | ✓ | | Same backbone pipeline; Objects365 (365 classes, ~600k images) added in YOLO11 for broader category coverage |
| `rtdetr-l` | ✓ | ✓ | ✓ | ✓ | Transformer encoder additionally pre-trained on Objects365 + OpenImages (~9M images); strongest generalisation of the group |

##### FRCNN models (torchvision weights)

All six FRCNN variants are loaded from torchvision's `DEFAULT` pretrained weights, which are trained on **COCO 2017** (train split, 118k images, 80 classes) via multi-scale training. No large-scale auxiliary datasets are used.

| Model | COCO 2017 | Notes |
| --- | --- | --- |
| `fasterrcnn_resnet50_v2` | ✓ | Trained with multi-scale augmentation, stronger than v1 |
| `fasterrcnn_resnet50` | ✓ | Classic Faster R-CNN training recipe |
| `fasterrcnn_mobilenet` | ✓ | Lighter backbone; same COCO training split |
| `retinanet` | ✓ | RetinaNet v2 with improved training schedule |
| `fcos` | ✓ | Anchor-free; COCO mAP competitive with Faster R-CNN |
| `ssdlite` | ✓ | 320×320 fixed input; fastest but weakest on small objects |

#### COCO → VisDrone class mapping

COCO → VisDrone mapping used:

| COCO | VisDrone |
| --- | --- |
| person | pedestrian |
| bicycle | bicycle |
| car | car |
| motorcycle | motor |
| bus | bus |
| truck | truck |

`van`, `people`, `tricycle`, `awning-tricycle` have no COCO equivalent — the baseline will always score 0 AP on those classes, which is accurate (the pretrained model genuinely can't detect them).

Usage:


# Single sequence, all models (will be slow — all 8 YOLO + 6 FRCNN):
python finetune_sequence/baseline_eval.py \
    --sequence datasets/VisDrone2019-VID-val/sequences/uav0000268_05773_v

# Just the two models you're fine-tuning, with annotated frames:
python finetune_sequence/baseline_eval.py \
    --sequence datasets/VisDrone2019-VID-val/sequences/uav0000268_05773_v \
    --yolo-models yolov8n \
    --frcnn-models fasterrcnn_resnet50_v2 \
    --save-frames

# All catalogue sequences, YOLO only, every 3rd frame:
python finetune_sequence/baseline_eval.py \
    --sequences-json datasets/sequences_categories.json \
    --frcnn-models none \
    --frame-stride 3
Results are written to runs/baseline_eval/baseline_results.json and a summary table is printed at the end.

## Visualization 

 Here's a summary of make_eval_gif.py:

What it does:

Re-runs inference on the requested time window (no pre-saved frames needed)
Draws GT boxes in green with filled label background, predictions in red with confidence score
Adds a top-left overlay per frame: model name, sequence mAP50 (if available from results JSON), and frame number
Appends a small legend bar at the bottom
Key flags:

Flag	Default	Description
--sequence	required	Path to sequence directory
--start	0	Start time in seconds
--duration	5	Duration in seconds
--min-map	0.0	Skip models below this mAP50 (requires --results)
--conf	0.25	Confidence threshold for displayed boxes
--output-fps	8	GIF playback rate
--scale	0.5	Resize factor (keeps GIF files small)
--side-by-side	off	Stitch all selected models into one wide GIF
--gt-only	off	Ground truth only, no inference
Usage patterns:


# GT-only reference clip
python finetune_sequence/make_eval_gif.py \
    --sequence datasets/VisDrone2019-VID-val/sequences/uav0000086_00000_v \
    --gt-only --start 5 --duration 8

# All models from a results file that beat 0.30 mAP50
python finetune_sequence/make_eval_gif.py \
    --sequence datasets/VisDrone2019-VID-val/sequences/uav0000086_00000_v \
    --results runs/baseline_eval/baseline_results.json \
    --min-map 0.30 --start 10 --duration 5

# Two models side by side
python finetune_sequence/make_eval_gif.py \
    --sequence datasets/VisDrone2019-VID-val/sequences/uav0000086_00000_v \
    --model yolo/yolov8n yolo/yolov8m --side-by-side --start 0 --duration 5

### Finetuning with a video sequence


### Hardware setup and script param setting

To determine training limitations, run 
`python finetune_sequence/probe_hardware.py`.

For the RTX A400 (3.67 GiB) with fasterrcnn_resnet50_v2 + temporal, the probe will recommend --batch-frcnn 1 --imgsz 640 --val-on-cpu. If even batch=1 is tight, switching to --frcnn-model fasterrcnn_mobilenet gets the base from ~700 MiB to ~250 MiB and comfortably fits.