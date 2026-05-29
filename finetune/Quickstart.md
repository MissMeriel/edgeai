# Scene-split, YOLO two-stage (default):
python train_by_distribution.py --split-by scene --family yolo --technique two_stage

# Weather-split, FRCNN with LoRA rank-8:
python train_by_distribution.py --split-by weather --family frcnn --technique lora --lora-rank 8

# All samples, both families, freeze + lora combined:
python train_by_distribution.py --split-by all --family both --technique freeze lora

# From pre-exported YOLO dir, split by time of day, custom classes:
python train_by_distribution.py \
  --data-dir runs/finetune_data \
  --annotation-json ../datasets/VisDrone2019-DET-train/annotation_dataset.json \
  --split-by time --family yolo --extra-classes drone uav

# Skip images with no scene tag (common in this dataset):
python train_by_distribution.py --split-by scene --skip-untagged



Here's the full format. Each row in the annotation .txt files is:


bbox_left, bbox_top, bbox_width, bbox_height, score, object_category, truncation, occlusion
Column meanings:

Col	Name	Meaning
1	bbox_left	X of top-left corner (pixels)
2	bbox_top	Y of top-left corner (pixels)
3	bbox_width	Width (pixels)
4	bbox_height	Height (pixels)
5	score	0 = ignored region (skip), 1 = valid annotation
6	object_category	Class ID (see below)
7	truncation	0 = fully in frame, 1 = partially outside
8	occlusion	0 = none, 1 = partial, 2 = heavy
Category IDs:

ID	Class
0	ignored region
1	pedestrian
2	people
3	bicycle
4	car
5	van
6	truck
7	tricycle
8	awning-tricycle
9	bus
10	motor
11	others


# All 6471 images, split by scene (3331 untagged will go into __untagged__ group)
python train_by_distribution.py \
    --visdrone ../datasets/VisDrone2019-DET-train \
    --annotation-json ../datasets/VisDrone2019-DET-train/annotation_dataset.json \
    --split-by scene --skip-untagged --family yolo --technique two_stage

# All 6471 images as one combined model (no scene splitting):
python train_by_distribution.py \
    --visdrone ../datasets/VisDrone2019-DET-train \
    --split-by all --family yolo --technique two_stage

# Split by time of day (all 6471 images have a time tag):
python train_by_distribution.py \
    --visdrone ../datasets/VisDrone2019-DET-train \
    --annotation-json ../datasets/VisDrone2019-DET-train/annotation_dataset.json \
    --split-by time --family yolo
