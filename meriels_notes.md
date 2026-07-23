# EdgeAI
## July 20 2026
finetune_sequence

in root:
python sequence_distance.py \
  datasets/VisDrone2019-VID-val/sequences/uav0000305_00000_v \
  datasets/VisDrone2019-VID-val/sequences/uav0000339_00001_v \
  --max-frames 60

## Instructions for Human Labellers
This pipeline relies on Python 3.12.8. Follow the [link](https://www.python.org/downloads/release/python-3128/) for installation instructions. After installation, check what python version you have active in your terminal by running `python --version`. Your output should match `Python 3.12.8`.

Next setup your environment:
```bash
./install_mac.sh # if your machine is a mac
./install.bat # if your machine is windows
./install.sh # if your machine is linux
```

Then unzip `fiftyone_labelling/annotation_dataset.zip` to wherever you want your data to live.

Download your dataset to the `datasets/` folder by following this link for the [VisDrone trainset (7.53 GB) under Task 2: Object Detection in Videos](https://github.com/VisDrone/VisDrone-Dataset)

Run the labelling UI:
```bash
cd fiftyone_labelling/
# setup your fiftyone dataset. this will persist between runs of the following scripts if you use the same --dataset-name because it's saved locally in a mongodb if you set persistent=True
python import_dataset.py --images /Users/mvonstein/edgeai/datasets/VisDrone2019-DET-train/images/ --json /Users/mvonstein/edgeai/datasets/VisDrone2019-DET-train/complete_dataset.json --dataset-name annotation_dataset
# new dataset, import all tags and export everything
python fiftyone_human_annotation.py /Users/mvonstein/edgeai/datasets/VisDrone2019-DET-train/images/ --dataset-name annotation_dataset  --import-tags tags.json --export annotation_dataset --export-tags annotation_dataset.json --format coco
# export just the new bboxes
python fiftyone_human_annotation.py ../datasets/VisDrone2019-DET-train/images/  --no-auto-predict  --import-tags tags.json --export annotation_dataset --format yolo
# existing dataset, no longer need to import tags
python fiftyone_human_annotation.py /Users/mvonstein/edgeai/datasets/VisDrone2019-DET-train/images/ --dataset-name annotation_dataset --export annotation_dataset --export-tags annotation_dataset.json --format coco
```

Export your dataset to a zip from your mongodb if you want to port it between systems or who knows what else:
```bash
python fix_export.py annotation_dataset 
```

The first 11 images are examples for how a properly labelled images will look. Use those as references. Spend some time playing with the interface so you understand how it works. Then, time how long it takes you to label 10 images by filling out the following chart:


## Warning box

| :exclamation:  If you use a different name for --dataset-name, you're creating a new mongodb reference. So, changes to annotation_dataset won't persist to new_dataset_a9e013f (for example)   |
|----------------------------------------------|

> [!CAUTION]
> Advises about risks or negative outcomes of certain actions.
Caution

## Quickstart
Installation:
```bash
./install.sh
```

Generate new data for human labelling using interfaces other than FiftyOne:
```bash
# Run video frame extraction for video datasets
python3 video_frame_extractor_fixed.py path/to/videos
# Run human annotation with web interface on provided test data (open in firefox)
python3 gradio_annotation_ui_synced_rectangles.py ./test_data/Explosion004_x264_24_20260327_104203_5324ec27/

```

## All Candidate Datasets
Download and unzip to `datasets/` directory, which is created by the `install.sh` script.

- VIRAT shaky drone footage: [VIRAT dataset](https://viratdata.org/#getting-data)

- CCTV: [kaggle download](https://www.kaggle.com/datasets/jonathannield/cctv-action-recognition-dataset)

- DCSASS: [kaggle download](https://www.kaggle.com/datasets/mateohervas/dcsass-dataset)

- xView natural disaster images: make an account at [xView2](https://xview2.org/) to access. I recommend starting with the Challenge training set (~7.8 GB).s

- VisDrone dataset: [github link to datasets](https://github.com/VisDrone/VisDrone-Dataset)

- Create your own classification dataset by running `python3 create_vehicle_dataset.py`

## Troubleshooting

If the image annoation UI does not load the files, use the fully qualified pathname to the image directory or run:
```bash
python3 fix_image_loading.py ./test_data/Explosion004_x264_24_20260327_104203_5324ec27/
```

## Project status


### Roadmap

### TODOs

### Support 
Meriel von Stein, CS PhD, RAND Info Scientist, mvonstein@rand.org

### Contributing

## Random notes
If you're exporting to YOLLO or COCO format you're constraining yourself to these [80 classes](https://gist.github.com/rcland12/dc48e1963268ff98c8b2c4543e7a9be8)