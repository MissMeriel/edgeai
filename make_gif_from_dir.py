import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw


BBox = Tuple[int, int, int, int]


def parse_framewise_bbox_file(path: Path) -> Dict[int, List[BBox]]:
    """
    Parse a VisDrone-style text file containing bounding boxes for multiple frames.

    Expected VisDrone VID format (CSV, one box per line):
        frame_id, target_id, left, top, width, height, score, object_category, truncation, occlusion

    We interpret:
        - frame_index = frame_id - 1 (0-based index)
        - (x_min, y_min) = (left, top)
        - (x_max, y_max) = (left + width, top + height)

    Lines starting with '#' or empty lines are ignored.
    """
    mapping: Dict[int, List[BBox]] = {}
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 6:
                continue
            try:
                frame_id = int(parts[0])
                # VisDrone frame ids are 1-based; convert to 0-based index
                frame_idx = frame_id - 1
                left = int(parts[2])
                top = int(parts[3])
                width = int(parts[4])
                height = int(parts[5])
            except ValueError:
                continue
            x_min, y_min = left, top
            x_max, y_max = left + width, top + height
            mapping.setdefault(frame_idx, []).append(
                (x_min, y_min, x_max, y_max)
            )
    return mapping


def draw_bboxes(image: Image.Image, boxes: List[BBox]) -> Image.Image:
    if not boxes:
        return image

    draw = ImageDraw.Draw(image)
    for x_min, y_min, x_max, y_max in boxes:
        draw.rectangle((x_min, y_min, x_max, y_max), outline="red", width=2)
    return image


def images_to_gif(
    image_dir: Path,
    output_path: Path,
    duration_ms: int = 100,
    bbox_file: Optional[Path] = None,
    max_seconds: Optional[float] = None,
) -> None:
    if not image_dir.is_dir():
        raise ValueError(f"{image_dir} is not a directory")

    # Collect and sort common image types by filename
    image_paths = sorted(
        [
            p
            for p in image_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        ]
    )

    if not image_paths:
        raise ValueError(f"No images found in {image_dir}")

    framewise_boxes: Dict[int, List[BBox]] = {}
    if bbox_file is not None:
        if not bbox_file.is_file():
            raise ValueError(f"{bbox_file} is not a file")
        framewise_boxes = parse_framewise_bbox_file(bbox_file)

    # If max_seconds is provided, truncate the number of frames accordingly.
    max_frames: Optional[int] = None
    if max_seconds is not None and max_seconds > 0:
        # duration_ms is per-frame duration
        frames_per_second = 1000.0 / duration_ms
        max_frames = int(max_seconds * frames_per_second)
        if max_frames <= 0:
            max_frames = 1

    frames = []
    for idx, p in enumerate(image_paths):
        if max_frames is not None and idx >= max_frames:
            break
        img = Image.open(p).convert("RGB")
        boxes = framewise_boxes.get(idx, [])
        frames.append(draw_bboxes(img, boxes))

    first, *rest = frames
    first.save(
        output_path,
        save_all=True,
        append_images=rest,
        duration=duration_ms,
        loop=0,
    )


def build_default_output_path(image_dir: Path) -> Path:
    """
    Build default output GIF path following user example:

    /.../datasets/VisDrone2019-VID-test-challenge/sequences/uav0000006_06900_v
      -> VisDrone2019-VID-test-challenge-uav0000006_06900_v.gif
    """
    # sequences directory is one level up from the image directory
    parent = image_dir.parent
    grandparent = parent.parent

    # If the directory structure matches .../<split>/sequences/<sequence_name>
    # we join the split directory name and sequence directory name.
    if grandparent.exists():
        return Path(
            f"{grandparent.name}-{image_dir.name}.gif"
        ).resolve()

    # Fallback: just use the directory name
    return Path(f"{image_dir.name}.gif").resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Turn an ordered directory of images into an animated GIF."
    )
    parser.add_argument(
        "image_dir",
        type=Path,
        help="Directory containing ordered image files.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output GIF path. If omitted, a name is inferred from the directory.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="Frames per second for the GIF (default: 10).",
    )
    parser.add_argument(
        "--bbox-file",
        type=Path,
        default=None,
        help=(
            "Optional text file with bounding boxes for frames. "
            "Each line: 'frame_index x_min y_min x_max y_max' with "
            "0-based frame_index matching the sorted image order."
        ),
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help=(
            "Optional maximum duration of the GIF in seconds. "
            "Frames beyond this duration (based on FPS) are skipped."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_dir: Path = args.image_dir

    if args.output is not None:
        output_path = args.output
    else:
        output_path = build_default_output_path(image_dir)

    duration_ms = int(1000 / args.fps)

    images_to_gif(
        image_dir,
        output_path,
        duration_ms=duration_ms,
        bbox_file=args.bbox_file,
        max_seconds=args.max_seconds,
    )
    print(f"Saved GIF to {output_path}")


if __name__ == "__main__":
    main()
