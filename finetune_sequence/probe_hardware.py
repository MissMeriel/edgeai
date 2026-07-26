"""
Probe the training machine's hardware and output recommended training settings.

Run this on the target training PC before starting a fine-tuning run:

    python finetune_sequence/probe_hardware.py

Outputs:
  - GPU name, VRAM, CUDA version, compute capability
  - CPU cores, system RAM
  - Per-model VRAM estimates at candidate batch sizes and image sizes
  - Recommended --batch-frcnn / --batch-yolo / --imgsz / --frame-stride flags
    for each model family given the available VRAM
  - Environment variable recommendations (PYTORCH_CUDA_ALLOC_CONF etc.)
"""

import json
import os
import platform
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# VRAM estimates (MiB) for a single image at 640px
# These are measured empirical peaks during training forward+backward,
# including activations, gradients, and optimizer state for SGD.
# ---------------------------------------------------------------------------

# (model_key, vram_per_image_MiB, model_base_MiB)
# base = weights + optimizer state; per_image = activations at 640px
FRCNN_VRAM = {
    "fasterrcnn_resnet50_v2": (340, 700),
    "fasterrcnn_resnet50":    (320, 650),
    "fasterrcnn_mobilenet":   (120, 250),
    "retinanet":              (300, 620),
    "fcos":                   (280, 580),
    "ssdlite":                (60,  160),
}

# YOLO: (vram_per_image_MiB at 640px, model_base_MiB)
YOLO_VRAM = {
    "yolov8n":  (35,  120),
    "yolov8s":  (60,  200),
    "yolov8m":  (110, 380),
    "yolov8l":  (170, 560),
    "yolo11n":  (38,  130),
    "yolo11s":  (65,  210),
    "yolo11m":  (120, 400),
    "rtdetr-l": (200, 700),
}

# VRAM scales roughly with (imgsz/640)^2 for feature maps
def vram_at_imgsz(per_image_640, imgsz):
    return per_image_640 * (imgsz / 640) ** 2


def safe_batch(vram_total_mib, model_base, per_image_640, imgsz,
               temporal=False, headroom=0.85):
    """
    Return the largest batch size that fits in vram_total_mib * headroom.
    temporal=True doubles the per-step activation cost (two forward passes).
    """
    available = vram_total_mib * headroom - model_base
    if available <= 0:
        return 0
    per_img = vram_at_imgsz(per_image_640, imgsz)
    if temporal:
        per_img *= 2  # primary + adjacent frame both resident simultaneously
    b = int(available / per_img)
    return max(0, b)


# ---------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------

def probe():
    info = {}

    # --- Python / OS ---
    info["python"] = sys.version.split()[0]
    info["platform"] = platform.platform()
    info["cpu_cores_logical"] = os.cpu_count()

    try:
        import psutil
        info["ram_total_GiB"] = round(psutil.virtual_memory().total / 1024**3, 1)
        info["ram_available_GiB"] = round(psutil.virtual_memory().available / 1024**3, 1)
    except ImportError:
        info["ram_total_GiB"] = "psutil not installed"

    # --- PyTorch / CUDA ---
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()

        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["cudnn_version"] = torch.backends.cudnn.version()
            gpus = []
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                vram_mib = props.total_memory // (1024 ** 2)
                gpus.append({
                    "index": i,
                    "name": props.name,
                    "vram_MiB": vram_mib,
                    "vram_GiB": round(vram_mib / 1024, 2),
                    "compute_capability": f"{props.major}.{props.minor}",
                    "multiprocessors": props.multi_processor_count,
                })
            info["gpus"] = gpus
        else:
            info["gpus"] = []
            info["mps_available"] = (
                torch.backends.mps.is_available()
                if hasattr(torch.backends, "mps") else False
            )
    except ImportError:
        info["torch_version"] = "not installed"

    # --- Ultralytics ---
    try:
        import ultralytics
        info["ultralytics_version"] = ultralytics.__version__
    except ImportError:
        info["ultralytics_version"] = "not installed"

    # --- torchvision ---
    try:
        import torchvision
        info["torchvision_version"] = torchvision.__version__
    except ImportError:
        info["torchvision_version"] = "not installed"

    return info


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------

def make_recommendations(info):
    recs = []

    gpus = info.get("gpus", [])
    if not gpus:
        vram = 0
        recs.append("No CUDA GPU detected — training will run on CPU (very slow).")
        recs.append("Set --device cpu and reduce --epochs-frcnn / --epochs-yolo.")
    else:
        # Use smallest GPU VRAM as the constraint (multi-GPU not yet supported)
        vram = min(g["vram_MiB"] for g in gpus)

    # Suggested imgsz tiers
    if vram >= 8000:
        imgsz_candidates = [640, 800]
    elif vram >= 5000:
        imgsz_candidates = [640]
    elif vram >= 3000:
        imgsz_candidates = [480, 640]
    else:
        imgsz_candidates = [416, 480]

    print("\n" + "=" * 70)
    print("RECOMMENDED SETTINGS")
    print("=" * 70)

    for imgsz in imgsz_candidates:
        print(f"\n--- Image size {imgsz}px ---")

        print("\n  FRCNN  (--family frcnn  --batch-frcnn N)")
        for key, (per_img, base) in FRCNN_VRAM.items():
            b_std = safe_batch(vram, base, per_img, imgsz, temporal=False)
            b_tmp = safe_batch(vram, base, per_img, imgsz, temporal=True)
            flag = ""
            if b_std == 0:
                flag = "  ← model too large for this GPU at this imgsz"
            elif b_tmp == 0:
                flag = "  ← temporal technique not feasible (OOM); use --batch-frcnn 1 or drop temporal"
            print(f"    {key:<30}  std: batch={b_std}  temporal: batch={b_tmp}{flag}")

        print("\n  YOLO   (--family yolo  --batch-yolo N)")
        for key, (per_img, base) in YOLO_VRAM.items():
            b = safe_batch(vram, base, per_img, imgsz, temporal=False)
            flag = "  ← too large" if b == 0 else ""
            print(f"    {key:<30}  batch={b}{flag}")

    # Environment variables
    print("\n--- Environment variable recommendations ---")
    if vram > 0 and vram < 5000:
        print("  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
        print("  # Reduces fragmentation on small VRAM cards.")
    print("  # Set before launching training:")
    print("  # export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")

    # Frame stride advice
    print("\n--- Frame stride advice (--frame-stride) ---")
    if vram < 4000:
        print("  Recommended: --frame-stride 3  (reduces dataset to ~33%, speeds up epochs)")
        print("  Minimum viable: --frame-stride 5  for very slow cards")
    elif vram < 8000:
        print("  Recommended: --frame-stride 2  (reduces dataset to ~50%)")
    else:
        print("  --frame-stride 1 is fine (full dataset)")

    # Suggested complete command
    print("\n--- Suggested command for this machine ---")
    if vram > 0:
        # Pick a feasible frcnn batch
        b_frcnn = max(1, safe_batch(vram, 700, 340, 640, temporal=True))
        b_yolo  = max(1, safe_batch(vram, 120, 35, 640))
        imgsz   = 640 if vram >= 3500 else 480
        stride  = 1 if vram >= 8000 else (2 if vram >= 5000 else 3)
        print(f"""
  python finetune_sequence/train_sequence.py \\
      --sequences-json datasets/sequences_categories.json \\
      --group-by scene \\
      --val-split-by random \\
      --family frcnn \\
      --technique two_stage \\
      --batch-frcnn {b_frcnn} \\
      --imgsz {imgsz} \\
      --frame-stride {stride} \\
      --val-on-cpu \\
      --device cuda

  # If using temporal technique, halve --batch-frcnn (two forward passes per step):
  # --batch-frcnn {max(1, b_frcnn // 2)} --technique two_stage temporal
""")
    return recs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    info = probe()

    print("=" * 70)
    print("HARDWARE PROBE")
    print("=" * 70)
    print(f"  Python        : {info['python']}")
    print(f"  Platform      : {info['platform']}")
    print(f"  CPU cores     : {info['cpu_cores_logical']}")
    if "ram_total_GiB" in info:
        print(f"  RAM           : {info['ram_total_GiB']} GiB total, "
              f"{info.get('ram_available_GiB', '?')} GiB available")
    print(f"  PyTorch       : {info.get('torch_version', 'n/a')}")
    print(f"  Ultralytics   : {info.get('ultralytics_version', 'n/a')}")
    print(f"  torchvision   : {info.get('torchvision_version', 'n/a')}")

    gpus = info.get("gpus", [])
    if gpus:
        print(f"  CUDA          : {info.get('cuda_version', 'n/a')}  "
              f"cuDNN {info.get('cudnn_version', 'n/a')}")
        for g in gpus:
            print(f"  GPU {g['index']}          : {g['name']}  "
                  f"{g['vram_GiB']} GiB VRAM  "
                  f"compute {g['compute_capability']}  "
                  f"{g['multiprocessors']} SMs")
    elif info.get("mps_available"):
        print("  Accelerator   : Apple MPS (Metal)")
    else:
        print("  Accelerator   : CPU only")

    make_recommendations(info)

    # Save full report
    out = Path("runs/hardware_probe.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(info, indent=2, default=str))
    print(f"\nFull report saved → {out}")
