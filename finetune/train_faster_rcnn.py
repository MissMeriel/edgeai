"""
Fine-tune torchvision object detection models on the exported 'fixed' samples.

Supported models (--model):
  fasterrcnn_resnet50    — Faster R-CNN ResNet-50 FPN v1 (default torchvision baseline)
  fasterrcnn_resnet50_v2 — Faster R-CNN ResNet-50 FPN v2 (stronger, recommended default)
  fasterrcnn_mobilenet   — Faster R-CNN MobileNetV3 (lightweight, edge-deployable)
  retinanet              — RetinaNet ResNet-50 FPN v2 (one-stage, focal loss, good recall)
  fcos                   — FCOS ResNet-50 FPN (anchor-free one-stage, good for small objects)
  ssdlite                — SSDLite MobileNetV3 (fastest, lowest memory, edge use)

All models are fine-tuned from COCO pretrained weights.

Fine-tuning strategy (--mode, applies to all models):
  head_only   — freeze backbone + FPN, train only box/class heads (best for <200 images)
  full        — fine-tune entire network with differential LR (best for 500+ images)
  progressive — head_only for first half, then unfreeze all (good default)

Usage:
    # First export data (if not already done):
    python export_for_training.py --output-dir runs/finetune_data

    # Recommended default (Faster R-CNN v2, progressive):
    python train_faster_rcnn.py --data-dir runs/finetune_data

    # Lightweight edge model:
    python train_faster_rcnn.py --data-dir runs/finetune_data --model fasterrcnn_mobilenet

    # Anchor-free, good for small/irregular objects:
    python train_faster_rcnn.py --data-dir runs/finetune_data --model fcos

    # Fastest inference, lowest memory:
    python train_faster_rcnn.py --data-dir runs/finetune_data --model ssdlite --batch 8

    # One-stage with focal loss (handles class imbalance well):
    python train_faster_rcnn.py --data-dir runs/finetune_data --model retinanet

    # Head-only (very small dataset <200 samples):
    python train_faster_rcnn.py --data-dir runs/finetune_data --mode head_only

    # Full fine-tune (large dataset):
    python train_faster_rcnn.py --data-dir runs/finetune_data --mode full --epochs 26

    # Resume from checkpoint:
    python train_faster_rcnn.py --data-dir runs/finetune_data --resume runs/frcnn_finetune/checkpoint_epoch10.pth
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torchvision.transforms.v2 as T
import torchvision.tv_tensors as tv_tensors
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class YOLODetectionDataset(Dataset):
    """Reads the YOLO-format export produced by export_for_training.py."""

    def __init__(self, images_dir: Path, labels_dir: Path, classes: list[str],
                 transforms=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.classes = classes
        self.transforms = transforms

        self.image_paths = sorted(
            p for p in images_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        )
        if not self.image_paths:
            raise FileNotFoundError(f"No images found in {images_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        w, h = image.size

        lbl_path = self.labels_dir / (img_path.stem + ".txt")
        boxes, labels = [], []
        if lbl_path.exists():
            for line in lbl_path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                cls_id = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:5])
                # Convert YOLO normalized cx/cy/w/h → absolute xyxy
                x1 = (cx - bw / 2) * w
                y1 = (cy - bh / 2) * h
                x2 = (cx + bw / 2) * w
                y2 = (cy + bh / 2) * h
                # Clamp and skip degenerate boxes
                x1, y1, x2, y2 = max(0.0, x1), max(0.0, y1), min(float(w), x2), min(float(h), y2)
                if x2 - x1 < 1 or y2 - y1 < 1:
                    continue
                boxes.append([x1, y1, x2, y2])
                labels.append(cls_id + 1)  # +1: background is class 0 in torchvision

        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # v2 transforms require BoundingBoxes tv_tensor so spatial transforms
        # (ScaleJitter, RandomCrop, SanitizeBoundingBoxes) can track the boxes.
        # Pass target as a dict so SanitizeBoundingBoxes can locate the labels.
        if self.transforms:
            image = T.ToImage()(image)
            boxes_tv = tv_tensors.BoundingBoxes(
                boxes, format="XYXY", canvas_size=(image.shape[-2], image.shape[-1])
            )
            image, tgt_out = self.transforms(image, {"boxes": boxes_tv, "labels": labels})
            boxes = tgt_out["boxes"]
            labels = tgt_out["labels"]

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
        }

        return image, target


def get_train_transforms():
    # ToImage/ToDtype are applied in __getitem__ before wrapping BoundingBoxes.
    return T.Compose([
        T.ToDtype(torch.float32, scale=True),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomPhotometricDistort(p=0.5),
        T.ScaleJitter(target_size=(800, 800), scale_range=(0.5, 2.0)),
        T.RandomCrop(size=(640, 640), pad_if_needed=True),
        T.SanitizeBoundingBoxes(),
    ])


def get_val_transforms():
    # ToImage/ToDtype are applied in __getitem__ before wrapping BoundingBoxes.
    return T.Compose([
        T.ToDtype(torch.float32, scale=True),
        T.Resize(size=640, max_size=1333),
    ])


def collate_fn(batch):
    return tuple(zip(*batch))


# ---------------------------------------------------------------------------
# Model catalogue
# ---------------------------------------------------------------------------

# Maps --model key → (builder_fn, weights_class, head_type)
# head_type: "fasterrcnn" | "retinanet" | "fcos" | "ssd"
_MODEL_REGISTRY: dict[str, tuple] = {}


def _register(key, builder_fn, weights_cls, head_type: str, description: str):
    _MODEL_REGISTRY[key] = (builder_fn, weights_cls, head_type, description)


def _populate_registry():
    import torchvision.models.detection as det
    _register(
        "fasterrcnn_resnet50_v2",
        det.fasterrcnn_resnet50_fpn_v2,
        det.FasterRCNN_ResNet50_FPN_V2_Weights,
        "fasterrcnn",
        "Faster R-CNN ResNet-50 FPN v2 — strongest two-stage model (recommended default)",
    )
    _register(
        "fasterrcnn_resnet50",
        det.fasterrcnn_resnet50_fpn,
        det.FasterRCNN_ResNet50_FPN_Weights,
        "fasterrcnn",
        "Faster R-CNN ResNet-50 FPN v1 — classic two-stage baseline",
    )
    _register(
        "fasterrcnn_mobilenet",
        det.fasterrcnn_mobilenet_v3_large_fpn,
        det.FasterRCNN_MobileNet_V3_Large_FPN_Weights,
        "fasterrcnn",
        "Faster R-CNN MobileNetV3 — lightweight, edge-deployable two-stage",
    )
    _register(
        "retinanet",
        det.retinanet_resnet50_fpn_v2,
        det.RetinaNet_ResNet50_FPN_V2_Weights,
        "retinanet",
        "RetinaNet ResNet-50 FPN v2 — one-stage, focal loss handles class imbalance",
    )
    _register(
        "fcos",
        det.fcos_resnet50_fpn,
        det.FCOS_ResNet50_FPN_Weights,
        "fcos",
        "FCOS ResNet-50 FPN — anchor-free one-stage, good for irregular/small objects",
    )
    _register(
        "ssdlite",
        det.ssdlite320_mobilenet_v3_large,
        det.SSDLite320_MobileNet_V3_Large_Weights,
        "ssd",
        "SSDLite MobileNetV3 320 — fastest inference, lowest memory, edge use",
    )


def build_model(num_classes: int, model_key: str = "fasterrcnn_resnet50_v2",
                pretrained: bool = True) -> torch.nn.Module:
    """
    Build and return a detection model with its head replaced for num_classes.
    num_classes must include background (i.e., your_classes + 1) for all models
    except SSD, which handles it the same way.

    Small-object anchor tuning is applied to two-stage (RPN-based) models:
    default COCO anchors start at 32px; we extend down to 8px for drone footage
    where pedestrians/cyclists occupy 10-40px at altitude.
    """
    _populate_registry()

    if model_key not in _MODEL_REGISTRY:
        sys.exit(f"Unknown model '{model_key}'. Choose from: {list(_MODEL_REGISTRY)}")

    builder_fn, weights_cls, head_type, desc = _MODEL_REGISTRY[model_key]
    print(f"Model: {model_key} — {desc}")

    weights = weights_cls.DEFAULT if pretrained else None
    model = builder_fn(weights=weights)

    if head_type == "fasterrcnn":
        # Replace RoI head classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        # Small-object anchors for drone imagery (8–128px instead of 32–512px)
        if hasattr(model, "rpn"):
            model.rpn.anchor_generator = AnchorGenerator(
                sizes=((8,), (16,), (32,), (64,), (128,)),
                aspect_ratios=((0.5, 1.0, 2.0),) * 5,
            )

    elif head_type == "retinanet":
        from torchvision.models.detection.retinanet import RetinaNetClassificationHead
        # num_anchors=9 is the RetinaNet default (3 scales × 3 ratios)
        model.head.classification_head = RetinaNetClassificationHead(
            in_channels=model.head.classification_head.conv[0][0].in_channels,
            num_anchors=model.head.classification_head.num_anchors,
            num_classes=num_classes,
        )
        # Smaller anchors for small objects
        model.anchor_generator = AnchorGenerator(
            sizes=tuple((s,) for s in (8, 16, 32, 64, 128)),
            aspect_ratios=((0.5, 1.0, 2.0),) * 5,
        )

    elif head_type == "fcos":
        from torchvision.models.detection.fcos import FCOSClassificationHead
        num_anchors = model.head.classification_head.num_anchors
        in_channels = model.head.classification_head.conv[0][0].in_channels
        model.head.classification_head = FCOSClassificationHead(
            in_channels=in_channels,
            num_anchors=num_anchors,
            num_classes=num_classes,
        )

    elif head_type == "ssd":
        # SSDLite uses a scoring head; replace the whole head for num_classes
        import torchvision.models.detection as det
        # SSD head replacement: rebuild with correct num_classes
        # The cleanest approach is to rebuild from scratch without pretrained head
        model_no_head = builder_fn(weights=None, num_classes=num_classes)
        # Copy backbone weights from pretrained model
        model_no_head.backbone.load_state_dict(model.backbone.state_dict())
        model = model_no_head

    return model


def freeze_backbone(model: torch.nn.Module):
    """Freeze backbone (and FPN if present), leaving detection heads trainable."""
    for name, param in model.named_parameters():
        if name.startswith("backbone") or name.startswith("fpn"):
            param.requires_grad = False


def unfreeze_all(model: torch.nn.Module):
    for param in model.parameters():
        param.requires_grad = True


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(model, optimizer, loader, device, epoch, print_freq=50):
    model.train()
    total_loss = 0.0
    n_batches = len(loader)

    for i, (images, targets) in enumerate(loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        # Gradient clipping: prevents exploding gradients on small datasets
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        total_loss += losses.item()
        if (i + 1) % print_freq == 0 or (i + 1) == n_batches:
            avg = total_loss / (i + 1)
            print(f"  Epoch {epoch} [{i+1}/{n_batches}]  avg_loss={avg:.4f}  "
                  f"cls={loss_dict.get('loss_classifier', 0):.3f}  "
                  f"box={loss_dict.get('loss_box_reg', 0):.3f}  "
                  f"obj={loss_dict.get('loss_objectness', 0):.3f}  "
                  f"rpn_box={loss_dict.get('loss_rpn_box_reg', 0):.3f}")

    return total_loss / n_batches


@torch.no_grad()
def evaluate(model, loader, device):
    """Compute average validation loss (model must stay in train mode for loss)."""
    model.train()  # torchvision detection models only produce loss in train mode
    total = 0.0
    for images, targets in loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        total += sum(loss_dict.values()).item()
    return total / len(loader)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train(
    data_dir: str,
    model_key: str = "fasterrcnn_resnet50_v2",
    mode: str = "progressive",
    epochs: int = 20,
    batch: int = 4,
    lr: float = 0.005,
    project: str = "runs/frcnn_finetune",
    resume: str | None = None,
    device_str: str = "",
):
    data_dir = Path(data_dir)
    yaml_path = data_dir / "dataset.yaml"
    if not yaml_path.exists():
        sys.exit(f"dataset.yaml not found at {yaml_path}. Run export_for_training.py first.")

    with yaml_path.open() as f:
        meta = yaml.safe_load(f)

    classes = meta["names"]
    num_classes = len(classes) + 1  # +1 for background

    print(f"\n{'='*60}")
    print(f"Fine-tuning {model_key} on {data_dir}")
    print(f"  classes ({len(classes)}): {classes}")
    print(f"  mode={mode}, epochs={epochs}, batch={batch}, lr={lr}")
    print(f"{'='*60}\n")

    # Device selection
    if device_str:
        device = torch.device(device_str)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        print("WARNING: no GPU found, training on CPU — expect slow performance")
    print(f"Device: {device}")

    # Datasets
    train_ds = YOLODetectionDataset(
        data_dir / "images" / "train",
        data_dir / "labels" / "train",
        classes,
        transforms=get_train_transforms(),
    )
    val_ds = YOLODetectionDataset(
        data_dir / "images" / "val",
        data_dir / "labels" / "val",
        classes,
        transforms=get_val_transforms(),
    )
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    # DataLoaders — num_workers=0 on MPS to avoid fork issues
    nw = 0 if str(device) == "mps" else 4
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True,
                               num_workers=nw, collate_fn=collate_fn,
                               pin_memory=(str(device) == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False,
                             num_workers=nw, collate_fn=collate_fn,
                             pin_memory=(str(device) == "cuda"))

    # Model
    model = build_model(num_classes, model_key=model_key, pretrained=True)
    model.to(device)

    # Load checkpoint if resuming
    start_epoch = 0
    if resume:
        ckpt = torch.load(resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"Resumed from {resume} (epoch {start_epoch})")

    # -----------------------------------------------------------------------
    # Optimizer & scheduler rationale:
    #
    # SGD with momentum=0.9:
    #   Standard for detection; Adam often diverges on small detection datasets.
    #
    # lr=0.005:
    #   Lower than detection training from scratch (0.02) to preserve pretrained features.
    #
    # Differential LR (full mode):
    #   backbone params get 0.1× the head LR — fine-tunes more carefully.
    #
    # weight_decay=0.0005:
    #   L2 reg — critical for small fine-tune datasets.
    #
    # MultiStepLR at [epoch*0.6, epoch*0.8]:
    #   Drops LR by 10× at two points in training. Standard detection schedule.
    #   Cosine decay is also fine but MultiStep is more interpretable for debugging.
    # -----------------------------------------------------------------------

    def _is_backbone_param(name: str) -> bool:
        return name.startswith("backbone") or name.startswith("fpn")

    if mode == "full":
        param_groups = [
            {"params": [p for n, p in model.named_parameters()
                        if _is_backbone_param(n) and p.requires_grad],
             "lr": lr * 0.1},
            {"params": [p for n, p in model.named_parameters()
                        if not _is_backbone_param(n) and p.requires_grad],
             "lr": lr},
        ]
    else:
        if mode in ("head_only", "progressive"):
            freeze_backbone(model)
        param_groups = [{"params": [p for p in model.parameters() if p.requires_grad]}]

    optimizer = torch.optim.SGD(param_groups, lr=lr, momentum=0.9,
                                 weight_decay=0.0005, nesterov=True)

    milestones = [max(1, int(epochs * 0.6)), max(2, int(epochs * 0.8))]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones,
                                                      gamma=0.1)

    out_dir = Path(project) / model_key
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    patience_counter = 0
    patience = 10  # early stopping

    history = []

    for epoch in range(start_epoch, epochs):
        # Progressive mode: unfreeze backbone at midpoint
        if mode == "progressive" and epoch == epochs // 2:
            print(f"\n--- Progressive mode: unfreezing backbone at epoch {epoch} ---")
            unfreeze_all(model)
            # Add backbone params to optimizer at reduced LR
            optimizer.add_param_group({
                "params": [p for n, p in model.named_parameters()
                           if _is_backbone_param(n)],
                "lr": lr * 0.01,  # very low LR for pretrained backbone
            })

        t0 = time.time()
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)
        val_loss = evaluate(model, val_loader, device)
        scheduler.step()
        elapsed = time.time() - t0

        current_lr = scheduler.get_last_lr()[0]
        print(f"\nEpoch {epoch:3d}/{epochs} | "
              f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
              f"lr={current_lr:.2e} | {elapsed:.0f}s")

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            ckpt_path = out_dir / f"checkpoint_epoch{epoch}.pth"
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "optimizer": optimizer.state_dict()}, ckpt_path)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_path = out_dir / "best.pth"
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "classes": classes,
                "num_classes": num_classes,
            }, best_path)
            print(f"  ✓ Saved best model (val_loss={val_loss:.4f}) → {best_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    # Save training history
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"\nTraining complete. Best val_loss={best_val_loss:.4f}")
    print(f"Best weights: {out_dir / 'best.pth'}")
    print(f"To load: model = build_model({num_classes}, model_key='{model_key}', pretrained=False); "
          f"model.load_state_dict(torch.load('best.pth')['model'])")


if __name__ == "__main__":
    _populate_registry()
    model_choices = list(_MODEL_REGISTRY)
    model_help = "\n".join(
        f"  {k}: {v[3]}" for k, v in _MODEL_REGISTRY.items()
    )

    parser = argparse.ArgumentParser(
        description="Fine-tune torchvision detection models on FiftyOne 'fixed' data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data-dir", default="runs/finetune_data",
                        help="Directory produced by export_for_training.py (default: runs/finetune_data)")
    parser.add_argument("--model", default="fasterrcnn_resnet50_v2",
                        choices=model_choices,
                        help=f"Model to fine-tune (default: fasterrcnn_resnet50_v2):\n{model_help}")
    parser.add_argument("--mode", default="progressive",
                        choices=["head_only", "full", "progressive"],
                        help="Fine-tuning strategy (default: progressive)\n"
                             "  head_only:   freeze backbone, best for <200 images\n"
                             "  full:        differential LR, best for 500+ images\n"
                             "  progressive: head_only → full at midpoint (default)")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Max training epochs (default: 20, early stopping at patience=10)")
    parser.add_argument("--batch", type=int, default=4,
                        help="Batch size (default: 4; two-stage models are memory-intensive)")
    parser.add_argument("--lr", type=float, default=0.005,
                        help="Base learning rate (default: 0.005)")
    parser.add_argument("--project", default="runs/frcnn_finetune",
                        help="Output directory (default: runs/frcnn_finetune)")
    parser.add_argument("--device", default="", dest="device_str",
                        help="Device: '' (auto), 'cpu', 'cuda', 'cuda:0', 'mps'")
    parser.add_argument("--resume", default=None,
                        help="Path to checkpoint .pth to resume training")
    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        model_key=args.model,
        mode=args.mode,
        epochs=args.epochs,
        batch=args.batch,
        lr=args.lr,
        project=args.project,
        resume=args.resume,
        device_str=args.device_str,
    )
