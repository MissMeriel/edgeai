"""
Pairwise distance between two image sequences.

Metrics:
  1. color_hist   — per-channel HSV histogram, Bhattacharyya distance
  2. places365    — ResNet50-Places365 penultimate-layer embeddings, Fréchet distance
  3. optical_flow — mean/variance of dense optical flow magnitude, Euclidean distance
  4. mmd          — RBF-kernel MMD on Places365 embeddings

The Fréchet and MMD values are also shown normalized to [0, 1] using the
theoretical / empirical max for the given embedding space.

Usage:
  python sequence_distance.py <seq_dir_a> <seq_dir_b> [--max-frames N] [--no-gpu]
                               [--plot <out.png>]
"""

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from scipy.linalg import sqrtm
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_frames(seq_dir: Path, max_frames: int) -> list[Path]:
    paths = sorted(seq_dir.glob("*.jpg")) + sorted(seq_dir.glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"No jpg/png images found in {seq_dir}")
    step = max(1, len(paths) // max_frames)
    return paths[::step][:max_frames]


# ---------------------------------------------------------------------------
# 1. Color histogram distance
# ---------------------------------------------------------------------------

def color_hist_signature(frame_paths: list[Path]) -> np.ndarray:
    """Mean per-channel HSV histogram over all frames (128 bins each → 384-d)."""
    hists = []
    for p in frame_paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h = []
        for ch, bins in zip(range(3), [180, 256, 256]):
            hist = cv2.calcHist([hsv], [ch], None, [128], [0, bins])
            hist = cv2.normalize(hist, hist).flatten()
            h.append(hist)
        hists.append(np.concatenate(h))
    return np.mean(hists, axis=0).astype(np.float32)


def color_hist_distance(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    """Bhattacharyya distance between two histogram signatures."""
    # Split back into per-channel histograms (128 bins each)
    scores = []
    for i in range(3):
        ha = sig_a[i * 128:(i + 1) * 128].reshape(-1, 1).astype(np.float32)
        hb = sig_b[i * 128:(i + 1) * 128].reshape(-1, 1).astype(np.float32)
        scores.append(cv2.compareHist(ha, hb, cv2.HISTCMP_BHATTACHARYYA))
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# 2. Places365 embeddings
# ---------------------------------------------------------------------------

PLACES365_WEIGHTS = Path(__file__).parent / "scenewise_data_cleaning" / "resnet50_places365.pth"

_TRANSFORM = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def build_places365_backbone(device: str) -> nn.Module:
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 365)

    if not PLACES365_WEIGHTS.exists():
        raise FileNotFoundError(
            f"Places365 weights not found at {PLACES365_WEIGHTS}. "
            "Run scenewise_data_cleaning/scene_classifier_places365.py once to download them."
        )

    state = torch.load(PLACES365_WEIGHTS, map_location="cpu", weights_only=False)
    # The .pth may be a plain state_dict or a checkpoint dict
    if isinstance(state, dict) and "state_dict" in state:
        state = {k.replace("module.", ""): v for k, v in state["state_dict"].items()}
    model.load_state_dict(state, strict=False)

    # Drop the final FC → use avgpool output as 2048-d embedding
    model.fc = nn.Identity()
    model.eval().to(device)
    return model


@torch.no_grad()
def places365_embeddings(frame_paths: list[Path], model: nn.Module, device: str) -> np.ndarray:
    """Return (N, 2048) embedding matrix for the sequence."""
    vecs = []
    for p in tqdm(frame_paths, desc="  embed", leave=False):
        img = Image.open(p).convert("RGB")
        x = _TRANSFORM(img).unsqueeze(0).to(device)
        vecs.append(model(x).squeeze(0).cpu().numpy())
    return np.stack(vecs)  # (N, 2048)


def frechet_distance_from_embeddings(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
    """
    Fréchet distance between two sets of embeddings.

    Projects to a PCA subspace of dimension min(n_samples-1, 256) before
    computing covariances so the matrices are always full-rank.
    """
    # PCA projection: fit on the joint set, apply to each
    all_emb = np.vstack([emb_a, emb_b])
    mean = all_emb.mean(0)
    centered = all_emb - mean
    # Truncated SVD — cheap for (N, 2048) with small N
    n_components = min(len(emb_a) - 1, len(emb_b) - 1, 256)
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    components = Vt[:n_components]  # (n_components, 2048)

    proj_a = (emb_a - mean) @ components.T
    proj_b = (emb_b - mean) @ components.T

    mu_a, mu_b = proj_a.mean(0), proj_b.mean(0)
    eps = 1e-6 * np.eye(n_components)
    cov_a = np.cov(proj_a, rowvar=False) + eps
    cov_b = np.cov(proj_b, rowvar=False) + eps

    diff = mu_a - mu_b
    covmean = sqrtm(cov_a @ cov_b)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fd = float(diff @ diff + np.trace(cov_a + cov_b - 2 * covmean))
    return fd


# ---------------------------------------------------------------------------
# 3. MMD (RBF kernel on Places365 embeddings)
# ---------------------------------------------------------------------------

def mmd_rbf(X: np.ndarray, Y: np.ndarray, gamma: float = None) -> float:
    """Unbiased MMD² estimate with RBF kernel."""
    if gamma is None:
        # median heuristic
        all_pts = np.vstack([X, Y])
        dists = np.sum((all_pts[:, None] - all_pts[None, :]) ** 2, axis=-1)
        gamma = 1.0 / (2 * np.median(dists[dists > 0]))

    def rbf(A, B):
        sq = np.sum((A[:, None] - B[None, :]) ** 2, axis=-1)
        return np.exp(-gamma * sq)

    n, m = len(X), len(Y)
    kxx = rbf(X, X)
    kyy = rbf(Y, Y)
    kxy = rbf(X, Y)
    # Unbiased: zero the diagonal for same-set terms
    np.fill_diagonal(kxx, 0)
    np.fill_diagonal(kyy, 0)
    return float(kxx.sum() / (n * (n - 1)) + kyy.sum() / (m * (m - 1)) - 2 * kxy.mean())


# ---------------------------------------------------------------------------
# 4. Optical flow statistics
# ---------------------------------------------------------------------------

def optical_flow_stats(frame_paths: list[Path]) -> np.ndarray:
    """Return [mean_mag, std_mag] of dense optical flow across consecutive frames."""
    mags, stds = [], []
    prev_gray = None
    for p in frame_paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
            )
            mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            mags.append(mag.mean())
            stds.append(mag.std())
        prev_gray = gray
    if not mags:
        return np.zeros(2)
    return np.array([np.mean(mags), np.mean(stds)])


def flow_distance(stats_a: np.ndarray, stats_b: np.ndarray) -> float:
    return float(np.linalg.norm(stats_a - stats_b))


# ---------------------------------------------------------------------------
# Max-value estimation
# ---------------------------------------------------------------------------

def frechet_max_from_embeddings(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
    """
    Upper bound for the Fréchet distance given these two embedding sets.

    Computed as FD(emb_a, -emb_a) — the distance between a distribution and
    its mirror image, which represents maximal displacement while keeping the
    same covariance structure. This is the largest FD achievable with
    distributions that have the same spread as the inputs.
    """
    return frechet_distance_from_embeddings(emb_a, -emb_a)


def mmd_max_from_embeddings(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
    """
    Empirical max MMD² for the given embedding sets, estimated as
    MMD²(emb_a, -emb_a) — a maximally separated pair with the same
    within-set structure as emb_a.
    """
    return mmd_rbf(emb_a, -emb_a)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_distances(metrics: dict, out_path: Path) -> None:
    """
    Bar chart of all distance metrics with Fréchet and MMD shown both as raw
    values and as a fraction of their estimated max.
    """
    labels = [
        "Color hist\n(Bhattacharyya)",
        "Optical flow\n(motion L2)",
        "Places365\nFréchet",
        "Places365\nFréchet / max",
        "Places365\nMMD²",
        "Places365\nMMD² / max",
    ]
    values = [
        metrics["color_hist"],
        metrics["flow"],
        metrics["frechet"],
        metrics["frechet"] / metrics["frechet_max"] if metrics["frechet_max"] > 0 else 0,
        metrics["mmd"],
        metrics["mmd"] / metrics["mmd_max"] if metrics["mmd_max"] > 0 else 0,
    ]
    colors = ["#4C72B0", "#4C72B0", "#DD8452", "#DD8452", "#55A868", "#55A868"]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.8)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.01,
            f"{val:.4f}",
            ha="center", va="bottom", fontsize=9,
        )

    ax.set_ylabel("Distance")
    ax.set_title(
        f"Sequence distance\n"
        f"A: {metrics['seq_a']}  vs  B: {metrics['seq_b']}"
    )
    ax.set_ylim(0, max(values) * 1.15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"  Plot saved → {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Pairwise sequence distance")
    parser.add_argument("seq_a", type=Path)
    parser.add_argument("seq_b", type=Path)
    parser.add_argument("--max-frames", type=int, default=60,
                        help="Max frames to sample per sequence (default 60)")
    parser.add_argument("--no-gpu", action="store_true")
    parser.add_argument("--plot", type=Path, default=None,
                        help="Save a bar-chart PNG to this path")
    args = parser.parse_args()

    device = "cpu" if args.no_gpu or not torch.cuda.is_available() else "cuda"
    print(f"Device: {device}")

    print(f"\nSequence A: {args.seq_a.name}")
    print(f"Sequence B: {args.seq_b.name}")

    print(f"\nLoading frames (max {args.max_frames} per sequence)...")
    frames_a = load_frames(args.seq_a, args.max_frames)
    frames_b = load_frames(args.seq_b, args.max_frames)
    print(f"  A: {len(frames_a)} frames,  B: {len(frames_b)} frames")

    # --- Color histogram ---
    print("\n[1/4] Color histogram distance...")
    sig_a = color_hist_signature(frames_a)
    sig_b = color_hist_signature(frames_b)
    d_color = color_hist_distance(sig_a, sig_b)

    # --- Optical flow ---
    print("[2/4] Optical flow statistics...")
    flow_a = optical_flow_stats(frames_a)
    flow_b = optical_flow_stats(frames_b)
    d_flow = flow_distance(flow_a, flow_b)

    # --- Places365 embeddings ---
    print("[3/4] Places365 embeddings...")
    model = build_places365_backbone(device)
    print("  Embedding sequence A...")
    emb_a = places365_embeddings(frames_a, model, device)
    print("  Embedding sequence B...")
    emb_b = places365_embeddings(frames_b, model, device)

    print("[4/4] Computing Fréchet distance and MMD (+ max estimates)...")
    d_frechet = frechet_distance_from_embeddings(emb_a, emb_b)
    d_mmd = mmd_rbf(emb_a, emb_b)
    d_frechet_max = frechet_max_from_embeddings(emb_a, emb_b)
    d_mmd_max = mmd_max_from_embeddings(emb_a, emb_b)

    # --- Report ---
    W = 56
    print("\n" + "=" * W)
    print(f"  Sequences compared")
    print(f"    A: {args.seq_a.name}  ({len(frames_a)} frames)")
    print(f"    B: {args.seq_b.name}  ({len(frames_b)} frames)")
    print("=" * W)
    print(f"  Color histogram (Bhattacharyya)  : {d_color:.4f}")
    print(f"    (0 = identical, 1 = maximally different)")
    print(f"  Optical flow (motion stats L2)   : {d_flow:.4f}")
    print(f"    A flow: mean={flow_a[0]:.2f} std={flow_a[1]:.2f}")
    print(f"    B flow: mean={flow_b[0]:.2f} std={flow_b[1]:.2f}")
    print(f"  Places365 Fréchet distance       : {d_frechet:.2f}")
    print(f"    estimated max                  : {d_frechet_max:.2f}")
    print(f"    normalized (val / max)         : {d_frechet / d_frechet_max:.4f}" if d_frechet_max > 0 else "    normalized: n/a")
    print(f"  Places365 MMD²                   : {d_mmd:.6f}")
    print(f"    estimated max                  : {d_mmd_max:.6f}")
    print(f"    normalized (val / max)         : {d_mmd / d_mmd_max:.4f}" if d_mmd_max > 0 else "    normalized: n/a")
    print("=" * W)

    if args.plot:
        print("\nGenerating plot...")
        plot_distances(
            {
                "seq_a": args.seq_a.name,
                "seq_b": args.seq_b.name,
                "color_hist": d_color,
                "flow": d_flow,
                "frechet": d_frechet,
                "frechet_max": d_frechet_max,
                "mmd": d_mmd,
                "mmd_max": d_mmd_max,
            },
            args.plot,
        )


if __name__ == "__main__":
    main()
