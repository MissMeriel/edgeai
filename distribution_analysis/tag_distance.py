"""
Cross-tag sequence distance using datasets/sequences_categories.json.

Pools all frames belonging to each tag value, embeds them with Places365,
then computes Fréchet distance and MMD² between the target tag and every
other tag. Also supports any tag field (scene, time, weather, quality).

Usage:
  # Compare city_street vs all other scene tags
  python tag_distance.py --tag scene --value city_street

  # Compare night vs all time-of-day tags, save plot
  python tag_distance.py --tag time --value night --plot night_vs_others.png

  # Limit frames per sequence for speed
  python tag_distance.py --tag scene --value city_street --max-frames 20
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from sequence_distance import (
    build_places365_backbone,
    frechet_distance_from_embeddings,
    frechet_max_from_embeddings,
    load_frames,
    mmd_max_from_embeddings,
    mmd_rbf,
    places365_embeddings,
)

CATEGORIES_JSON = Path(__file__).parent / "datasets" / "sequences_categories.json"
REPO_ROOT = Path(__file__).parent


def load_tag_groups(tag_field: str) -> dict[str, list[Path]]:
    """Return {tag_value: [sequence_dir, ...]} for the given field."""
    data = json.loads(CATEGORIES_JSON.read_text())
    groups: dict[str, list[Path]] = defaultdict(list)
    for seq_path_str, attrs in data.items():
        val = attrs.get(tag_field)
        if val is None:
            continue
        seq_dir = REPO_ROOT / seq_path_str.rstrip("/")
        if seq_dir.exists():
            groups[val].append(seq_dir)
        else:
            print(f"  [warn] missing: {seq_dir}")
    return dict(groups)


def embed_group(
    seq_dirs: list[Path],
    model,
    device: str,
    max_frames: int,
    label: str,
) -> np.ndarray:
    """Pool all frames from a group of sequences into one embedding matrix."""
    all_emb = []
    for seq_dir in seq_dirs:
        frames = load_frames(seq_dir, max_frames)
        emb = places365_embeddings(frames, model, device)
        all_emb.append(emb)
        print(f"    {seq_dir.name}: {len(frames)} frames → {emb.shape[0]} embeddings")
    pooled = np.vstack(all_emb)
    print(f"  [{label}] total embeddings: {pooled.shape[0]}")
    return pooled


def plot_tag_distances(
    target_tag: str,
    results: list[dict],
    out_path: Path,
) -> None:
    """
    Two-panel bar chart: Fréchet (raw + normalized) and MMD² (raw + normalized).
    Each panel shows a bar per comparison tag, with the target shown separately.
    """
    other_tags = [r["tag"] for r in results]
    fd_vals = [r["frechet"] for r in results]
    fd_norm = [r["frechet_norm"] for r in results]
    mmd_vals = [r["mmd"] for r in results]
    mmd_norm = [r["mmd_norm"] for r in results]

    x = np.arange(len(other_tags))
    fig, axes = plt.subplots(2, 2, figsize=(max(10, len(other_tags) * 2), 9))
    fig.suptitle(
        f"Distance from  \"{target_tag}\"  to other tag values",
        fontsize=13, fontweight="bold",
    )

    _bar(axes[0, 0], x, fd_vals, other_tags,
         "Places365 Fréchet distance (raw)", "#DD8452")
    _bar(axes[0, 1], x, fd_norm, other_tags,
         "Places365 Fréchet distance (normalized)", "#DD8452", ylim=(0, 1.15))
    _bar(axes[1, 0], x, mmd_vals, other_tags,
         "Places365 MMD² (raw)", "#55A868")
    _bar(axes[1, 1], x, mmd_norm, other_tags,
         "Places365 MMD² (normalized)", "#55A868", ylim=(0, 1.15))

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\n  Plot saved → {out_path}")
    plt.close(fig)


def _bar(ax, x, vals, labels, title, color, ylim=None):
    bars = ax.bar(x, vals, color=color, edgecolor="white", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax.set_title(title, fontsize=10)
    if ylim:
        ax.set_ylim(*ylim)
    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (ylim[1] if ylim else max(vals)) * 0.01,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=8,
        )


def main():
    parser = argparse.ArgumentParser(description="Cross-tag sequence distance")
    parser.add_argument("--tag", default="scene",
                        help="Tag field to compare on (scene/time/weather/quality)")
    parser.add_argument("--value", required=True,
                        help="Target tag value (e.g. city_street)")
    parser.add_argument("--max-frames", type=int, default=40,
                        help="Max frames sampled per sequence (default 40)")
    parser.add_argument("--no-gpu", action="store_true")
    parser.add_argument("--plot", type=Path, default=None,
                        help="Save bar-chart PNG to this path")
    args = parser.parse_args()

    device = "cpu" if args.no_gpu or not torch.cuda.is_available() else "cuda"
    print(f"Device: {device}")

    groups = load_tag_groups(args.tag)
    if args.value not in groups:
        print(f"ERROR: tag value '{args.value}' not found for field '{args.tag}'.")
        print(f"  Available values: {sorted(groups)}")
        raise SystemExit(1)

    print(f"\nTag field : {args.tag}")
    print(f"Target    : {args.value}  ({len(groups[args.value])} sequences)")
    print(f"Others    : {sorted(k for k in groups if k != args.value)}\n")

    model = build_places365_backbone(device)

    print(f"Embedding target group: {args.value}")
    emb_target = embed_group(groups[args.value], model, device, args.max_frames, args.value)

    results = []
    other_tags = sorted(k for k in groups if k != args.value)

    for other in other_tags:
        print(f"\nEmbedding comparison group: {other}")
        emb_other = embed_group(groups[other], model, device, args.max_frames, other)

        fd = frechet_distance_from_embeddings(emb_target, emb_other)
        fd_max = frechet_max_from_embeddings(emb_target, emb_other)
        mmd = mmd_rbf(emb_target, emb_other)
        mmd_max = mmd_max_from_embeddings(emb_target, emb_other)

        results.append({
            "tag": other,
            "n_seqs": len(groups[other]),
            "frechet": fd,
            "frechet_max": fd_max,
            "frechet_norm": fd / fd_max if fd_max > 0 else 0.0,
            "mmd": mmd,
            "mmd_max": mmd_max,
            "mmd_norm": mmd / mmd_max if mmd_max > 0 else 0.0,
        })

    # --- Report ---
    W = 72
    print("\n" + "=" * W)
    print(f"  Distances from \"{args.value}\" ({args.tag})")
    print("=" * W)
    hdr = f"  {'Tag':<20} {'N seqs':>6}  {'Fréchet':>10} {'FD/max':>8}  {'MMD²':>10} {'MMD/max':>8}"
    print(hdr)
    print("  " + "-" * (W - 2))
    for r in results:
        fd_norm_str = f"{r['frechet_norm']:.4f}" if r['frechet_max'] > 0 else "  n/a  "
        mmd_norm_str = f"{r['mmd_norm']:.4f}" if r['mmd_max'] > 0 else "  n/a  "
        print(
            f"  {r['tag']:<20} {r['n_seqs']:>6}  "
            f"{r['frechet']:>10.2f} {fd_norm_str:>8}  "
            f"{r['mmd']:>10.6f} {mmd_norm_str:>8}"
        )
    print("=" * W)

    if args.plot:
        print("\nGenerating plot...")
        plot_tag_distances(f"{args.tag}={args.value}", results, args.plot)


if __name__ == "__main__":
    main()
