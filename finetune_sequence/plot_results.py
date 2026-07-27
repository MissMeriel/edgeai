"""
Plot fine-tuning results against the pretrained baseline.

Discovers all training runs under runs/ automatically by reading
training_summary.json + training_config.json from each run directory.
Also reads runs/baseline_eval/baseline_results.json if present.

Produces two charts per model family that has results:

  Chart 1 — Technique comparison
    X-axis : fine-tuning technique (ordered cheapest → most expensive)
    Y-axis : mAP50 (YOLO) or normalised val-loss (FRCNN, inverted so higher=better)
    Series : one bar group per condition (scene group or "all")
    Reference line: pretrained baseline mAP50 for each sequence

  Chart 2 — Sequence scaling
    X-axis : number of training sequences used
    Y-axis : same metric as chart 1
    Series : one line per technique
    Reference line: same pretrained baseline

Usage:

  # Auto-discover runs in default location
  python finetune_sequence/plot_results.py

  # Explicit runs directory and baseline file
  python finetune_sequence/plot_results.py \\
      --runs-dir runs \\
      --baseline runs/baseline_eval/baseline_results.json \\
      --out-dir runs/plots

  # Only plot YOLO
  python finetune_sequence/plot_results.py --family yolo
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Technique ordering (cheapest → most expensive)
# ---------------------------------------------------------------------------

TECHNIQUE_ORDER = ["freeze", "lora", "two_stage", "full", "cosine", "temporal"]
TECHNIQUE_LABELS = {
    "freeze":    "Freeze\n(head only)",
    "lora":      "LoRA\n(adapters)",
    "two_stage": "Two-stage\n(bb→head)",
    "full":      "Full\nfine-tune",
    "cosine":    "Cosine\nwarm-up",
    "temporal":  "Temporal\nconsistency",
}

# For combined technique sets like {"two_stage", "temporal"} → display label
def technique_set_label(techniques: list[str]) -> str:
    ordered = [t for t in TECHNIQUE_ORDER if t in techniques]
    remaining = [t for t in techniques if t not in TECHNIQUE_ORDER]
    all_t = ordered + remaining
    return " + ".join(all_t) if all_t else "unknown"


def technique_set_rank(techniques: list[str]) -> float:
    """Sort key: index of the most expensive technique in the set."""
    indices = [TECHNIQUE_ORDER.index(t) if t in TECHNIQUE_ORDER else 99
               for t in techniques]
    return max(indices) if indices else 99


# ---------------------------------------------------------------------------
# Discover and load runs
# ---------------------------------------------------------------------------

def load_run(run_dir: Path) -> dict | None:
    summary_p = run_dir / "training_summary.json"
    config_p  = run_dir / "training_config.json"
    if not summary_p.exists() or not config_p.exists():
        return None
    try:
        summary = json.loads(summary_p.read_text())
        config  = json.loads(config_p.read_text())
    except json.JSONDecodeError:
        return None
    return {"dir": run_dir, "summary": summary, "config": config}


def discover_runs(runs_dir: Path) -> list[dict]:
    runs = []
    for child in sorted(runs_dir.iterdir()):
        if not child.is_dir():
            continue
        r = load_run(child)
        if r is not None:
            runs.append(r)
    return runs


# ---------------------------------------------------------------------------
# Extract per-run records
# ---------------------------------------------------------------------------

def extract_records(runs: list[dict]) -> list[dict]:
    """
    Return a flat list of records, one per (run, group, family).
    Each record:
      family         : "yolo" | "frcnn"
      model_key      : e.g. "yolov8n"
      techniques     : frozenset of technique strings
      technique_label: display string
      technique_rank : float (for ordering)
      group          : condition/scene label from training
      n_train_seq    : number of training sequences
      n_train_frames : n_train from summary
      metric         : mAP50 (YOLO) or best_val_loss (FRCNN, lower=better)
      metric_yolo    : mAP50 if YOLO else None
      metric_frcnn   : best_val_loss if FRCNN else None
      run_dir        : Path
    """
    records = []
    for run in runs:
        cfg     = run["config"]
        summary = run["summary"]
        techniques = frozenset(cfg.get("techniques", []))
        tech_label = technique_set_label(list(techniques))
        tech_rank  = technique_set_rank(list(techniques))

        for group_label, group_data in summary.items():
            n_train_seq = len(group_data.get("train_sequences", []))

            for family in ("yolo", "frcnn"):
                fam_data = group_data.get(family)
                if not fam_data or "error" in fam_data:
                    continue

                model_key = cfg.get(f"{family}_model", "unknown")

                if family == "yolo":
                    metric = fam_data.get("mAP50")
                else:
                    metric = fam_data.get("best_val_loss")

                if metric is None:
                    continue

                records.append({
                    "family":          family,
                    "model_key":       model_key,
                    "techniques":      techniques,
                    "technique_label": tech_label,
                    "technique_rank":  tech_rank,
                    "group":           group_label,
                    "n_train_seq":     n_train_seq,
                    "n_train_frames":  fam_data.get("n_train", 0),
                    "metric":          metric,
                    "run_dir":         run["dir"],
                })
    return records


# ---------------------------------------------------------------------------
# Load baseline
# ---------------------------------------------------------------------------

def load_baseline(baseline_path: Path) -> dict:
    """
    Returns {model_key: {sequence_name: mAP50}} for YOLO entries.
    FRCNN baseline entries are also stored if present.
    """
    if not baseline_path or not baseline_path.exists():
        return {}
    try:
        entries = json.loads(baseline_path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}

    by_model: dict[str, list[float]] = defaultdict(list)
    for e in entries:
        if "error" in e:
            continue
        key   = e.get("model", "")
        map50 = e.get("mAP50")
        if map50 is not None:
            by_model[key].append(map50)

    # Average across sequences for a single reference value per model
    return {k: sum(v) / len(v) for k, v in by_model.items() if v}


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _family_metric_label(family: str) -> str:
    return "mAP50" if family == "yolo" else "Val loss (lower = better)"


def _family_metric_direction(family: str) -> str:
    return "higher" if family == "yolo" else "lower"


def _colour_cycle(n: int):
    import matplotlib
    cmap = matplotlib.colormaps["tab10"]
    return [cmap(i % 10) for i in range(n)]


# ---------------------------------------------------------------------------
# Chart 1: technique comparison (bar chart per condition)
# ---------------------------------------------------------------------------

def plot_technique_comparison(
    records: list[dict],
    baseline: dict,
    family: str,
    model_key: str,
    out_dir: Path,
):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    subset = [r for r in records
              if r["family"] == family and r["model_key"] == model_key]
    if not subset:
        return

    # Collect all technique labels sorted by rank
    tech_keys = sorted({r["technique_label"] for r in subset},
                       key=lambda t: min(r["technique_rank"] for r in subset
                                         if r["technique_label"] == t))
    groups = sorted({r["group"] for r in subset})

    x = np.arange(len(tech_keys))
    n_groups = len(groups)
    bar_w = min(0.7 / max(n_groups, 1), 0.25)

    fig, ax = plt.subplots(figsize=(max(8, len(tech_keys) * 1.6 + 2), 5))
    colours = _colour_cycle(n_groups)

    for gi, (group, colour) in enumerate(zip(groups, colours)):
        heights = []
        for tk in tech_keys:
            vals = [r["metric"] for r in subset
                    if r["group"] == group and r["technique_label"] == tk]
            heights.append(vals[0] if vals else float("nan"))

        offset = (gi - n_groups / 2 + 0.5) * bar_w
        bars = ax.bar(x + offset, heights, bar_w, label=group,
                      color=colour, alpha=0.82, edgecolor="white", linewidth=0.6)

        # Value labels on bars
        for bar, h in zip(bars, heights):
            if h != h:  # nan
                continue
            va = "bottom" if family == "yolo" else "top"
            dy = 0.003 if family == "yolo" else -0.003
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + dy,
                    f"{h:.3f}", ha="center", va="bottom" if family == "yolo" else "top",
                    fontsize=7, color="black")

    # Baseline reference line
    b_val = baseline.get(model_key)
    if b_val is not None:
        ax.axhline(b_val, color="crimson", linestyle="--", linewidth=1.4,
                   label=f"Pretrained baseline ({b_val:.3f})", zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels(tech_keys, fontsize=9)
    ax.set_xlabel("Fine-tuning technique  (cheapest → most expensive)", fontsize=10)
    ax.set_ylabel(_family_metric_label(family), fontsize=10)
    ax.set_title(
        f"{family.upper()} — {model_key}\n"
        f"Technique comparison by condition group",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=8, loc="upper left", framealpha=0.85)
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out_path = out_dir / f"{family}_{model_key}_technique_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ---------------------------------------------------------------------------
# Chart 2: sequence scaling
# ---------------------------------------------------------------------------

def plot_sequence_scaling(
    records: list[dict],
    baseline: dict,
    family: str,
    model_key: str,
    out_dir: Path,
):
    import matplotlib.pyplot as plt
    import numpy as np

    subset = [r for r in records
              if r["family"] == family and r["model_key"] == model_key]
    if not subset:
        return

    # Group by technique label, sorted by rank
    by_tech: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for r in subset:
        by_tech[r["technique_label"]].append((r["n_train_seq"], r["metric"]))

    tech_keys_sorted = sorted(
        by_tech.keys(),
        key=lambda t: min(r["technique_rank"] for r in subset
                          if r["technique_label"] == t),
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    colours = _colour_cycle(len(tech_keys_sorted))
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]

    for i, (tk, colour) in enumerate(zip(tech_keys_sorted, colours)):
        pts = sorted(by_tech[tk])  # sort by n_train_seq
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        marker = markers[i % len(markers)]
        ax.plot(xs, ys, marker=marker, color=colour, linewidth=1.6,
                markersize=7, label=tk, alpha=0.88, zorder=4)
        for xi, yi in zip(xs, ys):
            ax.annotate(f"{yi:.3f}", (xi, yi),
                        textcoords="offset points", xytext=(4, 4),
                        fontsize=7, color=colour)

    # Baseline reference line
    b_val = baseline.get(model_key)
    if b_val is not None:
        ax.axhline(b_val, color="crimson", linestyle="--", linewidth=1.4,
                   label=f"Pretrained baseline ({b_val:.3f})", zorder=5)

    # X-axis: integer sequence counts
    all_xs = sorted({r["n_train_seq"] for r in subset})
    if all_xs:
        ax.set_xticks(all_xs)

    ax.set_xlabel("Number of training sequences", fontsize=10)
    ax.set_ylabel(_family_metric_label(family), fontsize=10)
    ax.set_title(
        f"{family.upper()} — {model_key}\n"
        f"Performance vs training sequences per technique",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=8, loc="upper left", framealpha=0.85)
    ax.grid(alpha=0.3, linestyle=":")
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out_path = out_dir / f"{family}_{model_key}_sequence_scaling.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        sys.exit("matplotlib not installed — activate .venv-eai and run: pip install matplotlib")

    parser = argparse.ArgumentParser(
        description="Plot fine-tuning results vs pretrained baseline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--runs-dir", default="runs", metavar="PATH",
                        help="Directory containing training run subdirectories "
                             "(default: runs)")
    parser.add_argument("--baseline", default=None, metavar="PATH",
                        help="Path to baseline_results.json from baseline_eval.py "
                             "(auto-detected as runs/baseline_eval/baseline_results.json)")
    parser.add_argument("--out-dir", default="runs/plots", metavar="PATH",
                        help="Output directory for PNGs (default: runs/plots)")
    parser.add_argument("--family", choices=["yolo", "frcnn", "both"], default="both",
                        help="Which model family to plot (default: both)")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    if not runs_dir.is_dir():
        sys.exit(f"runs-dir not found: {runs_dir}")

    # Auto-detect baseline
    baseline_path = None
    if args.baseline:
        baseline_path = Path(args.baseline)
    else:
        candidate = runs_dir / "baseline_eval" / "baseline_results.json"
        if candidate.exists():
            baseline_path = candidate
            print(f"  Using baseline: {baseline_path}")
        else:
            print("  No baseline_results.json found — reference lines will be omitted.")
            print("  Run baseline_eval.py first, or pass --baseline PATH.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = discover_runs(runs_dir)
    # Exclude the baseline_eval subdirectory itself if it appears
    runs = [r for r in runs if r["dir"].name != "baseline_eval"]

    if not runs:
        sys.exit(f"No training runs found in {runs_dir}. "
                 "Each run needs training_summary.json + training_config.json.")

    print(f"  Discovered {len(runs)} training run(s):")
    for r in runs:
        cfg = r["config"]
        print(f"    {r['dir'].name}  "
              f"family={cfg.get('family')}  "
              f"techniques={cfg.get('techniques')}  "
              f"group_by={cfg.get('group_by')}")

    records = extract_records(runs)
    if not records:
        sys.exit("No usable results found (all runs may have errors). "
                 "Check the training_summary.json files in each run directory.")

    baseline = load_baseline(baseline_path) if baseline_path else {}

    # Collect all (family, model_key) pairs that have data
    model_pairs = sorted({(r["family"], r["model_key"]) for r in records})
    families_requested = {"yolo", "frcnn"} if args.family == "both" else {args.family}
    model_pairs = [(f, m) for f, m in model_pairs if f in families_requested]

    if not model_pairs:
        sys.exit(f"No records for family={args.family}.")

    print(f"\n  Generating charts for: {model_pairs}")
    for family, model_key in model_pairs:
        print(f"\n  [{family.upper()}] {model_key}")
        plot_technique_comparison(records, baseline, family, model_key, out_dir)
        plot_sequence_scaling(records, baseline, family, model_key, out_dir)

    print(f"\nAll charts written to {out_dir}/")
