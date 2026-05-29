"""
Training Runs Dashboard

Reads all completed runs from finetune/runs/ and displays:
- Per-run summary metrics (mAP50, mAP50-95, precision, recall)
- Cross-run comparison charts
- Per-scene per-epoch training curves (loss, mAP)
- Confusion matrices and YOLO plot images

Launch:
    streamlit run finetune/training_dashboard.py
"""

import json
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from PIL import Image

st.set_page_config(
    page_title="Training Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

RUNS_DIR = Path(__file__).parent / "runs"

METRIC_LABELS = {
    "mAP50": "mAP@50",
    "mAP50_95": "mAP@50-95",
    "precision": "Precision",
    "recall": "Recall",
}

SCENE_ORDER = ["__untagged__", "city_street", "highway", "indoor", "parking_lot", "residential", "rural"]

CSV_COLS = {
    "train/box_loss": "Train Box Loss",
    "train/cls_loss": "Train Cls Loss",
    "train/dfl_loss": "Train DFL Loss",
    "val/box_loss": "Val Box Loss",
    "val/cls_loss": "Val Cls Loss",
    "val/dfl_loss": "Val DFL Loss",
    "metrics/precision(B)": "Precision",
    "metrics/recall(B)": "Recall",
    "metrics/mAP50(B)": "mAP@50",
    "metrics/mAP50-95(B)": "mAP@50-95",
}

PLOT_IMAGES = {
    "results.png": "Training Overview",
    "BoxPR_curve.png": "Precision-Recall",
    "BoxF1_curve.png": "F1 Curve",
    "BoxP_curve.png": "Precision Curve",
    "BoxR_curve.png": "Recall Curve",
    "confusion_matrix_normalized.png": "Confusion Matrix (norm.)",
    "confusion_matrix.png": "Confusion Matrix",
}


# ── data loading ──────────────────────────────────────────────────────────────

@st.cache_data
def load_runs() -> dict:
    """Return {run_name: {scene: {model_family: metrics_dict}}}"""
    runs = {}
    if not RUNS_DIR.exists():
        return runs
    for run_dir in sorted(RUNS_DIR.iterdir()):
        summary_path = run_dir / "training_summary.json"
        if not summary_path.exists():
            continue
        with open(summary_path) as f:
            summary = json.load(f)
        config_path = run_dir / "training_config.json"
        config = {}
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
        runs[run_dir.name] = {"summary": summary, "config": config, "path": run_dir}
    return runs


@st.cache_data
def load_csv(csv_path: str) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        return df
    except Exception:
        return None


def summary_to_df(runs: dict) -> pd.DataFrame:
    rows = []
    for run_name, run_data in runs.items():
        for scene, families in run_data["summary"].items():
            for family, metrics in families.items():
                rows.append({
                    "run": run_name,
                    "scene": scene,
                    "family": family,
                    "mAP50": metrics.get("mAP50"),
                    "mAP50_95": metrics.get("mAP50_95"),
                    "precision": metrics.get("precision"),
                    "recall": metrics.get("recall"),
                    "n_train": metrics.get("n_train"),
                    "n_val": metrics.get("n_val"),
                    "train_time_s": metrics.get("train_time_seconds"),
                })
    return pd.DataFrame(rows)


# ── sidebar ───────────────────────────────────────────────────────────────────

runs = load_runs()

if not runs:
    st.error(f"No runs found in `{RUNS_DIR}`. Make sure the path is correct.")
    st.stop()

run_names = list(runs.keys())

with st.sidebar:
    st.markdown("## Training Dashboard")
    st.markdown(f"**{len(run_names)} run(s) found**")
    st.markdown("---")

    page = st.radio(
        "View",
        ["Overview", "Run Comparison", "Epoch Curves", "Images"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("### Filters")
    selected_runs = st.multiselect("Runs", run_names, default=run_names)
    all_scenes = sorted({
        scene
        for r in runs.values()
        for scene in r["summary"].keys()
    })
    selected_scenes = st.multiselect("Scenes", all_scenes, default=all_scenes)
    selected_metric = st.selectbox("Primary metric", list(METRIC_LABELS.keys()), index=1)


df_all = summary_to_df({k: v for k, v in runs.items() if k in selected_runs})
df = df_all[df_all["scene"].isin(selected_scenes)] if not df_all.empty else df_all


# ── helpers ───────────────────────────────────────────────────────────────────

def metric_color_scale(val, col):
    if pd.isna(val):
        return ""
    # green for high mAP/precision/recall, neutral for losses
    if col in ("mAP50", "mAP50_95", "precision", "recall"):
        g = int(val * 200)
        return f"background-color: rgba(0,{g},0,0.15)"
    return ""


def fmt(v):
    if pd.isna(v):
        return "—"
    return f"{v:.4f}"


# ── pages ─────────────────────────────────────────────────────────────────────

if page == "Overview":
    st.markdown("## Overview")

    if df.empty:
        st.info("No data matches the current filters.")
        st.stop()

    # Top-line aggregate cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Runs", len(selected_runs))
    col2.metric("Scenes", len(selected_scenes))
    best_map = df["mAP50_95"].max()
    best_row = df.loc[df["mAP50_95"].idxmax()]
    col3.metric("Best mAP@50-95", f"{best_map:.4f}", f"{best_row['run']} / {best_row['scene']}")
    total_train = df["n_train"].sum()
    col4.metric("Total training images", f"{int(total_train):,}" if not pd.isna(total_train) else "—")

    st.markdown("---")

    # Summary table
    st.markdown("### All results")
    display_df = df[["run", "scene", "family", "mAP50", "mAP50_95", "precision", "recall", "n_train", "n_val", "train_time_s"]].copy()
    display_df["train_time_s"] = display_df["train_time_s"].apply(lambda x: f"{x:.0f}s" if not pd.isna(x) else "—")
    display_df["n_train"] = display_df["n_train"].apply(lambda x: f"{int(x):,}" if not pd.isna(x) else "—")
    display_df["n_val"] = display_df["n_val"].apply(lambda x: f"{int(x):,}" if not pd.isna(x) else "—")
    for col in ("mAP50", "mAP50_95", "precision", "recall"):
        display_df[col] = display_df[col].apply(fmt)
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Per-run config
    for run_name in selected_runs:
        cfg = runs[run_name].get("config")
        if cfg:
            with st.expander(f"Config: {run_name}"):
                st.json(cfg)


elif page == "Run Comparison":
    st.markdown("## Run Comparison")

    if df.empty:
        st.info("No data matches the current filters.")
        st.stop()

    metric_label = METRIC_LABELS[selected_metric]

    # Bar chart: metric per scene, grouped by run
    fig = px.bar(
        df,
        x="scene",
        y=selected_metric,
        color="run",
        barmode="group",
        title=f"{metric_label} by scene and run",
        labels={"scene": "Scene", selected_metric: metric_label, "run": "Run"},
        height=420,
    )
    fig.update_xaxes(categoryorder="array", categoryarray=SCENE_ORDER)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Radar chart comparing runs averaged across scenes
    avg_by_run = df.groupby("run")[["mAP50", "mAP50_95", "precision", "recall"]].mean().reset_index()
    radar_metrics = ["mAP50", "mAP50_95", "precision", "recall"]
    radar_labels = [METRIC_LABELS[m] for m in radar_metrics]

    fig_radar = go.Figure()
    for _, row in avg_by_run.iterrows():
        vals = [row[m] for m in radar_metrics]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=radar_labels + [radar_labels[0]],
            fill="toself",
            name=row["run"],
            opacity=0.6,
        ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title="Average metrics per run (all scenes)",
        height=420,
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown("---")

    # Heat-map: scenes × runs for selected metric
    if len(selected_runs) > 1:
        pivot = df.pivot_table(index="scene", columns="run", values=selected_metric)
        fig_hm = px.imshow(
            pivot,
            text_auto=".3f",
            color_continuous_scale="Greens",
            title=f"{metric_label} — scene × run heatmap",
            aspect="auto",
        )
        st.plotly_chart(fig_hm, use_container_width=True)

    # Dataset sizes per scene
    st.markdown("### Dataset sizes")
    size_df = df[["run", "scene", "n_train", "n_val"]].dropna()
    fig_size = px.bar(
        size_df,
        x="scene",
        y="n_train",
        color="run",
        barmode="group",
        title="Training samples per scene",
        labels={"n_train": "Train images", "scene": "Scene"},
        height=350,
    )
    fig_size.update_xaxes(categoryorder="array", categoryarray=SCENE_ORDER)
    st.plotly_chart(fig_size, use_container_width=True)


elif page == "Epoch Curves":
    st.markdown("## Epoch Curves")

    run_choice = st.selectbox("Run", selected_runs)
    run_path: Path = runs[run_choice]["path"]

    scenes_in_run = sorted(
        [s for s in all_scenes if (run_path / "models" / s / "yolo" / "train" / "results.csv").exists()],
        key=lambda s: SCENE_ORDER.index(s) if s in SCENE_ORDER else 99
    )

    if not scenes_in_run:
        st.info("No results.csv files found for this run.")
        st.stop()

    scene_choice = st.selectbox("Scene", scenes_in_run)

    # Some runs have phase1 (two-stage); offer phase selector if present
    phases = []
    for phase_name in ("phase1", "train"):
        csv_path = run_path / "models" / scene_choice / "yolo" / phase_name / "results.csv"
        if csv_path.exists():
            phases.append(phase_name)

    if len(phases) > 1:
        phase_choice = st.radio("Phase", phases, horizontal=True)
    else:
        phase_choice = phases[0] if phases else "train"

    csv_path = run_path / "models" / scene_choice / "yolo" / phase_choice / "results.csv"
    df_csv = load_csv(str(csv_path))

    if df_csv is None or df_csv.empty:
        st.warning(f"Could not load {csv_path}")
        st.stop()

    st.markdown(f"**{run_choice}** / **{scene_choice}** / {phase_choice} — {len(df_csv)} epochs")

    # Loss curves
    loss_tab, map_tab, pr_tab, lr_tab, raw_tab = st.tabs(["Loss", "mAP", "Precision / Recall", "LR", "Raw CSV"])

    with loss_tab:
        fig = go.Figure()
        for col, label, dash in [
            ("train/box_loss", "Train Box Loss", "solid"),
            ("val/box_loss",   "Val Box Loss",   "dot"),
            ("train/cls_loss", "Train Cls Loss",  "solid"),
            ("val/cls_loss",   "Val Cls Loss",    "dot"),
            ("train/dfl_loss", "Train DFL Loss",  "solid"),
            ("val/dfl_loss",   "Val DFL Loss",    "dot"),
        ]:
            if col in df_csv.columns:
                fig.add_trace(go.Scatter(
                    x=df_csv["epoch"], y=df_csv[col],
                    mode="lines", name=label,
                    line=dict(dash=dash),
                ))
        fig.update_layout(title="Training & Validation Loss", xaxis_title="Epoch", yaxis_title="Loss", hovermode="x unified", height=420)
        st.plotly_chart(fig, use_container_width=True)

    with map_tab:
        fig = go.Figure()
        for col, label, color in [
            ("metrics/mAP50(B)",    "mAP@50",      "#2196F3"),
            ("metrics/mAP50-95(B)", "mAP@50-95",   "#9C27B0"),
        ]:
            if col in df_csv.columns:
                fig.add_trace(go.Scatter(
                    x=df_csv["epoch"], y=df_csv[col],
                    mode="lines+markers", name=label,
                    line=dict(color=color, width=2),
                ))
        fig.update_layout(title="Mean Average Precision", xaxis_title="Epoch", yaxis_title="mAP", hovermode="x unified", height=420)
        st.plotly_chart(fig, use_container_width=True)

    with pr_tab:
        fig = go.Figure()
        for col, label, color in [
            ("metrics/precision(B)", "Precision", "#4CAF50"),
            ("metrics/recall(B)",    "Recall",    "#FF5722"),
        ]:
            if col in df_csv.columns:
                fig.add_trace(go.Scatter(
                    x=df_csv["epoch"], y=df_csv[col],
                    mode="lines+markers", name=label,
                    line=dict(color=color, width=2),
                ))
        fig.update_layout(title="Precision & Recall", xaxis_title="Epoch", yaxis_title="Value", hovermode="x unified", height=420)
        st.plotly_chart(fig, use_container_width=True)

    with lr_tab:
        fig = go.Figure()
        for col in [c for c in df_csv.columns if c.startswith("lr/")]:
            fig.add_trace(go.Scatter(x=df_csv["epoch"], y=df_csv[col], mode="lines", name=col))
        fig.update_layout(title="Learning Rate", xaxis_title="Epoch", yaxis_title="LR", hovermode="x unified", height=350)
        st.plotly_chart(fig, use_container_width=True)

    with raw_tab:
        st.dataframe(df_csv, use_container_width=True, hide_index=True)

    # Multi-scene overlay for selected metric
    st.markdown("---")
    st.markdown("### All scenes overlay")
    overlay_metric_col = st.selectbox(
        "Metric",
        [c for c in ["metrics/mAP50-95(B)", "metrics/mAP50(B)", "metrics/precision(B)", "metrics/recall(B)", "val/box_loss"] if True],
        index=0,
        key="overlay_metric",
    )

    fig_ov = go.Figure()
    for sc in scenes_in_run:
        ov_csv_path = run_path / "models" / sc / "yolo" / phase_choice / "results.csv"
        if not ov_csv_path.exists():
            continue
        df_ov = load_csv(str(ov_csv_path))
        if df_ov is None or overlay_metric_col not in df_ov.columns:
            continue
        fig_ov.add_trace(go.Scatter(
            x=df_ov["epoch"], y=df_ov[overlay_metric_col],
            mode="lines", name=sc,
        ))
    fig_ov.update_layout(
        title=f"{overlay_metric_col} — all scenes ({run_choice})",
        xaxis_title="Epoch",
        hovermode="x unified",
        height=400,
    )
    st.plotly_chart(fig_ov, use_container_width=True)


elif page == "Images":
    st.markdown("## Training Images")

    run_choice = st.selectbox("Run", selected_runs)
    run_path: Path = runs[run_choice]["path"]

    scenes_in_run = sorted([
        d.name for d in (run_path / "models").iterdir()
        if d.is_dir() and (d / "yolo" / "train").exists()
    ], key=lambda s: SCENE_ORDER.index(s) if s in SCENE_ORDER else 99)

    if not scenes_in_run:
        st.info("No model output directories found for this run.")
        st.stop()

    scene_choice = st.selectbox("Scene", scenes_in_run)

    phases = []
    for p in ("phase1", "train"):
        if (run_path / "models" / scene_choice / "yolo" / p).exists():
            phases.append(p)
    phase_choice = st.radio("Phase", phases, horizontal=True) if len(phases) > 1 else (phases[0] if phases else "train")

    phase_dir = run_path / "models" / scene_choice / "yolo" / phase_choice

    # Show plots in a grid
    plot_files = {k: v for k, v in PLOT_IMAGES.items() if (phase_dir / k).exists()}

    if not plot_files:
        st.info("No plot images found.")
    else:
        img_names = list(plot_files.keys())
        img_labels = list(plot_files.values())
        cols_per_row = 2
        for i in range(0, len(img_names), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col in enumerate(cols):
                idx = i + j
                if idx >= len(img_names):
                    break
                img_path = phase_dir / img_names[idx]
                try:
                    img = Image.open(img_path)
                    col.image(img, caption=img_labels[idx], use_container_width=True)
                except Exception as e:
                    col.warning(f"Could not load {img_names[idx]}: {e}")

    # Sample batch images
    st.markdown("---")
    st.markdown("### Sample batches")
    batch_images = sorted(phase_dir.glob("train_batch*.jpg")) + sorted(phase_dir.glob("val_batch*_pred.jpg"))
    if batch_images:
        chosen = st.selectbox("Image", [p.name for p in batch_images])
        img_path = phase_dir / chosen
        try:
            st.image(Image.open(img_path), caption=chosen, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not load image: {e}")
    else:
        st.info("No sample batch images found.")
