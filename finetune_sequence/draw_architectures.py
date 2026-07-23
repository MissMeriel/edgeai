"""
Generate Graphviz diagrams showing YOLO and FRCNN model architectures
and which parts each fine-tuning technique modifies.

Outputs:
  diagrams/yolo_techniques.svg
  diagrams/frcnn_techniques.svg
  diagrams/overview.svg   — side-by-side summary across all techniques

Usage:
    python finetune_sequence/draw_architectures.py
    python finetune_sequence/draw_architectures.py --out-dir my_dir --format pdf
"""

import argparse
from pathlib import Path


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
# frozen      — slate blue   (unchanged weights)
# trained     — coral/red    (weights updated this technique)
# lora        — gold         (LoRA A/B matrices injected)
# partial     — teal         (updated at reduced LR or only in phase 2)
# new_head    — green        (newly initialised head, always trained)
# temporal    — purple       (extra loss path / adjacent-frame branch)
# background  — light grey

C = {
    "frozen":    "#B0BEC5",   # blue-grey  — frozen
    "trained":   "#EF5350",   # red        — fully trained
    "partial":   "#26A69A",   # teal       — differential / phase-2 LR
    "lora":      "#FFC107",   # amber      — LoRA adapters
    "new_head":  "#66BB6A",   # green      — new head (always trained)
    "temporal":  "#AB47BC",   # purple     — temporal loss path
    "bg":        "#F5F5F5",
    "edge":      "#37474F",
    "white":     "#FFFFFF",
    "label_fg":  "#212121",
}

FONT = "Helvetica"


def _node(name, label, fillcolor, shape="box", style="filled,rounded",
          fontcolor="#212121", width="1.8", height="0.5"):
    return (f'  {name} [label="{label}", fillcolor="{fillcolor}", '
            f'shape={shape}, style="{style}", fontname="{FONT}", '
            f'fontsize=11, fontcolor="{fontcolor}", '
            f'width={width}, height={height}];\n')


def _edge(a, b, color=None, style="solid", label=""):
    col = color or C["edge"]
    lbl = f', label="{label}", fontname="{FONT}", fontsize=9' if label else ""
    return f'  {a} -> {b} [color="{col}", style={style}{lbl}];\n'


def _legend_items(items):
    """items: list of (color, label)"""
    rows = "".join(
        f'<TR><TD BGCOLOR="{c}" WIDTH="18" HEIGHT="14"> </TD>'
        f'<TD ALIGN="LEFT"><FONT FACE="{FONT}" POINT-SIZE="10"> {lbl}</FONT></TD></TR>'
        for c, lbl in items
    )
    return (
        '  legend [shape=plaintext, label=<<TABLE BORDER="1" CELLBORDER="0" '
        f'CELLSPACING="3" BGCOLOR="{C["bg"]}">'
        f'<TR><TD COLSPAN="2"><B><FONT FACE="{FONT}" POINT-SIZE="11">'
        f'Legend</FONT></B></TD></TR>{rows}</TABLE>>];\n'
    )


# ---------------------------------------------------------------------------
# YOLO architecture diagram
# ---------------------------------------------------------------------------

def build_yolo_diagram() -> str:
    """
    YOLO architecture: Input → Backbone (CSP-DarkNet) → Neck (FPN/PAN) → Head (Detect)
    One cluster per fine-tuning technique showing which layers are frozen/trained/partial.
    """

    legend = _legend_items([
        (C["frozen"],   "frozen  (weights unchanged)"),
        (C["trained"],  "trained  (full LR)"),
        (C["partial"],  "trained  (reduced LR or phase 2 only)"),
        (C["new_head"], "new head  (always trained)"),
        (C["temporal"], "temporal loss path"),
    ])

    def yolo_cluster(cluster_id, title, bb_col, neck_col, head_col,
                     extra_nodes="", extra_edges="", phase_label=""):
        phase = f'\\n<FONT POINT-SIZE="9">({phase_label})</FONT>' if phase_label else ""
        return (
            f'  subgraph cluster_{cluster_id} {{\n'
            f'    label=<<B>{title}</B>{phase}>;\n'
            f'    fontname="{FONT}"; fontsize=12;\n'
            f'    bgcolor="{C["bg"]}";\n'
            f'    style="rounded";\n'
            f'    {cluster_id}_input  [label="Input\\nframes", fillcolor="{C["white"]}", '
            f'shape=box, style="filled,rounded", fontname="{FONT}", fontsize=10, width=1.4];\n'
            f'    {cluster_id}_bb     [label="Backbone\\n(CSP-DarkNet)", fillcolor="{bb_col}", '
            f'shape=box, style="filled,rounded", fontname="{FONT}", fontsize=10, width=1.8];\n'
            f'    {cluster_id}_neck   [label="Neck\\n(FPN/PAN)", fillcolor="{neck_col}", '
            f'shape=box, style="filled,rounded", fontname="{FONT}", fontsize=10, width=1.8];\n'
            f'    {cluster_id}_head   [label="Detect Head\\n(box+cls)", fillcolor="{head_col}", '
            f'shape=box, style="filled,rounded", fontname="{FONT}", fontsize=10, width=1.8];\n'
            + extra_nodes +
            f'    {cluster_id}_input -> {cluster_id}_bb   [color="{C["edge"]}"];\n'
            f'    {cluster_id}_bb    -> {cluster_id}_neck [color="{C["edge"]}"];\n'
            f'    {cluster_id}_neck  -> {cluster_id}_head [color="{C["edge"]}"];\n'
            + extra_edges +
            f'  }}\n'
        )

    # freeze: backbone + neck frozen, head trained
    g_freeze = yolo_cluster(
        "freeze", "freeze",
        bb_col=C["frozen"], neck_col=C["frozen"], head_col=C["trained"],
    )

    # two_stage: phase1 head-only, phase2 all trained (backbone at partial LR)
    two_stage_extra_nodes = (
        f'    two_stage_p1 [label="Phase 1\\n(head only)", fillcolor="{C["frozen"]}", '
        f'shape=note, style="filled", fontname="{FONT}", fontsize=9, width=1.4];\n'
        f'    two_stage_p2 [label="Phase 2\\n(all, bb@0.1×LR)", fillcolor="{C["partial"]}", '
        f'shape=note, style="filled", fontname="{FONT}", fontsize=9, width=1.4];\n'
    )
    g_two_stage = yolo_cluster(
        "two_stage", "two_stage",
        bb_col=C["partial"], neck_col=C["partial"], head_col=C["trained"],
        extra_nodes=two_stage_extra_nodes,
    )

    # full: everything trained, backbone at 0.1× LR
    g_full = yolo_cluster(
        "full", "full",
        bb_col=C["partial"], neck_col=C["partial"], head_col=C["trained"],
    )

    # lora (YOLO approximation): freeze=10 + lower LR — annotated as approximation
    lora_extra = (
        f'    yolo_lora_note [label="YOLO: freeze first 10\\nlayers + low LR\\n'
        f'(true LoRA on FRCNN)", fillcolor="#FFF9C4", '
        f'shape=note, style="filled", fontname="{FONT}", fontsize=9, width=2.0];\n'
    )
    g_lora = yolo_cluster(
        "yolo_lora", "lora (YOLO approx.)",
        bb_col=C["frozen"], neck_col=C["partial"], head_col=C["trained"],
        extra_nodes=lora_extra,
    )

    # temporal: full training + adjacent frame loss
    temporal_extra_nodes = (
        f'    yolo_adj [label="Adjacent\\nframe", fillcolor="{C["temporal"]}", '
        f'shape=box, style="filled,rounded", fontname="{FONT}", fontsize=10, width=1.4];\n'
        f'    yolo_aug [label="copy_paste 0.3\\nclose_mosaic 20", fillcolor="{C["temporal"]}", '
        f'shape=note, style="filled", fontname="{FONT}", fontsize=9, width=1.8];\n'
    )
    temporal_extra_edges = (
        f'    yolo_adj -> temporal_head [color="{C["temporal"]}", style=dashed, '
        f'xlabel="augment", fontname="{FONT}", fontsize=9];\n'
    )
    g_temporal = yolo_cluster(
        "temporal", "temporal",
        bb_col=C["partial"], neck_col=C["partial"], head_col=C["trained"],
        extra_nodes=temporal_extra_nodes,
        extra_edges=temporal_extra_edges,
    )

    dot = (
        'digraph yolo_techniques {\n'
        f'  graph [rankdir=TB, bgcolor="{C["white"]}", splines=curved, nodesep=0.4, ranksep=0.6,\n'
        f'         label=<<B><FONT FACE="{FONT}" POINT-SIZE="16">'
        'YOLO Architecture — Fine-tuning Techniques</FONT></B>>,\n'
        f'         labelloc=t, fontname="{FONT}"];\n'
        '  node [margin="0.1,0.05"];\n\n'
        + g_freeze
        + g_two_stage
        + g_full
        + g_lora
        + g_temporal
        + legend
        + '}\n'
    )
    return dot


# ---------------------------------------------------------------------------
# FRCNN architecture diagram
# ---------------------------------------------------------------------------

def build_frcnn_diagram() -> str:
    """
    FRCNN / RetinaNet / FCOS / SSDLite architecture:
      Input → Backbone (ResNet/MobileNet) → FPN Neck → RPN / Proposals → RoI Head → Output

    Techniques: freeze, two_stage, full, lora, cosine, temporal
    """

    def frcnn_cluster(cid, title, bb_col, fpn_col, rpn_col, roi_col,
                      extra_nodes="", extra_edges=""):
        return (
            f'  subgraph cluster_{cid} {{\n'
            f'    label=<<B>{title}</B>>;\n'
            f'    fontname="{FONT}"; fontsize=12;\n'
            f'    bgcolor="{C["bg"]}";\n'
            f'    style="rounded";\n'
            # nodes
            f'    {cid}_in   [label="Input", fillcolor="{C["white"]}", '
            f'shape=box, style="filled,rounded", fontname="{FONT}", fontsize=10, width=1.2];\n'
            f'    {cid}_bb   [label="Backbone\\n(ResNet/MobileNet)", fillcolor="{bb_col}", '
            f'shape=box, style="filled,rounded", fontname="{FONT}", fontsize=10, width=1.9];\n'
            f'    {cid}_fpn  [label="FPN Neck", fillcolor="{fpn_col}", '
            f'shape=box, style="filled,rounded", fontname="{FONT}", fontsize=10, width=1.5];\n'
            f'    {cid}_rpn  [label="RPN /\\nProposals", fillcolor="{rpn_col}", '
            f'shape=box, style="filled,rounded", fontname="{FONT}", fontsize=10, width=1.5];\n'
            f'    {cid}_roi  [label="RoI Head\\n(box+cls)", fillcolor="{roi_col}", '
            f'shape=box, style="filled,rounded", fontname="{FONT}", fontsize=10, width=1.5];\n'
            + extra_nodes
            + f'    {cid}_in  -> {cid}_bb  [color="{C["edge"]}"];\n'
            f'    {cid}_bb  -> {cid}_fpn [color="{C["edge"]}"];\n'
            f'    {cid}_fpn -> {cid}_rpn [color="{C["edge"]}"];\n'
            f'    {cid}_rpn -> {cid}_roi [color="{C["edge"]}"];\n'
            + extra_edges
            + f'  }}\n'
        )

    legend = _legend_items([
        (C["frozen"],   "frozen  (weights unchanged)"),
        (C["trained"],  "trained  (full LR)"),
        (C["partial"],  "trained  (reduced LR or phase 2 only)"),
        (C["lora"],     "LoRA adapters  (A·B low-rank update)"),
        (C["new_head"], "new head  (always trained)"),
        (C["temporal"], "temporal loss path"),
    ])

    # freeze
    g_freeze = frcnn_cluster(
        "fr_freeze", "freeze",
        bb_col=C["frozen"], fpn_col=C["frozen"],
        rpn_col=C["trained"], roi_col=C["new_head"],
    )

    # two_stage
    ts_extra = (
        f'    fr_ts_p1 [label="Phase 1: head only", fillcolor="{C["frozen"]}", '
        f'shape=note, style="filled", fontname="{FONT}", fontsize=9, width=1.8];\n'
        f'    fr_ts_p2 [label="Phase 2: bb @ 0.01×LR", fillcolor="{C["partial"]}", '
        f'shape=note, style="filled", fontname="{FONT}", fontsize=9, width=1.8];\n'
    )
    g_two_stage = frcnn_cluster(
        "fr_two", "two_stage",
        bb_col=C["partial"], fpn_col=C["partial"],
        rpn_col=C["trained"], roi_col=C["new_head"],
        extra_nodes=ts_extra,
    )

    # full — differential LR
    g_full = frcnn_cluster(
        "fr_full", "full",
        bb_col=C["partial"], fpn_col=C["partial"],
        rpn_col=C["trained"], roi_col=C["new_head"],
    )

    # lora — LoRA on backbone Linear layers, rest frozen except heads
    lora_extra_nodes = (
        f'    fr_lora_A [label="LoRA A\\n(rank×in)", fillcolor="{C["lora"]}", '
        f'shape=box, style="filled,rounded", fontname="{FONT}", fontsize=9, width=1.3];\n'
        f'    fr_lora_B [label="LoRA B\\n(out×rank)", fillcolor="{C["lora"]}", '
        f'shape=box, style="filled,rounded", fontname="{FONT}", fontsize=9, width=1.3];\n'
        f'    fr_lora_eq [label="ΔW = B · A  (rank=4)", fillcolor="#FFF9C4", '
        f'shape=note, style="filled", fontname="{FONT}", fontsize=9, width=1.9];\n'
    )
    lora_extra_edges = (
        f'    fr_lora_bb -> fr_lora_A [color="{C["lora"]}", style=dashed];\n'
        f'    fr_lora_A  -> fr_lora_B [color="{C["lora"]}", style=dashed];\n'
        f'    fr_lora_B  -> fr_lora_rpn [color="{C["lora"]}", style=dashed, '
        f'xlabel="inject", fontname="{FONT}", fontsize=9];\n'
    )
    g_lora = frcnn_cluster(
        "fr_lora", "lora",
        bb_col=C["frozen"], fpn_col=C["frozen"],
        rpn_col=C["trained"], roi_col=C["new_head"],
        extra_nodes=lora_extra_nodes,
        extra_edges=lora_extra_edges,
    )

    # cosine — same layers as full/two_stage but different LR schedule
    cosine_extra = (
        f'    fr_cos_sched [label="CosineAnnealingLR\\n(instead of MultiStep)", '
        f'fillcolor="{C["temporal"]}", shape=note, style="filled", '
        f'fontname="{FONT}", fontsize=9, width=2.2];\n'
    )
    g_cosine = frcnn_cluster(
        "fr_cos", "cosine",
        bb_col=C["partial"], fpn_col=C["partial"],
        rpn_col=C["trained"], roi_col=C["new_head"],
        extra_nodes=cosine_extra,
    )

    # temporal — explicit adjacent-frame loss
    temporal_extra_nodes = (
        f'    fr_t_adj  [label="Adjacent\\nframe (t±1)", fillcolor="{C["temporal"]}", '
        f'shape=box, style="filled,rounded", fontname="{FONT}", fontsize=10, width=1.5];\n'
        f'    fr_t_loss [label="Temporal loss\\n(weight 0.3×)", fillcolor="{C["temporal"]}", '
        f'shape=hexagon, style="filled", fontname="{FONT}", fontsize=9, width=1.7];\n'
    )
    temporal_extra_edges = (
        f'    fr_t_adj  -> fr_t_bb   [color="{C["temporal"]}", style=dashed];\n'
        f'    fr_t_bb   -> fr_t_fpn  [color="{C["temporal"]}", style=dashed];\n'
        f'    fr_t_fpn  -> fr_t_rpn  [color="{C["temporal"]}", style=dashed];\n'
        f'    fr_t_rpn  -> fr_t_loss [color="{C["temporal"]}", style=dashed];\n'
        f'    fr_t_roi  -> fr_t_loss [color="{C["temporal"]}", style=dashed,\n'
        f'                xlabel="main loss\\n+ 0.3× adj loss", fontname="{FONT}", fontsize=9];\n'
    )
    g_temporal = frcnn_cluster(
        "fr_t", "temporal",
        bb_col=C["partial"], fpn_col=C["partial"],
        rpn_col=C["trained"], roi_col=C["new_head"],
        extra_nodes=temporal_extra_nodes,
        extra_edges=temporal_extra_edges,
    )

    dot = (
        'digraph frcnn_techniques {\n'
        f'  graph [rankdir=TB, bgcolor="{C["white"]}", splines=curved, nodesep=0.5, ranksep=0.6,\n'
        f'         label=<<B><FONT FACE="{FONT}" POINT-SIZE="16">'
        'Faster R-CNN / RetinaNet / FCOS — Fine-tuning Techniques</FONT></B>>,\n'
        f'         labelloc=t, fontname="{FONT}"];\n'
        '  node [margin="0.1,0.05"];\n\n'
        + g_freeze
        + g_two_stage
        + g_full
        + g_lora
        + g_cosine
        + g_temporal
        + legend
        + '}\n'
    )
    return dot


# ---------------------------------------------------------------------------
# Overview diagram — technique × model layer matrix
# ---------------------------------------------------------------------------

def build_overview_diagram() -> str:
    """
    HTML-table node showing a matrix:
      rows = layers (Input, Backbone, FPN/Neck, RPN/Proposals, RoI/Detect Head)
      cols = techniques (freeze, two_stage, full, lora, cosine, temporal)
    Separate tables for YOLO and FRCNN.
    """

    # (row_label, freeze, lora, two_stage, full, cosine, temporal)
    # ordered cheapest → most expensive resource consumption
    # cell values: "frozen" | "trained" | "partial" | "lora" | "new_head" | "n/a" | "temporal"
    YOLO_ROWS = [
        ("Input",                    "frozen",   "frozen",  "frozen",   "frozen",  "frozen",  "frozen"),
        ("Backbone\\n(CSP-DarkNet)", "frozen",   "frozen",  "partial",  "partial", "partial", "partial"),
        ("Neck\\n(FPN/PAN)",         "frozen",   "partial", "partial",  "partial", "partial", "partial"),
        ("Detect Head",              "trained",  "trained", "trained",  "trained", "trained", "trained"),
        ("Adj-frame aug",            "—",        "—",       "—",        "—",       "—",       "temporal"),
    ]
    FRCNN_ROWS = [
        ("Input",          "frozen",    "frozen",  "frozen",    "frozen",  "frozen",  "frozen"),
        ("Backbone",       "frozen",    "frozen",  "partial",   "partial", "partial", "partial"),
        ("FPN Neck",       "frozen",    "frozen",  "partial",   "partial", "partial", "partial"),
        ("LoRA A/B",       "—",         "lora",    "—",         "—",       "—",       "—"),
        ("RPN/Proposals",  "trained",   "trained", "trained",   "trained", "trained", "trained"),
        ("RoI/Class Head", "new_head",  "new_head","new_head",  "new_head","new_head","new_head"),
        ("Adj-frame loss", "—",         "—",       "—",         "—",       "—",       "temporal"),
        ("LR schedule",    "MultiStep", "MultiStep","MultiStep","MultiStep","Cosine", "MultiStep"),
    ]

    CELL_COLOR = {
        "frozen":   C["frozen"],
        "trained":  C["trained"],
        "partial":  C["partial"],
        "lora":     C["lora"],
        "new_head": C["new_head"],
        "temporal": C["temporal"],
        "—":        "#EEEEEE",
        "MultiStep":"#E3F2FD",
        "Cosine":   C["temporal"],
    }
    CELL_LABEL = {
        "frozen":   "frozen",
        "trained":  "trained",
        "partial":  "low LR",
        "lora":     "LoRA",
        "new_head": "trained (new)",
        "temporal": "enabled",
        "—":        "—",
        "MultiStep":"MultiStep",
        "Cosine":   "Cosine",
    }

    TECHNIQUES = ["freeze", "lora", "two_stage", "full", "cosine", "temporal"]
    HDR_COLOR = "#455A64"
    HDR_FONT = "white"

    def make_table(title, rows):
        hdr_cells = "".join(
            f'<TD BGCOLOR="{HDR_COLOR}"><FONT COLOR="{HDR_FONT}" FACE="{FONT}" '
            f'POINT-SIZE="10"><B>{t}</B></FONT></TD>'
            for t in TECHNIQUES
        )
        header = (
            f'<TR><TD BGCOLOR="{HDR_COLOR}"><FONT COLOR="{HDR_FONT}" FACE="{FONT}" '
            f'POINT-SIZE="10"><B>Layer</B></FONT></TD>{hdr_cells}</TR>'
        )
        body = ""
        for row in rows:
            label = row[0]
            cells = row[1:]
            row_cells = "".join(
                f'<TD BGCOLOR="{CELL_COLOR[c]}"><FONT FACE="{FONT}" POINT-SIZE="9">'
                f'{CELL_LABEL[c]}</FONT></TD>'
                for c in cells
            )
            row_label = (
                f'<TD BGCOLOR="#ECEFF1" ALIGN="LEFT"><FONT FACE="{FONT}" '
                f'POINT-SIZE="10">{label}</FONT></TD>'
            )
            body += f"<TR>{row_label}{row_cells}</TR>"

        return (
            f'<TABLE BORDER="1" CELLBORDER="0" CELLSPACING="2" BGCOLOR="{C["bg"]}">'
            f'<TR><TD COLSPAN="{1 + len(TECHNIQUES)}" BGCOLOR="{HDR_COLOR}">'
            f'<FONT COLOR="white" FACE="{FONT}" POINT-SIZE="13"><B>{title}</B></FONT>'
            f'</TD></TR>{header}{body}</TABLE>'
        )

    yolo_table = make_table("YOLO (YOLOv8 / YOLOv11 / RT-DETR)", YOLO_ROWS)
    frcnn_table = make_table(
        "Faster R-CNN / RetinaNet / FCOS / SSDLite", FRCNN_ROWS)

    legend = _legend_items([
        (C["frozen"],   "frozen  — weights unchanged"),
        (C["trained"],  "trained  — full LR update"),
        (C["partial"],  "low LR  — trained at reduced learning rate (0.01–0.1×)"),
        (C["lora"],     "LoRA  — trainable low-rank A·B matrices injected"),
        (C["new_head"], "trained (new)  — freshly initialised head, always trained"),
        (C["temporal"], "enabled  — temporal consistency loss / augmentation active"),
    ])

    dot = (
        'digraph overview {\n'
        f'  graph [rankdir=TB, bgcolor="{C["white"]}", '
        f'label=<<B><FONT FACE="{FONT}" POINT-SIZE="17">'
        'Fine-tuning Techniques — Layer Impact Summary</FONT></B>>,\n'
        f'         labelloc=t, fontname="{FONT}"];\n\n'
        f'  yolo_tbl  [shape=plaintext, label=<{yolo_table}>];\n'
        f'  frcnn_tbl [shape=plaintext, label=<{frcnn_table}>];\n'
        + legend
        + '}\n'
    )
    return dot


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

def render(dot_src: str, out_path: Path, fmt: str):
    try:
        import graphviz
    except ImportError:
        sys.exit("graphviz Python package not found. Run: pip install graphviz")

    src = graphviz.Source(dot_src)
    # graphviz.Source.render writes <out_path>.<fmt> and optionally <out_path>
    rendered = src.render(
        filename=str(out_path.with_suffix("")),
        format=fmt,
        cleanup=True,
    )
    print(f"  Written: {rendered}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate architecture + fine-tuning technique diagrams")
    parser.add_argument("--out-dir", default="finetune_sequence/diagrams",
                        help="Output directory (default: finetune_sequence/diagrams)")
    parser.add_argument("--format", default="svg",
                        choices=["svg", "png", "pdf"],
                        help="Output format (default: svg)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    diagrams = [
        ("yolo_techniques",   build_yolo_diagram()),
        ("frcnn_techniques",  build_frcnn_diagram()),
        ("overview",          build_overview_diagram()),
    ]

    for name, dot in diagrams:
        print(f"Rendering {name}...")
        render(dot, out_dir / name, args.format)

    print(f"\nAll diagrams written to {out_dir}/")
