"""Pareto frontier plotter for CBV-RAG vs baselines and ablations.

Reads one or more JSON result files (produced by ``run_cbvrag_eval.py``,
``run_baselines.py``, and ``run_ablation.py``) and plots a **Pareto frontier**
showing EM vs total tokens.  Each system is drawn with a distinct marker.  The
Pareto-optimal region (no system dominates another on both axes) is shaded.

Output files
------------
* ``<out>`` — PNG raster image.
* ``<out>.pdf`` — Vector PDF (publication-quality, same content).

Usage
-----
    python scripts/plot_frontier.py \\
        --inputs \\
            logs/eval_heuristic.json \\
            logs/eval_il.json \\
            logs/baselines_hotpotqa.json \\
            logs/ablation_hotpotqa.json \\
        --out logs/frontier_hotpotqa.png \\
        --title HotpotQA
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ---- marker / colour catalogue (one per named system) ----------------------

_SYSTEM_STYLES: Dict[str, Dict] = {
    # CBV-RAG variants
    "cbvrag_heuristic":   {"marker": "o", "color": "#1f77b4", "zorder": 4},
    "cbvrag_il":          {"marker": "s", "color": "#2ca02c", "zorder": 4},
    "cbvrag_offline":     {"marker": "D", "color": "#d62728", "zorder": 4},
    # Ablations
    "full":               {"marker": "o", "color": "#1f77b4", "zorder": 4},
    "no_null_branch":     {"marker": "v", "color": "#9467bd", "zorder": 3},
    "gd_only":            {"marker": "^", "color": "#8c564b", "zorder": 3},
    "no_nli_verifier":    {"marker": "<", "color": "#e377c2", "zorder": 3},
    # Baselines
    "vanilla_rag":        {"marker": "P", "color": "#ff7f0e", "zorder": 2},
    "cfrag":              {"marker": "X", "color": "#17becf", "zorder": 2},
    "vericite":           {"marker": "*", "color": "#bcbd22", "zorder": 2},
    "cfrag_baseline":     {"marker": "X", "color": "#17becf", "zorder": 2},
}

_DEFAULT_STYLE = {"marker": "o", "color": "#7f7f7f", "zorder": 1}


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_systems(paths: List[str]) -> Dict[str, Tuple[float, float]]:
    """Load (em, avg_total_tokens) for each named system across all input files.

    Each JSON file may contain top-level keys that are system names, each with
    an ``"aggregate"`` or direct ``"accuracy"``/``"avg_total_tokens"`` dict.
    Both ``run_cbvrag_eval.py`` and ``run_baselines.py`` / ``run_ablation.py``
    output formats are supported.

    Returns
    -------
    dict mapping system_name → (em, avg_total_tokens).
    """
    systems: Dict[str, Tuple[float, float]] = {}

    for path in paths:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        for key, val in data.items():
            if not isinstance(val, dict):
                continue
            # Handle run_ablation.py / run_baselines.py format: val["aggregate"]
            agg = val.get("aggregate") or val
            # EM field may be "em" (new scripts) or "accuracy" (old eval script)
            em = float(agg.get("em", agg.get("accuracy", 0.0)))
            tokens = float(agg.get("avg_total_tokens", 0.0))
            if em > 0 or tokens > 0:
                systems[key] = (em, tokens)

    return systems


# ---------------------------------------------------------------------------
# Pareto frontier computation
# ---------------------------------------------------------------------------

def _pareto_frontier(
    systems: Dict[str, Tuple[float, float]],
) -> List[str]:
    """Return names of systems on the Pareto frontier (higher EM, lower tokens).

    A system is Pareto-optimal if no other system has both strictly higher EM
    *and* strictly lower (or equal) token count.
    """
    pareto = []
    names = list(systems.keys())
    for name, (em_i, tok_i) in systems.items():
        dominated = False
        for other, (em_j, tok_j) in systems.items():
            if other == name:
                continue
            if em_j >= em_i and tok_j <= tok_i and (em_j > em_i or tok_j < tok_i):
                dominated = True
                break
        if not dominated:
            pareto.append(name)
    return pareto


# ---------------------------------------------------------------------------
# Main plotting
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Plot EM vs Tokens Pareto frontier for CBV-RAG.")
    ap.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="One or more JSON result files from run_cbvrag_eval / run_baselines / run_ablation.",
    )
    ap.add_argument("--out", default="logs/frontier.png", help="Output PNG path.")
    ap.add_argument("--title", default="", help="Dataset name shown in the plot title.")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument(
        "--no_shading",
        action="store_true",
        help="Disable the Pareto-optimal region shading.",
    )
    args = ap.parse_args()

    systems = _load_systems(args.inputs)
    if not systems:
        print("No valid system data found in the provided input files.")
        return 1

    pareto_names = _pareto_frontier(systems)

    fig, ax = plt.subplots(figsize=(8, 5))

    # ---- Shaded Pareto region -----------------------------------------------
    if not args.no_shading and pareto_names:
        # Sort Pareto systems by token count to draw a step-wise region.
        pareto_pts = sorted([(systems[n][1], systems[n][0]) for n in pareto_names])
        xs_p = [p[0] for p in pareto_pts]
        ys_p = [p[1] for p in pareto_pts]

        # Extend to axes limits.
        x_min = 0.0
        y_max = max(em for _, em in systems.values()) * 1.1
        fill_xs = [x_min] + xs_p + [max(xs_p)]
        fill_ys = [ys_p[0]] + ys_p + [0.0]
        ax.fill(fill_xs, fill_ys, alpha=0.08, color="#1f77b4", label="_pareto_region")

        # Draw the Pareto step line.
        ax.step(xs_p, ys_p, where="post", color="#1f77b4", linewidth=1.2,
                linestyle="--", alpha=0.5, label="_pareto_line")

    # ---- Scatter points -------------------------------------------------------
    legend_handles = []
    for name, (em, tokens) in systems.items():
        style = _SYSTEM_STYLES.get(name, _DEFAULT_STYLE)
        is_pareto = name in pareto_names
        edge = "black" if is_pareto else "none"
        size = 120 if is_pareto else 80
        ax.scatter(
            tokens,
            em * 100,  # convert to percentage
            marker=style["marker"],
            color=style["color"],
            s=size,
            edgecolors=edge,
            linewidths=1.2,
            zorder=style["zorder"],
        )
        ax.annotate(
            name,
            (tokens, em * 100),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=7.5,
            color=style["color"],
        )
        legend_handles.append(
            mpatches.Patch(color=style["color"], label=name)
        )

    # ---- Pareto-optimal region legend patch ----------------------------------
    if not args.no_shading and pareto_names:
        legend_handles.append(
            mpatches.Patch(color="#1f77b4", alpha=0.15, label="Pareto-optimal region")
        )

    # ---- Labels & formatting --------------------------------------------------
    ax.set_xlabel("Total Tokens Used", fontsize=11)
    ax.set_ylabel("Exact Match (%)", fontsize=11)
    title = f"EM vs Total Tokens — {args.title}" if args.title else "EM vs Total Tokens"
    ax.set_title(title, fontsize=12)
    ax.legend(handles=legend_handles, fontsize=8, loc="lower right", framealpha=0.85)
    ax.grid(True, alpha=0.3, linestyle=":")

    # Force EM axis to percentage scale.
    y_vals = [em * 100 for em, _ in [(em, tok) for _, (em, tok) in systems.items()]]
    if y_vals:
        ax.set_ylim(bottom=max(0, min(y_vals) - 5), top=min(100, max(y_vals) + 8))

    fig.tight_layout()

    # ---- Save PNG + PDF -------------------------------------------------------
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi)
    print(f"Saved PNG: {out_path}")

    pdf_path = out_path.with_suffix(".pdf")
    fig.savefig(pdf_path)
    print(f"Saved PDF: {pdf_path}")

    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
