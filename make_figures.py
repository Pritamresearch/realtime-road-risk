"""README figures: docs/results.png (top-1 hit and frame AUROC by method) and docs/timings.png (per-stage latency, before/after).

    python make_figures.py          # needs out/eval_frames.csv from evaluate.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from evaluate import auroc, bootstrap_draws, ci, loco

HERE = Path(__file__).resolve().parent
DOCS = HERE / "docs"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
OURS, OTHER = "#2a78d6", "#8a8984"  # categorical slot 1 for this system; neutral for references

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})


def results_figure() -> None:
    df = pd.read_csv(HERE / "out" / "eval_frames.csv")
    grid = yaml.safe_load((HERE / "eval_grid.yaml").read_text())
    refs = ["v1 box height", "v1 box bottom", "v1 semantic only", "previous realtime (seg, corridor)"]
    cands = [v for v in df.variant.unique() if v not in refs]
    sel, _ = loco(df, cands)
    draws, keys = bootstrap_draws(df[df.variant == refs[0]], grid["n_boot"], grid["seed"])

    def hits(fdf):
        return keys.merge(fdf[fdf.boxed][["clip", "frame", "hit"]], on=["clip", "frame"], how="left")["hit"].to_numpy(float)

    rows = [("This system (held-out LOCO)", sel, True), ("v1 box height (no depth)", df[df.variant == refs[0]], False),
            ("v1 box bottom (no depth)", df[df.variant == refs[1]], False), ("v1 class weight only", df[df.variant == refs[2]], False),
            ("First real-time version", df[df.variant == refs[3]], False)]
    stats = [(name, *ci(hits(f), draws), auroc(f.score.to_numpy(float), f.boxed.to_numpy(bool)), ours) for name, f, ours in rows]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.4), gridspec_kw={"width_ratios": [1.6, 1]}, sharey=True)
    y = np.arange(len(stats))[::-1]
    for yi, (name, m, lo, hi, au, ours) in zip(y, stats):
        c = OURS if ours else OTHER
        a1.plot([lo, hi], [yi, yi], color=c, lw=2, solid_capstyle="round")
        a1.plot(m, yi, "o", ms=8, color=c, mec=SURFACE, mew=2)
        a1.text(hi + 0.012, yi, f"{m:.3f}", va="center", color=INK, fontsize=9)
        a2.plot(au, yi, "o", ms=8, color=c, mec=SURFACE, mew=2)  # dots, not bars: the axis does not start at 0
        a2.text(au + 0.006, yi, f"{au:.3f}", va="center", color=INK, fontsize=9)
    a1.set_yticks(y, [s[0] for s in stats])
    a1.set_xlim(0, 0.75)
    a1.set_xlabel("Top-1 object hit (IoU >= 0.5), 95% CI")
    a2.set_xlim(0.8, 1.0)
    a2.set_xlabel("Frame AUROC, critical vs calm frames\n(axis starts at 0.8)")
    for a in (a1, a2):
        a.grid(axis="x", color=GRID, lw=0.8)
        a.set_axisbelow(True)
        for s in ("top", "right", "left"):
            a.spines[s].set_visible(False)
        a.tick_params(axis="y", length=0)
    fig.suptitle("242 annotated frames, 8 clips, 1 annotator: comparable to box height, not better (difference CIs include 0)",
                 color=INK, fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(DOCS / "results.png", dpi=160)
    plt.close(fig)


def timings_figure() -> None:
    """Medians measured on the RTX 3050 6 GB Laptop GPU on battery (see README, 'Speed')."""
    stages = ["DA3 metric depth", "Segmentation", "Detection (YOLO26s-seg)", "Fusion + render (serial)"]
    before = [460, 210, 0, 45]
    after = [264, 120, 30, 0]  # fusion + render now overlap the GPU stage
    fig, ax = plt.subplots(figsize=(8.5, 2.9))
    y = np.arange(len(stages))[::-1]
    ax.barh(y + 0.19, before, height=0.34, color=OTHER, label="First version: 1.2-1.7 FPS")
    ax.barh(y - 0.19, after, height=0.34, color=OURS, label="Now: 2.3-2.4 FPS")
    for yi, b, a in zip(y, before, after):
        ax.text(b + 6, yi + 0.19, f"{b} ms" if b else "not used", va="center", color=INK2, fontsize=9)
        ax.text(a + 6, yi - 0.19, f"{a} ms" if a else "overlapped", va="center", color=INK, fontsize=9)
    ax.set_yticks(y, stages)
    ax.set_xlabel("Median latency per frame (ms), laptop on battery")
    ax.set_xlim(0, 560)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(frameon=False, loc="lower right", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(DOCS / "timings.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    DOCS.mkdir(exist_ok=True)
    results_figure()
    timings_figure()
    print("wrote", DOCS / "results.png", DOCS / "timings.png")
