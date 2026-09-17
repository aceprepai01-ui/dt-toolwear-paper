#!/usr/bin/env python3
"""Paper figures (Measurement/Elsevier style). Vector PDF + 300dpi PNG.

Requires: results/w6_hb_matrix.json, results/w5_downstream.json,
results/w6b_flag_cp.json, data/processed/*, QIT under data/raw/scidata.

Palette: Okabe-Ito subset, validated (dataviz six checks, 2026-08-13):
pooled #0072B2 / handbook #E69F00 / B2 #009E73 / B3 #CC79A7 / diffusion #D55E00.
Direct value labels satisfy the CVD 6-8 band relief requirement.
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mdrdc.data.qit import load_qit_tool                                    # noqa: E402
from mdrdc.data.resample import isotonize                                   # noqa: E402
from mdrdc.experiments.protocol import (ROTATIONS, label_indices,           # noqa: E402
                                        load_processed, test_indices)
from mdrdc.twin.calibrate import calibrate                                  # noqa: E402
from mdrdc.twin.hierarchical import calibrate_pooled, fit_hyperprior        # noqa: E402

C = {"pooled": "#0072B2", "handbook": "#E69F00", "B2": "#009E73",
     "B3": "#CC79A7", "OURS": "#D55E00", "real": "#222222"}
FIGDIR = "paper/figures"

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 8.5, "axes.labelsize": 9,
    "axes.titlesize": 9, "legend.fontsize": 7.5, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": "#E6E6E6",
    "grid.linewidth": 0.5, "axes.axisbelow": True, "lines.linewidth": 1.6,
})


def save(fig, name):
    os.makedirs(FIGDIR, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{FIGDIR}/{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {name}")


def load_results():
    with open("results/w6_hb_matrix.json") as f:
        h = json.load(f)
    with open("results/w5_downstream.json") as f:
        w5 = json.load(f)
    return h, w5


def forecasts_at(data, wears, rot, k=0.10):
    tgt = rot["target"]
    tw = wears[tgt]
    sources = {n: wears[n] for n in list(rot["sources"]) + ["qit"]}
    hyper = fit_hyperprior(sources)
    idx = label_indices(len(tw), k)
    labeled = np.maximum.accumulate(tw[idx])
    p_pool = calibrate_pooled(labeled, idx, hyper).extrapolate(len(tw), vb0=float(labeled[0]))
    p_hand = calibrate(labeled, indices=idx).extrapolate(len(tw), vb0=float(labeled[0]))
    return tw, idx, p_pool, p_hand


def fig2_headline(h, w5):
    ks = (5, 10)
    systems = ["pooled", "handbook", "B2", "B3", "OURS"]
    labels = {"pooled": "Pooled twin (ours)", "handbook": "Handbook-prior twin",
              "B2": "TCN few-shot", "B3": "TCN + twin aug.", "OURS": "TCN + diffusion aug."}
    means, sems = {}, {}
    for k in ks:
        for s in ("pooled", "handbook"):
            v = [r["mae"] for r in h["H1"] if r["k"] == k / 100 and r["system"] == s]
            means[(s, k)], sems[(s, k)] = np.mean(v), 0.0
        for s in ("B2", "B3", "OURS"):
            e = w5["summary"][f"{s}@{k}%"]
            means[(s, k)], sems[(s, k)] = e["mae"], e["sem"]

    fig, ax = plt.subplots(figsize=(5.6, 2.6))
    x = np.arange(len(ks))
    w = 0.16
    for i, s in enumerate(systems):
        vals = [means[(s, k)] for k in ks]
        errs = [sems[(s, k)] for k in ks]
        pos = x + (i - 2) * w
        ax.bar(pos, vals, w * 0.92, color=C[s], label=labels[s],
               yerr=[e if e > 0 else None or 0 for e in errs],
               error_kw={"lw": 0.8, "capsize": 2}, zorder=3)
        for p, v in zip(pos, vals):
            ax.text(p, v + 1.2, f"{v:.0f}", ha="center", fontsize=7)
    ax.axhline(21.0, color="#444444", ls="--", lw=1.0, zorder=2)
    ax.text(-0.44, 17.2, "DL upper bound (80% labels)", fontsize=7, color="#444444",
            ha="left", va="top")
    ax.set_xticks(x, [f"k = {k}% labels" for k in ks])
    ax.set_ylabel("Tail wear MAE (μm)")
    ax.legend(ncol=3, frameon=False, loc="lower center",
              bbox_to_anchor=(0.5, 1.01), columnspacing=1.0, handlelength=1.2)
    ax.set_ylim(0, 62)
    save(fig, "fig2_headline")


def fig3_curves(data, wears):
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.3), sharex=False)
    for ax, rot in zip(axes, ROTATIONS):
        tw, idx, p_pool, p_hand = forecasts_at(data, wears, rot)
        n = len(tw)
        ti = test_indices(n)
        ax.axvspan(ti[0], n - 1, color="#F0F0F0", zorder=0)
        ax.plot(tw, color=C["real"], lw=1.2, label="Measured")
        ax.plot(p_pool, color=C["pooled"], lw=1.6, label="Pooled twin")
        ax.plot(p_hand, color=C["handbook"], lw=1.4, ls="--", label="Handbook twin")
        ax.plot(idx, tw[idx], ".", color=C["real"], ms=2.5, label="Labels (k=10%)")
        ax.axhline(165, color="#888888", ls=":", lw=0.8)
        ax.set_title(f"Target {rot['target']}")
        ax.set_xlabel("Cut number")
    axes[0].set_ylabel("Flank wear VB (μm)")
    axes[0].legend(frameon=False, loc="upper left", fontsize=6.5)
    axes[2].text(8, 168, "VB threshold", fontsize=6.5, color="#888888")
    save(fig, "fig3_extrapolation")


def fig4_boundary(data, wears):
    with open("results/w6b_flag_cp.json") as f:
        b = json.load(f)
    with open("results/w6_hb_matrix.json") as f:
        h = json.load(f)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.4),
                                 gridspec_kw={"width_ratios": [1.5, 1]})
    rot = next(r for r in ROTATIONS if r["target"] == "c4")
    tw, idx, p_pool, p_hand = forecasts_at(data, wears, rot)
    n = len(tw)
    a1.axvspan(int(0.8 * n), n - 1, color="#F0F0F0", zorder=0)
    a1.plot(tw, color=C["real"], lw=1.2, label="Measured (c4)")
    a1.plot(p_pool, color=C["pooled"], lw=1.6, label="Pooled twin")
    a1.plot(p_hand, color=C["handbook"], lw=1.4, ls="--", label="Handbook twin")
    a1.plot(idx, tw[idx], ".", color=C["real"], ms=2.5)
    a1.annotate("regime change begins\nafter the label pool", xy=(265, 160),
                xytext=(120, 185), fontsize=7,
                arrowprops={"arrowstyle": "->", "lw": 0.8})
    a1.set_xlabel("Cut number")
    a1.set_ylabel("Flank wear VB (μm)")
    a1.legend(frameon=False, fontsize=6.5, loc="upper left")
    a1.set_title("(a) The undetectable tool")

    tools = ["c1", "c4", "c6"]
    mis = [next(r["misfit_rmse_um"] for r in h["H5"] if r["target"] == t) for t in tools]
    div = [next(r["divergence_um"] for r in b["H5b"] if r["target"] == t) for t in tools]
    x = np.arange(3)
    a2.bar(x - 0.18, mis, 0.34, color=C["pooled"], label="Label-window misfit", zorder=3)
    a2.bar(x + 0.18, div, 0.34, color=C["handbook"], label="Prior-sensitivity div.", zorder=3)
    for xi, (m, d) in enumerate(zip(mis, div)):
        a2.text(xi - 0.18, m + 0.15, f"{m:.1f}", ha="center", fontsize=6.5)
        a2.text(xi + 0.18, d + 0.15, f"{d:.1f}", ha="center", fontsize=6.5)
    a2.set_xticks(x, tools)
    a2.set_ylabel("Detector value (μm)")
    a2.set_title("(b) Neither detector separates c4")
    a2.set_ylim(0, 11.5)
    a2.legend(frameon=False, fontsize=6.5, loc="upper left", handlelength=1.2)
    save(fig, "fig4_boundary")


def fig5_robustness(h):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.4, 2.2))
    tools = ["c1", "c4", "c6"]
    for i, t in enumerate(tools):
        v = [r["mae"] for r in h["H3"] if r["target"] == t]
        full = next(r["mae"] for r in h["H1"]
                    if r["target"] == t and r["k"] == 0.10 and r["system"] == "pooled")
        a1.plot([i] * len(v), v, "o", color=C["pooled"], ms=4, alpha=0.7, zorder=3)
        a1.plot(i, full, "D", color=C["real"], ms=4, zorder=4)
    a1.set_xticks(range(3), tools)
    a1.set_ylabel("Tail wear MAE (μm)")
    a1.set_title("(a) Leave-one-source-out")
    a1.plot([], [], "o", color=C["pooled"], label="LOO hyperprior")
    a1.plot([], [], "D", color=C["real"], label="All sources")
    a1.legend(frameon=False, fontsize=6.5, loc="center left")

    for t in tools:
        xs = [0, 1, 2]
        ys = [next(r["mae"] for r in h["H4"] if r["target"] == t and r["tau_scale"] == f)
              for f in (0.5, 1.0, 2.0)]
        a2.plot(xs, ys, "o-", color=C["pooled"], ms=3.5, alpha=0.75)
        a2.text(2.1, ys[-1], t, fontsize=7, va="center")
    a2.set_xticks([0, 1, 2], ["×0.5", "×1", "×2"])
    a2.set_xlabel("Hyperprior scale factor")
    a2.set_title("(b) Prior-scale sensitivity")
    a2.set_xlim(-0.3, 2.5)
    save(fig, "fig5_robustness")


def main():
    h, w5 = load_results()
    data = {t: load_processed("data/processed", t) for t in ("c1", "c4", "c6")}
    wears = {t: isotonize(d.wear) for t, d in data.items()}
    wears["qit"] = isotonize(load_qit_tool("data/raw/scidata/Milling dataset from QIT").wear)
    fig2_headline(h, w5)
    fig3_curves(data, wears)
    fig4_boundary(data, wears)
    fig5_robustness(h)
    print("all figures done ->", FIGDIR)


if __name__ == "__main__":
    main()
