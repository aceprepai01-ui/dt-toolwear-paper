#!/usr/bin/env python3
"""Teaching figures for the Part-1 lecture (data pipeline + twin + baselines).

Requires data/processed/*.npz and results/w2_baselines.json.
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
from mdrdc.data.resample import isotonize, path_to_u, resample_path, u_to_path  # noqa: E402
from mdrdc.experiments.protocol import load_processed                           # noqa: E402
from mdrdc.twin.calibrate import calibrate_early_fraction                       # noqa: E402

C = {"c1": "#0072B2", "c4": "#D55E00", "c6": "#009E73", "grey": "#888888"}
OUT = "paper/lecture_figs"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#E8E8E8", "axes.axisbelow": True})


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(f"{OUT}/{name}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def figL1_data(data):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.2))
    for t, d in data.items():
        a1.plot(d.wear, color=C[t], lw=1.6, label=f"{t} (max {d.wear.max():.0f} μm)")
    a1.axhline(165, color=C["grey"], ls=":", lw=1)
    a1.text(5, 168, "报废阈值 VB=165 μm", fontsize=8, color=C["grey"])
    a1.set_xlabel("切削序号 (cut)")
    a1.set_ylabel("后刀面磨损 VB (μm)")
    a1.set_title("(a) 三把标注刀具的全寿命磨损曲线")
    a1.legend(frameon=False, fontsize=8)
    for t, d in data.items():
        a2.plot(d.features[:, 0], color=C[t], lw=1.2, label=t)
    a2.set_xlabel("切削序号 (cut)")
    a2.set_ylabel("Fx RMS (N)")
    a2.set_title("(b) 力特征随磨损单调漂移 —— 可观测代理")
    a2.legend(frameon=False, fontsize=8)
    save(fig, "L1_data_overview")


def figL2_codec(data):
    w = isotonize(data["c1"].wear)
    p64 = resample_path(w, 64)
    w0, u = path_to_u(w, 64)
    back = u_to_path(w0, u)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    axes[0].plot(w, color=C["c1"], lw=1.4)
    axes[0].plot(np.linspace(0, len(w) - 1, 64), p64, "o", ms=2.5, color=C["c4"])
    axes[0].set_title("① 等张化 + PCHIP 重采样到 64 点")
    axes[0].set_xlabel("cut")
    axes[1].plot(u, color=C["c4"], lw=1.4)
    axes[1].set_title("② u = softplus⁻¹(增量)：无约束表征")
    axes[1].set_xlabel("网格位置")
    axes[2].plot(p64, color=C["c1"], lw=2.4, alpha=0.4, label="原路径")
    axes[2].plot(back, "--", color="#222222", lw=1.2, label="解码路径")
    axes[2].set_title("③ 解码 = w₀+cumsum(softplus(u))，必单调")
    axes[2].set_xlabel("网格位置")
    axes[2].legend(frameon=False, fontsize=8)
    save(fig, "L2_codec")


def figL3_twin(data):
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharex=True)
    for ax, (t, d) in zip(axes, data.items()):
        w = isotonize(d.wear)
        res = calibrate_early_fraction(w, fraction=0.15)
        sim = res.extrapolate(len(w), vb0=float(w[0]))
        n_cal = res.n_calib
        ax.axvspan(0, n_cal, color="#DDEBF7", zorder=0)
        ax.plot(w, color="#222222", lw=1.3, label="实测")
        ax.plot(sim, color=C[t], lw=1.6, ls="--", label="孪生外推")
        ax.set_title(f"{t}: 前15%标定 R²={res.r2_fit:.3f}")
        ax.set_xlabel("cut")
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    axes[0].set_ylabel("VB (μm)")
    axes[0].text(8, axes[0].get_ylim()[1] * 0.55, "标定窗", fontsize=8, color="#4472C4")
    save(fig, "L3_twin_calibration")


def figL4_baselines():
    with open("results/w2_baselines.json") as f:
        s = json.load(f)["summary"]
    order = ["B1", "B2@5%", "B2@10%", "B2@20%", "B3@5%", "B3@10%", "B3@20%", "LIT"]
    labels = ["B1\n上界", "B2\n5%", "B2\n10%", "B2\n20%", "B3\n5%", "B3\n10%", "B3\n20%", "LIT\n文献协议"]
    colors = ["#222222"] + ["#009E73"] * 3 + ["#CC79A7"] * 3 + ["#888888"]
    fig, ax = plt.subplots(figsize=(8, 3.2))
    x = np.arange(len(order))
    vals = [s[k]["mae_mean"] for k in order]
    errs = [s[k]["mae_std"] for k in order]
    ax.bar(x, vals, 0.62, color=colors, yerr=errs, error_kw={"lw": 0.8, "capsize": 2}, zorder=3)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 3, f"{v:.0f}", ha="center", fontsize=8)
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_ylabel("磨损 MAE (μm)")
    ax.set_title("W2 基线全景（末段20%测试窗，15 次运行均值±std）")
    save(fig, "L4_baselines")


def main():
    plt.rcParams["font.family"] = ["Arial Unicode MS", "PingFang SC", "sans-serif"]
    data = {t: load_processed("data/processed", t) for t in ("c1", "c4", "c6")}
    figL1_data(data)
    figL2_codec(data)
    figL3_twin(data)
    figL4_baselines()
    print("done ->", OUT)


if __name__ == "__main__":
    main()
