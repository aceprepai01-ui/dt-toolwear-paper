#!/usr/bin/env python3
"""Part-2 lecture figures — STANDALONE (the mdrdc package and raw data were
deleted from disk on ~2026-08-22; every number below is transcribed from the
archived run reports in refine-logs/EXPERIMENT_TRACKER.md and session records).

Figure provenance, stated in every caption:
  [存档数字]  redrawn from archived run results
  [合成示意]  synthetic illustration of a documented phenomenon
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

BLUE, ORANGE, GREEN, PINK, VERM, GREY = ("#0072B2", "#E69F00", "#009E73",
                                          "#CC79A7", "#D55E00", "#888888")
OUT = "/Users/jixia/Downloads/dt-toolwear-paper/code/paper/lecture_figs"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#E8E8E8", "axes.axisbelow": True,
                     "font.family": ["Arial Unicode MS", "PingFang SC", "sans-serif"]})

# ---- archived numbers (refine-logs + session reports, 2026-08-12/13) ----
CAMPAIGN = [("v0 冒烟\n(计分bug)", 40.5), ("v1 删失\n修正", 37.7), ("v2 协议\n对齐", 35.3),
            ("v3 编码+\n逐位置", 29.3), ("v4 局部化+\n拒绝选项", 27.0)]
TWIN_BASE = 24.3
W5 = {"B2": {5: (51.39, 2.49), 10: (52.43, 3.63)},
      "B3": {5: (53.46, 4.46), 10: (48.46, 3.52)},
      "OURS": {5: (53.69, 5.62), 10: (48.36, 4.73)}}
H1 = {"pooled": {5: 22.65, 10: 25.01}, "handbook": {5: 31.82, 10: 32.41}}
DL_UB = 21.0
DETECT = {"c1": (4.17, 7.5), "c4": (5.03, 6.9), "c6": (6.56, 7.8)}  # (misfit, divergence)


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(f"{OUT}/{name}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def synth_wear(n=315, plateau=True, explode=False, seed=0):
    """Three-phase synthetic wear curve for illustrative figures only."""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    w = 45 * (1 - np.exp(-t / 22)) + 0.22 * t + 55 * (t / n) ** 4 + 30
    if plateau:
        w[110:190] = w[110]  # measurement plateau
    if explode:
        w[250:] = w[250] + 1.25 * (t[250:] - 250) ** 1.15
    w = np.maximum.accumulate(w + rng.normal(0, 0.6, n))
    return w


def _box(ax, x, y, w, h, text, fc, fs=9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                fc=fc, ec="#555555", lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def _arrow(ax, x0, y0, x1, y1, label=None):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="->",
                                 mutation_scale=12, lw=1.1, color="#333333"))
    if label:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.06, label, ha="center",
                fontsize=7.5, color="#333333")


def figL5_pipeline():
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.2)
    ax.axis("off")
    _box(ax, 0.1, 2.0, 1.9, 0.9, "源刀全寿命\n(2×PHM2010 + QIT)", "#DDEBF7")
    _box(ax, 0.1, 0.4, 1.9, 0.9, "目标刀 k% 标注\n(均匀抽样)", "#DDEBF7")
    _box(ax, 2.6, 2.0, 2.15, 0.9, "标定变体库\n每刀16-32个孪生变体\n残差 r = u实−u孪", "#FFF2CC")
    _box(ax, 2.6, 0.4, 2.15, 0.9, "目标孪生 MAP 标定\n→ u_twin，失配 m", "#FFF2CC")
    _box(ax, 5.35, 1.2, 2.0, 1.0, "条件残差扩散\n0.19M 参数 1-D DDPM\n条件=[u_twin, m, 尾斜率]", "#E2EFDA")
    _box(ax, 7.95, 1.2, 1.95, 1.0, "解码（必单调）\nw=w₀+Σsoftplus(u_twin+r̂)\n→ 50 条修正路径", "#FCE4EC")
    _arrow(ax, 2.0, 2.45, 2.6, 2.45)
    _arrow(ax, 2.0, 0.85, 2.6, 0.85)
    _arrow(ax, 4.75, 2.3, 5.45, 2.05, "训练")
    _arrow(ax, 4.75, 1.0, 5.45, 1.35, "条件")
    _arrow(ax, 7.35, 1.7, 7.95, 1.7, "采样")
    ax.text(5.0, 0.04, "设计要点：扩散只学“孪生与真实的差”；单调性由 softplus-累加结构保证，与 r̂ 取值无关",
            fontsize=8.5, color="#555555", ha="center")
    save(fig, "L5_mdrdc_pipeline")


def figL6_campaign():
    labels = [c[0] for c in CAMPAIGN]
    crps = [c[1] for c in CAMPAIGN]
    fixes = ["门槛虚高", "两条路径\n同规则删失", "库与部署\n同标定协议", "ENC_EPS=0.05\n消灭−13.8尖峰",
             "同域检索+核加权\n修正不显著即回退"]
    fig, ax = plt.subplots(figsize=(8.2, 3.5))
    x = np.arange(len(labels))
    ax.plot(x, crps, "o-", color=VERM, lw=2, ms=6, zorder=3)
    for xi, v in zip(x, crps):
        ax.text(xi + 0.08, v + 0.6, f"{v}", fontsize=9, color=VERM)
    ax.axhline(TWIN_BASE, color=BLUE, ls="--", lw=1.4)
    ax.text(3.95, 22.9, f"孪生先验基准 {TWIN_BASE}（低于它修正才算有效）",
            fontsize=8.5, color=BLUE, ha="right")
    for xi, f in zip(x, fixes):
        ax.text(xi, 43.6, f, ha="center", fontsize=7, color="#555555")
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_ylabel("聚合穿越时间 CRPS（刀数）")
    ax.set_ylim(20, 47)
    save(fig, "L6_campaign")


def figL7_uspike():
    w = synth_wear(plateau=True)
    # PCHIP-free illustrative resample: strided pick + isotonic already holds
    idx = np.linspace(0, len(w) - 1, 64).round().astype(int)
    p64 = w[idx]
    inc = np.clip(np.diff(p64), 0.0, None)

    def sp_inv(y, eps):
        y = np.clip(y, eps, None)
        return y + np.log(-np.expm1(-y))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3), sharex=True)
    u1 = sp_inv(inc, 1e-6)
    a1.plot(u1, color=VERM, lw=1.4)
    a1.set_title("(a) 下限 1e-6：平台段爆出约 −13.8 的尖峰")
    a1.set_xlabel("网格位置")
    a1.set_ylabel("u 值")
    a1.annotate("测量平台 → 增量≈0\n→ softplus⁻¹ 发散", xy=(int(np.argmin(u1)), float(u1.min())),
                xytext=(30, -9.5), fontsize=8, arrowprops={"arrowstyle": "->", "lw": 0.8})
    a2.plot(sp_inv(inc, 0.05), color=BLUE, lw=1.4)
    a2.set_title("(b) ENC_EPS=0.05 μm：u 有界于 ≈−3")
    a2.set_xlabel("网格位置")
    save(fig, "L7_uspike")


def figL8_arbitration():
    names = [("B2", GREEN, "TCN 少样本"), ("B3", PINK, "TCN+孪生增强"),
             ("OURS", VERM, "TCN+扩散修正增强")]
    fig, ax = plt.subplots(figsize=(7.5, 3.3))
    x = np.arange(2)
    for i, (k_, color, label) in enumerate(names):
        vals = [W5[k_][k][0] for k in (5, 10)]
        errs = [W5[k_][k][1] for k in (5, 10)]
        pos = x + (i - 1) * 0.22
        ax.bar(pos, vals, 0.20, color=color, label=label, yerr=errs,
               error_kw={"lw": 0.9, "capsize": 3}, zorder=3)
        for p_, v in zip(pos, vals):
            ax.text(p_, v + 2.2, f"{v:.1f}", ha="center", fontsize=8.5)
    ax.set_xticks(x, ["k = 5%", "k = 10%"])
    ax.set_ylabel("末段磨损 MAE (μm)")
    ax.set_ylim(0, 68)
    ax.set_title("W5 下游仲裁（预登记规则）：误差棒全交叠 → NO SEPARATION，K1 否决")
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="lower center",
              bbox_to_anchor=(0.5, -0.36))
    save(fig, "L8_arbitration")


def figL9_reframe():
    systems = [("池化孪生（换帅后）", BLUE, [H1["pooled"][5], H1["pooled"][10]], [0, 0]),
               ("手册先验孪生", ORANGE, [H1["handbook"][5], H1["handbook"][10]], [0, 0]),
               ("TCN 少样本", GREEN, [W5["B2"][5][0], W5["B2"][10][0]], [W5["B2"][5][1], W5["B2"][10][1]]),
               ("TCN+孪生增强", PINK, [W5["B3"][5][0], W5["B3"][10][0]], [W5["B3"][5][1], W5["B3"][10][1]]),
               ("TCN+扩散增强", VERM, [W5["OURS"][5][0], W5["OURS"][10][0]], [W5["OURS"][5][1], W5["OURS"][10][1]])]
    fig, ax = plt.subplots(figsize=(8.4, 3.4))
    x = np.arange(2)
    for i, (label, color, vals, errs) in enumerate(systems):
        pos = x + (i - 2) * 0.155
        ax.bar(pos, vals, 0.145, color=color, label=label,
               yerr=[e if e else 0 for e in errs], error_kw={"lw": 0.8, "capsize": 2}, zorder=3)
        for p_, v in zip(pos, vals):
            ax.text(p_, v + 1.4, f"{v:.0f}", ha="center", fontsize=7.5)
    ax.axhline(DL_UB, color="#444444", ls="--", lw=1.0)
    ax.text(-0.42, 17.6, f"DL 上界（80%标注）= {DL_UB}", fontsize=7.5, color="#444444", va="top")
    ax.set_xticks(x, ["k = 5% 标注", "k = 10% 标注"])
    ax.set_ylabel("末段磨损 MAE (μm)")
    ax.set_ylim(0, 64)
    ax.legend(ncol=3, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0),
              columnspacing=0.9, handlelength=1.1, fontsize=7.5)
    save(fig, "L9_reframe")


def figL10_boundary():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.4, 3.1),
                                 gridspec_kw={"width_ratios": [1.4, 1]})
    w = synth_wear(plateau=False, explode=True, seed=3)
    n = len(w)
    twin = w.copy().astype(float)
    twin[250:] = twin[250] + 0.32 * (np.arange(n - 250))  # keeps pre-break trend
    lab = np.linspace(0, int(0.8 * n) - 1, 31).round().astype(int)
    a1.axvspan(int(0.8 * n), n - 1, color="#F0F0F0", zorder=0)
    a1.plot(w, color="#222222", lw=1.3, label="实测（c4 形态示意）")
    a1.plot(twin, color=BLUE, lw=1.6, ls="--", label="孪生外推（池化/手册同败）")
    a1.plot(lab, w[lab], ".", color="#222222", ms=3)
    a1.annotate("状态切换发生在\n标注池结束之后", xy=(275, w[275]), xytext=(120, w[-1] * 0.68),
                fontsize=8, arrowprops={"arrowstyle": "->", "lw": 0.8})
    a1.set_xlabel("切削序号")
    a1.set_ylabel("VB (μm)")
    a1.set_title("(a) c4 型状态切换（合成示意）")
    a1.legend(frameon=False, fontsize=7.5, loc="upper left")
    tools = list(DETECT)
    x = np.arange(3)
    a2.bar(x - 0.18, [DETECT[t][0] for t in tools], 0.34, color=BLUE,
           label="标注窗失配", zorder=3)
    a2.bar(x + 0.18, [DETECT[t][1] for t in tools], 0.34, color=ORANGE,
           label="先验敏感度分歧", zorder=3)
    for xi, t in enumerate(tools):
        a2.text(xi - 0.18, DETECT[t][0] + 0.15, f"{DETECT[t][0]:.1f}", ha="center", fontsize=7)
        a2.text(xi + 0.18, DETECT[t][1] + 0.15, f"{DETECT[t][1]:.1f}", ha="center", fontsize=7)
    a2.set_xticks(x, tools)
    a2.set_ylim(0, 11.5)
    a2.set_ylabel("探测器读数 (μm)")
    a2.set_title("(b) 两个探测器都不认识 c4")
    a2.legend(frameon=False, fontsize=7.5, loc="upper left")
    save(fig, "L10_boundary")


def main():
    figL5_pipeline()
    figL6_campaign()
    figL7_uspike()
    figL8_arbitration()
    figL9_reframe()
    figL10_boundary()
    print("done ->", OUT)


if __name__ == "__main__":
    main()
