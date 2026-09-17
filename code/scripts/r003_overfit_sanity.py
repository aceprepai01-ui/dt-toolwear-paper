#!/usr/bin/env python3
"""R003: pipeline sanity — a tiny TCN must overfit one tool (train MAE -> ~0).

Proves feature -> wear learnability and the training loop before any real
experiment. Gate: train MAE < 3 um after a few hundred epochs.
Runs on synthetic data with --synthetic (no dataset needed).
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tests.synthetic import synth_wear  # noqa: E402

GATE_MAE_UM = 3.0


class TinyTCN(nn.Module):
    """Minimal causal dilated conv stack for per-cut wear regression."""

    def __init__(self, in_dim: int, hidden: int = 32, levels: int = 4):
        super().__init__()
        layers = []
        ch = in_dim
        for i in range(levels):
            d = 2 ** i
            layers += [nn.Conv1d(ch, hidden, 3, padding=d, dilation=d), nn.ReLU()]
            ch = hidden
        self.net = nn.Sequential(*layers)
        self.head = nn.Conv1d(hidden, 1, 1)

    def forward(self, x):  # x: (B, C, T)
        h = self.net(x)[..., : x.shape[-1]]
        return self.head(h)[:, 0, :]


def load_xy(args) -> tuple[np.ndarray, np.ndarray]:
    if args.synthetic:
        wear = synth_wear(315, seed=0)
        rng = np.random.default_rng(0)
        feats = np.stack([wear * s + rng.normal(0, 1.0, len(wear))
                          for s in (0.01, 0.02, 0.005, 0.03)] * 3, axis=1)
        return feats, wear
    d = np.load(os.path.join(args.processed, "c1.npz"))
    return d["features"], d["wear"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--epochs", type=int, default=500)
    args = ap.parse_args()

    feats, wear = load_xy(args)
    mu, sd = feats.mean(0), feats.std(0) + 1e-9
    x = torch.tensor(((feats - mu) / sd).T[None], dtype=torch.float32)  # (1, C, T)
    y = torch.tensor(wear[None], dtype=torch.float32)

    torch.manual_seed(0)
    model = TinyTCN(in_dim=x.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for ep in range(args.epochs):
        opt.zero_grad()
        loss = nn.functional.huber_loss(model(x), y, delta=5.0)
        loss.backward()
        opt.step()
        if (ep + 1) % 100 == 0:
            with torch.no_grad():
                m = (model(x) - y).abs().mean().item()
            print(f"epoch {ep + 1}: train MAE = {m:.3f} um")

    with torch.no_grad():
        final_mae = (model(x) - y).abs().mean().item()
    ok = final_mae < GATE_MAE_UM
    print(f"\n=== R003 REPORT ===\ntrain MAE = {final_mae:.3f} um (gate < {GATE_MAE_UM})")
    print("GATE:", "PASS — pipeline learns; ready for W2 baselines" if ok else "FAIL — debug loop")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
