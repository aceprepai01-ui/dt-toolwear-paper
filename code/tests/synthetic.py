"""Synthetic three-phase wear curves + fake PHM2010 directory tree for tests."""
from __future__ import annotations

import os

import numpy as np


def synth_wear(n_cuts: int = 315, seed: int = 0, noise: float = 0.0) -> np.ndarray:
    """Three-phase wear curve in micrometers: break-in, steady, accelerating."""
    rng = np.random.default_rng(seed)
    n = np.arange(n_cuts, dtype=float)
    path = 40.0 * (1.0 - np.exp(-n / 25.0)) + 0.25 * n + 60.0 * (n / n_cuts) ** 4
    if noise > 0:
        path = path + rng.normal(0.0, noise, n_cuts)
    return path


def make_fake_phm2010(root: str, tools=("c1", "c4", "c6"), n_cuts: int = 12,
                      n_samples: int = 2000, seed: int = 0) -> None:
    """Write a miniature dataset with the real directory layout."""
    rng = np.random.default_rng(seed)
    for tool in tools:
        idx = tool[1:]
        sig_dir = os.path.join(root, tool, tool)
        os.makedirs(sig_dir, exist_ok=True)
        wear = synth_wear(n_cuts, seed=hash(tool) % 2 ** 16)
        rows = ["cut,flute_1,flute_2,flute_3"]
        for c in range(n_cuts):
            base = wear[c]
            rows.append(f"{c + 1},{base - 1:.3f},{base:.3f},{base + 1:.3f}")
        with open(os.path.join(root, tool, f"{tool}_wear.csv"), "w") as f:
            f.write("\n".join(rows) + "\n")
        for c in range(n_cuts):
            amp = 1.0 + 0.01 * wear[c]  # force grows with wear
            sig = rng.normal(0.0, amp, (n_samples, 7))
            np.savetxt(os.path.join(sig_dir, f"c_{idx}_{c + 1:03d}.csv"), sig, delimiter=",")
