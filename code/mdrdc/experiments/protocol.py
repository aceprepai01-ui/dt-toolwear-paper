"""W2+ experimental protocol: rotations, label budgets, fixed test window, RUL.

- Test window: LAST 20% of the target tool's cuts (high-wear, decision-critical),
  shared by every system and every k — cross-k comparable.
- Label budget: uniform-strided k% over the FIRST 80% of cuts.
- RUL(n) = max(0, t_cross(wear, W_TH) - n), threshold W_TH = 165 um (crossed by
  all three labeled PHM2010 tools).
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from ..metrics import crossing_time

W_TH = 165.0
TEST_FRACTION = 0.2
K_FRACTIONS = (0.05, 0.10, 0.20)
B1_LABEL_FRACTION = 0.80

ROTATIONS = (
    {"target": "c1", "sources": ("c4", "c6")},
    {"target": "c4", "sources": ("c1", "c6")},
    {"target": "c6", "sources": ("c1", "c4")},
)


@dataclass(frozen=True)
class ToolData:
    tool: str
    features: np.ndarray  # (n, 12)
    wear: np.ndarray      # (n,)


def load_processed(processed_dir: str, tool: str) -> ToolData:
    path = os.path.join(processed_dir, f"{tool}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} missing — run scripts/r001_parse_features.py first")
    d = np.load(path)
    feats, wear = d["features"], d["wear"]
    if len(feats) != len(wear):
        raise ValueError(f"{tool}: feature/wear length mismatch")
    return ToolData(tool=tool, features=feats, wear=wear)


def label_indices(n_cuts: int, k_frac: float) -> np.ndarray:
    """Uniform-strided k% label budget over the first 80% of cuts.

    Uniform sampling (standard in the few-shot tool-wear literature) gives every
    system visibility across the observed wear range; a contiguous early-only
    budget makes the scarcity floor a pure-extrapolation task dominated by
    normalization artifacts (observed: B2@20% worse than B2@5%). The last-20%
    test window is never sampled.
    """
    if not 0.0 < k_frac <= B1_LABEL_FRACTION:
        raise ValueError(f"k_frac out of range: {k_frac}")
    pool_end = int(round(n_cuts * B1_LABEL_FRACTION))
    m = min(pool_end, max(10, int(round(n_cuts * k_frac))))
    return np.unique(np.round(np.linspace(0, pool_end - 1, m)).astype(int))


def test_indices(n_cuts: int) -> np.ndarray:
    """Last 20% of cuts — fixed evaluation window."""
    start = int(round(n_cuts * (1.0 - TEST_FRACTION)))
    return np.arange(start, n_cuts)


def rul_from_wear(wear: np.ndarray, w_th: float = W_TH) -> np.ndarray:
    """RUL in cuts against threshold crossing; 0 after crossing.

    Requires the (isotonized) path to cross w_th — PHM2010 labeled tools all do.
    """
    t_cross = crossing_time(np.maximum.accumulate(np.asarray(wear, dtype=float)), w_th)
    if not np.isfinite(t_cross):
        raise ValueError(f"Wear path never crosses threshold {w_th} um")
    n = np.arange(len(wear), dtype=float)
    return np.clip(t_cross - n, 0.0, None)


def rul_pred_from_wear_pred(wear_pred: np.ndarray, w_th: float = W_TH) -> np.ndarray:
    """Derive predicted RUL from a predicted wear sequence (isotonized first).

    If the predicted path never crosses, censor at sequence end (max RUL).
    """
    iso = np.maximum.accumulate(np.asarray(wear_pred, dtype=float))
    t_cross = crossing_time(iso, w_th)
    if not np.isfinite(t_cross):
        t_cross = float(len(iso))
    n = np.arange(len(iso), dtype=float)
    return np.clip(t_cross - n, 0.0, None)
