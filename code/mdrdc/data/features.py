"""Per-cut force feature extraction (force-only, per FINAL_PROPOSAL main line).

For each force channel (Fx, Fy, Fz): RMS, absolute peak, mean(|x|), std.
Identical schema is used for real and twin-synthesized features to avoid
domain-cue leakage into the predictor.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FORCE_CHANNELS = ("fx", "fy", "fz")
STATS = ("rms", "peak", "absmean", "std")

FEATURE_NAMES = tuple(f"{ch}_{st}" for ch in FORCE_CHANNELS for st in STATS)


def extract_cut_features(signals: pd.DataFrame) -> np.ndarray:
    """Return the 12-dim force feature vector for one cut. Pure function."""
    missing = [c for c in FORCE_CHANNELS if c not in signals.columns]
    if missing:
        raise ValueError(f"Missing force channels: {missing}")
    feats = []
    for ch in FORCE_CHANNELS:
        x = signals[ch].to_numpy(dtype=float)
        if not np.all(np.isfinite(x)):
            x = x[np.isfinite(x)]
            if len(x) == 0:
                raise ValueError(f"Channel {ch} contains no finite samples")
        feats.extend([
            float(np.sqrt(np.mean(x ** 2))),
            float(np.max(np.abs(x))),
            float(np.mean(np.abs(x))),
            float(np.std(x)),
        ])
    return np.asarray(feats, dtype=float)


def feature_matrix(per_cut_frames) -> np.ndarray:
    """Stack per-cut feature vectors into an (n_cuts, 12) matrix."""
    rows = [extract_cut_features(df) for df in per_cut_frames]
    if not rows:
        raise ValueError("No cuts provided")
    return np.stack(rows, axis=0)
