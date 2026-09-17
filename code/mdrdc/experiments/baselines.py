"""Training-set assembly for W2 baselines (B1 / B2 / B3).

B1  source full + target first-80% labels          -> upper bound
B2  target k% labels ONLY                          -> scarcity floor
B3  source full + twin-synth target + k% labels    -> "is generation needed?"

The twin in B3 is calibrated on the SAME k% budget (no hidden labels);
its force features come from the source-fitted mechanistic+ridge force map.
"""
from __future__ import annotations

import numpy as np

from ..twin.calibrate import calibrate
from ..twin.force_model import fit_force_map
from .protocol import B1_LABEL_FRACTION, ToolData, label_indices


def _seq(td: ToolData, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return td.features[idx], td.wear[idx]


def build_b1(target: ToolData, sources: list) -> list:
    idx = label_indices(len(target.wear), B1_LABEL_FRACTION)
    return [(s.features, s.wear) for s in sources] + [_seq(target, idx)]


def build_b2(target: ToolData, k_frac: float) -> list:
    return [_seq(target, label_indices(len(target.wear), k_frac))]


def build_b3(target: ToolData, sources: list, k_frac: float,
             noise_seed: int = 0) -> list:
    idx = label_indices(len(target.wear), k_frac)
    labeled_wear = np.maximum.accumulate(target.wear[idx])
    cal = calibrate(labeled_wear, indices=idx, n_starts=6, seed=noise_seed)
    twin_path = cal.extrapolate(len(target.wear), vb0=float(labeled_wear[0]))

    src_wear = np.concatenate([np.maximum.accumulate(s.wear) for s in sources])
    src_feat = np.concatenate([s.features for s in sources], axis=0)
    fmap = fit_force_map(src_wear, src_feat)

    synth_feat = fmap.predict(twin_path)
    rng = np.random.default_rng(noise_seed)
    resid_scale = np.std(src_feat, axis=0) * 0.05  # small jitter, source-scaled
    synth_feat = synth_feat + rng.normal(0.0, 1.0, synth_feat.shape) * resid_scale[None, :]

    return ([(s.features, s.wear) for s in sources]
            + [(synth_feat, twin_path)]
            + [_seq(target, idx)])
