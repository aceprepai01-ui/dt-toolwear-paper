"""Training-sample bank for M-DRDC: calibration-variant augmentation.

Only 2-3 full source wear paths exist per rotation — far too few for a
generative model. Each variant re-calibrates the twin on a random uniform k%
label budget (5-25%) of a jittered copy of the real path, yielding a distinct
(u_twin, residual, cond) triple per variant. CRITICAL: the bank uses the SAME
calibration protocol as the target (uniform budget over the first 80%);
contiguous early-fraction calibration produced systematically larger twin errors
than deployment, so the DDPM learned oversized residuals that damaged
already-good target twins (W3 gate failure, 2026-08).

Cond vec (4-d): [misfit/10, w_start/100, twin_rise/100, labeled-tail-slope*10].
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..data.resample import (ENC_EPS, T_GRID, isotonize, path_to_u,
                             resample_path, softplus_inv, to_increments)
from ..metrics import rmse
from ..twin.calibrate import calibrate
from .protocol import ToolData, label_indices

JITTER_SD_UM = 1.5


@dataclass(frozen=True)
class BankSample:
    residual: np.ndarray  # (L,) u-space residual, L = T_GRID - 1
    u_twin: np.ndarray    # (L,)
    cond_vec: np.ndarray  # (4,): [misfit/10, w_start/100, twin_rise/100, tail_slope*10]
    tool: str


def encode_twin(twin_path: np.ndarray) -> np.ndarray:
    """Twin path (n_cuts,) -> u-space (T_GRID-1,). Same ENC_EPS floor as
    path_to_u — residuals only make sense with both paths in one encoding."""
    p = resample_path(twin_path, T_GRID)
    return softplus_inv(to_increments(p), eps=ENC_EPS)


def labeled_tail_slope(wear_labeled: np.ndarray, idx: np.ndarray) -> float:
    """Wear slope (um/cut) over the last half of the labeled window — the
    deployment-observable degradation-regime descriptor used for gating."""
    half = len(idx) // 2
    x = np.asarray(idx[half:], dtype=float)
    y = np.asarray(wear_labeled[half:], dtype=float)
    if len(x) < 3 or x[-1] == x[0]:
        return 0.0
    return float(np.polyfit(x, y, 1)[0])


def make_cond_vec(misfit_rmse: float, w_start: float, twin_rise: float,
                  tail_slope: float) -> np.ndarray:
    return np.array([misfit_rmse / 10.0, w_start / 100.0, twin_rise / 100.0,
                     tail_slope * 10.0], dtype=float)


def regime_of(slope_x10: float) -> int:
    """Tail-slope regime: 0 plateau (<4), 1 climb (4-8), 2 steep (>8)."""
    return 0 if slope_x10 < 4.0 else (1 if slope_x10 < 8.0 else 2)


def gate_and_weight(bank: list, target_cond: np.ndarray,
                    ess_min: float = 12.0, bandwidth: float = 1.0) -> tuple:
    """Retrieval localization (per external methods review, 2026-08-12):
    keep only same-regime bank samples (adjacent regime added if effective
    sample size is too small), and weight them by a Gaussian kernel over
    standardized [misfit, tail-slope]. Returns (subset, weights)."""
    tr = regime_of(target_cond[3])
    for widen in (0, 1):
        subset = [b for b in bank if abs(regime_of(b.cond_vec[3]) - tr) <= widen]
        if len(subset) < 4:
            continue
        feats = np.array([[b.cond_vec[0], b.cond_vec[3]] for b in subset])
        mu, sd = feats.mean(0), feats.std(0) + 1e-9
        tgt = (np.array([target_cond[0], target_cond[3]]) - mu) / sd
        d2 = np.sum(((feats - mu) / sd - tgt) ** 2, axis=1)
        w = np.exp(-d2 / (2 * bandwidth ** 2))
        if w.sum() <= 0:
            continue
        w = w / w.sum()
        ess = 1.0 / np.sum(w ** 2)
        if ess >= ess_min or widen == 1:
            return subset, w
    raise ValueError("Bank too small for regime gating — increase variants")


def build_bank(sources: list, n_variants: int = 16, seed: int = 0) -> list:
    """Bank of BankSamples from full-life SOURCE tools only."""
    rng = np.random.default_rng(seed)
    bank = []
    for td in sources:
        real = isotonize(td.wear)
        for v in range(n_variants):
            jittered = isotonize(real + rng.normal(0.0, JITTER_SD_UM, len(real)))
            k = float(rng.uniform(0.05, 0.25))
            idx = label_indices(len(jittered), k)
            cal = calibrate(jittered[idx], indices=idx, n_starts=4, seed=seed * 1000 + v)
            twin = cal.extrapolate(len(jittered), vb0=float(jittered[idx[0]]))
            u_twin = encode_twin(twin)
            _, u_real = path_to_u(jittered, T_GRID)
            misfit = rmse(jittered[idx], twin[idx])
            slope = labeled_tail_slope(jittered[idx], idx)
            bank.append(BankSample(
                residual=u_real - u_twin, u_twin=u_twin,
                cond_vec=make_cond_vec(misfit, float(jittered[idx[0]]),
                                       float(twin[-1] - twin[0]), slope),
                tool=td.tool))
    if len(bank) < 4:
        raise ValueError("Bank too small — increase n_variants or sources")
    return bank


def target_conditioning(target: ToolData, label_idx: np.ndarray,
                        seed: int = 0) -> tuple[np.ndarray, np.ndarray, float]:
    """Twin encoding + cond vec for the TARGET tool from its k% label budget only.

    Returns (u_twin, cond_vec, w_start). Target wear outside label_idx is never touched.
    """
    labeled = np.maximum.accumulate(target.wear[label_idx])
    cal = calibrate(labeled, indices=label_idx, n_starts=6, seed=seed)
    twin = cal.extrapolate(len(target.wear), vb0=float(labeled[0]))
    u_twin = encode_twin(twin)
    misfit = rmse(labeled, twin[label_idx])
    slope = labeled_tail_slope(labeled, label_idx)
    cond = make_cond_vec(misfit, float(labeled[0]), float(twin[-1] - twin[0]), slope)
    return u_twin, cond, float(labeled[0])
