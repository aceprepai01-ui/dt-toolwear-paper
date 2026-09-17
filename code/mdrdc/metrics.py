"""Evaluation metrics: regression, RUL scoring, UQ, and distributional (CRPS).

PHM scoring uses the standard asymmetric exponential convention (late RUL
predictions are penalized harder than early ones): a_early=13, a_late=10.
Verify against published PHM2010 baselines in R001 before trusting comparisons.
"""
from __future__ import annotations

import numpy as np


def _pair(y_true, y_pred):
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    if yt.shape != yp.shape:
        raise ValueError(f"Shape mismatch: {yt.shape} vs {yp.shape}")
    if not (np.all(np.isfinite(yt)) and np.all(np.isfinite(yp))):
        raise ValueError("Non-finite values in metric inputs")
    return yt, yp


def mae(y_true, y_pred) -> float:
    yt, yp = _pair(y_true, y_pred)
    return float(np.mean(np.abs(yt - yp)))


def rmse(y_true, y_pred) -> float:
    yt, yp = _pair(y_true, y_pred)
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def r2(y_true, y_pred) -> float:
    yt, yp = _pair(y_true, y_pred)
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - np.mean(yt)) ** 2)
    if ss_tot == 0:
        raise ValueError("Constant y_true: R^2 undefined")
    return float(1.0 - ss_res / ss_tot)


def phm_score(rul_true, rul_pred, a_early: float = 13.0, a_late: float = 10.0) -> float:
    """Asymmetric RUL score (lower is better). d>0 means late (dangerous)."""
    yt, yp = _pair(rul_true, rul_pred)
    d = yp - yt
    return float(np.sum(np.where(d < 0, np.exp(-d / a_early) - 1.0, np.exp(d / a_late) - 1.0)))


def picp(y_true, lo, hi) -> float:
    """Prediction-interval coverage probability."""
    yt, lo_ = _pair(y_true, lo)
    _, hi_ = _pair(y_true, hi)
    if np.any(hi_ < lo_):
        raise ValueError("Interval with hi < lo")
    return float(np.mean((yt >= lo_) & (yt <= hi_)))


def nmpiw(y_true, lo, hi) -> float:
    """Normalized mean prediction-interval width."""
    yt, lo_ = _pair(y_true, lo)
    _, hi_ = _pair(y_true, hi)
    rng = float(np.max(yt) - np.min(yt))
    if rng == 0:
        raise ValueError("Constant y_true: NMPIW undefined")
    return float(np.mean(hi_ - lo_) / rng)


def crps_empirical(samples: np.ndarray, y: float) -> float:
    """Sample-based CRPS: E|X - y| - 0.5 E|X - X'| (energy form).

    Used for threshold-crossing-time distributions from sampled wear paths.
    """
    s = np.asarray(samples, dtype=float)
    if s.ndim != 1 or len(s) < 2:
        raise ValueError("samples must be 1-D with >=2 entries")
    term1 = np.mean(np.abs(s - float(y)))
    term2 = 0.5 * np.mean(np.abs(s[:, None] - s[None, :]))
    return float(term1 - term2)


def crossing_time(path: np.ndarray, threshold: float) -> float:
    """First index (fractional, linear-interpolated) where path >= threshold.

    Returns +inf if the path never crosses — callers must handle censoring.
    """
    p = np.asarray(path, dtype=float)
    above = np.nonzero(p >= threshold)[0]
    if len(above) == 0:
        return float("inf")
    i = int(above[0])
    if i == 0:
        return 0.0
    frac = (threshold - p[i - 1]) / (p[i] - p[i - 1])
    return float(i - 1 + frac)
