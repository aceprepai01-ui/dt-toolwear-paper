"""Wear-path resampling onto the unified T'=64 increment grid (M-DRDC input format).

Uses PCHIP (monotone cubic) interpolation so resampling never introduces
non-monotone artifacts into an isotonically-preprocessed wear path.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import PchipInterpolator

T_GRID = 64
_EPS = 1e-6
# Increment floor for PATH ENCODING (micrometers/grid-step). Real wear paths
# contain measurement plateaus whose ~0 increments explode under softplus-inv
# (u -> -13.8), dominating residual statistics (W3 diagnosis, 2026-08). A 0.05 um
# floor bounds u >= ~-3.0 at a worst-case round-trip cost of ~3 um over a full
# life — well under measurement noise.
ENC_EPS = 0.05


def isotonize(wear: np.ndarray) -> np.ndarray:
    """Return the running-max envelope (measurement noise can make raw VB dip;
    physical wear cannot decrease). Pure function — input is not modified."""
    w = np.asarray(wear, dtype=float)
    if w.ndim != 1 or len(w) < 2:
        raise ValueError("wear must be a 1-D array with >=2 points")
    return np.maximum.accumulate(w)


def resample_path(wear: np.ndarray, t_grid: int = T_GRID) -> np.ndarray:
    """Resample a monotone wear path to `t_grid` points on normalized life [0, 1]."""
    w = isotonize(wear)
    x = np.linspace(0.0, 1.0, len(w))
    xi = np.linspace(0.0, 1.0, t_grid)
    return PchipInterpolator(x, w)(xi)


def to_increments(path: np.ndarray) -> np.ndarray:
    """Monotone path (T,) -> non-negative increments (T-1,)."""
    d = np.diff(np.asarray(path, dtype=float))
    if np.any(d < -1e-9):
        raise ValueError("Path is not monotone; call isotonize/resample_path first")
    return np.clip(d, 0.0, None)


def softplus(x: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, x)


def softplus_inv(y: np.ndarray, eps: float = _EPS) -> np.ndarray:
    """Inverse softplus with an epsilon floor for plateau (zero-increment) regions."""
    y = np.clip(np.asarray(y, dtype=float), eps, None)
    return y + np.log(-np.expm1(-y))


def path_to_u(wear: np.ndarray, t_grid: int = T_GRID) -> tuple[float, np.ndarray]:
    """Full encoding: raw wear path -> (w_start, u) with u = softplus_inv(increments).

    Uses the ENC_EPS increment floor — see its comment for why.
    """
    p = resample_path(wear, t_grid)
    return float(p[0]), softplus_inv(to_increments(p), eps=ENC_EPS)


def u_to_path(w_start: float, u: np.ndarray) -> np.ndarray:
    """Decode: (w_start, u) -> monotone wear path of length len(u)+1."""
    return w_start + np.concatenate([[0.0], np.cumsum(softplus(np.asarray(u, dtype=float)))])


def upsample_path(path_grid: np.ndarray, n_cuts: int) -> np.ndarray:
    """Grid-resolution wear path (T_GRID,) -> per-cut path (n_cuts,), monotone-preserving."""
    p = np.asarray(path_grid, dtype=float)
    if p.ndim != 1 or len(p) < 2:
        raise ValueError("path_grid must be 1-D with >=2 points")
    x = np.linspace(0.0, 1.0, len(p))
    return PchipInterpolator(x, p)(np.linspace(0.0, 1.0, n_cuts))
