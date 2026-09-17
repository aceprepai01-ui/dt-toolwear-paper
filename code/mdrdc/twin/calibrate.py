"""MAP calibration of the wear-ODE twin (S1) with labeled wear data.

theta* = argmin  sum((VB_sim - VB_meas)^2) / sigma^2  +  sum(((theta - mu) / tau)^2)

Gaussian priors in (log-)parameter space encode handbook ranges; multi-start
L-BFGS-B guards against local minima. Gate (R002): early-15% fit R^2 >= 0.8.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from ..metrics import r2
from .wear_ode import PARAM_NAMES, TwinParams, simulate

# Prior means/stds in parameter space (log-space where applicable).
PRIOR_MU = TwinParams().to_vector()
PRIOR_TAU = np.array([2.0, 0.5, 0.5, 1.5, 1.0, 1.5, 0.7])
SIGMA_MEAS = 5.0  # micrometers, microscope measurement noise scale
# PHM2010 is single-condition: freeze (a, b) at prior mean there (unidentifiable).
FIT_MASK_SINGLE_COND = np.array([True, False, False, True, True, True, True])


@dataclass(frozen=True)
class CalibrationResult:
    params: TwinParams
    r2_fit: float          # on the calibration window
    n_calib: int
    loss: float

    def extrapolate(self, n_cuts: int, vb0: float = 0.0, vc=None, fz=None) -> np.ndarray:
        kwargs = {}
        if vc is not None:
            kwargs["vc"] = vc
        if fz is not None:
            kwargs["fz"] = fz
        return simulate(self.params, n_cuts, vb0=vb0, **kwargs)


def _objective(free: np.ndarray, wear: np.ndarray, indices: np.ndarray,
               mask: np.ndarray, prior_mu: np.ndarray, prior_tau: np.ndarray) -> float:
    theta = prior_mu.copy()
    theta[mask] = free
    params = TwinParams.from_vector(theta)
    sim = simulate(params, int(indices[-1]) + 1, vb0=float(wear[0]))
    data_term = float(np.sum((sim[indices] - wear) ** 2)) / SIGMA_MEAS ** 2
    prior_term = float(np.sum(((theta - prior_mu) / prior_tau) ** 2))
    return data_term + prior_term


def calibrate(wear_meas: np.ndarray, indices: np.ndarray | None = None,
              n_starts: int = 8, seed: int = 0, single_condition: bool = True,
              prior_mu: np.ndarray | None = None,
              prior_tau: np.ndarray | None = None) -> CalibrationResult:
    """MAP-fit twin parameters on labeled wear measurements (micrometers).

    indices: cut indices of the measurements (default: contiguous 0..n-1).
    Supports uniform-strided label budgets — the ODE is simulated over the full
    horizon and matched at the labeled cuts.
    prior_mu/prior_tau: override the handbook prior (used by the empirical-Bayes
    pooled twin, mdrdc.twin.hierarchical). Defaults preserve legacy behavior.
    """
    wear = np.asarray(wear_meas, dtype=float)
    if wear.ndim != 1 or len(wear) < 10:
        raise ValueError("Need a 1-D wear array with >=10 measurements")
    if not np.all(np.isfinite(wear)):
        raise ValueError("Non-finite wear measurements")
    idx = np.arange(len(wear)) if indices is None else np.asarray(indices, dtype=int)
    if idx.shape != wear.shape or idx[0] != 0 or np.any(np.diff(idx) <= 0):
        raise ValueError("indices must be strictly increasing, start at 0, match wear length")
    mask = FIT_MASK_SINGLE_COND if single_condition else np.ones(len(PARAM_NAMES), bool)
    pmu = PRIOR_MU if prior_mu is None else np.asarray(prior_mu, dtype=float)
    ptau = PRIOR_TAU if prior_tau is None else np.asarray(prior_tau, dtype=float)
    if pmu.shape != PRIOR_MU.shape or ptau.shape != PRIOR_TAU.shape or np.any(ptau <= 0):
        raise ValueError("prior_mu/prior_tau must match parameter dimension with tau > 0")

    rng = np.random.default_rng(seed)
    best = None
    for i in range(n_starts):
        x0 = pmu[mask] + (0.0 if i == 0 else rng.normal(0, 0.5, mask.sum()) * ptau[mask])
        res = minimize(_objective, x0, args=(wear, idx, mask, pmu, ptau), method="L-BFGS-B")
        if best is None or res.fun < best.fun:
            best = res
    theta = pmu.copy()
    theta[mask] = best.x
    params = TwinParams.from_vector(theta)
    sim = simulate(params, int(idx[-1]) + 1, vb0=float(wear[0]))
    return CalibrationResult(params=params, r2_fit=r2(wear, sim[idx]),
                             n_calib=len(wear), loss=float(best.fun))


def calibrate_early_fraction(wear_full: np.ndarray, fraction: float = 0.15,
                             **kwargs) -> CalibrationResult:
    """Convenience: calibrate on the first `fraction` of a full-life path (contiguous)."""
    if not 0.02 < fraction <= 1.0:
        raise ValueError(f"fraction out of range: {fraction}")
    n = max(10, int(round(len(wear_full) * fraction)))
    return calibrate(np.asarray(wear_full, dtype=float)[:n], **kwargs)
