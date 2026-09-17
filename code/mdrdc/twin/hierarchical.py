"""Empirical-Bayes pooled twin (v3 main method).

Fit the wear ODE independently on each full-life SOURCE tool, form an empirical
hyperprior (per-parameter mean/sd of the source MAP estimates), then MAP-calibrate
the TARGET under that hyperprior from its k% label budget.

Positioning (per novelty cross-verification 2026-08-13): hierarchical/EB pooling
is a borrowed tool (arXiv 2601.15942; J Manuf Process 2026-06 drilling); the
claimed delta is the milling wear-trajectory combination and the benchmark
evidence, not the Bayesian machinery.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..metrics import rmse
from .calibrate import PRIOR_TAU, CalibrationResult, calibrate

MIN_TAU_FLOOR = 0.05  # never let an empirical sd collapse to ~0 (2-3 source tools)


@dataclass(frozen=True)
class Hyperprior:
    mu: np.ndarray            # (P,) empirical mean of source MAP fits
    tau: np.ndarray           # (P,) empirical sd (floored, scalable)
    source_fits: tuple        # per-source CalibrationResult, for LOO/audit
    source_names: tuple

    def scaled(self, factor: float) -> "Hyperprior":
        """Prior-scale robustness (H4): widen/narrow tau by `factor`."""
        if factor <= 0:
            raise ValueError("scale factor must be positive")
        return Hyperprior(mu=self.mu, tau=self.tau * factor,
                          source_fits=self.source_fits, source_names=self.source_names)


def fit_hyperprior(source_wears: dict, n_starts: int = 8, seed: int = 0,
                   tau_shrink: float = 1.0) -> Hyperprior:
    """source_wears: {name: full-life isotonic wear array}. Each source is
    MAP-fitted under the wide handbook prior; the hyperprior pools those fits."""
    if len(source_wears) < 2:
        raise ValueError("Need >=2 source tools to pool")
    fits, names = [], []
    for name, wear in source_wears.items():
        w = np.asarray(wear, dtype=float)
        if len(w) < 20 or np.any(np.diff(w) < -1e-9):
            raise ValueError(f"{name}: need a long isotonic full-life path")
        fits.append(calibrate(w, n_starts=n_starts, seed=seed))
        names.append(name)
    thetas = np.stack([f.params.to_vector() for f in fits])
    mu = thetas.mean(axis=0)
    # Empirical sd with a floor: 2-3 sources under-disperse; blend toward the
    # handbook tau so the prior stays honest about population spread.
    sd = np.maximum(thetas.std(axis=0, ddof=1) * tau_shrink, MIN_TAU_FLOOR)
    tau = np.minimum(sd, PRIOR_TAU)  # never wider than the handbook prior
    return Hyperprior(mu=mu, tau=tau, source_fits=tuple(fits), source_names=tuple(names))


def calibrate_pooled(target_wear: np.ndarray, indices: np.ndarray,
                     hyper: Hyperprior, n_starts: int = 8,
                     seed: int = 0) -> CalibrationResult:
    """MAP calibration of the target under the pooled empirical hyperprior."""
    return calibrate(target_wear, indices=indices, n_starts=n_starts, seed=seed,
                     prior_mu=hyper.mu, prior_tau=hyper.tau)


def misfit_flag(result: CalibrationResult, target_wear: np.ndarray,
                indices: np.ndarray, threshold_um: float = 8.0) -> tuple[bool, float]:
    """Out-of-population detector (H5): labeled-window RMSE of the pooled fit.

    NOTE (archived W6/W6b finding): this flag does NOT separate c4-type regime
    changes — kept as the honest negative result, not as a working detector.
    Returns (flagged, rmse_um)."""
    idx = np.asarray(indices, dtype=int)
    sim = result.extrapolate(int(idx[-1]) + 1, vb0=float(target_wear[0]))
    m = rmse(np.asarray(target_wear, dtype=float), sim[idx])
    return bool(m > threshold_um), float(m)
