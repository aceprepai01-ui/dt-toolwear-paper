"""Mechanistic tool-wear twin: calibratable wear-rate ODE (S1).

    dVB/dn = g(p) * [ beta_brk * exp(-VB / vb_brk) + 1 + beta_acc * (VB / vb_acc)^2 ]
    g(p)   = k1 * (Vc/Vc0)^a * (fz/fz0)^b

Three-phase shape: break-in (exp term), steady (constant), accelerating (quadratic).
Parameters are fitted in log-space for positivity. NOTE: PHM2010 runs a single
cutting condition, so (a, b) are unidentifiable there and fold into k1 — they are
exposed for the multi-condition source pool and held at prior mean otherwise.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

# Reference condition (PHM2010 nominal): Vc-equivalent spindle 10400 rpm, feed 1555 mm/min
VC0 = 10400.0
FZ0 = 1555.0

PARAM_NAMES = ("log_k1", "a", "b", "log_beta_brk", "log_vb_brk", "log_beta_acc", "log_vb_acc")


@dataclass(frozen=True)
class TwinParams:
    """Immutable twin parameters (log-space where positivity is required)."""

    log_k1: float = 0.0
    a: float = 1.0
    b: float = 0.5
    log_beta_brk: float = 1.0
    log_vb_brk: float = np.log(20.0)   # micrometers
    log_beta_acc: float = 0.0
    log_vb_acc: float = np.log(150.0)  # micrometers

    def to_vector(self) -> np.ndarray:
        return np.array([getattr(self, n) for n in PARAM_NAMES], dtype=float)

    @staticmethod
    def from_vector(v: np.ndarray) -> "TwinParams":
        v = np.asarray(v, dtype=float)
        if v.shape != (len(PARAM_NAMES),):
            raise ValueError(f"Expected {len(PARAM_NAMES)} params, got {v.shape}")
        return TwinParams(**dict(zip(PARAM_NAMES, v.tolist())))

    def with_updates(self, **kwargs) -> "TwinParams":
        return replace(self, **kwargs)


def wear_rate(vb: np.ndarray, params: TwinParams, vc: float = VC0, fz: float = FZ0) -> np.ndarray:
    """Per-cut wear rate (micrometers/cut). Vectorized over vb."""
    if vc <= 0 or fz <= 0:
        raise ValueError("Cutting parameters must be positive")
    vb = np.clip(np.asarray(vb, dtype=float), 0.0, None)
    k1 = np.exp(params.log_k1)
    g = k1 * (vc / VC0) ** params.a * (fz / FZ0) ** params.b
    brk = np.exp(params.log_beta_brk) * np.exp(-vb / np.exp(params.log_vb_brk))
    acc = np.exp(params.log_beta_acc) * (vb / np.exp(params.log_vb_acc)) ** 2
    return g * (brk + 1.0 + acc)


def simulate(params: TwinParams, n_cuts: int, vb0: float = 0.0,
             vc: float = VC0, fz: float = FZ0) -> np.ndarray:
    """Explicit-Euler integration over cut index; returns (n_cuts,) wear path.

    Monotone by construction (rate is non-negative). Rate is capped to keep the
    accelerating phase numerically stable far outside the calibrated regime.
    """
    if n_cuts < 2:
        raise ValueError("n_cuts must be >= 2")
    path = np.empty(n_cuts, dtype=float)
    vb = float(vb0)
    for i in range(n_cuts):
        path[i] = vb
        rate = float(wear_rate(np.array([vb]), params, vc, fz)[0])
        vb = vb + min(rate, 50.0)  # cap: 50 um/cut is far beyond physical rates
    return path
