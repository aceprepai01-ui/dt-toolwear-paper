"""Twin force map (S3): mechanistic wear->force-feature map + ridge residual.

Mechanistic part: force features scale linearly with wear via K(VB) = K0*(1+kappa*VB)
(Altintas linear-edge-force reasoning). The ridge part is a closed-form correction
on explicit basis functions — deliberately low-capacity (7 basis coefficients per
output feature, 84 total across 12 features; the vc/fz columns degenerate under a
single cutting condition) so the paper's gains cannot be attributed to it (K4).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _basis(vb: np.ndarray, vc: np.ndarray, fz: np.ndarray) -> np.ndarray:
    """Design matrix: [1, vb, vb^2, vb*vc, vb*fz, vc, fz] (normalized inputs)."""
    cols = [np.ones_like(vb), vb, vb ** 2, vb * vc, vb * fz, vc, fz]
    return np.stack(cols, axis=1)


@dataclass(frozen=True)
class ForceMap:
    """f(vb, p) = base * (1 + kappa*vb) + Basis(vb, p) @ W_ridge   (per feature)."""

    base: np.ndarray     # (n_features,)
    kappa: np.ndarray    # (n_features,)
    ridge_w: np.ndarray  # (n_basis, n_features)
    vb_scale: float
    lam: float

    def predict(self, vb: np.ndarray, vc: float = 1.0, fz: float = 1.0) -> np.ndarray:
        vbn = np.asarray(vb, dtype=float)[:, None] / self.vb_scale
        mech = self.base[None, :] * (1.0 + self.kappa[None, :] * vbn)
        vcv = np.full(len(vbn), float(vc))
        fzv = np.full(len(vbn), float(fz))
        return mech + _basis(vbn[:, 0], vcv, fzv) @ self.ridge_w


def fit_force_map(vb: np.ndarray, features: np.ndarray,
                  vc: np.ndarray | None = None, fz: np.ndarray | None = None,
                  lam: float = 1e-2) -> ForceMap:
    """Fit on SOURCE data only. Two stages, both closed-form:
    1) per-feature least squares for (base, kappa) of the mechanistic line;
    2) ridge regression on the mechanistic residual (lam guards the exact
       collinearity of constant vc/fz columns under a single condition).
    """
    vb = np.asarray(vb, dtype=float)
    X = np.asarray(features, dtype=float)
    if vb.ndim != 1 or X.ndim != 2 or len(vb) != len(X):
        raise ValueError("vb must be (n,), features (n, d), matching lengths")
    if len(vb) < 10:
        raise ValueError("Need >=10 samples to fit the force map")
    vb_scale = float(np.max(vb)) or 1.0
    vbn = vb / vb_scale
    vcv = np.ones_like(vbn) if vc is None else np.asarray(vc, dtype=float)
    fzv = np.ones_like(vbn) if fz is None else np.asarray(fz, dtype=float)

    A = np.stack([np.ones_like(vbn), vbn], axis=1)          # (n, 2)
    coef, *_ = np.linalg.lstsq(A, X, rcond=None)            # (2, d)
    base = coef[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        kappa = np.where(np.abs(base) > 1e-12, coef[1] / base, 0.0)
    mech = base[None, :] * (1.0 + kappa[None, :] * vbn[:, None])

    B = _basis(vbn, vcv, fzv)                               # (n, k)
    R = X - mech
    ridge_w = np.linalg.solve(B.T @ B + lam * np.eye(B.shape[1]), B.T @ R)
    return ForceMap(base=base, kappa=kappa, ridge_w=ridge_w, vb_scale=vb_scale, lam=lam)
