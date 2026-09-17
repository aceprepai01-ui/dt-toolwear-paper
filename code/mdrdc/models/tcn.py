"""Causal TCN wear regressor (S4) — shared backbone for ALL systems (fairness).

Trains on variable-length per-cut feature sequences; Huber loss on wear.
Deliberately standard: the backbone is a declared non-contribution.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


class TCN(nn.Module):
    def __init__(self, in_dim: int = 12, hidden: int = 48, levels: int = 5,
                 dropout: float = 0.1):
        super().__init__()
        blocks = []
        ch = in_dim
        for i in range(levels):
            d = 2 ** i
            blocks += [nn.Conv1d(ch, hidden, 3, padding=d, dilation=d),
                       nn.ReLU(), nn.Dropout(dropout)]
            ch = hidden
        self.net = nn.Sequential(*blocks)
        self.head = nn.Conv1d(hidden, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, T) -> (B, T)
        h = self.net(x)[..., : x.shape[-1]]
        return self.head(h)[:, 0, :]


@dataclass(frozen=True)
class Normalizer:
    mu: np.ndarray
    sd: np.ndarray

    @staticmethod
    def fit(feature_arrays) -> "Normalizer":
        stacked = np.concatenate([np.asarray(a, dtype=float) for a in feature_arrays], axis=0)
        return Normalizer(mu=stacked.mean(0), sd=stacked.std(0) + 1e-9)

    def apply(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=float) - self.mu) / self.sd


@dataclass(frozen=True)
class FittedTCN:
    """Model plus input/output normalization — one immutable inference unit."""

    model: TCN
    x_norm: Normalizer
    y_mu: float
    y_sd: float


def train_tcn(sequences, seed: int = 0, epochs: int = 300, lr: float = 2e-3,
              hidden: int = 48, device: str | None = None) -> FittedTCN:
    """sequences: list of (features (T_i, 12), wear (T_i,)) training pairs.

    Wear targets are z-normalized internally (unbounded raw range slows Huber
    convergence and lets scarce-data models extrapolate absurdly — W2 smoke bug).
    """
    if not sequences:
        raise ValueError("No training sequences")
    dev = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    x_norm = Normalizer.fit([f for f, _ in sequences])
    all_w = np.concatenate([np.asarray(w, dtype=float) for _, w in sequences])
    y_mu, y_sd = float(all_w.mean()), float(all_w.std() + 1e-9)
    tensors = [(torch.tensor(x_norm.apply(f).T[None], dtype=torch.float32, device=dev),
                torch.tensor(((np.asarray(w, dtype=float) - y_mu) / y_sd)[None],
                             dtype=torch.float32, device=dev))
               for f, w in sequences]
    torch.manual_seed(seed)
    model = TCN(in_dim=sequences[0][0].shape[1], hidden=hidden).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for _ in range(epochs):
        opt.zero_grad()
        loss = sum(nn.functional.huber_loss(model(x), y, delta=1.0) for x, y in tensors)
        (loss / len(tensors)).backward()
        opt.step()
        sched.step()
    model.eval()
    return FittedTCN(model=model, x_norm=x_norm, y_mu=y_mu, y_sd=y_sd)


def predict_tcn(fitted: FittedTCN, features: np.ndarray) -> np.ndarray:
    dev = next(fitted.model.parameters()).device
    x = torch.tensor(fitted.x_norm.apply(features).T[None], dtype=torch.float32, device=dev)
    with torch.no_grad():
        z = fitted.model(x)[0].cpu().numpy()
    return z * fitted.y_sd + fitted.y_mu
