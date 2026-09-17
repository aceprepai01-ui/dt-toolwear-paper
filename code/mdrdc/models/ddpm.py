"""M-DRDC core: conditional 1-D DDPM over wear-increment residuals (S2).

Object: r = u_real - u_twin in R^{63} (u = softplus-inverse increment encoding on
the T'=64 grid, ENC_EPS floor). Decoding w = w0 + cumsum(softplus(u_twin + r)) is
monotone for ANY r — degradation consistency is structural, not penalized.

Standard pieces (declared non-contributions): cosine noise schedule, eps-
prediction loss, ancestral sampling, classifier-free guidance, EMA weights.
~0.19M params. Per-position normalization: residuals are strongly heteroscedastic.

Archived verdict (W3/W5 2026-08): this corrector never beat its own twin prior on
the crossing gate nor added downstream value — kept as the controlled negative
result and as baselines B5/B5+/A1 infrastructure.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

T_STEPS = 200
CFG_DROP_P = 0.15
EMA_DECAY = 0.998


def cosine_alpha_bar(t_steps: int = T_STEPS, s: float = 0.008) -> torch.Tensor:
    t = torch.linspace(0, t_steps, t_steps + 1) / t_steps
    f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    ab = (f / f[0]).clamp(1e-5, 1.0)
    return ab[1:]  # (T,)


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 2), nn.SiLU(), nn.Linear(dim * 2, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
        ang = t.float()[:, None] * freqs[None, :]
        return self.mlp(torch.cat([ang.sin(), ang.cos()], dim=-1))


class FiLMConvBlock(nn.Module):
    def __init__(self, ch: int, dilation: int, emb_dim: int):
        super().__init__()
        self.conv = nn.Conv1d(ch, ch, 3, padding=dilation, dilation=dilation)
        self.norm = nn.GroupNorm(8, ch)
        self.film = nn.Linear(emb_dim, ch * 2)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.norm(self.conv(x))
        scale, shift = self.film(emb)[:, :, None].chunk(2, dim=1)
        return x + nn.functional.silu(h * (1 + scale) + shift)


class ResidualDenoiser(nn.Module):
    """eps_theta(r_t, t, cond): input channels = [r_t, u_twin]; FiLM = t + cond_vec."""

    def __init__(self, hidden: int = 64, emb_dim: int = 64, cond_dim: int = 3):
        super().__init__()
        self.time_emb = TimeEmbedding(emb_dim)
        self.cond_mlp = nn.Sequential(nn.Linear(cond_dim, emb_dim), nn.SiLU(),
                                      nn.Linear(emb_dim, emb_dim))
        self.null_cond = nn.Parameter(torch.zeros(emb_dim))  # CFG null embedding
        self.inp = nn.Conv1d(2, hidden, 3, padding=1)
        self.blocks = nn.ModuleList([FiLMConvBlock(hidden, d, emb_dim)
                                     for d in (1, 2, 4, 8, 8, 4, 2, 1)])
        self.out = nn.Conv1d(hidden, 1, 3, padding=1)

    def forward(self, r_t: torch.Tensor, t: torch.Tensor, u_twin: torch.Tensor,
                cond_vec: torch.Tensor | None) -> torch.Tensor:
        emb = self.time_emb(t)
        if cond_vec is None:  # unconditional branch for CFG
            emb = emb + self.null_cond[None, :]
        else:
            emb = emb + self.cond_mlp(cond_vec)
        h = self.inp(torch.stack([r_t, u_twin], dim=1))
        for blk in self.blocks:
            h = blk(h, emb)
        return self.out(h)[:, 0, :]


@dataclass(frozen=True)
class FittedDDPM:
    model: ResidualDenoiser
    alpha_bar: torch.Tensor
    r_mu: np.ndarray  # (L,) per-position — residuals are strongly heteroscedastic
    r_sd: np.ndarray  # (L,)


def train_ddpm(residuals: np.ndarray, u_twins: np.ndarray, cond_vecs: np.ndarray,
               epochs: int = 4000, lr: float = 2e-4, seed: int = 0,
               device: str | None = None, log_every: int = 1000,
               sample_weights: np.ndarray | None = None,
               cfg_drop_p: float = CFG_DROP_P) -> FittedDDPM:
    """residuals/u_twins: (N, L); cond_vecs: (N, c). Full-batch Adam (N is small).

    sample_weights: retrieval-localization kernel weights (normalized inside).
    cfg_drop_p=0 disables classifier-free-guidance dropout (use with guidance 0).
    EMA weights (decay 0.998) are loaded into the returned model."""
    if residuals.shape != u_twins.shape or len(residuals) != len(cond_vecs):
        raise ValueError("Shape mismatch between residuals, u_twins, cond_vecs")
    if len(residuals) < 4:
        raise ValueError("Need >=4 training sequences (use the calibration-variant bank)")
    dev = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    r_mu = residuals.mean(axis=0)
    r_sd = residuals.std(axis=0) + 1e-6
    r0 = torch.tensor((residuals - r_mu[None, :]) / r_sd[None, :],
                      dtype=torch.float32, device=dev)
    ut = torch.tensor((u_twins - u_twins.mean()) / (u_twins.std() + 1e-9),
                      dtype=torch.float32, device=dev)
    cv = torch.tensor(cond_vecs, dtype=torch.float32, device=dev)

    torch.manual_seed(seed)
    model = ResidualDenoiser(cond_dim=cond_vecs.shape[1]).to(dev)
    ab = cosine_alpha_bar().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()}
    n = len(r0)
    if sample_weights is None:
        sw = torch.ones(n, device=dev) / n
    else:
        w = np.asarray(sample_weights, dtype=float)
        if w.shape != (n,) or w.sum() <= 0:
            raise ValueError("sample_weights must be (N,) with positive sum")
        sw = torch.tensor(w / w.sum(), dtype=torch.float32, device=dev)
    for ep in range(epochs):
        t = torch.randint(0, T_STEPS, (n,), device=dev)
        eps = torch.randn_like(r0)
        a = ab[t][:, None]
        r_t = a.sqrt() * r0 + (1 - a).sqrt() * eps
        drop = torch.rand(n, device=dev) < cfg_drop_p
        loss = 0.0
        if (~drop).any():
            pred_c = model(r_t[~drop], t[~drop], ut[~drop], cv[~drop])
            per = ((pred_c - eps[~drop]) ** 2).mean(dim=1)
            loss = loss + (per * sw[~drop]).sum() / sw[~drop].sum().clamp_min(1e-9)
        if drop.any():
            pred_u = model(r_t[drop], t[drop], ut[drop], None)
            per = ((pred_u - eps[drop]) ** 2).mean(dim=1)
            loss = loss + (per * sw[drop]).sum() / sw[drop].sum().clamp_min(1e-9)
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if v.dtype.is_floating_point:
                    ema[k].mul_(EMA_DECAY).add_(v, alpha=1 - EMA_DECAY)
                else:
                    ema[k] = v.detach().clone()
        if log_every and (ep + 1) % log_every == 0:
            print(f"    ddpm epoch {ep + 1}/{epochs}: loss={float(loss):.4f}", flush=True)
    model.load_state_dict(ema)
    model.eval()
    return FittedDDPM(model=model, alpha_bar=ab, r_mu=r_mu, r_sd=r_sd)


@torch.no_grad()
def sample_residuals(fitted: FittedDDPM, u_twin: np.ndarray, cond_vec: np.ndarray,
                     n_samples: int = 50, guidance: float = 2.0,
                     seed: int = 0) -> np.ndarray:
    """Ancestral DDPM sampling with CFG; returns (n_samples, L) residuals (denormalized)."""
    dev = fitted.alpha_bar.device
    L = len(u_twin)
    torch.manual_seed(seed)
    ut = torch.tensor((u_twin - u_twin.mean()) / (u_twin.std() + 1e-9),
                      dtype=torch.float32, device=dev)[None, :].repeat(n_samples, 1)
    cv = torch.tensor(cond_vec, dtype=torch.float32, device=dev)[None, :].repeat(n_samples, 1)
    ab = fitted.alpha_bar
    alphas = ab / torch.cat([torch.ones(1, device=dev), ab[:-1]])
    x = torch.randn(n_samples, L, device=dev)
    for ti in reversed(range(T_STEPS)):
        t = torch.full((n_samples,), ti, device=dev, dtype=torch.long)
        eps_c = fitted.model(x, t, ut, cv)
        if guidance == 0.0:
            eps = eps_c  # null branch untrained when cfg_drop_p=0; never touch it
        else:
            eps_u = fitted.model(x, t, ut, None)
            eps = (1 + guidance) * eps_c - guidance * eps_u
        a_t, ab_t = alphas[ti], ab[ti]
        x = (x - (1 - a_t) / (1 - ab_t).sqrt() * eps) / a_t.sqrt()
        if ti > 0:
            sigma = ((1 - ab[ti - 1]) / (1 - ab_t) * (1 - a_t)).sqrt()
            x = x + sigma * torch.randn_like(x)
    return (x.cpu().numpy() * fitted.r_sd[None, :]) + fitted.r_mu[None, :]
