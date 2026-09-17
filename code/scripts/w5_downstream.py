#!/usr/bin/env python3
"""W5: DOWNSTREAM ARBITRATION — the paper's actual K1 claim.

Does training the wear/RUL TCN on M-DRDC-corrected synthetic paths beat
(B2) few-shot-only and (B3) twin-only augmentation at k in {5,10}%?

Verdict rule (pre-registered): OURS beats B3 AND B2 on mean wear MAE at k=5%
and k=10% with non-overlapping +/-1 SEM -> K1 alive; else contingency.
ARCHIVED VERDICT (2026-08-12): NO SEPARATION at both k — K1 rejected; the
paper's headline moved to the EB pooled twin (see FINAL_PROPOSAL_v3).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mdrdc.data.qit import load_qit_tool                                      # noqa: E402
from mdrdc.data.resample import isotonize, u_to_path, upsample_path           # noqa: E402
from mdrdc.experiments import baselines as bl                                 # noqa: E402
from mdrdc.experiments.protocol import (ROTATIONS, label_indices,             # noqa: E402
                                        load_processed, rul_from_wear,
                                        rul_pred_from_wear_pred, test_indices)
from mdrdc.experiments.residual_bank import (build_bank, gate_and_weight,     # noqa: E402
                                             target_conditioning)
from mdrdc.metrics import crossing_time, mae, phm_score, rmse                 # noqa: E402
from mdrdc.models.ddpm import sample_residuals, train_ddpm                    # noqa: E402
from mdrdc.models.tcn import predict_tcn, train_tcn                           # noqa: E402
from mdrdc.twin.force_model import fit_force_map                              # noqa: E402

AUG_PATHS = 10
JITTER_FRAC = 0.05


def synth_sequences(paths_grid: np.ndarray, n_cuts: int, sources: list,
                    seed: int) -> list:
    """Corrected grid paths -> (features, wear) training sequences via the
    source-fitted force map — same recipe and jitter as B3 for fairness."""
    src_wear = np.concatenate([isotonize(s.wear) for s in sources])
    src_feat = np.concatenate([s.features for s in sources], axis=0)
    fmap = fit_force_map(src_wear, src_feat)
    rng = np.random.default_rng(seed)
    resid_scale = np.std(src_feat, axis=0) * JITTER_FRAC
    out = []
    for p in paths_grid:
        wear_cuts = upsample_path(p, n_cuts)
        feats = fmap.predict(wear_cuts)
        feats = feats + rng.normal(0.0, 1.0, feats.shape) * resid_scale[None, :]
        out.append((feats, wear_cuts))
    return out


def build_ours(target, sources, bank_sources, k_frac: float, seed: int,
               args) -> tuple[list, dict]:
    n_cuts = len(target.wear)
    idx = label_indices(n_cuts, k_frac)
    u_twin, cond_vec, w_start = target_conditioning(target, idx, seed=seed)
    bank = build_bank(bank_sources, n_variants=args.variants, seed=seed)
    subset, weights = gate_and_weight(bank, cond_vec)
    two = lambda c: np.array([c[0], c[3]])
    fitted = train_ddpm(
        residuals=np.stack([b.residual for b in subset]),
        u_twins=np.stack([b.u_twin for b in subset]),
        cond_vecs=np.stack([two(b.cond_vec) for b in subset]),
        sample_weights=weights, cfg_drop_p=0.0,
        epochs=args.ddpm_epochs, seed=seed, log_every=0)
    res = sample_residuals(fitted, u_twin, two(cond_vec), n_samples=50,
                           guidance=0.0, seed=seed)
    paths = np.stack([u_to_path(w_start, u_twin + r) for r in res])
    twin64 = u_to_path(w_start, u_twin)

    t_twin = crossing_time(twin64, 165.0)
    t_twin = t_twin if np.isfinite(t_twin) else float(len(twin64))
    t_samp = np.array([crossing_time(p, 165.0) for p in paths])
    t_samp = np.where(np.isfinite(t_samp), t_samp, float(len(twin64)))
    shift = t_samp - t_twin
    iqr = float(np.percentile(shift, 75) - np.percentile(shift, 25))
    applied = abs(float(np.median(shift))) >= max(1.0, 1.5 * iqr)  # grid units: 5 cuts ~= 1

    # Physical-plausibility filter: an undertrained/mis-scaled sampler can emit
    # runaway paths (observed: poisoned TCN normalization, MAE ~1e4 um). Cap at
    # 2x the max wear ever observed on the source tools; need >=3 survivors.
    path_cap = 2.0 * max(float(isotonize(s.wear).max()) for s in sources)
    valid = paths[(paths[:, -1] <= path_cap) & np.all(np.isfinite(paths), axis=1)]
    if applied and len(valid) < 3:
        applied = False
    aug_grid = valid[:AUG_PATHS] if applied else np.stack([twin64])
    seqs = ([(s.features, s.wear) for s in sources]
            + synth_sequences(aug_grid, n_cuts, sources, seed)
            + [(target.features[idx], target.wear[idx])])
    return seqs, {"correction_applied": bool(applied)}


def run_cell(system: str, k: float, rot: dict, data: dict, qit, seed: int,
             args) -> dict:
    target = data[rot["target"]]
    sources = [data[s] for s in rot["sources"]]
    extra = {}
    if system == "B2":
        seqs = bl.build_b2(target, k)
    elif system == "B3":
        seqs = bl.build_b3(target, sources, k, noise_seed=seed)
    elif system == "OURS":
        bank_sources = sources + ([qit] if qit is not None else [])
        seqs, extra = build_ours(target, sources, bank_sources, k, seed, args)
    else:
        raise ValueError(system)
    fitted = train_tcn(seqs, seed=seed, epochs=args.tcn_epochs)
    wear_pred = predict_tcn(fitted, target.features)
    ti = test_indices(len(target.wear))
    rul_true = rul_from_wear(target.wear)[ti]
    rul_pred = rul_pred_from_wear_pred(wear_pred)[ti]
    return {"system": system, "k": k, "target": rot["target"], "seed": seed,
            "wear_mae": mae(target.wear[ti], wear_pred[ti]),
            "wear_rmse": rmse(target.wear[ti], wear_pred[ti]),
            "rul_phm_score": phm_score(rul_true, rul_pred), **extra}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--qit-root", default="data/raw/scidata/Milling dataset from QIT")
    ap.add_argument("--systems", default="B2,B3,OURS")
    ap.add_argument("--ks", default="0.05,0.10")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--variants", type=int, default=24)
    ap.add_argument("--ddpm-epochs", type=int, default=3000)
    ap.add_argument("--tcn-epochs", type=int, default=400)
    ap.add_argument("--out", default="results/w5_downstream.json")
    args = ap.parse_args()

    data = {t: load_processed(args.processed, t) for t in ("c1", "c4", "c6")}
    qit = load_qit_tool(args.qit_root) if args.qit_root else None
    systems = args.systems.split(",")
    ks = [float(x) for x in args.ks.split(",")]

    rows, t0 = [], time.time()
    total = len(systems) * len(ks) * len(ROTATIONS) * args.seeds
    for k in ks:
        for system in systems:
            for rot in ROTATIONS:
                for seed in range(args.seeds):
                    rows.append(run_cell(system, k, rot, data, qit, seed, args))
                    r = rows[-1]
                    print(f"[{len(rows)}/{total}] {system} k={k} {r['target']} s{seed}: "
                          f"MAE={r['wear_mae']:.2f} ({time.time() - t0:.0f}s)"
                          + (" [applied]" if r.get("correction_applied") else ""), flush=True)

    print("\n=== W5 DOWNSTREAM ARBITRATION (wear MAE, last-20% window) ===")
    summary = {}
    for k in ks:
        for system in systems:
            v = [r["wear_mae"] for r in rows if r["system"] == system and r["k"] == k]
            m, sem = float(np.mean(v)), float(np.std(v) / np.sqrt(len(v)))
            summary[f"{system}@{int(k * 100)}%"] = {"mae": round(m, 2), "sem": round(sem, 2)}
            print(f"  {system}@{int(k * 100)}%: {m:6.2f} ± {sem:.2f} (SEM, n={len(v)})")

    verdicts = {}
    for k in ks:
        kk = f"@{int(k * 100)}%"
        if "OURS" in systems and "B3" in systems:
            o, b3 = summary["OURS" + kk], summary["B3" + kk]
            b2 = summary.get("B2" + kk, {"mae": np.inf, "sem": 0})
            win = (o["mae"] + o["sem"] < b3["mae"] - b3["sem"]) and \
                  (o["mae"] + o["sem"] < b2["mae"] - b2["sem"])
            verdicts[kk] = "OURS WINS" if win else "NO SEPARATION"
            print(f"  K1{kk}: {verdicts[kk]}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"rows": rows, "summary": summary, "verdicts": verdicts,
                   "config": vars(args)}, f, indent=2)
    print(f"Saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
