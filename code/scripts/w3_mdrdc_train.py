#!/usr/bin/env python3
"""W3-4 (M2): train M-DRDC per rotation; gate on generative-object quality.

Per rotation: bank from source tools (+QIT via --qit-root) -> train residual DDPM
-> sample N corrected paths for the target (labels used ONLY for evaluation) ->
compare against the raw twin.

GATE (M2): mean crossing-time CRPS of corrected samples < mean |twin crossing
error| (horizon-censored on BOTH sides) across rotations.
ARCHIVED VERDICT: never passed (final 26.3-27.6 vs twin 24.3) — kept as the
controlled negative result; see EXPERIMENT_TRACKER R020.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mdrdc.data.resample import T_GRID, isotonize, resample_path, u_to_path   # noqa: E402
from mdrdc.experiments.protocol import (ROTATIONS, label_indices,             # noqa: E402
                                        load_processed)
from mdrdc.experiments.residual_bank import (build_bank, gate_and_weight,     # noqa: E402
                                             target_conditioning)
from mdrdc.metrics import crps_empirical, crossing_time, rmse                 # noqa: E402
from mdrdc.models.ddpm import sample_residuals, train_ddpm                    # noqa: E402

W_TH = 165.0


def run_rotation(rot: dict, data: dict, k_frac: float, args) -> dict:
    target = data[rot["target"]]
    sources = [data[s] for s in rot["sources"]]
    if args.qit_root:
        from mdrdc.data.qit import load_qit_tool
        sources = sources + [load_qit_tool(args.qit_root)]
    n_cuts = len(target.wear)
    grid_to_cuts = n_cuts / T_GRID

    src_names = [s.tool for s in sources]
    print(f"  bank: {src_names} x {args.variants} variants ...", flush=True)
    bank = build_bank(sources, n_variants=args.variants, seed=args.seed)
    idx = label_indices(n_cuts, k_frac)
    u_twin, cond_vec, w_start = target_conditioning(target, idx, seed=args.seed)

    if args.localized:
        # Retrieval-localized residual diffusion (external review prescription):
        # same-slope-regime bank, kernel-weighted loss, 2-d FiLM cond, no CFG.
        subset, weights = gate_and_weight(bank, cond_vec)
        two = lambda c: np.array([c[0], c[3]])  # [misfit, tail-slope] only
        print(f"  localized: {len(subset)}/{len(bank)} bank samples, "
              f"ESS={1.0 / float(np.sum(weights ** 2)):.1f}", flush=True)
        fitted = train_ddpm(
            residuals=np.stack([b.residual for b in subset]),
            u_twins=np.stack([b.u_twin for b in subset]),
            cond_vecs=np.stack([two(b.cond_vec) for b in subset]),
            sample_weights=weights, cfg_drop_p=0.0,
            epochs=args.epochs, seed=args.seed, log_every=args.epochs // 4)
        res = sample_residuals(fitted, u_twin, two(cond_vec),
                               n_samples=args.samples, guidance=0.0, seed=args.seed)
    else:
        fitted = train_ddpm(
            residuals=np.stack([b.residual for b in bank]),
            u_twins=np.stack([b.u_twin for b in bank]),
            cond_vecs=np.stack([b.cond_vec for b in bank]),
            epochs=args.epochs, seed=args.seed, log_every=args.epochs // 4)
        res = sample_residuals(fitted, u_twin, cond_vec, n_samples=args.samples,
                               guidance=args.guidance, seed=args.seed)
    paths = np.stack([u_to_path(w_start, u_twin + r) for r in res])  # (N, 64)
    twin64 = u_to_path(w_start, u_twin)

    real64 = resample_path(isotonize(target.wear), T_GRID)
    t_real = crossing_time(real64, W_TH) * grid_to_cuts
    t_twin = crossing_time(twin64, W_TH) * grid_to_cuts
    t_samp = np.array([crossing_time(p, W_TH) for p in paths]) * grid_to_cuts
    censored = ~np.isfinite(t_samp)
    t_samp = np.where(censored, float(n_cuts), t_samp)
    # Censoring: a path that never crosses is treated as crossing at the horizon
    # (same rule for twin and samples — anything else inflates the prior's error).
    t_twin_c = t_twin if np.isfinite(t_twin) else float(n_cuts)

    # Selective application (reject option): correct only when the predicted
    # crossing-time shift is decisive; otherwise emit the twin unchanged.
    applied = True
    if args.selective:
        shift = t_samp - t_twin_c
        iqr = float(np.percentile(shift, 75) - np.percentile(shift, 25))
        applied = abs(float(np.median(shift))) >= max(5.0, 1.5 * iqr)
        if not applied:
            t_samp = np.array([t_twin_c, t_twin_c])  # degenerate: twin forecast
            paths = np.stack([twin64, twin64])
            censored = np.zeros(2, dtype=bool)

    row = {
        "correction_applied": bool(applied),
        "target": rot["target"], "k": k_frac,
        "crossing_true_cuts": round(t_real, 1),
        "crossing_twin_cuts": round(t_twin_c, 1),
        "twin_censored": bool(~np.isfinite(t_twin)),
        "twin_abs_err_cuts": round(abs(t_twin_c - t_real), 1),
        "mdrdc_crps_cuts": round(crps_empirical(t_samp, t_real), 1),
        "mdrdc_median_err_cuts": round(abs(float(np.median(t_samp)) - t_real), 1),
        "censored_frac": round(float(censored.mean()), 2),
        "path_rmse_twin_um": round(rmse(real64, twin64), 2),
        "path_rmse_mdrdc_um": round(rmse(real64, paths.mean(axis=0)), 2),
    }
    print(f"  {rot['target']} k={k_frac}: twin_err={row['twin_abs_err_cuts']} cuts, "
          f"CRPS={row['mdrdc_crps_cuts']} cuts, "
          f"pathRMSE twin={row['path_rmse_twin_um']} -> mdrdc={row['path_rmse_mdrdc_um']} um",
          flush=True)
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--k", type=float, default=0.10)
    ap.add_argument("--epochs", type=int, default=4000)
    ap.add_argument("--variants", type=int, default=16)
    ap.add_argument("--samples", type=int, default=50)
    ap.add_argument("--guidance", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--qit-root", default=None,
                    help="QIT-CEMC dataset dir; adds its wear path to every rotation's source bank")
    ap.add_argument("--localized", action="store_true",
                    help="regime-gated kernel-weighted bank, 2-d cond, no CFG")
    ap.add_argument("--selective", action="store_true",
                    help="reject option: emit twin when correction is not decisive")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="results/w3_mdrdc.json")
    args = ap.parse_args()
    if args.smoke:
        args.epochs, args.variants, args.samples = 600, 6, 20

    data = {t: load_processed(args.processed, t) for t in ("c1", "c4", "c6")}
    t0 = time.time()
    rows = [run_rotation(rot, data, args.k, args) for rot in ROTATIONS]

    twin_err = float(np.mean([r["twin_abs_err_cuts"] for r in rows]))
    crps = float(np.mean([r["mdrdc_crps_cuts"] for r in rows]))
    rmse_twin = float(np.mean([r["path_rmse_twin_um"] for r in rows]))
    rmse_ours = float(np.mean([r["path_rmse_mdrdc_um"] for r in rows]))
    gate = crps < twin_err
    print(f"\n=== W3 M-DRDC REPORT (k={args.k}, {time.time() - t0:.0f}s) ===")
    print(f"crossing-time: twin abs err {twin_err:.1f} cuts -> M-DRDC CRPS {crps:.1f} cuts")
    print(f"path RMSE:     twin {rmse_twin:.1f} um -> M-DRDC mean-path {rmse_ours:.1f} um")
    print(f"GATE (CRPS < twin err): {'PASS — corrector beats its prior' if gate else 'FAIL'}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"rows": rows, "gate_crps_lt_twin": gate,
                   "mean": {"twin_err": twin_err, "crps": crps,
                            "rmse_twin": rmse_twin, "rmse_mdrdc": rmse_ours},
                   "config": vars(args)}, f, indent=2)
    print(f"Saved {args.out}")
    return 0 if gate else 1


if __name__ == "__main__":
    raise SystemExit(main())
