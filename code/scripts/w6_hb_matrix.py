#!/usr/bin/env python3
"""W6: v3 experiment matrix for the EB pooled twin (H1, H3, H4, H5, H6).

All CPU-deterministic. H2 (headline vs DL) reuses stored W2/W5 tables.

H1  pooled vs handbook-prior vs zero-shot(population mean, no target labels)
H3  leave-one-source-out hyperprior sensitivity
H4  prior-scale robustness tau x {0.5, 1, 2}
H5  misfit-flag operating characteristics across thresholds
H6  population-calibrated split-conformal intervals on the tail (PICP/NMPIW)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mdrdc.data.qit import load_qit_tool                                    # noqa: E402
from mdrdc.data.resample import isotonize                                   # noqa: E402
from mdrdc.experiments.protocol import (ROTATIONS, label_indices,           # noqa: E402
                                        load_processed, rul_from_wear,
                                        rul_pred_from_wear_pred, test_indices)
from mdrdc.metrics import mae, phm_score, picp, nmpiw, rmse                 # noqa: E402
from mdrdc.twin.calibrate import calibrate                                  # noqa: E402
from mdrdc.twin.hierarchical import (calibrate_pooled, fit_hyperprior,      # noqa: E402
                                     misfit_flag)
from mdrdc.twin.wear_ode import TwinParams, simulate                        # noqa: E402

FLAG_THRESHOLDS = (4.0, 6.0, 8.0, 10.0, 12.0)


def tail_eval(target_wear: np.ndarray, pred_path: np.ndarray) -> dict:
    ti = test_indices(len(target_wear))
    rul_true = rul_from_wear(target_wear)[ti]
    rul_pred = rul_pred_from_wear_pred(pred_path)[ti]
    return {"mae": round(mae(target_wear[ti], pred_path[ti]), 2),
            "rmse": round(rmse(target_wear[ti], pred_path[ti]), 2),
            "rul_phm": round(phm_score(rul_true, rul_pred), 1)}


def pooled_forecast(target_wear, k, hyper, seed=0):
    idx = label_indices(len(target_wear), k)
    labeled = np.maximum.accumulate(target_wear[idx])
    res = calibrate_pooled(labeled, idx, hyper, seed=seed)
    path = res.extrapolate(len(target_wear), vb0=float(labeled[0]))
    return res, path, idx, labeled


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--qit-root", default="data/raw/scidata/Milling dataset from QIT")
    ap.add_argument("--out", default="results/w6_hb_matrix.json")
    args = ap.parse_args()

    data = {t: load_processed(args.processed, t) for t in ("c1", "c4", "c6")}
    qit = load_qit_tool(args.qit_root)
    wears = {t: isotonize(d.wear) for t, d in data.items()}
    wears["qit"] = isotonize(qit.wear)

    out = {"H1": [], "H3": [], "H4": [], "H5": [], "H6": []}
    for rot in ROTATIONS:
        tgt = rot["target"]
        target_wear = wears[tgt]
        src_names = list(rot["sources"]) + ["qit"]
        sources = {n: wears[n] for n in src_names}
        hyper = fit_hyperprior(sources)

        # --- H1 ---
        for k in (0.05, 0.10):
            _, path_p, idx, labeled = pooled_forecast(target_wear, k, hyper)
            res_h = calibrate(labeled, indices=idx)
            path_h = res_h.extrapolate(len(target_wear), vb0=float(labeled[0]))
            path_z = simulate(TwinParams.from_vector(hyper.mu), len(target_wear),
                              vb0=float(target_wear[0]))
            for name, path in (("pooled", path_p), ("handbook", path_h), ("zeroshot", path_z)):
                out["H1"].append({"target": tgt, "k": k, "system": name,
                                  **tail_eval(target_wear, path)})

        # --- H3: leave-one-source-out ---
        for drop in src_names:
            keep = {n: w for n, w in sources.items() if n != drop}
            hp_loo = fit_hyperprior(keep)
            _, path, _, _ = pooled_forecast(target_wear, 0.10, hp_loo)
            out["H3"].append({"target": tgt, "dropped": drop,
                              **tail_eval(target_wear, path)})

        # --- H4: prior-scale robustness ---
        for f in (0.5, 1.0, 2.0):
            _, path, _, _ = pooled_forecast(target_wear, 0.10, hyper.scaled(f))
            out["H4"].append({"target": tgt, "tau_scale": f,
                              **tail_eval(target_wear, path)})

        # --- H5: misfit flag ---
        res, _, idx, labeled = pooled_forecast(target_wear, 0.10, hyper)
        _, m = misfit_flag(res, labeled, idx)
        out["H5"].append({"target": tgt, "misfit_rmse_um": round(m, 2),
                          "flag_at": {str(th): bool(m > th) for th in FLAG_THRESHOLDS}})

        # --- H6: population-calibrated split-CP on the tail ---
        cal_scores = []
        for held in src_names:
            others = {n: w for n, w in sources.items() if n != held}
            if len(others) < 2:
                continue
            hp_h = fit_hyperprior(others)
            hw = sources[held]
            _, hpath, _, _ = pooled_forecast(hw, 0.10, hp_h)
            hti = test_indices(len(hw))
            cal_scores.extend(np.abs(hw[hti] - hpath[hti]).tolist())
        q = float(np.quantile(np.asarray(cal_scores),
                              min(1.0, np.ceil((len(cal_scores) + 1) * 0.9) / len(cal_scores))))
        _, tpath, _, _ = pooled_forecast(target_wear, 0.10, hyper)
        ti = test_indices(len(target_wear))
        lo, hi = tpath[ti] - q, tpath[ti] + q
        out["H6"].append({"target": tgt, "q90_um": round(q, 2),
                          "picp": round(picp(target_wear[ti], lo, hi), 3),
                          "nmpiw": round(nmpiw(target_wear[ti], lo, hi), 3),
                          "n_cal": len(cal_scores)})
        print(f"{tgt} done", flush=True)

    print("\n=== H1 (tail wear MAE, um) ===")
    for k in (0.05, 0.10):
        for sysname in ("pooled", "handbook", "zeroshot"):
            v = [r["mae"] for r in out["H1"] if r["k"] == k and r["system"] == sysname]
            print(f"  {sysname}@{int(k*100)}%: mean {np.mean(v):6.2f}  per-tool {v}")
    print("=== H3 LOO spread (pooled@10% MAE) ===")
    for tgt in ("c1", "c4", "c6"):
        v = [r["mae"] for r in out["H3"] if r["target"] == tgt]
        print(f"  {tgt}: {min(v):.1f} - {max(v):.1f}")
    print("=== H4 tau-scale ===")
    for f in (0.5, 1.0, 2.0):
        v = [r["mae"] for r in out["H4"] if r["tau_scale"] == f]
        print(f"  x{f}: mean {np.mean(v):.2f}")
    print("=== H5 misfit ===")
    for r in out["H5"]:
        print(f"  {r['target']}: rmse {r['misfit_rmse_um']}")
    print("=== H6 CP ===")
    for r in out["H6"]:
        print(f"  {r['target']}: PICP {r['picp']} NMPIW {r['nmpiw']} (q={r['q90_um']}, n={r['n_cal']})")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
