#!/usr/bin/env python3
"""W2 (M1): run LIT / B1 / B2 / B3 baselines over 3 rotations x seeds.

Evaluation: fixed last-20% test window of the target tool (LIT: full trajectory).
Gates: (a) LIT full-trajectory cross-tool MAE in the published cross-tool range;
       (b) B2 MAE degrades monotonically as k shrinks (within SEM tolerance).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mdrdc.experiments import baselines as bl                      # noqa: E402
from mdrdc.experiments.protocol import (K_FRACTIONS, ROTATIONS,    # noqa: E402
                                        load_processed, rul_from_wear,
                                        rul_pred_from_wear_pred, test_indices)
from mdrdc.metrics import mae, phm_score, rmse                     # noqa: E402
from mdrdc.models.tcn import predict_tcn, train_tcn                # noqa: E402

# Gate range corrected 2026-08: initial (8, 25) was calibrated on SAME-TOOL
# literature numbers; the LIT check is CROSS-TOOL force-only, whose published
# spread is roughly 20-50 um MAE. Documented in EXPERIMENT_TRACKER (R010).
B1_GATE = (10.0, 45.0)


def run_one(system: str, k: float | None, rot: dict, data: dict,
            seed: int, epochs: int) -> dict:
    target, sources = data[rot["target"]], [data[s] for s in rot["sources"]]
    if system == "LIT":
        # Literature protocol: train on the two source tools only, evaluate on the
        # FULL target trajectory — the apples-to-apples check against published
        # cross-tool PHM2010 numbers (R010 gate).
        seqs = [(s.features, s.wear) for s in sources]
    elif system == "B1":
        seqs = bl.build_b1(target, sources)
    elif system == "B2":
        seqs = bl.build_b2(target, k)
    elif system == "B3":
        seqs = bl.build_b3(target, sources, k, noise_seed=seed)
    else:
        raise ValueError(f"Unknown system {system!r}")

    fitted = train_tcn(seqs, seed=seed, epochs=epochs)
    wear_pred = predict_tcn(fitted, target.features)

    ti = (np.arange(len(target.wear)) if system == "LIT"
          else test_indices(len(target.wear)))
    rul_true = rul_from_wear(target.wear)[ti]
    rul_pred = rul_pred_from_wear_pred(wear_pred)[ti]
    return {
        "system": system, "k": k, "target": rot["target"], "seed": seed,
        "wear_mae": mae(target.wear[ti], wear_pred[ti]),
        "wear_rmse": rmse(target.wear[ti], wear_pred[ti]),
        "rul_phm_score": phm_score(rul_true, rul_pred),
    }


def summarize(rows: list) -> dict:
    out = {}
    for r in rows:
        key = f"{r['system']}@{int(r['k'] * 100)}%" if r["k"] else r["system"]
        out.setdefault(key, []).append(r["wear_mae"])
    return {k: {"mae_mean": round(float(np.mean(v)), 2),
                "mae_std": round(float(np.std(v)), 2), "n": len(v)}
            for k, v in sorted(out.items())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--smoke", action="store_true", help="1 seed, 60 epochs, quick check")
    ap.add_argument("--out", default="results/w2_baselines.json")
    args = ap.parse_args()
    if args.smoke:
        args.seeds, args.epochs = 1, 60

    data = {t: load_processed(args.processed, t) for t in ("c1", "c4", "c6")}
    configs = ([("LIT", None), ("B1", None)]
               + [(s, k) for s in ("B2", "B3") for k in K_FRACTIONS])

    rows, t0 = [], time.time()
    total = len(configs) * len(ROTATIONS) * args.seeds
    for system, k in configs:
        for rot in ROTATIONS:
            for seed in range(args.seeds):
                rows.append(run_one(system, k, rot, data, seed, args.epochs))
                done = len(rows)
                print(f"[{done}/{total}] {system} k={k} target={rot['target']} "
                      f"seed={seed}: MAE={rows[-1]['wear_mae']:.2f} um "
                      f"({time.time() - t0:.0f}s)", flush=True)

    summary = summarize(rows)
    print("\n=== W2 SUMMARY (wear MAE on last-20% window, um) ===")
    for key, s in summary.items():
        print(f"  {key:8s}: {s['mae_mean']:6.2f} ± {s['mae_std']:.2f}  (n={s['n']})")

    lit = summary["LIT"]["mae_mean"]
    gate_a = B1_GATE[0] <= lit <= B1_GATE[1]
    # Noise-aware monotonicity: a violation smaller than the standard error of
    # the mean is noise, not a real inversion.
    b2 = [summary[f"B2@{int(k * 100)}%"]["mae_mean"] for k in K_FRACTIONS]
    sem = [summary[f"B2@{int(k * 100)}%"]["mae_std"] / np.sqrt(summary[f"B2@{int(k * 100)}%"]["n"])
           for k in K_FRACTIONS]
    gate_b = all(b2[i] >= b2[i + 1] - max(sem[i], sem[i + 1]) for i in range(len(b2) - 1))
    print(f"\nGATE A (LIT full-trajectory cross-tool in {B1_GATE} um): "
          f"{'PASS' if gate_a else 'FAIL'} ({lit:.2f})")
    print(f"GATE B (B2 monotone 5%>=10%>=20% within SEM): {'PASS' if gate_b else 'FAIL'} "
          f"({b2[0]:.2f} / {b2[1]:.2f} / {b2[2]:.2f})")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"rows": rows, "summary": summary,
                   "gates": {"A_lit_literature_range": gate_a, "B_b2_monotone": gate_b}}, f, indent=2)
    print(f"Saved {args.out}")
    return 0 if (gate_a and gate_b) else 1


if __name__ == "__main__":
    raise SystemExit(main())
