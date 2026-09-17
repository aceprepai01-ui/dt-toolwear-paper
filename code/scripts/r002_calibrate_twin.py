#!/usr/bin/env python3
"""R002: MAP-calibrate the wear-ODE twin per tool on the early 15% of life.

Gate: early-fit R^2 >= 0.8 for every labeled tool.
Also reports full-life extrapolation MAE (context only, not gated in W1).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mdrdc.data.resample import isotonize  # noqa: E402
from mdrdc.metrics import mae  # noqa: E402
from mdrdc.twin.calibrate import calibrate_early_fraction  # noqa: E402

R2_GATE = 0.8


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--fraction", type=float, default=0.15)
    ap.add_argument("--out", default="results/r002_twin_calibration.json")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.processed, "c*.npz")))
    if not files:
        print(f"No processed tools in {args.processed} — run r001 first.", file=sys.stderr)
        return 1

    report, all_ok = [], True
    for f in files:
        tool = os.path.basename(f).split(".")[0]
        wear = isotonize(np.load(f)["wear"])
        res = calibrate_early_fraction(wear, fraction=args.fraction)
        extrap = res.extrapolate(len(wear), vb0=float(wear[0]))
        ok = res.r2_fit >= R2_GATE
        all_ok &= ok
        row = {"tool": tool, "r2_early": round(res.r2_fit, 4), "n_calib": res.n_calib,
               "extrap_mae_um": round(mae(wear, extrap), 2),
               "params": dict(zip(("log_k1", "a", "b", "log_beta_brk", "log_vb_brk",
                                   "log_beta_acc", "log_vb_acc"),
                                  np.round(res.params.to_vector(), 4).tolist())),
               "gate": "PASS" if ok else "FAIL"}
        report.append(row)
        print(f"[{row['gate']}] {tool}: early R2={row['r2_early']} "
              f"(gate >= {R2_GATE}), extrap MAE={row['extrap_mae_um']} um")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nSaved {args.out}")
    print("GATE:", "PASS — twin usable, proceed to W2" if all_ok
          else "FAIL — widen priors / try hierarchical pooling (see plan M0 mitigation)")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
