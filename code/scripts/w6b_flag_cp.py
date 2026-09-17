#!/usr/bin/env python3
"""W6b: rescue attempts for H5 (regime flag) and H6 (conformal intervals).

H5b  Deployment-observable flag: divergence between the pooled and the
     handbook-prior extrapolations (both computable from labels alone).
H6b  Misfit-normalized nonconformity: score = |err| / (misfit_labeled + c).

ARCHIVED VERDICT (2026-08-13): both rescues FAILED — divergence is flat across
tools (7.5/6.9/7.8) and normalization worsens c4 coverage (0.286 -> 0.159).
Kept as the triple-confirmation of the c4 undetectability boundary finding.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mdrdc.data.qit import load_qit_tool                                    # noqa: E402
from mdrdc.data.resample import isotonize                                   # noqa: E402
from mdrdc.experiments.protocol import (ROTATIONS, label_indices,           # noqa: E402
                                        load_processed, test_indices)
from mdrdc.metrics import nmpiw, picp                                       # noqa: E402
from mdrdc.twin.calibrate import calibrate                                  # noqa: E402
from mdrdc.twin.hierarchical import calibrate_pooled, fit_hyperprior, misfit_flag  # noqa: E402

C_NORM = 2.0  # um, stabilizer in normalized score


def forecasts(wear, k, hyper, seed=0):
    idx = label_indices(len(wear), k)
    labeled = np.maximum.accumulate(wear[idx])
    res_p = calibrate_pooled(labeled, idx, hyper, seed=seed)
    res_h = calibrate(labeled, indices=idx, seed=seed)
    path_p = res_p.extrapolate(len(wear), vb0=float(labeled[0]))
    path_h = res_h.extrapolate(len(wear), vb0=float(labeled[0]))
    _, mis = misfit_flag(res_p, labeled, idx)
    return path_p, path_h, mis


def main() -> int:
    data = {t: load_processed("data/processed", t) for t in ("c1", "c4", "c6")}
    qit = load_qit_tool("data/raw/scidata/Milling dataset from QIT")
    wears = {t: isotonize(d.wear) for t, d in data.items()}
    wears["qit"] = isotonize(qit.wear)

    out = {"H5b": [], "H6b": []}
    for rot in ROTATIONS:
        tgt = rot["target"]
        tw = wears[tgt]
        src_names = list(rot["sources"]) + ["qit"]
        sources = {n: wears[n] for n in src_names}
        hyper = fit_hyperprior(sources)
        ti = test_indices(len(tw))

        # H5b: prior-sensitivity divergence (deployment-observable)
        path_p, path_h, mis = forecasts(tw, 0.10, hyper)
        div = float(np.mean(np.abs(path_p[ti] - path_h[ti])))
        out["H5b"].append({"target": tgt, "divergence_um": round(div, 2),
                           "misfit_um": round(mis, 2)})

        # H6b: misfit-normalized CP calibrated on held-out sources
        scores = []
        for held in src_names:
            others = {n: w for n, w in sources.items() if n != held}
            if len(others) < 2:
                continue
            hp_h = fit_hyperprior(others)
            hw = sources[held]
            hp_path, _, h_mis = forecasts(hw, 0.10, hp_h)
            hti = test_indices(len(hw))
            scores.extend((np.abs(hw[hti] - hp_path[hti]) / (h_mis + C_NORM)).tolist())
    # noqa: E501
        scores = np.asarray(scores)
        q = float(np.quantile(scores, min(1.0, np.ceil((len(scores) + 1) * 0.9) / len(scores))))
        width = q * (mis + C_NORM)
        lo, hi = path_p[ti] - width, path_p[ti] + width
        out["H6b"].append({"target": tgt, "width_um": round(width, 2),
                           "picp": round(picp(tw[ti], lo, hi), 3),
                           "nmpiw": round(nmpiw(tw[ti], lo, hi), 3)})
        print(f"{tgt}: divergence={div:.1f}um misfit={mis:.1f} | "
              f"CP width={width:.1f} PICP={out['H6b'][-1]['picp']}", flush=True)

    print("\nH5b verdict: c4 divergence should exceed c1/c6 if the flag works")
    os.makedirs("results", exist_ok=True)
    with open("results/w6b_flag_cp.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Saved results/w6b_flag_cp.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
