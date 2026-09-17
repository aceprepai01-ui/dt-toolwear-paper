#!/usr/bin/env python3
"""R001: parse PHM2010, verify cut counts, extract per-cut force features.

Writes data/processed/<tool>.npz with: wear, wear_per_flute, features, feature_names.
Gate: each labeled tool has 315 cuts and finite features.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mdrdc.data import phm2010  # noqa: E402
from mdrdc.data.features import FEATURE_NAMES, extract_cut_features  # noqa: E402


def process_tool(root: str, tool: str, out_dir: str) -> dict:
    rec = phm2010.load_tool(root, tool)
    feats = np.empty((rec.n_cuts, len(FEATURE_NAMES)), dtype=float)
    for i, path in enumerate(rec.signal_files):
        feats[i] = extract_cut_features(phm2010.read_cut_signals(path))
        if (i + 1) % 50 == 0:
            print(f"  {tool}: {i + 1}/{rec.n_cuts} cuts", flush=True)
    out = os.path.join(out_dir, f"{tool}.npz")
    np.savez_compressed(out, wear=rec.wear, wear_per_flute=rec.wear_per_flute,
                        features=feats, feature_names=np.array(FEATURE_NAMES))
    ok_counts = rec.n_cuts == phm2010.EXPECTED_CUTS
    ok_finite = bool(np.all(np.isfinite(feats)))
    return {"tool": tool, "n_cuts": rec.n_cuts, "expected": phm2010.EXPECTED_CUTS,
            "counts_ok": ok_counts, "finite_ok": ok_finite, "out": out,
            "wear_range_um": (float(rec.wear.min()), float(rec.wear.max()))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw/phm2010")
    ap.add_argument("--out", default="data/processed")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    results = [process_tool(args.root, t, args.out) for t in phm2010.LABELED_TOOLS]
    print("\n=== R001 REPORT ===")
    all_ok = True
    for r in results:
        status = "PASS" if (r["counts_ok"] and r["finite_ok"]) else "FAIL"
        all_ok &= status == "PASS"
        print(f"[{status}] {r['tool']}: {r['n_cuts']}/{r['expected']} cuts, "
              f"wear {r['wear_range_um'][0]:.1f}-{r['wear_range_um'][1]:.1f} um -> {r['out']}")
    print("GATE:", "PASS — proceed to R002" if all_ok else "FAIL — inspect dataset layout")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
