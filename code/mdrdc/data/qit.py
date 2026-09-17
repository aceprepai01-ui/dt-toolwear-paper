"""QIT-CEMC dataset (Nature Sci Data, figshare 27323346): single coated end-mill,
~69 machining cycles, wear measured per cycle on 4 side + 4 end edges.

Used as an additional SOURCE full-life wear path (cross-machine diversity) for
the residual bank and the EB hyperprior. Signals are not parsed here — only the
wear labels ('tool wear.xls') are needed for source-pool roles.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from ..experiments.protocol import ToolData

WEAR_FILE = "tool wear.xls"
MM_TO_UM = 1000.0
# Sheet layout: rows 0-3 are a 3-level header; data rows carry
# [Cycle, side edge1 (VBmax, VB1/2ap, S), edge2 (...), ..., end teeth (...)].
SIDE_VBMAX_COLS = (1, 4, 7, 10)


def load_qit_wear(root: str) -> np.ndarray:
    """Return the per-cycle wear path in micrometers: max VBmax over the 4 side
    edges (consistent with the PHM2010 max-over-flutes target choice)."""
    path = os.path.join(root, WEAR_FILE)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} missing — extract the QIT-CEMC archive (figshare 27323346) here")
    df = pd.ExcelFile(path).parse("tool wear", header=None)
    rows = df.iloc[4:].reset_index(drop=True)
    cycles = pd.to_numeric(rows[0], errors="coerce")
    rows = rows[cycles.notna()]
    vb = rows[list(SIDE_VBMAX_COLS)].apply(pd.to_numeric, errors="coerce")
    if vb.isna().any().any():
        vb = vb.dropna()
    wear_mm = vb.max(axis=1).to_numpy(dtype=float)
    if len(wear_mm) < 20:
        raise ValueError(f"QIT wear table suspiciously short: {len(wear_mm)} cycles")
    wear_um = np.maximum.accumulate(wear_mm * MM_TO_UM)
    if not np.all(np.isfinite(wear_um)) or wear_um[-1] < 100.0:
        raise ValueError("QIT wear path failed sanity checks (units? layout?)")
    return wear_um


def load_qit_tool(root: str) -> ToolData:
    """ToolData wrapper; features are empty — QIT serves wear-only source roles."""
    wear = load_qit_wear(root)
    return ToolData(tool="qit", features=np.empty((len(wear), 0)), wear=wear)
