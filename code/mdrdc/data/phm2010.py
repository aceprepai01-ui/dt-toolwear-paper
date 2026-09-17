"""PHM2010 milling dataset parsing.

Layout expected under a tool directory (Kaggle mirror / IEEE DataPort archive):
    <root>/c1/c1_wear.csv           wear per cut: cut, flute_1, flute_2, flute_3 (in 1e-3 mm)
    <root>/c1/c1/c_1_001.csv ...    per-cut 7-channel signals @50 kHz:
                                    Fx, Fy, Fz, Vx, Vy, Vz, AE
Only c1, c4, c6 carry wear labels (the challenge training tools).
"""
from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

LABELED_TOOLS = ("c1", "c4", "c6")
EXPECTED_CUTS = 315
SIGNAL_COLUMNS = ("fx", "fy", "fz", "vx", "vy", "vz", "ae")
FORCE_COLUMNS = ("fx", "fy", "fz")


@dataclass(frozen=True)
class ToolRecord:
    """Immutable container for one tool's parsed data."""

    tool: str
    wear: np.ndarray          # (n_cuts,) regression target in micrometers
    wear_per_flute: np.ndarray  # (n_cuts, 3)
    signal_files: tuple       # sorted per-cut signal file paths, len == n_cuts

    @property
    def n_cuts(self) -> int:
        return len(self.wear)


def _find_wear_csv(root: str, tool: str) -> str:
    candidates = sorted(
        glob.glob(os.path.join(root, tool, "**", f"{tool}_wear.csv"), recursive=True)
    )
    if not candidates:
        raise FileNotFoundError(
            f"No wear file '{tool}_wear.csv' under {os.path.join(root, tool)!r}. "
            "Check the dataset was downloaded and unpacked (scripts/download_phm2010.sh)."
        )
    return candidates[0]


def _find_signal_files(root: str, tool: str) -> tuple:
    idx = tool[1:]  # 'c1' -> '1'
    pattern = os.path.join(root, tool, "**", f"c_{idx}_*.csv")
    files = [f for f in glob.glob(pattern, recursive=True) if "wear" not in os.path.basename(f)]

    def cut_number(path: str) -> int:
        m = re.search(r"_(\d+)\.csv$", os.path.basename(path))
        if m is None:
            raise ValueError(f"Cannot parse cut number from {path!r}")
        return int(m.group(1))

    return tuple(sorted(files, key=cut_number))


def load_wear(root: str, tool: str, target: str = "max") -> tuple[np.ndarray, np.ndarray]:
    """Return (target_wear, per_flute_wear) in micrometers.

    target: 'max' (default, conservative — standard failure criterion) or 'mean'.
    The choice must be declared in the paper; both are kept for reporting.
    """
    if target not in ("max", "mean"):
        raise ValueError(f"target must be 'max' or 'mean', got {target!r}")
    df = pd.read_csv(_find_wear_csv(root, tool))
    flute_cols = [c for c in df.columns if "flute" in c.lower()]
    if len(flute_cols) != 3:
        raise ValueError(f"Expected 3 flute columns in wear file, got {flute_cols}")
    per_flute = df[flute_cols].to_numpy(dtype=float)
    if not np.all(np.isfinite(per_flute)):
        raise ValueError(f"Non-finite wear values in {tool} wear file")
    agg = per_flute.max(axis=1) if target == "max" else per_flute.mean(axis=1)
    return agg, per_flute


def load_tool(root: str, tool: str, target: str = "max", strict: bool = True) -> ToolRecord:
    """Parse one labeled tool; validates cut counts against the wear table."""
    if tool not in LABELED_TOOLS:
        raise ValueError(f"{tool!r} is not a labeled tool; labeled tools are {LABELED_TOOLS}")
    wear, per_flute = load_wear(root, tool, target=target)
    files = _find_signal_files(root, tool)
    if len(files) != len(wear):
        msg = (f"{tool}: {len(files)} signal files but {len(wear)} wear rows "
               f"(literature reports {EXPECTED_CUTS} cuts)")
        if strict:
            raise ValueError(msg)
        n = min(len(files), len(wear))
        wear, per_flute, files = wear[:n], per_flute[:n], files[:n]
    return ToolRecord(tool=tool, wear=wear, wear_per_flute=per_flute, signal_files=files)


def read_cut_signals(path: str) -> pd.DataFrame:
    """Read one per-cut signal CSV (7 unnamed columns) with validation."""
    df = pd.read_csv(path, header=None)
    if df.shape[1] != len(SIGNAL_COLUMNS):
        raise ValueError(f"{path}: expected {len(SIGNAL_COLUMNS)} columns, got {df.shape[1]}")
    df.columns = list(SIGNAL_COLUMNS)
    if len(df) < 1000:
        raise ValueError(f"{path}: suspiciously short signal ({len(df)} samples)")
    return df
