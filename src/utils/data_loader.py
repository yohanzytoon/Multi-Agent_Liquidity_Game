"""Helpers to load synthetic calibration data (correlation matrices, node stats)."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


def load_correlation_matrix(path: Path) -> Tuple[List[str], Optional[np.ndarray]]:
    """
    Load asset_graph_edges.csv into a symmetric correlation matrix and ordered asset list.
    Returns (assets, matrix) where matrix[i,j] is the correlation between assets[i] and assets[j].
    If the file is missing or empty, returns ([], None).
    """
    if not path.exists():
        return [], None
    df = pd.read_csv(path)
    if df.empty:
        return [], None
    assets = sorted(set(df["asset_i"]).union(df["asset_j"]))
    idx = {a: i for i, a in enumerate(assets)}
    mat = np.ones((len(assets), len(assets)))
    for _, row in df.iterrows():
        i, j = idx[row["asset_i"]], idx[row["asset_j"]]
        mat[i, j] = mat[j, i] = float(row["correlation"])
    return assets, mat


def load_depth_stats(path: Path) -> Optional[pd.DataFrame]:
    """Load real_depth_statistics.csv if present."""
    if not path.exists():
        return None
    return pd.read_csv(path)


def load_depth_profiles(path: Path) -> Optional[pd.DataFrame]:
    """Load synthetic_depth_profiles.csv if present."""
    if not path.exists():
        return None
    return pd.read_csv(path)


def load_orderflow_events(path: Path) -> Optional[pd.DataFrame]:
    """Load synthetic_orderflow_events.csv if present."""
    if not path.exists():
        return None
    return pd.read_csv(path).sort_values("timestamp").reset_index(drop=True)


def load_regime_series(path: Path) -> Optional[pd.DataFrame]:
    """Load regimes.csv if present."""
    if not path.exists():
        return None
    return pd.read_csv(path).sort_values("timestamp").reset_index(drop=True)
