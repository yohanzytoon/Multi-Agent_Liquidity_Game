"""
Streamlit dashboard for the Multi-Agent Liquidity Game.

Features:
- Load and plot rewards from artifacts (single/multi-agent runs).
- Inspect synthetic datasets (depth profiles, orderflow events, regimes).
- Visualize correlation matrix from asset_graph_edges.csv.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

import pandas as pd
import numpy as np
import streamlit as st


# -------------------- Config -------------------- #
DEFAULT_ARTIFACTS = {
    "single_maker": Path("artifacts/single_maker/rewards.csv"),
    "multi_agent_basic": Path("artifacts/multi_agent_basic/rewards.csv"),
    "multi_asset_gnn": Path("artifacts/multi_asset_gnn/rewards.csv"),
}
DEFAULT_DATA_DIR = Path("data_big")


# -------------------- Data loaders -------------------- #
@st.cache_data
def load_rewards(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def load_edges(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def load_dataset(path: Path, nrows: int = 2000) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, nrows=nrows)


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    assets = sorted(set(df["asset_i"]).union(df["asset_j"]))
    mat = pd.DataFrame(np.ones((len(assets), len(assets))), index=assets, columns=assets)
    for _, row in df.iterrows():
        a, b, c = row["asset_i"], row["asset_j"], row["correlation"]
        mat.loc[a, b] = mat.loc[b, a] = c
    return mat


# -------------------- UI helpers -------------------- #
def show_rewards_section() -> None:
    st.header("Reward Explorer")
    run = st.selectbox("Choose run", list(DEFAULT_ARTIFACTS.keys()))
    default_path = DEFAULT_ARTIFACTS[run]
    path = st.text_input("Rewards CSV path", value=str(default_path))
    df = load_rewards(Path(path))
    if df.empty:
        st.warning("No data loaded.")
        return
    st.write("Loaded rows:", len(df))
    st.line_chart(df.set_index("episode"))
    st.write("Summary:", df.describe())


def show_datasets_section(data_dir: Path) -> None:
    st.header("Synthetic Datasets")
    st.caption(f"Showing samples from {data_dir}")
    cols = st.columns(2)
    # Depth profiles
    depth = load_dataset(data_dir / "synthetic_depth_profiles.csv", nrows=2000)
    cols[0].subheader("Depth Profiles (head)")
    cols[0].dataframe(depth.head(10))
    # Orderflow events
    events = load_dataset(data_dir / "synthetic_orderflow_events.csv", nrows=2000)
    cols[1].subheader("Orderflow Events (head)")
    cols[1].dataframe(events.head(10))
    # Regime series
    regimes = load_dataset(data_dir / "regimes.csv", nrows=2000)
    st.subheader("Regimes (head)")
    st.dataframe(regimes.head(10))


def show_graph_section(data_dir: Path) -> None:
    st.header("Asset Graph")
    edges = load_edges(data_dir / "asset_graph_edges.csv")
    if edges.empty:
        st.info("No edges file found.")
        return
    st.write("Edges (head):")
    st.dataframe(edges.head(20))
    mat = correlation_matrix(edges)
    if not mat.empty:
        st.subheader("Correlation Heatmap")
        st.dataframe(mat.style.background_gradient(cmap="coolwarm"))


# -------------------- Main -------------------- #
def main() -> None:
    st.title("Multi-Agent Liquidity Game Dashboard")
    data_dir = Path(st.sidebar.text_input("Data directory", value=str(DEFAULT_DATA_DIR)))
    section = st.sidebar.radio("Section", ["Rewards", "Datasets", "Asset Graph"])

    if section == "Rewards":
        show_rewards_section()
    elif section == "Datasets":
        show_datasets_section(data_dir)
    elif section == "Asset Graph":
        show_graph_section(data_dir)


if __name__ == "__main__":
    main()

