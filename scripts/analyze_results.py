"""Simple analysis helper for logged reward CSVs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.utils.plotting import plot_rewards
from src.utils.logging_utils import get_logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze reward logs.")
    parser.add_argument("--csv", type=str, default="artifacts/single_maker/rewards.csv", help="Path to rewards.csv")
    parser.add_argument("--out", type=str, default=None, help="Optional output plot path")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found")

    logger = get_logger("analyze_results")
    df = pd.read_csv(csv_path)
    logger.info("Loaded %d rows from %s", len(df), csv_path)

    reward_cols = [c for c in df.columns if c != "episode"]
    reward_history = {col.replace("_reward", ""): df[col].tolist() for col in reward_cols}

    out_path = Path(args.out) if args.out else csv_path.with_name("analysis.png")
    plot_rewards(reward_history, out_path)
    for col in reward_cols:
        logger.info("%s: mean %.4f | std %.4f | final %.4f", col, df[col].mean(), df[col].std(), df[col].iloc[-1])
    logger.info("Saved plot to %s", out_path)


if __name__ == "__main__":
    main()
