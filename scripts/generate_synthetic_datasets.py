"""
Generate synthetic datasets for the liquidity sandbox without external deps.
Outputs:
  synthetic_depth_profiles.csv
  synthetic_orderflow_events.csv
  asset_graph_nodes.csv
  asset_graph_edges.csv
  regimes.csv
  real_depth_statistics.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import random
from pathlib import Path
from typing import List


def generate_assets(k: int) -> List[str]:
    return [f"ASSET_{chr(ord('A') + i)}" for i in range(k)]


def write_csv(path: Path, fieldnames: List[str], rows: List[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def generate_depth_profiles(outdir: Path, assets: List[str], n: int, rng: random.Random) -> None:
    rows = []
    regimes = ["normal", "stress"]
    for i in range(n):
        ts = i
        asset = rng.choice(assets)
        base = 100 + assets.index(asset) * 2
        vol = rng.uniform(0.01, 0.05)
        regime = rng.choices(regimes, weights=[0.8, 0.2])[0]
        spread = rng.uniform(0.01, 0.05) * (2 if regime == "stress" else 1)
        mid = base * (1 + rng.gauss(0, vol * 0.1))
        bid_sizes = [rng.gammavariate(2.0, 50) for _ in range(5)]
        ask_sizes = [rng.gammavariate(2.0, 50) for _ in range(5)]
        imbalance = rng.uniform(-1, 1)
        row = {
            "timestamp": ts,
            "asset": asset,
            "midprice": round(mid, 4),
            "spread": round(spread, 4),
            "volatility": round(vol, 4),
            "regime": regime,
            "orderflow_imbalance": round(imbalance, 4),
        }
        for j in range(5):
            row[f"bid_size_{j}"] = round(bid_sizes[j], 4)
            row[f"ask_size_{j}"] = round(ask_sizes[j], 4)
        rows.append(row)

    fieldnames = ["timestamp", "asset", "midprice", "spread", "volatility", "regime", "orderflow_imbalance"] + [
        f"bid_size_{j}" for j in range(5)
    ] + [f"ask_size_{j}" for j in range(5)]
    write_csv(outdir / "synthetic_depth_profiles.csv", fieldnames, rows)


def generate_orderflow_events(outdir: Path, assets: List[str], n: int, rng: random.Random) -> None:
    types = ["LIMIT_ADD", "LIMIT_CANCEL", "MARKET_ORDER", "REGIME_SWITCH"]
    regimes = ["normal", "stress"]
    rows = []
    ts = 0
    for _ in range(n):
        ts += rng.randint(1, 3)
        asset = rng.choice(assets)
        regime = rng.choices(regimes, weights=[0.8, 0.2])[0]
        typ = rng.choices(types, weights=[0.4, 0.2, 0.35, 0.05])[0]
        side = rng.choice(["buy", "sell"])
        price = 100 + assets.index(asset) * 2 + rng.gauss(0, 0.5)
        size = max(0.1, rng.gauss(100, 30))
        rows.append(
            {
                "timestamp": ts,
                "asset": asset,
                "type": typ,
                "side": side,
                "price": round(price, 4),
                "size": round(size, 4),
                "regime": regime,
            }
        )
    fieldnames = ["timestamp", "asset", "type", "side", "price", "size", "regime"]
    write_csv(outdir / "synthetic_orderflow_events.csv", fieldnames, rows)


def generate_asset_graph(outdir: Path, assets: List[str], rng: random.Random) -> None:
    node_rows = []
    for asset in assets:
        node_rows.append(
            {
                "asset_id": asset,
                "baseline_vol": round(rng.uniform(0.01, 0.05), 4),
                "baseline_spread": round(rng.uniform(0.01, 0.04), 4),
                "baseline_imbalance": round(rng.uniform(-0.2, 0.2), 4),
            }
        )
    write_csv(outdir / "asset_graph_nodes.csv", ["asset_id", "baseline_vol", "baseline_spread", "baseline_imbalance"], node_rows)

    edge_rows = []
    for i, a in enumerate(assets):
        for j, b in enumerate(assets):
            if i >= j:
                continue
            corr = rng.uniform(0.2, 0.9)
            edge_rows.append(
                {
                    "asset_i": a,
                    "asset_j": b,
                    "correlation": round(corr, 4),
                    "synthetic_hedge_ratio": round(rng.uniform(0.5, 1.5), 4),
                    "distance_metric": round(1 - corr, 4),
                }
            )
    write_csv(outdir / "asset_graph_edges.csv", ["asset_i", "asset_j", "correlation", "synthetic_hedge_ratio", "distance_metric"], edge_rows)


def generate_regimes(outdir: Path, n: int, rng: random.Random) -> None:
    rows = []
    ts = 0
    regime = "normal"
    for _ in range(n):
        ts += rng.randint(1, 5)
        if rng.random() < 0.1:
            regime = "stress" if regime == "normal" else "normal"
        vol = rng.uniform(0.01, 0.05) * (2 if regime == "stress" else 1)
        spread = rng.uniform(0.01, 0.05) * (2 if regime == "stress" else 1)
        depth = rng.uniform(50, 200) * (0.6 if regime == "stress" else 1.0)
        rows.append(
            {
                "timestamp": ts,
                "inferred_regime": regime,
                "vol": round(vol, 4),
                "spread": round(spread, 4),
                "depth": round(depth, 4),
            }
        )
    write_csv(outdir / "regimes.csv", ["timestamp", "inferred_regime", "vol", "spread", "depth"], rows)


def generate_real_depth_stats(outdir: Path, assets: List[str], rng: random.Random) -> None:
    rows = []
    for asset in assets:
        rows.append(
            {
                "asset": asset,
                "avg_spread": round(rng.uniform(0.01, 0.05), 4),
                "avg_depth_1": round(rng.uniform(50, 200), 4),
                "avg_depth_5": round(rng.uniform(200, 800), 4),
                "imbalance_mean": round(rng.uniform(-0.1, 0.1), 4),
                "imbalance_std": round(rng.uniform(0.05, 0.2), 4),
                "vol_mean": round(rng.uniform(0.01, 0.05), 4),
            }
        )
    write_csv(
        outdir / "real_depth_statistics.csv",
        ["asset", "avg_spread", "avg_depth_1", "avg_depth_5", "imbalance_mean", "imbalance_std", "vol_mean"],
        rows,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic datasets.")
    parser.add_argument("--outdir", type=str, default="data", help="Output directory")
    parser.add_argument("--assets", type=int, default=5, help="Number of assets")
    parser.add_argument("--snapshots", type=int, default=500, help="Number of depth snapshots")
    parser.add_argument("--events", type=int, default=1500, help="Number of order flow events")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    outdir = Path(args.outdir)
    os.makedirs(outdir, exist_ok=True)

    assets = generate_assets(args.assets)
    generate_depth_profiles(outdir, assets, args.snapshots, rng)
    generate_orderflow_events(outdir, assets, args.events, rng)
    generate_asset_graph(outdir, assets, rng)
    generate_regimes(outdir, n=max(200, args.snapshots // 2), rng=rng)
    generate_real_depth_stats(outdir, assets, rng)
    print(f"Generated datasets in {outdir}")


if __name__ == "__main__":
    main()
