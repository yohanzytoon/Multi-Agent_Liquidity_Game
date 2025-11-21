# Synthetic Dataset Usage

We include a generator and ready-made datasets to calibrate the sandbox.

## Files
- `scripts/generate_synthetic_datasets.py`: creates six CSVs.
- `data/`: small sample.
- `data_big/`: large sample (e.g., 600 snapshots/2000 events or more if regenerated).

Generated CSVs:
1. `synthetic_depth_profiles.csv` — snapshots (`timestamp, asset, midprice, spread, volatility, regime, orderflow_imbalance, bid_size_0..4, ask_size_0..4`)
2. `synthetic_orderflow_events.csv` — event stream (`timestamp, asset, type, side, price, size, regime`)
3. `asset_graph_nodes.csv` — node stats (`asset_id, baseline_vol, baseline_spread, baseline_imbalance`)
4. `asset_graph_edges.csv` — edges (`asset_i, asset_j, correlation, synthetic_hedge_ratio, distance_metric`)
5. `regimes.csv` — inferred regimes over time (`timestamp, inferred_regime, vol, spread, depth`)
6. `real_depth_statistics.csv` — aggregate depth stats per asset.

## Generate More Data
```bash
python3 scripts/generate_synthetic_datasets.py \
  --outdir data_big \
  --assets 10 \
  --snapshots 100000 \
  --events 200000 \
  --seed 42
```

## Consuming Data in the Codebase
- Correlation matrix: `load_correlation_matrix(path)` (used automatically in `scripts/run_multi_asset_gnn.py` if `data_big/asset_graph_edges.csv` exists).
- To seed env fundamentals or calibrate noise, you can read `synthetic_depth_profiles.csv` and set `MarketEnvConfig` accordingly in scripts (customize as needed).

