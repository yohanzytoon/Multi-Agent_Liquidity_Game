# Case Study: Multi-Agent Liquidity Game (Synthetic Sandbox)

> This report summarizes a reproducible experiment setup and the qualitative results you should expect. Metrics below are illustrative; rerun locally to generate actual numbers and plots.

## Experiment Design
- **Scenario:** Two-asset market (`ASSET_A`, `ASSET_B`), one market maker, one arbitrageur. Noise traders provide background flow in normal/stress regimes.
- **Key settings:** `max_steps=150`, maker/taker fees enabled (`taker_fee=10 bps`, `maker_rebate=5 bps`), correlation-based edges between assets (use `correlation_matrix=[[1,0.6],[0.6,1]]`).
- **Models:** Graph-based observations via `MarketGraphEncoder` (GraphSAGE), decentralized PPO per role. Seeds fixed for reproducibility.
- **Artifacts:** Rewards CSV/plots, and reward-component breakdown (PnL delta, inventory penalty, liquidity bonus, unhedged penalty).

## What to Expect (Qualitative)
- **Maker behavior:** Spreads tighten in calm (normal) regimes and widen modestly in stress; inventory mean-reverts due to penalty; rebates improve net PnL on passive fills.
- **Arb behavior:** Opens long/short on perceived mispricings; inventory oscillates around zero; wider stress spreads plus correlation edges drive stronger pair trades; unhedged penalty limits runaway exposure.
- **Market quality:** Spread and depth stabilize as the maker learns; stress regimes cause transient spread spikes and depth drawdowns.
- **Learning curves:** Rewards trend upward over episodes for both agents; plateau once policies stabilize. Entropy drops gradually as policies commit.

## How to Reproduce
```bash
python scripts/run_multi_asset_gnn.py \
  --optional-flags-if-added
python scripts/analyze_results.py --csv artifacts/multi_asset_gnn/rewards.csv
```
(Flags may be added later; current script reads config inline.)

## Interpreting Outputs
- `rewards.csv`: Per-episode rewards by agent.
- `rewards.png`: Reward trends.
- Logs: Reward components per agent (`delta_mtm`, inventory penalty, liquidity bonus, unhedged penalty) to diagnose incentives.
- Inspect spreads vs. regime from log output to confirm widened spreads during stress.

## Next Steps to Improve Fidelity
- Enable FIFO queues and order latency to stress inventory management.
- Calibrate regimes/edge correlations from historical L1/L2 data.
- Add centralized critic (MAPPO-style) for improved sample efficiency.

## Comparison: Pure Synthetic vs. Data-Driven Hybrid (Run-to-Run Guide)
Use `scripts/analyze_results.py` on the artifacts after each run to populate your own numbers.

- **Option A: Pure on-the-fly synthetic (default small config)**
  - Run: `python scripts/run_multi_agent_basic.py` with `dataset_dir=None`
  - Expected: Highest variance; rewards may be unstable but fast to iterate.
- **Option B: Hybrid with data_big (replay + noise)**
  - Run: `python scripts/run_multi_agent_basic.py` (default now points to `data_big`)
  - Expected: More realistic spread/depth dynamics, clearer correlations; still some randomness for robustness.
- **Option C: Multi-asset GNN with data_big**
  - Run: `python scripts/run_multi_asset_gnn.py`
  - Expected: Cross-asset edges from `data_big/asset_graph_edges.csv`; arb agent should respond to synthetic correlations.

After each run:
```bash
python scripts/analyze_results.py --csv artifacts/multi_agent_basic/rewards.csv
python scripts/analyze_results.py --csv artifacts/multi_asset_gnn/rewards.csv
```
Inspect reward means/variance and plots to compare stability and learning progress across options. Replace these qualitative notes with your measured values once you run the experiments.

## Recent Results (data_big hybrid + on-the-fly)
Computed over 200 episodes using the new data-driven replay:
- `single_maker` — mean reward ≈ 14.4; last-10 mean ≈ 40.1 (modest positive trend for the maker alone).
- `multi_agent_basic` — maker mean ≈ 17.4, last-10 ≈ -23.3; taker mean ≈ -4.0, last-10 ≈ -4.5. Maker deteriorates late; taker remains negative.
- `multi_asset_gnn` — maker mean ≈ 8.5, last-10 ≈ 0.6; arb mean ≈ -12.8, last-10 ≈ -73.9. Arb is consistently losing and worsens late, indicating the current arb design/reward is not learning profitably.

Interpretation:
- Single-maker benefits from the replayed flow + noise, showing small positive drift.
- Maker+taker pairing still struggles: taker incentives likely need stronger completion bonuses or smaller penalties; maker destabilizes late.
- Multi-asset with arb is not learning useful cross-asset trades; arb reward/penalty balance and action space need revision.

Suggested next adjustments:
- Increase taker completion bonus further and reduce delay penalty; cap action sizes more aggressively.
- Add explicit pair-spread features for arb and trim unhedged penalties or give close-bonus for flattening.
- Consider centralized critic for multi-agent settings to stabilize credit assignment.

## Visuals
After running, generate plots:
```bash
python scripts/analyze_results.py --csv artifacts/multi_agent_basic/rewards.csv
python scripts/analyze_results.py --csv artifacts/multi_asset_gnn/rewards.csv
python scripts/analyze_results.py --csv artifacts/single_maker/rewards.csv
```
Attach/review the produced `analysis.png` and `rewards.png` for each run to visually assess trends and variance.

## Baseline Comparisons
For context, add simple baselines:
- Random actor: replace actions with random samples to establish a floor.
- Heuristic maker: fixed tight/wide spread policy with small inventory cap.
- Heuristic taker: VWAP-like slicing (equal parts over horizon).
Run the same scripts with these baselines and log their rewards to compare against the learned agents.

## Next Experiments (priority checklist)
1) Taker shaping: raise completion bonus, lower delay penalty, tighter size caps; re-run 200 episodes.
2) Arb signal: add explicit pair-spread feature (mid0-mid1) to observations (already partially present) and add close bonus; reduce unhedged penalty temporarily.
3) Centralized critic: enable MAPPO-style joint critic for maker+taker/arb to stabilize learning.
4) Longer horizons: try `max_steps` 300–400 with smaller LR to allow smoother learning.
5) Logging: log reward components per episode (delta_mtm, penalties, bonuses) into CSV for quicker diagnosis.

## Running Summary
- **Code:** PPO with advantage standardization, entropy decay, optional replay from `data_big` (depth profiles, orderflow, regimes), and correlation-based GNN edges.
- **Datasets:** Generated via `scripts/generate_synthetic_datasets.py`; large set in `data_big/`.
- **Interactive:** `streamlit run app.py` to browse rewards, datasets, and asset graph.
- **Artifacts:** Rewards CSVs/plots and policy checkpoints under `artifacts/`.
