# Multi-Agent Liquidity Game (Research Sandbox)

This project is a self-contained research sandbox for studying market microstructure and multi-agent reinforcement learning. Multiple agent types (market makers, takers, and arbitrageurs) interact on synthetic limit order books across multiple assets/venues. Observations are built as graphs and encoded with a GNN, while policies are trained with a minimalist PPO implementation.

> **Disclaimer:** This code is for research and educational purposes only. It is highly simplified and **not** suitable for live trading or production deployment.

## Why This Is Interesting
- Captures **competition for liquidity** between heterogeneous agents.
- Simulates **cross-asset/venue relationships** that drive spreads and arbitrage opportunities.
- Uses **graph-based observations** instead of raw order events to make policies more structured and sample efficient.
- Provides a transparent **PPO baseline** for single- and multi-agent setups.

## Conceptual Design

### Microstructure & Order Book
- Each asset has a `LimitOrderBook` with aggregated price levels (no per-order IDs).
- Supports limit/market orders, partial fills, cancels, and top-of-book queries (best bid/ask, spread, depth arrays).
- Noise traders provide background order flow with configurable intensity, spread preference, and size distribution.

### Regimes
- Two built-in regimes: **normal** vs **stress**.
- Regime parameters: volatility of fundamentals, noise intensity, typical size, and spread preference.
- A regime is sampled at reset and can optionally switch mid-episode with a small probability.

### Agents & Roles
- **Market maker:** posts buy/sell quotes around the mid; rewarded for PnL and providing liquidity, penalized for inventory risk.
- **Execution taker:** must buy/sell a target quantity; rewarded for execution quality, penalized for delay/impact.
- **Arbitrageur:** trades pairs; rewarded for cross-asset PnL, penalized for unhedged exposure and leverage.

### Graph-Based Observation
- **Nodes:** one per asset (or venue-asset). Features include normalized midprice, spread, best bid/ask size, average depth, realized short-term volatility, order-flow imbalance, and a stress indicator.
- **Edges:** fully connected by default; edge attributes can be set to empirical correlations via `MarketEnvConfig.correlation_matrix` (fallback to ones if not provided).
- **Encoder:** `MarketGraphEncoder` (GraphSAGE) pools node embeddings into a global representation that feeds actor-critic policies.

### Reinforcement Learning Setup
- **Non-stationarity:** Each agent changes the environment for others; policies are trained independently (decentralized) but can share architectures.
- **PPO overview:** Rollouts collect `(obs, action, log_prob, reward, value)`; advantages via GAE; clipped objective with entropy bonus and value loss.
- **Policy organization:** One PPO instance per agent type in the provided scripts. Parameter sharing is possible by reusing the same policy object. Advantages are standardized and entropy decays over training for stability.

#### Reward Functions (per step, simplified)
- `ΔMTM = (cash + Σ inventory * mid) - last_mtm`
- **Maker:** `reward = ΔMTM - λ_inv * ||inventory||² + λ_liquidity * presence`
- **Taker:** `reward = ΔMTM - λ_inv * ||inventory||² - λ_delay * |remaining_target|`
- **Arb:** `reward = ΔMTM - λ_inv * ||inventory||² - λ_unhedged * Σ|inventory|`

## Repository Structure
```
├─ requirements.txt                # Core dependencies (PyTorch, PyG, etc.)
├─ pyproject.toml                  # Minimal packaging metadata
├─ src/
│  ├─ __init__.py
│  ├─ envs/
│  │  ├─ __init__.py
│  │  ├─ order_book.py             # Limit order book implementation
│  │  ├─ market_env.py             # Multi-asset environment + noise flow
│  │  └─ regimes.py                # Regime sampling/transition logic
│  ├─ agents/
│  │  ├─ __init__.py
│  │  ├─ policies.py               # Action dataclasses + actor-critic wrapper
│  │  ├─ maker_agent.py            # Market maker action mapping
│  │  ├─ taker_agent.py            # Execution agent action mapping
│  │  └─ arb_agent.py              # Cross-asset arbitrage action mapping
│  ├─ models/
│  │  ├─ __init__.py
│  │  ├─ gnn_encoder.py            # GraphSAGE encoder
│  │  └─ ppo.py                    # Minimal PPO trainer + rollout buffer
│  └─ utils/
│     ├─ __init__.py
│     ├─ config.py                 # Config dataclasses
│     ├─ logging_utils.py          # Logger factory
│     └─ plotting.py               # Quick plotting helpers
├─ scripts/
│  ├─ run_single_agent_maker.py    # Maker vs noise traders
│  ├─ run_multi_agent_basic.py     # Maker + taker interaction
│  ├─ run_multi_asset_gnn.py       # Multi-asset maker + arb with GNN
│  └─ analyze_results.py           # Plot rewards from saved CSVs
└─ tests/
   ├─ test_order_book.py           # Order book sanity checks
   └─ test_env_basic.py            # Env reset/step smoke test
```

## How the Environment Works

1. **Reset:** Clear books, sample a regime, seed symmetric depth around a fundamental price, reset agent inventories/cash and targets.
2. **Observe:** Each agent receives `PolicyInput` containing:
   - `vector` features: normalized time, cash, remaining target, inventory by asset, and basic per-asset market stats (mid, spread, best bid/ask).
   - `graph` features: `torch_geometric.data.Data` built from node/edge features.
3. **Act:** Agents return typed actions (`MakerAction`, `TakerAction`, `ArbitrageAction`).
4. **Match:** Actions are applied to the limit books; trades update inventories/cash. Noise traders add extra limit/market orders.
5. **Reward:** Mark-to-market PnL change minus role-specific penalties/bonuses.
6. **Terminate:** Episode ends at `max_steps`; `done["__all__"]` propagates to all agents.

### Order Book Simplifications
- Aggregated by price level (no FIFO).
- Maker resting depth is tracked separately to attribute fills from aggressors and noise.
- Limit orders that cross the spread trade immediately with residual resting depth.

### Regime Modeling
- `normal`: lower volatility, tighter default spreads.
- `stress`: higher volatility, larger order sizes, wider spreads.
- Switching probability is configurable; include the regime flag in node features for learnability.

### Fees / Rebates
- Passive makers receive a configurable rebate (`maker_rebate`), aggressors pay a taker fee (`taker_fee`). Both apply as fractions of notional on fills, and are logged in reward components.

## Running the Examples

### Do I Need Data?
- No external data is required. The environment generates synthetic fundamentals, order flow, and regimes on the fly.
- If you want to calibrate to real markets, you can:
  - Pull public historical L1/L2 data (e.g., from Polygon, Tiingo, Binance, Coinbase, or academic limit order book datasets).
  - Estimate spread/vol/size distributions and correlation matrices offline, then pass them as custom `RegimeConfig` lists and edge features in `MarketEnvConfig`.
  - Replace the noise-flow generator with a replay sampler that injects historical flow into the books.

### 1) Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
# PyTorch Geometric wheels depend on platform; see https://pytorch-geometric.readthedocs.io/
```

### 2) Single Maker vs Noise
```bash
python scripts/run_single_agent_maker.py
```
Outputs: `artifacts/single_maker/` with `rewards.csv`, `rewards.png`, and `maker_policy.pt`.

### 3) Maker + Taker Interaction
```bash
python scripts/run_multi_agent_basic.py
```
Outputs: `artifacts/multi_agent_basic/` with reward CSV/plot and both policies.

### 4) Multi-Asset with GNN Encoder (Maker + Arb)
```bash
python scripts/run_multi_asset_gnn.py
```
Outputs: `artifacts/multi_asset_gnn/` including GNN-trained policies.

### 5) Analyze Logged Rewards
```bash
python scripts/analyze_results.py --csv artifacts/single_maker/rewards.csv
```
Produces an aggregated plot and summary stats.

### 6) Run Tests
```bash
pytest -q
```

## Interpreting Results
- **Reward curves:** Upward trends suggest learning; plateauing may indicate convergence. Compare across agents to see co-adaptation.
- **Spreads/depth:** Makers should tighten spreads in calm regimes and widen in stress; depth should grow where market impact is low.
- **Taker behaviour:** Remaining target should decay smoothly; spikes imply hesitance or over-aggression.
- **Arbitrage:** Positions should oscillate around zero; persistent inventory hints at slippage or mis-specified penalties.
- **GNN diagnostics:** Node inputs encode mid/spread/depth/vol/imbalance. Swap `MarketGraphEncoder` for attention-based layers (e.g., GAT) to inspect attention weights.

## Implementation Notes
- **Central vs. decentral training:** Provided scripts use decentralized PPO (one trainer per role). Centralized critics or parameter sharing can be added by reusing policy objects across agents.
- **Episode flow:** Agents act → books update → noise flow hits → fundamentals evolve → rewards computed → observations rebuilt.
- **Agent action mapping:**
  - Maker policy outputs `(bid_offset, ask_offset, size)`; offsets are scaled by the configured spread.
  - Taker policy outputs `(direction_signal, size)`; direction is discretized to buy/sell/wait.
  - Arb policy outputs `(direction_signal, close_signal, size)`; direction sets long/short ordering of the asset pair.

## Extending the Project
1. **Microstructure realism:** Add latency, queue priority, per-order IDs, fees/rebates, and impact models.
2. **Regimes:** Implement richer scenarios (flash crash, liquidity drought, correlated shocks).
3. **Agents:** Add alpha-seeking trend followers or VWAP/TWAP style execution agents.
4. **Algorithms:** Swap PPO for MADDPG, QMIX, or IPPO; integrate curriculum learning over regimes.
5. **Analytics:** Record full depth snapshots, slippage metrics, and visualize GNN attention/importance.
6. **Calibration:** Fit noise trader parameters to real data or synthetic stylized facts (e.g., heavy tails, clustered volatility).

## Caveats & Simplifications
- No explicit latency, clearing, fees, or order identifiers.
- Inventory/impact penalties are stylized; reward shaping may need tuning for stability.
- Correlations for edges are placeholders (fully connected with unit edge features).
- PPO implementation is intentionally minimal; for scale, incorporate batching, vectorized environments, and checkpointing.

## Troubleshooting
- **PyG install issues:** Use the official wheel selector for your platform/CUDA version.
- **Exploding losses:** Lower `learning_rate`, increase `value_coef`, or clip gradients (already enabled).
- **Degenerate books:** Increase `base_spread` or noise intensity to keep depth populated at reset.

## How to Contribute/Modify
- Adjust defaults in `src/utils/config.py`.
- Extend node/edge features and update `gnn_encoder.py` to try different GNN layers.
- Modify reward coefficients in `MarketEnvConfig` to change incentives.
- Replace action mappings in agents to expand action spaces (e.g., multi-asset quoting).
