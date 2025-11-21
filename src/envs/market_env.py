"""Multi-asset / multi-venue market environment with microstructure dynamics."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import Data

from pathlib import Path

from ..agents.policies import (
    AgentPolicyOutput,
    ArbitrageAction,
    MakerAction,
    PolicyInput,
    TakerAction,
)
from ..utils.config import MarketEnvConfig, RegimeConfig
from ..utils.logging_utils import get_logger
from ..utils import (
    load_depth_profiles,
    load_orderflow_events,
    load_regime_series,
)

from .order_book import LimitOrderBook
from .regimes import RegimeManager


@dataclass
class AgentState:
    role: str
    inventory: Dict[str, float]
    cash: float
    pnl: float = 0.0
    target_inventory: float = 0.0
    last_mtm: float = 0.0
    remaining_target: float = 0.0
    last_remaining_target: float = 0.0


class MultiAssetMarketEnv:
    """
    Synthetic multi-asset microstructure environment.
    """

    def __init__(self, config: MarketEnvConfig, seed: int = 123) -> None:
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.logger = get_logger("market_env", config.log_dir)

        self.books: Dict[str, LimitOrderBook] = {
            asset: LimitOrderBook(
                tick_size=config.order_book.tick_size,
                depth_levels=config.order_book.depth_levels,
            )
            for asset in config.assets
        }
        self.fundamental: Dict[str, float] = {asset: 100.0 for asset in config.assets}
        self.price_history: Dict[str, List[float]] = {asset: [] for asset in config.assets}
        self.order_flow_history: Dict[str, List[int]] = {asset: [] for asset in config.assets}
        self.regime_manager = RegimeManager(config.regimes, config.regime_switch_prob, self.rng)
        self.correlation_matrix = (
            np.array(config.correlation_matrix) if config.correlation_matrix is not None else None
        )
        # Dataset-backed calibration
        self.depth_df = None
        self.events_df = None
        self.regime_df = None
        if self.config.dataset_dir:
            dpath = Path(self.config.dataset_dir)
            self.depth_df = load_depth_profiles(dpath / "synthetic_depth_profiles.csv")
            self.events_df = load_orderflow_events(dpath / "synthetic_orderflow_events.csv")
            self.regime_df = load_regime_series(dpath / "regimes.csv")
            if self.depth_df is not None:
                self.logger.info("Loaded depth profiles from %s", dpath)
            if self.events_df is not None:
                self.logger.info("Loaded orderflow events from %s", dpath)
            if self.regime_df is not None:
                self.logger.info("Loaded regime series from %s", dpath)

        self.agent_states: Dict[str, AgentState] = {}
        self.maker_outstanding: Dict[str, Dict[str, Dict[float, float]]] = {}
        self.time_step: int = 0
        self.done: bool = False
        self._event_pointer: int = 0
        self._event_start_ts: int = 0

    # ----------------------------------------------------------------------#
    # Lifecycle                                                             #
    # ----------------------------------------------------------------------#
    def register_agent(self, agent_id: str, role: str, target_inventory: float = 0.0) -> None:
        self.agent_states[agent_id] = AgentState(
            role=role,
            inventory={asset: 0.0 for asset in self.config.assets},
            cash=self.config.initial_cash,
            target_inventory=target_inventory,
            remaining_target=target_inventory,
            last_remaining_target=abs(target_inventory),
            last_mtm=self.config.initial_cash,
        )
        if role == "maker":
            self.maker_outstanding[agent_id] = {asset: {} for asset in self.config.assets}

    def reset(self) -> Dict[str, PolicyInput]:
        self.logger.info("Resetting environment")
        self.done = False
        self.time_step = 0
        for book in self.books.values():
            book.reset()
        self.regime_manager.sample()

        for asset in self.config.assets:
            if self.depth_df is not None and len(self.depth_df) > 0:
                row = self.depth_df.sample(1).iloc[0]
                start_price = float(row["midprice"])
                spread = float(row["spread"])
                self.fundamental[asset] = start_price
                self.price_history[asset] = [start_price]
                self.order_flow_history[asset] = []
                self._seed_from_profile(asset, start_price, spread, row)
            else:
                start_price = 100.0 + self.rng.normal(0, 0.5)
                self.fundamental[asset] = start_price
                self.price_history[asset] = [start_price]
                self.order_flow_history[asset] = []
                self._seed_liquidity(asset)

        for agent_id, state in self.agent_states.items():
            state.inventory = {asset: 0.0 for asset in self.config.assets}
            state.cash = self.config.initial_cash
            state.pnl = 0.0
            state.remaining_target = state.target_inventory
            state.last_remaining_target = abs(state.target_inventory)
            state.last_mtm = self.config.initial_cash
            if state.role == "maker":
                self.maker_outstanding[agent_id] = {asset: {} for asset in self.config.assets}

        if self.regime_df is not None and len(self.regime_df) > 0:
            row = self.regime_df.sample(1).iloc[0]
            # Override current regime based on sample
            self.regime_manager.current = RegimeConfig(
                name=row["inferred_regime"],
                volatility=float(row["vol"]),
                order_size_mean=100.0,
                noise_intensity=1.0,
                spread_preference=float(row["spread"]),
            )

        # Reset event pointer for replay
        if self.events_df is not None and len(self.events_df) > 0:
            self._event_pointer = int(self.rng.integers(0, len(self.events_df) // 2))
            self._event_start_ts = int(self.events_df.iloc[self._event_pointer]["timestamp"])

        return self._build_all_observations()

    # ----------------------------------------------------------------------#
    # Core step                                                             #
    # ----------------------------------------------------------------------#
    def step(
        self, actions: Dict[str, AgentPolicyOutput]
    ) -> Tuple[Dict[str, PolicyInput], Dict[str, float], Dict[str, bool], Dict[str, dict]]:
        if self.done:
            raise RuntimeError("Environment already terminated. Call reset().")

        self.regime_manager.maybe_switch()
        # Apply agent actions
        for agent_id, action in actions.items():
            self._apply_action(agent_id, action)

        # Simulate noise traders
        self._simulate_noise_flow()

        # Update fundamentals and mark-to-market
        self._update_fundamentals()

        rewards, info = self._compute_rewards()
        observations = self._build_all_observations()

        self.time_step += 1
        self.done = self.time_step >= self.config.max_steps
        dones = {agent_id: self.done for agent_id in self.agent_states}
        dones["__all__"] = self.done

        return observations, rewards, dones, info

    # ----------------------------------------------------------------------#
    # Actions                                                               #
    # ----------------------------------------------------------------------#
    def _apply_action(self, agent_id: str, action: AgentPolicyOutput) -> None:
        state = self.agent_states[agent_id]
        if isinstance(action, MakerAction):
            self._apply_maker_action(agent_id, action)
        elif isinstance(action, TakerAction):
            self._apply_taker_action(agent_id, action)
        elif isinstance(action, ArbitrageAction):
            self._apply_arb_action(agent_id, action)
        else:
            self.logger.warning("Unknown action type from %s", agent_id)

    def _apply_maker_action(self, agent_id: str, action: MakerAction) -> None:
        book = self.books[action.asset]
        mid = book.get_midprice() or self.fundamental[action.asset]
        # Cancel previous quotes from this maker on this asset
        outstanding = self.maker_outstanding.get(agent_id, {}).get(action.asset, {})
        for price, size in list(outstanding.items()):
            book.cancel("buy" if price < mid else "sell", price, size=size)
        self.maker_outstanding[agent_id][action.asset] = {}

        if action.cancel_only:
            return

        bid_price = max(0.01, mid - abs(action.bid_offset))
        ask_price = mid + abs(action.ask_offset)
        size = max(1e-3, action.size)

        # Add bid
        bid_fills = book.add_limit_order("buy", bid_price, size)
        bid_residual = size - sum(sz for _, sz in bid_fills)
        if bid_residual > 0:
            self.maker_outstanding[agent_id][action.asset][bid_price] = bid_residual
        # Add ask
        ask_fills = book.add_limit_order("sell", ask_price, size)
        ask_residual = size - sum(sz for _, sz in ask_fills)
        if ask_residual > 0:
            self.maker_outstanding[agent_id][action.asset][ask_price] = ask_residual

        # Update maker PnL for aggressive fills
        for price, qty in bid_fills:
            self._update_inventory_cash(agent_id, action.asset, "buy", price, qty, fee_rate=self.config.taker_fee)
        for price, qty in ask_fills:
            self._update_inventory_cash(agent_id, action.asset, "sell", price, qty, fee_rate=self.config.taker_fee)

    def _apply_taker_action(self, agent_id: str, action: TakerAction) -> None:
        if action.action == "wait" or action.size <= 0:
            return
        book = self.books[action.asset]
        size = max(1e-3, action.size)
        fills = book.add_market_order(action.action, size)
        self._allocate_fills(agent_id, action.asset, action.action, fills)
        # Track progress toward execution objective
        state = self.agent_states[agent_id]
        if action.action == "buy":
            state.remaining_target -= size
        elif action.action == "sell":
            state.remaining_target += size

    def _apply_arb_action(self, agent_id: str, action: ArbitrageAction) -> None:
        # Open or close a simple long-short pair
        if action.close:
            # Flatten inventory on both assets
            state = self.agent_states[agent_id]
            for asset, inv in state.inventory.items():
                if abs(inv) < 1e-6:
                    continue
                side = "sell" if inv > 0 else "buy"
                fills = self.books[asset].add_market_order(side, abs(inv))
                self._allocate_fills(agent_id, asset, side, fills)
            return

        size = max(1e-3, action.size)
        if action.long_asset == action.short_asset:
            return
        long_fills = self.books[action.long_asset].add_market_order("buy", size)
        short_fills = self.books[action.short_asset].add_market_order("sell", size)
        self._allocate_fills(agent_id, action.long_asset, "buy", long_fills)
        self._allocate_fills(agent_id, action.short_asset, "sell", short_fills)

    def _apply_replay_event(self, ev) -> None:
        asset = ev["asset"]
        if asset not in self.books:
            return
        etype = ev["type"]
        side = ev.get("side", "buy")
        book = self.books[asset]
        if etype == "LIMIT_ADD":
            book.add_limit_order(side, float(ev["price"]), float(ev["size"]))
        elif etype == "LIMIT_CANCEL":
            book.cancel(side, float(ev["price"]), size=float(ev["size"]))
        elif etype == "MARKET_ORDER":
            fills = book.add_market_order(side, float(ev["size"]))
            self._allocate_fills(None, asset, side, fills)
        elif etype == "REGIME_SWITCH" and self.regime_df is not None:
            name = ev.get("regime", "normal")
            self.regime_manager.current = RegimeConfig(
                name=name,
                volatility=self.regime_manager.current.volatility,
                order_size_mean=self.regime_manager.current.order_size_mean,
                noise_intensity=self.regime_manager.current.noise_intensity,
                spread_preference=self.regime_manager.current.spread_preference,
            )

    # ----------------------------------------------------------------------#
    # Noise and fundamentals                                                #
    # ----------------------------------------------------------------------#
    def _simulate_noise_flow(self) -> None:
        regime = self.regime_manager.current
        for asset, book in self.books.items():
            # Mix replayed events with synthetic noise
            use_replay = self.config.replay_flow and self.events_df is not None and len(self.events_df) > 0
            mix = self.config.replay_mix if use_replay else 0.0
            # Replay events whose timestamp is within current step window
            if use_replay:
                current_ts = self._event_start_ts + self.time_step
                while self._event_pointer < len(self.events_df):
                    ev = self.events_df.iloc[self._event_pointer]
                    if ev["timestamp"] > current_ts:
                        break
                    self._apply_replay_event(ev)
                    self._event_pointer += 1

            n_events = self.rng.poisson(lam=max(1e-3, regime.noise_intensity))
            for _ in range(n_events):
                if use_replay and self.rng.random() < mix:
                    # pull a random event near pointer
                    idx = int(self.rng.integers(max(0, self._event_pointer - 10), min(len(self.events_df), self._event_pointer + 10)))
                    ev = self.events_df.iloc[idx]
                    self._apply_replay_event(ev)
                    continue

                size = max(1e-3, self.rng.normal(regime.order_size_mean, regime.order_size_mean * 0.25))
                if self.rng.random() < 0.5:  # market order
                    side = "buy" if self.rng.random() < 0.5 else "sell"
                    fills = book.add_market_order(side, size)
                    self._allocate_fills(None, asset, side, fills)
                    self._record_order_flow(asset, side, size)
                else:  # limit order
                    direction = 1 if self.rng.random() < 0.5 else -1
                    price = self.fundamental[asset] + direction * (
                        self.config.base_spread * regime.spread_preference * self.rng.random()
                    )
                    side = "buy" if direction < 0 else "sell"
                    book.add_limit_order(side, price, size)

    def _update_fundamentals(self) -> None:
        regime = self.regime_manager.current
        for asset in self.config.assets:
            drift = 0.0
            vol = regime.volatility
            dt = 1.0
            shock = self.rng.normal(0, math.sqrt(dt))
            self.fundamental[asset] *= math.exp((drift - 0.5 * vol**2) * dt + vol * shock)
            mid = self.books[asset].get_midprice() or self.fundamental[asset]
            self.price_history[asset].append(mid)

    # ----------------------------------------------------------------------#
    # Rewards and observations                                              #
    # ----------------------------------------------------------------------#
    def _compute_rewards(self) -> Tuple[Dict[str, float], Dict[str, dict]]:
        rewards: Dict[str, float] = {}
        info: Dict[str, dict] = {}
        for agent_id, state in self.agent_states.items():
            mtm = self._mark_to_market(state)
            delta_mtm = mtm - state.last_mtm
            state.last_mtm = mtm

            inv_penalty = self.config.inventory_penalty * sum(v * v for v in state.inventory.values())
            reward = delta_mtm - inv_penalty
            components = {"delta_mtm": delta_mtm, "inventory_penalty": inv_penalty}

            if state.role == "maker":
                presence = self._maker_presence(agent_id)
                bonus = presence * self.config.maker_liquidity_penalty
                reward += bonus
                components["liquidity_bonus"] = bonus
            elif state.role == "taker":
                delay_penalty = self.config.taker_delay_penalty * abs(state.remaining_target)
                reward -= delay_penalty
                components["delay_penalty"] = delay_penalty
                # Progress bonus for reducing remaining target
                prev_remaining = state.last_remaining_target
                curr_remaining = abs(state.remaining_target)
                progress = max(0.0, prev_remaining - curr_remaining)
                completion_bonus = self.config.taker_completion_bonus * progress
                reward += completion_bonus
                components["completion_bonus"] = completion_bonus
                state.last_remaining_target = curr_remaining
            elif state.role == "arb":
                unhedged = sum(abs(v) for v in state.inventory.values())
                unhedged_pen = self.config.arb_unhedged_penalty * unhedged
                reward -= unhedged_pen
                components["unhedged_penalty"] = unhedged_pen
            # Global reward scaling for stability
            reward_scaled = reward * self.config.reward_scale
            for key in components:
                components[key] *= self.config.reward_scale

            rewards[agent_id] = reward_scaled
            info[agent_id] = components
            state.pnl += reward
        return rewards, info

    def _build_all_observations(self) -> Dict[str, PolicyInput]:
        graph = self._build_graph()
        observations: Dict[str, PolicyInput] = {}
        for agent_id, state in self.agent_states.items():
            vec = self._build_vector_obs(state)
            observations[agent_id] = PolicyInput(vector=vec, graph=graph)
        return observations

    def _build_vector_obs(self, state: AgentState) -> np.ndarray:
        time_feature = self.time_step / max(1, self.config.max_steps)
        market_features: List[float] = []
        mids = []
        for asset in self.config.assets:
            mid = self.books[asset].get_midprice() or self.fundamental[asset]
            mids.append(mid)
            spread = self.books[asset].get_spread() or self.config.base_spread
            bid = self.books[asset].get_best_bid() or 0.0
            ask = self.books[asset].get_best_ask() or 0.0
            mid_norm = mid / 100.0
            market_features.extend([mid_norm, spread, bid, ask])
        inv = [state.inventory[a] for a in self.config.assets]
        global_mid_mean = float(np.mean(mids)) if mids else 0.0
        pair_spread = mids[0] - mids[1] if len(mids) >= 2 else 0.0
        vec = np.array(
            [
                time_feature,
                state.cash / (self.config.initial_cash + 1e-6),
                state.remaining_target,
                *inv,
                *market_features,
                global_mid_mean / 100.0,
                pair_spread / 100.0,
            ],
            dtype=np.float32,
        )
        return vec

    def _build_graph(self) -> Data:
        node_features: List[List[float]] = []
        for asset in self.config.assets:
            mid = self.books[asset].get_midprice() or self.fundamental[asset]
            spread = self.books[asset].get_spread() or self.config.base_spread
            best_bid_size = self.books[asset].bids.peekitem(0)[1] if len(self.books[asset].bids) else 0.0
            best_ask_size = self.books[asset].asks.peekitem(0)[1] if len(self.books[asset].asks) else 0.0
            depth_prices, depth_sizes = self.books[asset].depth_arrays(self.config.order_book.depth_levels)
            avg_depth = float(np.mean(depth_sizes)) if len(depth_sizes) else 0.0
            vol = self._realized_vol(asset)
            imbalance = self._order_imbalance(asset)
            regime_flag = 1.0 if self.regime_manager.current.name == "stress" else 0.0
            node_features.append(
                [
                    mid / 100.0,
                    spread,
                    best_bid_size,
                    best_ask_size,
                    avg_depth,
                    vol,
                    imbalance,
                    regime_flag,
                ]
            )
        num_nodes = len(self.config.assets)
        # Fully connected graph (excluding self)
        edges = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i == j:
                    continue
                edges.append((i, j))

        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            if self.correlation_matrix is not None and self.correlation_matrix.shape == (num_nodes, num_nodes):
                attrs = [self.correlation_matrix[i, j] for i, j in edges]
                edge_attr = torch.tensor(attrs, dtype=torch.float).unsqueeze(-1)
            else:
                edge_attr = torch.ones((edge_index.shape[1], 1), dtype=torch.float)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 1), dtype=torch.float)
        x = torch.tensor(node_features, dtype=torch.float)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    def _realized_vol(self, asset: str) -> float:
        prices = self.price_history[asset]
        if len(prices) < 2:
            return 0.0
        returns = np.diff(prices[-self.config.lookback :])
        return float(np.std(returns)) if len(returns) > 0 else 0.0

    def _order_imbalance(self, asset: str) -> float:
        history = self.order_flow_history[asset][-self.config.lookback :]
        if not history:
            return 0.0
        buys = sum(1 for h in history if h > 0)
        sells = sum(1 for h in history if h < 0)
        total = buys + sells
        return (buys - sells) / total if total > 0 else 0.0

    # ----------------------------------------------------------------------#
    # Helpers                                                               #
    # ----------------------------------------------------------------------#
    def _allocate_fills(
        self,
        aggressor: Optional[str],
        asset: str,
        side: str,
        fills: List[Tuple[float, float]],
    ) -> None:
        """
        Update agent states given fills. Aggressor is the agent sending the market order.
        Makers who have resting liquidity at the fill price are allocated first-come.
        """
        remaining_sizes = [qty for _, qty in fills]
        # Aggressor trades first
        if aggressor is not None:
            for (price, qty) in fills:
                self._update_inventory_cash(
                    aggressor,
                    asset,
                    side,
                    price,
                    qty,
                    fee_rate=self.config.taker_fee,
                )
        # Allocate to makers
        for maker_id, outstanding_per_asset in self.maker_outstanding.items():
            outstanding = outstanding_per_asset.get(asset, {})
            to_delete = []
            for i, (price, qty) in enumerate(fills):
                if remaining_sizes[i] <= 0:
                    continue
                maker_qty = outstanding.get(price, 0.0)
                if maker_qty <= 0:
                    continue
                traded = min(remaining_sizes[i], maker_qty)
                maker_side = "sell" if side == "buy" else "buy"
                self._update_inventory_cash(
                    maker_id,
                    asset,
                    maker_side,
                    price,
                    traded,
                    fee_rate=-self.config.maker_rebate,
                )
                outstanding[price] -= traded
                remaining_sizes[i] -= traded
                if outstanding[price] <= 1e-9:
                    to_delete.append(price)
            for price in to_delete:
                del outstanding[price]

    def _update_inventory_cash(
        self, agent_id: str, asset: str, side: str, price: float, qty: float, fee_rate: float = 0.0
    ) -> None:
        state = self.agent_states[agent_id]
        notional = price * qty
        if side == "buy":
            state.inventory[asset] += qty
            state.cash -= notional
        else:
            state.inventory[asset] -= qty
            state.cash += notional
        # Positive fee_rate reduces cash; negative fee_rate adds rebate
        state.cash -= fee_rate * notional

    def _record_order_flow(self, asset: str, side: str, size: float) -> None:
        signed = size if side == "buy" else -size
        self.order_flow_history[asset].append(int(np.sign(signed)))

    def _mark_to_market(self, state: AgentState) -> float:
        value = state.cash
        for asset, inv in state.inventory.items():
            mid = self.books[asset].get_midprice() or self.fundamental[asset]
            value += inv * mid
        return value

    def _maker_presence(self, agent_id: str) -> float:
        outstanding = self.maker_outstanding.get(agent_id, {})
        presence = 0.0
        for asset_data in outstanding.values():
            if len(asset_data) >= 2:
                presence += 1.0
        return presence

    def _seed_liquidity(self, asset: str) -> None:
        """
        Add initial symmetric book depth around the fundamental price.
        """
        book = self.books[asset]
        mid = self.fundamental[asset]
        for level in range(1, self.config.order_book.depth_levels + 1):
            offset = self.config.base_spread * level
            size = 50.0 / level
            book.add_limit_order("buy", mid - offset, size)
            book.add_limit_order("sell", mid + offset, size)

    def _seed_from_profile(self, asset: str, mid: float, spread: float, row) -> None:
        """
        Seed order book from a depth profile row.
        """
        book = self.books[asset]
        for level in range(5):
            offset = spread * (level + 1)
            bid_size = float(row.get(f"bid_size_{level}", 0.0))
            ask_size = float(row.get(f"ask_size_{level}", 0.0))
            if bid_size > 0:
                book.add_limit_order("buy", mid - offset, bid_size)
            if ask_size > 0:
                book.add_limit_order("sell", mid + offset, ask_size)
