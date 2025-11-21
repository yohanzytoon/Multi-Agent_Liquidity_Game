"""Configuration dataclasses used across the project."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class OrderBookConfig:
    tick_size: float = 0.01
    depth_levels: int = 5


@dataclass
class RegimeConfig:
    """Defines volatility/stress regime parameters."""

    name: str
    volatility: float
    order_size_mean: float
    noise_intensity: float
    spread_preference: float


@dataclass
class MarketEnvConfig:
    assets: List[str] = field(default_factory=lambda: ["ASSET_A"])
    max_steps: int = 200
    lookback: int = 20
    dataset_dir: Optional[str] = None  # path containing synthetic CSVs (depth profiles, orderflow, regimes)
    replay_flow: bool = True  # whether to replay events from dataset_dir if available
    replay_mix: float = 0.5  # fraction of background events from replay vs noise
    order_book: OrderBookConfig = field(default_factory=OrderBookConfig)
    regimes: Optional[List[RegimeConfig]] = None
    regime_switch_prob: float = 0.0
    initial_cash: float = 1_000_000.0
    maker_liquidity_penalty: float = 1e-4
    inventory_penalty: float = 1e-5
    taker_impact_penalty: float = 1e-4
    taker_delay_penalty: float = 5e-3
    arb_unhedged_penalty: float = 1e-4
    arb_leverage_penalty: float = 1e-4
    base_spread: float = 0.01
    maker_rebate: float = 0.00005  # fraction of notional received by passive maker
    taker_fee: float = 0.0001  # fraction of notional paid by aggressor
    correlation_matrix: Optional[List[List[float]]] = None
    reward_scale: float = 0.01  # global reward scaling to stabilize critic targets
    taker_completion_bonus: float = 0.3  # bonus per unit of target reduced
    log_dir: str = "logs"


@dataclass
class PPOConfig:
    learning_rate: float = 5e-5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.12
    entropy_coef: float = 0.005
    entropy_decay: float = 0.995
    entropy_min: float = 0.0005
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    update_epochs: int = 5
    batch_size: int = 64
    rollout_length: int = 512


@dataclass
class TrainingConfig:
    episodes: int = 10
    seed: int = 123
    log_interval: int = 1
    save_path: str = "artifacts"
    device: str = "cpu"
    extra_params: Dict[str, float] = field(default_factory=dict)
