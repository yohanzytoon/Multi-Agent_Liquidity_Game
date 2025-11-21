"""Arbitrage agent for cross-asset spread trading."""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn.functional as F

from ..utils.config import MarketEnvConfig
from .policies import ActorCriticPolicy, ArbitrageAction, PolicyInput


class ArbitrageAgent:
    """Simplified statistical arbitrage agent opening/closing pairs."""

    def __init__(
        self,
        assets: List[str],
        env_config: MarketEnvConfig,
        hidden_sizes: list[int] | None = None,
        device: str = "cpu",
        use_gnn: bool = True,
    ) -> None:
        if len(assets) < 2:
            raise ValueError("ArbitrageAgent requires at least two assets.")
        self.assets = assets[:2]
        self.device = device
        obs_dim = 5 + 5 * len(env_config.assets)
        self.policy = ActorCriticPolicy(
            obs_dim=obs_dim,
            action_dim=3,
            hidden_sizes=hidden_sizes,
            use_gnn=use_gnn,
            gnn_params={"input_dim": 8, "hidden_dim": 32, "num_layers": 2},
            device=device,
        ).to(device)
        self.size_scale = 5.0

    def act(self, obs: PolicyInput) -> Tuple[ArbitrageAction, torch.Tensor, torch.Tensor, torch.Tensor]:
        action, log_prob, value = self.policy.get_action(obs)
        action = action.detach()
        direction_signal = torch.tanh(action[0]).item()
        close_signal = torch.sigmoid(action[1]).item()
        size = torch.clamp(F.softplus(action[2]), max=3.0).item() * self.size_scale

        if close_signal > 0.7:
            arb_action = ArbitrageAction(
                long_asset=self.assets[0],
                short_asset=self.assets[1],
                size=0.0,
                close=True,
            )
        else:
            if direction_signal >= 0:
                long_asset, short_asset = self.assets[0], self.assets[1]
            else:
                long_asset, short_asset = self.assets[1], self.assets[0]
            arb_action = ArbitrageAction(
                long_asset=long_asset,
                short_asset=short_asset,
                size=size,
                close=False,
            )
        return arb_action, log_prob, value, action
