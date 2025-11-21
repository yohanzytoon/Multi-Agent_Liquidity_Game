"""Execution/taker agent that submits market orders or waits."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F

from ..utils.config import MarketEnvConfig
from .policies import ActorCriticPolicy, PolicyInput, TakerAction


class TakerAgent:
    """Agent focused on executing a target quantity efficiently."""

    def __init__(
        self,
        asset: str,
        env_config: MarketEnvConfig,
        hidden_sizes: list[int] | None = None,
        device: str = "cpu",
        use_gnn: bool = False,
    ) -> None:
        self.asset = asset
        self.device = device
        obs_dim = 5 + 5 * len(env_config.assets)
        self.policy = ActorCriticPolicy(
            obs_dim=obs_dim,
            action_dim=2,
            hidden_sizes=hidden_sizes,
            use_gnn=use_gnn,
            gnn_params={"input_dim": 8, "hidden_dim": 32, "num_layers": 2},
            device=device,
        ).to(device)
        self.size_scale = 8.0

    def act(self, obs: PolicyInput) -> Tuple[TakerAction, torch.Tensor, torch.Tensor, torch.Tensor]:
        action, log_prob, value = self.policy.get_action(obs)
        action = action.detach()
        signal = torch.tanh(action[0]).item()
        size = torch.clamp(F.softplus(action[1]), max=5.0).item() * self.size_scale
        if signal > 0.33:
            act = "buy"
        elif signal < -0.33:
            act = "sell"
        else:
            act = "wait"
        taker_action = TakerAction(asset=self.asset, action=act, size=size)
        return taker_action, log_prob, value, action
