"""Market maker agent that translates policy outputs into quoting actions."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F

from ..utils.config import MarketEnvConfig
from .policies import ActorCriticPolicy, MakerAction, PolicyInput


class MakerAgent:
    """Wraps an actor-critic policy to generate maker actions."""

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
        # time + cash + remaining + inventory per asset + 4 per-asset market stats + 2 global extras
        obs_dim = 5 + 5 * len(env_config.assets)
        self.policy = ActorCriticPolicy(
            obs_dim=obs_dim,
            action_dim=3,
            hidden_sizes=hidden_sizes,
            use_gnn=use_gnn,
            gnn_params={"input_dim": 8, "hidden_dim": 32, "num_layers": 2},
            device=device,
        ).to(device)
        self.offset_scale = env_config.base_spread
        self.size_scale = 5.0

    def act(self, obs: PolicyInput) -> Tuple[MakerAction, torch.Tensor, torch.Tensor, torch.Tensor]:
        action, log_prob, value = self.policy.get_action(obs)
        action = action.detach()
        bid_offset = max(0.0, abs(action[0].item())) * self.offset_scale
        ask_offset = max(0.0, abs(action[1].item())) * self.offset_scale
        size = torch.clamp(F.softplus(action[2]), max=5.0).item() * self.size_scale
        maker_action = MakerAction(asset=self.asset, bid_offset=bid_offset, ask_offset=ask_offset, size=size)
        return maker_action, log_prob, value, action
