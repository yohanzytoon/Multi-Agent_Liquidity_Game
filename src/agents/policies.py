"""Policy helpers and action dataclasses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Protocol, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch_geometric.data import Data

from ..models.gnn_encoder import MarketGraphEncoder


# ---------------------------------------------------------------------------#
# Action dataclasses                                                         #
# ---------------------------------------------------------------------------#
@dataclass
class MakerAction:
    asset: str
    bid_offset: float
    ask_offset: float
    size: float
    cancel_only: bool = False


@dataclass
class TakerAction:
    asset: str
    action: str  # "buy", "sell", or "wait"
    size: float


@dataclass
class ArbitrageAction:
    long_asset: str
    short_asset: str
    size: float
    close: bool = False


AgentPolicyOutput = MakerAction | TakerAction | ArbitrageAction


@dataclass
class PolicyInput:
    vector: np.ndarray
    graph: Optional[Data] = None


class PolicyNetwork(Protocol):
    def get_action(self, obs: PolicyInput) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ...

    def evaluate_actions(
        self, obs_batch: List[PolicyInput], actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ...


class ActorCriticPolicy(nn.Module):
    """
    Lightweight actor-critic policy supporting vector and optional graph inputs.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: Optional[List[int]] = None,
        use_gnn: bool = False,
        gnn_params: Optional[dict] = None,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.use_gnn = use_gnn
        self.gnn_encoder = MarketGraphEncoder(**gnn_params) if use_gnn else None
        hidden_sizes = hidden_sizes or [128, 64]
        self.device = torch.device(device)
        self.to(self.device)

        feature_dim = obs_dim + (self.gnn_encoder.output_dim if self.gnn_encoder else 0)
        layers: List[nn.Module] = []
        last = feature_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            last = h
        self.body = nn.Sequential(*layers)
        self.actor_mean = nn.Linear(last, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        self.critic = nn.Linear(last, 1)

    def _process_obs(self, obs: PolicyInput) -> torch.Tensor:
        vec = torch.as_tensor(obs.vector, dtype=torch.float32, device=self.device)
        if self.use_gnn and obs.graph is not None and self.gnn_encoder is not None:
            graph = obs.graph.to(self.device)
            g_feat = self.gnn_encoder(graph)
            vec = torch.cat([vec, g_feat], dim=-1)
        return vec

    def forward(self, obs: PolicyInput) -> Tuple[Normal, torch.Tensor]:
        feat = self.body(self._process_obs(obs))
        mean = self.actor_mean(feat)
        std = torch.exp(self.actor_log_std)
        dist = Normal(mean, std)
        value = self.critic(feat).squeeze(-1)
        return dist, value

    def value(self, obs: PolicyInput) -> torch.Tensor:
        """Compute critic value only."""
        _, value = self.forward(obs)
        return value

    def get_action(self, obs: PolicyInput) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, value = self.forward(obs)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob, value

    def evaluate_actions(
        self, obs_batch: List[PolicyInput], actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        log_probs = []
        entropies = []
        values = []
        for obs, action in zip(obs_batch, actions):
            dist, value = self.forward(obs)
            log_probs.append(dist.log_prob(action).sum(-1))
            entropies.append(dist.entropy().sum(-1))
            values.append(value)
        return torch.stack(log_probs), torch.stack(entropies), torch.stack(values)
