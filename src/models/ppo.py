"""Minimal PPO implementation supporting both single and multi-agent usage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from torch import optim
import torch.nn as nn

from ..utils.config import PPOConfig

if TYPE_CHECKING:
    from ..agents.policies import PolicyInput, PolicyNetwork
else:
    PolicyInput = Any
    PolicyNetwork = Any


class CentralizedValue(nn.Module):
    """
    Simple centralized value network (MAPPO-style) that can consume concatenated observations
    from multiple agents. This is a scaffold; caller is responsible for constructing joint inputs.
    """

    def __init__(self, input_dim: int, hidden_sizes: Optional[List[int]] = None) -> None:
        super().__init__()
        hidden_sizes = hidden_sizes or [256, 128]
        layers = []
        last = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            last = h
        layers.append(nn.Linear(last, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).squeeze(-1)


@dataclass
class RolloutBuffer:
    observations: List[PolicyInput]
    actions: List[torch.Tensor]
    log_probs: List[torch.Tensor]
    rewards: List[float]
    dones: List[bool]
    values: List[torch.Tensor]
    advantages: Optional[torch.Tensor] = None
    returns: Optional[torch.Tensor] = None

    def compute_advantages(self, config: PPOConfig, last_value: float = 0.0, device: str = "cpu") -> None:
        gae = 0.0
        advantages = []
        values = torch.stack(self.values + [torch.tensor(last_value, device=device)])
        for step in reversed(range(len(self.rewards))):
            reward_t = torch.tensor(self.rewards[step], device=device)
            mask = 1 - float(self.dones[step])
            delta = reward_t + config.gamma * values[step + 1] * mask - values[step]
            gae = delta + config.gamma * config.gae_lambda * (1 - float(self.dones[step])) * gae
            advantages.insert(0, gae)
        adv_tensor = torch.tensor(advantages, dtype=torch.float32, device=device)
        # Advantage standardization for stability
        adv_mean = adv_tensor.mean()
        adv_std = adv_tensor.std(unbiased=False) + 1e-8
        adv_tensor = (adv_tensor - adv_mean) / adv_std
        self.advantages = adv_tensor
        self.returns = self.advantages + values[:-1]

    def clear(self) -> None:
        self.observations.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()
        self.advantages = None
        self.returns = None


class PPOTrainer:
    """
    Minimal PPO trainer operating on a given policy and rollout buffer.
    Supports entropy decay and advantage standardization.
    """

    def __init__(self, policy: PolicyNetwork, config: PPOConfig, device: str = "cpu") -> None:
        self.policy = policy
        self.config = config
        self.device = device
        self.optimizer = optim.Adam(self.policy.parameters(), lr=config.learning_rate)
        self.current_entropy_coef = config.entropy_coef

    def update(self, buffer: RolloutBuffer) -> dict:
        if buffer.advantages is None or buffer.returns is None:
            buffer.compute_advantages(self.config, device=self.device)

        actions = torch.stack(buffer.actions).detach().to(self.device)
        old_log_probs = torch.stack(buffer.log_probs).detach().to(self.device)
        advantages = buffer.advantages.detach().to(self.device)
        returns = buffer.returns.detach().to(self.device)

        stats = {}
        for _ in range(self.config.update_epochs):
            log_probs, entropies, values = self.policy.evaluate_actions(buffer.observations, actions)
            ratios = torch.exp(log_probs - old_log_probs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(values.squeeze(-1), returns)
            entropy_loss = entropies.mean()

            loss = policy_loss + self.config.value_coef * value_loss - self.current_entropy_coef * entropy_loss

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
            self.optimizer.step()

            stats = {
                "policy_loss": policy_loss.item(),
                "value_loss": value_loss.item(),
                "entropy": entropy_loss.item(),
                "entropy_coef": self.current_entropy_coef,
            }
            # Decay entropy coefficient for later updates
            self.current_entropy_coef = max(
                self.config.entropy_min, self.current_entropy_coef * self.config.entropy_decay
            )
        buffer.clear()
        return stats
