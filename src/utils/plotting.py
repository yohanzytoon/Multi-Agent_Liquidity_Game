"""Plotting helpers for quick diagnostics."""

from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np


def plot_rewards(reward_history: Dict[str, List[float]], path: str) -> None:
    plt.figure(figsize=(8, 4))
    for agent, rewards in reward_history.items():
        plt.plot(rewards, label=agent)
    plt.xlabel("Episode")
    plt.ylabel("Average reward")
    plt.title("Reward trends")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_spread_by_regime(
    spread_history: Sequence[float],
    regime_history: Sequence[str],
    path: str,
) -> None:
    plt.figure(figsize=(8, 4))
    regimes = list(dict.fromkeys(regime_history))  # preserve order
    regime_to_idx = {r: i for i, r in enumerate(regimes)}
    colors = plt.cm.viridis(np.linspace(0, 1, len(regimes)))
    for regime, color in zip(regimes, colors):
        idxs = [i for i, r in enumerate(regime_history) if r == regime]
        plt.plot(idxs, [spread_history[i] for i in idxs], ".", color=color, label=regime)
    plt.xlabel("Step")
    plt.ylabel("Spread")
    plt.title("Spread by regime")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

