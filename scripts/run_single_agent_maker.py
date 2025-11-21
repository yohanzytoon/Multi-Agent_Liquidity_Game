"""Train a single market maker against noise traders."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import pandas as pd
from tqdm import trange

from src.agents.maker_agent import MakerAgent
from src.models.ppo import PPOTrainer, RolloutBuffer
from src.utils.config import MarketEnvConfig, PPOConfig, TrainingConfig
from src.utils.logging_utils import get_logger
from src.utils.plotting import plot_rewards
from src.utils import set_global_seeds
from src.envs.market_env import MultiAssetMarketEnv


def main() -> None:
    env_config = MarketEnvConfig(assets=["ASSET_A"], max_steps=150, dataset_dir="data_big")
    training = TrainingConfig(episodes=200, device="cpu", save_path="artifacts/single_maker")
    ppo_config = PPOConfig(rollout_length=env_config.max_steps)
    os.makedirs(training.save_path, exist_ok=True)
    set_global_seeds(training.seed)

    logger = get_logger("single_agent_maker", env_config.log_dir)
    env = MultiAssetMarketEnv(env_config, seed=training.seed)
    env.register_agent("maker_0", role="maker")
    maker = MakerAgent(asset="ASSET_A", env_config=env_config, use_gnn=False, device=training.device)

    trainer = PPOTrainer(policy=maker.policy, config=ppo_config, device=training.device)
    reward_history = []

    for episode in trange(training.episodes, desc="Episodes"):
        obs = env.reset()
        buffer = RolloutBuffer([], [], [], [], [], [])
        ep_rewards = []
        done = False
        while not done:
            maker_action, log_prob, value, raw_action = maker.act(obs["maker_0"])
            buffer.observations.append(obs["maker_0"])
            buffer.actions.append(raw_action)
            buffer.log_probs.append(log_prob)
            buffer.values.append(value)
            next_obs, rewards, dones, _ = env.step({"maker_0": maker_action})
            buffer.rewards.append(rewards["maker_0"])
            buffer.dones.append(dones["maker_0"])
            ep_rewards.append(rewards["maker_0"])
            obs = next_obs
            done = dones["maker_0"]

        bootstrap_value = 0.0 if done else maker.policy.value(obs["maker_0"]).detach().item()
        buffer.compute_advantages(ppo_config, last_value=bootstrap_value, device=training.device)
        stats = trainer.update(buffer)
        logger.info("Episode %s reward %.4f | stats %s", episode, sum(ep_rewards), stats)
        reward_history.append(sum(ep_rewards))

    pd.DataFrame({"episode": list(range(len(reward_history))), "maker_reward": reward_history}).to_csv(
        Path(training.save_path) / "rewards.csv", index=False
    )
    plot_rewards({"maker": reward_history}, Path(training.save_path) / "rewards.png")
    torch.save(maker.policy.state_dict(), Path(training.save_path) / "maker_policy.pt")
    logger.info("Training finished. Artifacts in %s", training.save_path)


if __name__ == "__main__":
    main()
