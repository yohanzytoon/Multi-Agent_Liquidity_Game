"""Train a maker and taker together on a single asset."""

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
from src.agents.taker_agent import TakerAgent
from src.models.ppo import PPOTrainer, RolloutBuffer
from src.utils.config import MarketEnvConfig, PPOConfig, TrainingConfig
from src.utils.logging_utils import get_logger
from src.utils.plotting import plot_rewards
from src.utils import set_global_seeds
from src.envs.market_env import MultiAssetMarketEnv


def main() -> None:
    env_config = MarketEnvConfig(
        assets=["ASSET_A"],
        max_steps=180,
        dataset_dir="data_big",
    )
    training = TrainingConfig(episodes=200, device="cpu", save_path="artifacts/multi_agent_basic")
    ppo_config = PPOConfig(rollout_length=env_config.max_steps)
    os.makedirs(training.save_path, exist_ok=True)
    set_global_seeds(training.seed)

    logger = get_logger("multi_agent_basic", env_config.log_dir)
    env = MultiAssetMarketEnv(env_config, seed=training.seed)
    env.register_agent("maker_0", role="maker")
    env.register_agent("taker_0", role="taker", target_inventory=500.0)

    maker = MakerAgent(asset="ASSET_A", env_config=env_config, use_gnn=False, device=training.device)
    taker = TakerAgent(asset="ASSET_A", env_config=env_config, use_gnn=False, device=training.device)

    maker_trainer = PPOTrainer(policy=maker.policy, config=ppo_config, device=training.device)
    taker_trainer = PPOTrainer(policy=taker.policy, config=ppo_config, device=training.device)

    reward_history = {"maker": [], "taker": []}

    for episode in trange(training.episodes, desc="Episodes"):
        obs = env.reset()
        maker_buffer = RolloutBuffer([], [], [], [], [], [])
        taker_buffer = RolloutBuffer([], [], [], [], [], [])
        done = False
        ep_rewards_m = []
        ep_rewards_t = []
        while not done:
            maker_action, m_log_prob, m_value, m_raw = maker.act(obs["maker_0"])
            taker_action, t_log_prob, t_value, t_raw = taker.act(obs["taker_0"])

            maker_buffer.observations.append(obs["maker_0"])
            maker_buffer.actions.append(m_raw)
            maker_buffer.log_probs.append(m_log_prob)
            maker_buffer.values.append(m_value)

            taker_buffer.observations.append(obs["taker_0"])
            taker_buffer.actions.append(t_raw)
            taker_buffer.log_probs.append(t_log_prob)
            taker_buffer.values.append(t_value)

            next_obs, rewards, dones, _ = env.step({"maker_0": maker_action, "taker_0": taker_action})
            maker_buffer.rewards.append(rewards["maker_0"])
            maker_buffer.dones.append(dones["maker_0"])
            taker_buffer.rewards.append(rewards["taker_0"])
            taker_buffer.dones.append(dones["taker_0"])

            ep_rewards_m.append(rewards["maker_0"])
            ep_rewards_t.append(rewards["taker_0"])

            obs = next_obs
            done = dones["maker_0"]

        maker_bootstrap = 0.0 if done else maker.policy.value(obs["maker_0"]).detach().item()
        taker_bootstrap = 0.0 if done else taker.policy.value(obs["taker_0"]).detach().item()
        maker_buffer.compute_advantages(ppo_config, last_value=maker_bootstrap, device=training.device)
        taker_buffer.compute_advantages(ppo_config, last_value=taker_bootstrap, device=training.device)
        m_stats = maker_trainer.update(maker_buffer)
        t_stats = taker_trainer.update(taker_buffer)
        logger.info(
            "Episode %s maker_reward %.4f taker_reward %.4f | maker %s | taker %s",
            episode,
            sum(ep_rewards_m),
            sum(ep_rewards_t),
            m_stats,
            t_stats,
        )
        reward_history["maker"].append(sum(ep_rewards_m))
        reward_history["taker"].append(sum(ep_rewards_t))

    df = pd.DataFrame(
        {
            "episode": list(range(len(reward_history["maker"]))),
            "maker_reward": reward_history["maker"],
            "taker_reward": reward_history["taker"],
        }
    )
    df.to_csv(Path(training.save_path) / "rewards.csv", index=False)
    plot_rewards(
        reward_history,
        Path(training.save_path) / "rewards.png",
    )
    torch.save(maker.policy.state_dict(), Path(training.save_path) / "maker_policy.pt")
    torch.save(taker.policy.state_dict(), Path(training.save_path) / "taker_policy.pt")
    logger.info("Training finished. Artifacts in %s", training.save_path)


if __name__ == "__main__":
    main()
