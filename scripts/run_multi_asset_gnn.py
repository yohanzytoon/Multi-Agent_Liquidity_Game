"""Cross-asset example with GNN observations and maker + arbitrage agents."""

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

from src.agents.arb_agent import ArbitrageAgent
from src.agents.maker_agent import MakerAgent
from src.models.ppo import PPOTrainer, RolloutBuffer
from src.utils import set_global_seeds, load_correlation_matrix
from src.utils.config import MarketEnvConfig, PPOConfig, TrainingConfig
from src.utils.logging_utils import get_logger
from src.utils.plotting import plot_rewards
from src.envs.market_env import MultiAssetMarketEnv


def main() -> None:
    data_dir = Path("data_big")
    edges_path = data_dir / "asset_graph_edges.csv"
    assets, corr_matrix = load_correlation_matrix(edges_path)
    if not assets:
        assets = ["ASSET_A", "ASSET_B"]

    env_config = MarketEnvConfig(
        assets=assets[:5],  # use first few for tractable training
        max_steps=200,
        correlation_matrix=corr_matrix.tolist() if corr_matrix is not None else None,
        dataset_dir=str(data_dir),
    )
    training = TrainingConfig(episodes=200, device="cpu", save_path="artifacts/multi_asset_gnn")
    ppo_config = PPOConfig(rollout_length=env_config.max_steps)
    os.makedirs(training.save_path, exist_ok=True)
    set_global_seeds(training.seed)

    logger = get_logger("multi_asset_gnn", env_config.log_dir)
    env = MultiAssetMarketEnv(env_config, seed=training.seed)
    env.register_agent("maker_0", role="maker")
    env.register_agent("arb_0", role="arb")

    maker = MakerAgent(asset="ASSET_A", env_config=env_config, use_gnn=True, device=training.device)
    arb = ArbitrageAgent(assets=env_config.assets, env_config=env_config, use_gnn=True, device=training.device)

    maker_trainer = PPOTrainer(policy=maker.policy, config=ppo_config, device=training.device)
    arb_trainer = PPOTrainer(policy=arb.policy, config=ppo_config, device=training.device)

    reward_history = {"maker": [], "arb": []}

    for episode in trange(training.episodes, desc="Episodes"):
        obs = env.reset()
        maker_buffer = RolloutBuffer([], [], [], [], [], [])
        arb_buffer = RolloutBuffer([], [], [], [], [], [])
        done = False
        ep_rewards_m = []
        ep_rewards_a = []
        while not done:
            maker_action, m_log_prob, m_value, m_raw = maker.act(obs["maker_0"])
            arb_action, a_log_prob, a_value, a_raw = arb.act(obs["arb_0"])

            maker_buffer.observations.append(obs["maker_0"])
            maker_buffer.actions.append(m_raw)
            maker_buffer.log_probs.append(m_log_prob)
            maker_buffer.values.append(m_value)

            arb_buffer.observations.append(obs["arb_0"])
            arb_buffer.actions.append(a_raw)
            arb_buffer.log_probs.append(a_log_prob)
            arb_buffer.values.append(a_value)

            next_obs, rewards, dones, _ = env.step({"maker_0": maker_action, "arb_0": arb_action})
            maker_buffer.rewards.append(rewards["maker_0"])
            maker_buffer.dones.append(dones["maker_0"])
            arb_buffer.rewards.append(rewards["arb_0"])
            arb_buffer.dones.append(dones["arb_0"])

            ep_rewards_m.append(rewards["maker_0"])
            ep_rewards_a.append(rewards["arb_0"])

            obs = next_obs
            done = dones["maker_0"]

        maker_bootstrap = 0.0 if done else maker.policy.value(obs["maker_0"]).detach().item()
        arb_bootstrap = 0.0 if done else arb.policy.value(obs["arb_0"]).detach().item()
        maker_buffer.compute_advantages(ppo_config, last_value=maker_bootstrap, device=training.device)
        arb_buffer.compute_advantages(ppo_config, last_value=arb_bootstrap, device=training.device)
        m_stats = maker_trainer.update(maker_buffer)
        a_stats = arb_trainer.update(arb_buffer)
        logger.info(
            "Episode %s maker_reward %.4f arb_reward %.4f | maker %s | arb %s",
            episode,
            sum(ep_rewards_m),
            sum(ep_rewards_a),
            m_stats,
            a_stats,
        )
        reward_history["maker"].append(sum(ep_rewards_m))
        reward_history["arb"].append(sum(ep_rewards_a))

    df = pd.DataFrame(
        {
            "episode": list(range(len(reward_history["maker"]))),
            "maker_reward": reward_history["maker"],
            "arb_reward": reward_history["arb"],
        }
    )
    df.to_csv(Path(training.save_path) / "rewards.csv", index=False)
    plot_rewards(reward_history, Path(training.save_path) / "rewards.png")
    torch.save(maker.policy.state_dict(), Path(training.save_path) / "maker_policy.pt")
    torch.save(arb.policy.state_dict(), Path(training.save_path) / "arb_policy.pt")
    logger.info("Training finished. Artifacts in %s", training.save_path)


if __name__ == "__main__":
    main()
