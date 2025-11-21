from src.agents.policies import MakerAction
from src.envs.market_env import MultiAssetMarketEnv
from src.utils.config import MarketEnvConfig


def test_env_step_shapes():
    env_config = MarketEnvConfig(assets=["ASSET_A"], max_steps=5)
    env = MultiAssetMarketEnv(env_config, seed=42)
    env.register_agent("maker_0", role="maker")
    obs = env.reset()
    assert "maker_0" in obs
    action = MakerAction(asset="ASSET_A", bid_offset=0.01, ask_offset=0.01, size=1.0, cancel_only=True)
    next_obs, rewards, dones, _ = env.step({"maker_0": action})
    assert "maker_0" in next_obs
    assert "maker_0" in rewards
    assert "maker_0" in dones

