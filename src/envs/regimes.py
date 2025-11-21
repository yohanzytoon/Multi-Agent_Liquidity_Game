"""Volatility/stress regime handling."""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ..utils.config import RegimeConfig


@dataclass
class RegimeManager:
    """Samples and tracks market regimes."""

    regimes: Optional[List[RegimeConfig]] = None
    switch_prob: float = 0.0
    rng: np.random.Generator = np.random.default_rng()

    def __post_init__(self) -> None:
        if not self.regimes:
            self.regimes = [
                RegimeConfig(
                    name="normal",
                    volatility=0.01,
                    order_size_mean=100.0,
                    noise_intensity=1.0,
                    spread_preference=1.0,
                ),
                RegimeConfig(
                    name="stress",
                    volatility=0.05,
                    order_size_mean=150.0,
                    noise_intensity=1.5,
                    spread_preference=2.0,
                ),
            ]
        self.current: RegimeConfig = self.regimes[0]

    def sample(self) -> RegimeConfig:
        """Sample a new regime uniformly."""
        self.current = self.rng.choice(self.regimes)
        return self.current

    def maybe_switch(self) -> RegimeConfig:
        """Switch regimes with a small probability."""
        if self.rng.random() < self.switch_prob:
            self.sample()
        return self.current
