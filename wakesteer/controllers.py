"""Controllers share one interface: ``controller(obs, env) -> action in [-1, 1]^N``.

Baselines may read the environment's *measured* wind (``env.wd_meas``,
``env.ws_meas``) -- the same information the RL agent observes -- but never
the true wind.
"""

from __future__ import annotations

import numpy as np

from .lut import YawLUT


class GreedyController:
    """Industry-standard control: every turbine tracks the measured wind (zero offset)."""

    name = "Greedy"

    def __call__(self, obs, env):
        return np.zeros((env.n, env.n_turbines))


class LUTController:
    """Look up FLORIS-optimal offsets at the measured wind (model-based wake steering)."""

    name = "LUT (FLORIS-SR)"

    def __init__(self, lut: YawLUT):
        self.lut = lut

    def __call__(self, obs, env):
        return self.lut.lookup(env.wd_meas, env.ws_meas) / env.cfg.max_offset


class RandomController:
    name = "Random"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def __call__(self, obs, env):
        return self.rng.uniform(-1.0, 1.0, (env.n, env.n_turbines))


class PolicyController:
    """Deterministic (mean) action of a trained PPO policy."""

    def __init__(self, model, name: str = "PPO (ours)", deterministic: bool = True):
        self.model, self.name, self.deterministic = model, name, deterministic

    def __call__(self, obs, env):
        return self.model.act(obs, deterministic=self.deterministic)
