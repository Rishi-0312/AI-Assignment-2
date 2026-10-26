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


class FilteredLUTController:
    """LUT driven by exponentially smoothed wind measurements (common industrial practice).

    The nacelles track ``filtered direction - LUT(filtered wind)``, so measurement
    noise is low-pass filtered before it reaches the yaw drives.  ``alpha`` is the
    weight of the newest measurement (1 = no filtering).
    """

    def __init__(self, lut: YawLUT, alpha: float = 0.3):
        self.lut, self.alpha = lut, alpha
        self.name = f"LUT + filter (alpha={alpha:g})"
        self._sin = self._cos = self._speed = None

    def __call__(self, obs, env):
        theta = np.deg2rad(env.wd_meas)
        if env.t == 0 or self._sin is None or len(self._sin) != env.n:
            self._sin, self._cos, self._speed = np.sin(theta), np.cos(theta), env.ws_meas.copy()
        else:
            a = self.alpha
            self._sin = (1 - a) * self._sin + a * np.sin(theta)
            self._cos = (1 - a) * self._cos + a * np.cos(theta)
            self._speed = (1 - a) * self._speed + a * env.ws_meas
        wd_filtered = np.mod(np.rad2deg(np.arctan2(self._sin, self._cos)), 360.0)
        offsets = self.lut.lookup(wd_filtered, self._speed)
        # Set-points are relative to the raw measurement; shift them so the nacelle aims at the filtered target.
        setpoint = (np.mod(env.wd_meas - wd_filtered + 180.0, 360.0) - 180.0)[:, None] + offsets
        return setpoint / env.cfg.max_offset


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
