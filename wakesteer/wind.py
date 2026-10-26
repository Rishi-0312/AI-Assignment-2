"""Stochastic, exogenous wind conditions for a batch of environments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WindConfig:
    # Direction: integrated AR(1) process ("persistent random walk"): the wind
    # veers in trends that last several minutes rather than as white jitter.
    direction_persistence: float = 0.7  # AR(1) coefficient of the per-step change
    direction_innovation: float = 0.8  # [deg] innovation std of the per-step change
    direction_range: tuple[float, float] = (0.0, 360.0)  # initial-direction sector
    # Speed: Ornstein-Uhlenbeck process around an episode-mean speed.
    mean_speed_range: tuple[float, float] = (5.0, 12.0)  # [m/s]
    speed_reversion: float = 0.1  # mean-reversion rate per step
    speed_innovation: float = 0.25  # [m/s]
    speed_bounds: tuple[float, float] = (3.5, 15.0)  # [m/s]
    # Controllers only observe noisy 1-min measurements of the true wind.
    direction_noise: float = 2.0  # [deg]
    speed_noise: float = 0.2  # [m/s]


class WindProcess:
    """Vectorised wind model for ``n`` independent environments."""

    def __init__(self, cfg: WindConfig, n: int):
        self.cfg = cfg
        self.n = n
        self.direction = np.zeros(n)  # [deg], meteorological convention
        self.trend = np.zeros(n)  # [deg/step], current per-step direction change
        self.speed = np.zeros(n)  # [m/s]
        self.mean_speed = np.zeros(n)  # [m/s]

    def reset(self, rng: np.random.Generator, directions=None, speeds=None) -> None:
        c = self.cfg
        lo, hi = c.direction_range
        self.direction = np.mod(rng.uniform(lo, hi, self.n), 360.0)
        rho = c.direction_persistence
        self.trend = rng.normal(0.0, c.direction_innovation / np.sqrt(1.0 - rho**2), self.n)
        self.mean_speed = rng.uniform(*c.mean_speed_range, self.n)
        k = c.speed_reversion
        self.speed = self.mean_speed + rng.normal(0.0, c.speed_innovation / np.sqrt(1.0 - (1.0 - k) ** 2), self.n)
        self.speed = np.clip(self.speed, *c.speed_bounds)
        # Optional overrides (used for steady-state sweeps).
        if directions is not None:
            self.direction = np.mod(np.broadcast_to(np.asarray(directions, dtype=float), (self.n,)).copy(), 360.0)
        if speeds is not None:
            self.speed = np.broadcast_to(np.asarray(speeds, dtype=float), (self.n,)).copy()
            self.mean_speed = self.speed.copy()

    def step(self, rng: np.random.Generator) -> None:
        c = self.cfg
        self.trend = c.direction_persistence * self.trend + c.direction_innovation * rng.standard_normal(self.n)
        self.direction = np.mod(self.direction + self.trend, 360.0)
        self.speed += c.speed_reversion * (self.mean_speed - self.speed) + c.speed_innovation * rng.standard_normal(self.n)
        self.speed = np.clip(self.speed, *c.speed_bounds)

    def measure(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """Noisy measurements (direction [deg], speed [m/s]) of the current wind."""
        wd = np.mod(self.direction + self.cfg.direction_noise * rng.standard_normal(self.n), 360.0)
        ws = np.maximum(self.speed + self.cfg.speed_noise * rng.standard_normal(self.n), 0.0)
        return wd, ws
