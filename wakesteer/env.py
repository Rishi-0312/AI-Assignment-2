"""Wake-steering MDP built on FLORIS.

Every control interval (dt = 60 s) the agent sends a yaw-offset set-point to
each turbine.  The turbine yaw drives move the nacelles towards
``measured wind direction - set-point`` at a limited rate, the wind evolves
stochastically, and FLORIS computes the resulting turbine powers.

Observation (dimension 4 + 2N for N turbines), all roughly in [-1, 1]:
    [0]      sin(measured wind direction)
    [1]      cos(measured wind direction)
    [2]      (measured wind speed - 8.5) / 3.5
    [3]      change of measured direction since the last step / 5 deg
    [4:4+N]  measured yaw offset of each turbine / max_offset
    [4+N:]   power of each turbine / rated power (omitted if observe_powers=False)

Action (dimension N), in [-1, 1]:  yaw-offset set-point / max_offset.

Reward:
    r = 100 * (P_farm - P_greedy(wind)) / P_capacity  -  c_yaw * sum_i |nacelle travel_i|
i.e. power gained over the all-aligned farm, in percent of farm capacity,
minus a yaw-actuator wear penalty per degree of nacelle rotation.
"""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np

from .farm import FarmConfig, FarmSimulator, GreedyReference
from .wind import WindConfig, WindProcess


def wrap180(angle):
    """Wrap angles [deg] to [-180, 180)."""
    return np.mod(np.asarray(angle) + 180.0, 360.0) - 180.0


@dataclass(frozen=True)
class EnvConfig:
    episode_length: int = 60  # control steps per episode (one hour)
    dt: float = 60.0  # control interval [s]
    yaw_rate: float = 0.3  # [deg/s], yaw-drive limit of the NREL 5-MW turbine
    max_offset: float = 25.0  # [deg], bound on commanded yaw offsets
    yaw_cost: float = 0.005  # reward penalty per degree of nacelle travel
    reward_scale: float = 100.0  # power term in percent of farm capacity
    random_initial_offsets: bool = False  # exploring starts (used for training)
    observe_powers: bool = True  # include turbine powers in the observation


class WakeSteeringVecEnv:
    """A batch of ``n_envs`` independent wake-steering MDPs sharing one FLORIS evaluator.

    All environments advance in lock-step and share the episode length, so a
    single (batched) FLORIS call per step serves the whole batch.
    """

    def __init__(
        self,
        n_envs: int,
        farm: FarmConfig = FarmConfig(),
        wind: WindConfig = WindConfig(),
        cfg: EnvConfig = EnvConfig(),
        simulator: FarmSimulator | None = None,
        reference: GreedyReference | None = None,
    ):
        self.n = n_envs
        self.cfg = cfg
        self.sim = simulator if simulator is not None else FarmSimulator(farm)
        self.ref = reference if reference is not None else GreedyReference.load_or_build(self.sim)
        self.n_turbines = self.sim.n_turbines
        self.obs_dim = 4 + (2 if cfg.observe_powers else 1) * self.n_turbines
        self.act_dim = self.n_turbines
        self.capacity = self.n_turbines * self.sim.rated_power
        self.max_step = cfg.yaw_rate * cfg.dt  # max nacelle rotation per step [deg]
        self.wind = WindProcess(wind, n_envs)
        self.t = 0
        self._seed_rngs(None)

    # ------------------------------------------------------------------ helpers
    def _seed_rngs(self, seed) -> None:
        # The wind (and its measurement noise) has its own random stream, so a
        # given seed produces the same wind realisation whatever the controller
        # does.  Controllers are therefore compared on identical weather.
        wind_seq, aux_seq = np.random.SeedSequence(seed).spawn(2)
        self.wind_rng = np.random.default_rng(wind_seq)
        self.aux_rng = np.random.default_rng(aux_seq)

    def yaw_offsets(self) -> np.ndarray:
        """True rotor misalignment (FLORIS yaw angle) of every turbine [deg]."""
        return wrap180(self.wind.direction[:, None] - self.heading)

    def measured_offsets(self) -> np.ndarray:
        """Misalignment as seen by the controller (relative to the measured direction)."""
        return wrap180(self.wd_meas[:, None] - self.heading)

    def _simulate(self) -> np.ndarray:
        return self.sim.turbine_powers(self.wind.direction, self.wind.speed, self.yaw_offsets())

    def _observe(self) -> np.ndarray:
        theta = np.deg2rad(self.wd_meas)
        trend = wrap180(self.wd_meas - self.wd_meas_prev)
        parts = [
            np.sin(theta)[:, None],
            np.cos(theta)[:, None],
            ((self.ws_meas - 8.5) / 3.5)[:, None],
            (trend / 5.0)[:, None],
            self.measured_offsets() / self.cfg.max_offset,
        ]
        if self.cfg.observe_powers:
            parts.append(self.powers / self.sim.rated_power)
        return np.concatenate(parts, axis=1).astype(np.float32)

    # --------------------------------------------------------------------- API
    def reset(self, seed=None, initial_offsets=None, directions=None, speeds=None) -> np.ndarray:
        """Start new episodes for all environments and return the first observation.

        ``initial_offsets`` (deg) sets the starting yaw offsets; by default the
        turbines start aligned with the measured wind (or at random offsets if
        ``random_initial_offsets``).  ``directions``/``speeds`` override the
        initial wind (used for steady-state sweeps).
        """
        if seed is not None:
            self._seed_rngs(seed)
        self.wind.reset(self.wind_rng, directions, speeds)
        self.wd_meas, self.ws_meas = self.wind.measure(self.wind_rng)
        self.wd_meas_prev = self.wd_meas.copy()
        if initial_offsets is None:
            if self.cfg.random_initial_offsets:
                initial_offsets = self.aux_rng.uniform(-1.0, 1.0, (self.n, self.n_turbines)) * self.cfg.max_offset
            else:
                initial_offsets = np.zeros((self.n, self.n_turbines))
        initial_offsets = np.broadcast_to(np.asarray(initial_offsets, dtype=float), (self.n, self.n_turbines))
        self.heading = np.mod(self.wd_meas[:, None] - initial_offsets, 360.0)
        self.t = 0
        self.powers = self._simulate()
        return self._observe()

    def step(self, action: np.ndarray):
        """Advance all environments by one control interval.

        Returns ``(obs, reward, done, info)``; ``done`` is a single bool because
        the environments run in lock-step (episodes end by time limit only).
        """
        setpoints = np.clip(np.asarray(action, dtype=float), -1.0, 1.0) * self.cfg.max_offset
        target = self.wd_meas[:, None] - setpoints  # desired nacelle heading [deg]
        move = np.clip(wrap180(target - self.heading), -self.max_step, self.max_step)
        self.heading = np.mod(self.heading + move, 360.0)
        travel = np.abs(move)

        self.wind.step(self.wind_rng)
        self.powers = self._simulate()
        farm_power = self.powers.sum(axis=1)
        greedy_power = self.ref(self.wind.direction, self.wind.speed)
        power_term = self.cfg.reward_scale * (farm_power - greedy_power) / self.capacity
        reward = power_term - self.cfg.yaw_cost * travel.sum(axis=1)

        info = {
            "farm_power": farm_power,
            "greedy_power": greedy_power,
            "turbine_powers": self.powers.copy(),
            "yaw_travel": travel,
            "yaw_offsets": self.yaw_offsets(),
            "setpoints": setpoints,
            "wind_direction": self.wind.direction.copy(),
            "wind_speed": self.wind.speed.copy(),
            "power_term": power_term,
        }
        self.wd_meas_prev = self.wd_meas
        self.wd_meas, self.ws_meas = self.wind.measure(self.wind_rng)
        info["measured_direction"] = self.wd_meas.copy()
        self.t += 1
        return self._observe(), reward, self.t >= self.cfg.episode_length, info

    def close(self) -> None:
        self.sim.close()


class WakeSteeringEnv(gym.Env):
    """Single-environment Gymnasium interface (e.g. for ``gymnasium.make`` or other RL libraries)."""

    metadata = {"render_modes": []}

    def __init__(self, farm: FarmConfig = FarmConfig(), wind: WindConfig = WindConfig(), cfg: EnvConfig = EnvConfig()):
        self._vec = WakeSteeringVecEnv(1, farm, wind, cfg)
        n = self._vec.n_turbines
        # Features are O(1) by construction; the box is a generous envelope, not a clip.
        self.observation_space = gym.spaces.Box(-10.0, 10.0, shape=(self._vec.obs_dim,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(n,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is None and not hasattr(self, "_seeded"):
            seed = int(self.np_random.integers(2**31))
        self._seeded = True
        obs = self._vec.reset(seed=seed, **(options or {}))
        return obs[0], {}

    def step(self, action):
        obs, reward, done, info = self._vec.step(np.asarray(action, dtype=float)[None, :])
        info = {k: v[0] for k, v in info.items()}
        return obs[0], float(reward[0]), False, bool(done), info


gym.register(id="WakeSteer-v0", entry_point="wakesteer.env:WakeSteeringEnv")
