"""Environment tests (run with ``pytest -q``)."""

from dataclasses import replace

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import wakesteer.env  # noqa: F401  (registers WakeSteer-v0)
from wakesteer.env import EnvConfig, WakeSteeringVecEnv, wrap180
from wakesteer.evaluation import static_wind
from wakesteer.farm import FarmConfig, FarmSimulator, GreedyReference
from wakesteer.wind import WindConfig


@pytest.fixture(scope="module")
def sim():
    return FarmSimulator(FarmConfig())


@pytest.fixture(scope="module")
def ref(sim):
    return GreedyReference.load_or_build(sim)


def make(sim, ref, n=4, wind=WindConfig(), **cfg):
    return WakeSteeringVecEnv(n, wind=wind, cfg=replace(EnvConfig(), **cfg), simulator=sim, reference=ref)


def test_gymnasium_api():
    env = gym.make("WakeSteer-v0")
    check_env(env.unwrapped, skip_render_check=True)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (22,) and obs.dtype == np.float32
    assert env.action_space.shape == (9,)


def test_yaw_rate_limit(sim, ref):
    env = make(sim, ref)
    env.reset(seed=1)
    heading = env.heading.copy()
    _, _, _, info = env.step(np.ones((env.n, env.n_turbines)))  # request +25 deg everywhere
    max_step = env.cfg.yaw_rate * env.cfg.dt
    assert np.all(np.abs(wrap180(env.heading - heading)) <= max_step + 1e-9)
    assert np.all(info["yaw_travel"] <= max_step + 1e-9)


def test_hold_action_does_not_move(sim, ref):
    env = make(sim, ref)
    obs = env.reset(seed=2, initial_offsets=10.0)
    hold = env.measured_offsets() / env.cfg.max_offset
    _, _, _, info = env.step(hold)
    assert np.allclose(info["yaw_travel"], 0.0, atol=1e-9)


def test_same_seed_same_weather(sim, ref):
    env = make(sim, ref)
    env.reset(seed=3)
    rng = np.random.default_rng(0)
    winds_a = [env.step(rng.uniform(-1, 1, (env.n, env.n_turbines)))[3]["wind_direction"] for _ in range(5)]
    env.reset(seed=3)
    winds_b = [env.step(np.zeros((env.n, env.n_turbines)))[3]["wind_direction"] for _ in range(5)]
    assert np.allclose(winds_a, winds_b)


def test_aligned_farm_has_zero_power_reward(sim, ref):
    # Constant, noise-free wind on a reference-grid point: greedy control matches the greedy reference.
    env = make(sim, ref, n=3, wind=static_wind(WindConfig()))
    env.reset(seed=4, directions=[270.0, 45.0, 300.0], speeds=8.0)
    _, reward, _, info = env.step(np.zeros((3, env.n_turbines)))
    assert np.allclose(info["power_term"], 0.0, atol=1e-6)
    assert np.allclose(reward, 0.0, atol=1e-6)


def test_wake_steering_increases_power_in_aligned_wind(sim, ref):
    env = make(sim, ref, n=1, wind=static_wind(WindConfig()), yaw_cost=0.0)
    env.reset(seed=5, directions=270.0, speeds=8.0)
    steer = np.zeros((1, env.n_turbines))
    steer[0, [0, 3, 6]] = 1.0  # yaw the upstream column by +25 deg
    for _ in range(3):  # rate limit: needs two steps to reach 25 deg
        _, reward, _, _ = env.step(steer)
    assert reward[0] > 2.0  # > 2 % of farm capacity gained


def test_observation_without_powers(sim, ref):
    env = make(sim, ref, observe_powers=False)
    assert env.reset(seed=6).shape == (env.n, 4 + env.n_turbines)


def test_filtered_lut_without_filtering_equals_lut(sim, ref):
    from wakesteer.controllers import FilteredLUTController, LUTController
    from wakesteer.lut import YawLUT, lut_path

    lut = YawLUT.load(lut_path(FarmConfig()))
    env = make(sim, ref)
    obs = env.reset(seed=8)
    plain, unfiltered = LUTController(lut), FilteredLUTController(lut, alpha=1.0)
    for _ in range(3):
        a = plain(obs, env)
        assert np.allclose(a, unfiltered(obs, env))
        obs, _, _, _ = env.step(a)


def test_episode_length(sim, ref):
    env = make(sim, ref, n=2, episode_length=3)
    env.reset(seed=7)
    dones = [env.step(np.zeros((2, env.n_turbines)))[2] for _ in range(3)]
    assert dones == [False, False, True]
