"""Roll out controllers on identical weather and summarise the results."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace

import numpy as np

from .env import WakeSteeringVecEnv
from .wind import WindConfig

RECORD_KEYS = ("farm_power", "greedy_power", "yaw_travel", "power_term")


def rollout(env: WakeSteeringVecEnv, controller, seed: int, initial_offsets=None, keys=RECORD_KEYS, **reset_kw):
    """Run one episode in every environment; returns arrays shaped (T, n_envs, ...)."""
    obs = env.reset(seed=seed, initial_offsets=initial_offsets, **reset_kw)
    traj = defaultdict(list)
    done = False
    while not done:
        action = controller(obs, env)
        obs, reward, done, info = env.step(action)
        traj["reward"].append(reward)
        for k in keys:
            traj[k].append(info[k])
    return {k: np.stack(v) for k, v in traj.items()}


def summarize(traj: dict, baseline: dict | None = None, dt: float = 60.0) -> dict:
    """Energy, yaw activity and reward of a rollout; gains are relative to ``baseline`` (same seeds)."""
    T, n = traj["farm_power"].shape
    hours = T * dt / 3600.0
    energy = traj["farm_power"].sum(axis=0) * dt / 3600.0 / 1e6  # MWh per episode
    out = {
        "mean_power_MW": float(traj["farm_power"].mean() / 1e6),
        "energy_MWh": float(energy.mean()),
        "yaw_travel_deg_per_turbine_h": float(traj["yaw_travel"].sum(axis=0).mean() / hours),
        "mean_reward": float(traj["reward"].mean()),
        "episodes": int(n),
    }
    if baseline is not None:
        base_energy = baseline["farm_power"].sum(axis=0) * dt / 3600.0 / 1e6
        per_episode = 100.0 * (energy / base_energy - 1.0)
        out["energy_gain_pct"] = float(100.0 * (energy.sum() / base_energy.sum() - 1.0))
        out["energy_gain_pct_sem"] = float(per_episode.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        out["episodes_better_pct"] = float(100.0 * np.mean(energy > base_energy + 1e-9))
    return out


def steady_state_sweep(env_template: WakeSteeringVecEnv, controller, directions, speed=8.0, settle_steps=12):
    """Farm power after the controller settles in constant, noise-free wind.

    Returns (turbine powers (n_dir, N) [W], yaw offsets (n_dir, N) [deg]).
    """
    env = WakeSteeringVecEnv(
        len(directions),
        wind=static_wind(env_template.wind.cfg),
        cfg=replace(env_template.cfg, episode_length=settle_steps),
        simulator=env_template.sim,
        reference=env_template.ref,
    )
    obs = env.reset(seed=0, initial_offsets=0.0, directions=directions, speeds=speed)
    done = False
    info = None
    while not done:
        obs, _, done, info = env.step(controller(obs, env))
    return info["turbine_powers"], info["yaw_offsets"]


def static_wind(cfg: WindConfig) -> WindConfig:
    return replace(cfg, direction_innovation=0.0, speed_innovation=0.0, speed_reversion=0.0, direction_noise=0.0, speed_noise=0.0)
