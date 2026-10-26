"""Proximal Policy Optimisation (clipped surrogate objective) written for the batched wake-steering environment.

Implementation notes
--------------------
* Gaussian policy with a state-independent, learned log-std; sampled actions
  are clipped to [-1, 1] by the environment (log-probs use the raw sample).
* Separate actor and critic MLPs; gradient norms are clipped per network so the
  (large) value-loss gradients never shrink the policy update.
* Episodes end only by the time limit, which is not a true terminal state of
  the wind farm: at those boundaries GAE bootstraps from V(final observation).
* The wind direction enters the network through Fourier features
  sin(k*theta), cos(k*theta): wake interactions in a grid farm repeat every
  45-90 degrees, which low-order sin/cos inputs alone represent poorly.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


@dataclass(frozen=True)
class PPOConfig:
    total_steps: int = 5_000_000
    n_envs: int = 256
    n_steps: int = 30  # rollout length per environment and iteration
    learning_rate: float = 3e-4
    anneal_lr: bool = True
    gamma: float = 0.95
    gae_lambda: float = 0.9
    clip_coef: float = 0.2
    update_epochs: int = 10
    n_minibatches: int = 8
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.02
    hidden_sizes: tuple[int, ...] = (256, 256)
    log_std_init: float = -0.7
    direction_harmonics: tuple[int, ...] = (2, 3, 4, 8)


def _mlp(in_dim: int, out_dim: int, hidden: tuple[int, ...], out_gain: float) -> nn.Sequential:
    layers, d = [], in_dim
    for h in hidden:
        lin = nn.Linear(d, h)
        nn.init.orthogonal_(lin.weight, np.sqrt(2.0))
        nn.init.zeros_(lin.bias)
        layers += [lin, nn.Tanh()]
        d = h
    out = nn.Linear(d, out_dim)
    nn.init.orthogonal_(out.weight, out_gain)
    nn.init.zeros_(out.bias)
    layers.append(out)
    return nn.Sequential(*layers)


class DirectionFeatures(nn.Module):
    """Appends sin(k*theta), cos(k*theta) of the measured wind direction (obs[0] = sin, obs[1] = cos)."""

    def __init__(self, harmonics: tuple[int, ...]):
        super().__init__()
        self.register_buffer("k", torch.tensor(harmonics, dtype=torch.float32))

    @property
    def extra_dim(self) -> int:
        return 2 * len(self.k)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if len(self.k) == 0:
            return obs
        theta = torch.atan2(obs[..., :1], obs[..., 1:2])
        kt = theta * self.k
        return torch.cat([obs, torch.sin(kt), torch.cos(kt)], dim=-1)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden=(256, 256), log_std_init=-0.7, harmonics=(2, 3, 4, 8)):
        super().__init__()
        self.obs_dim, self.act_dim = obs_dim, act_dim
        self.features = DirectionFeatures(tuple(harmonics))
        in_dim = obs_dim + self.features.extra_dim
        self.actor = _mlp(in_dim, act_dim, tuple(hidden), out_gain=0.01)  # starts near zero offsets = greedy
        self.critic = _mlp(in_dim, 1, tuple(hidden), out_gain=1.0)
        self.log_std = nn.Parameter(torch.full((act_dim,), float(log_std_init)))

    def distribution(self, obs: torch.Tensor) -> Normal:
        mean = self.actor(self.features(obs))
        return Normal(mean, self.log_std.exp().expand_as(mean))

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(self.features(obs)).squeeze(-1)

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        if deterministic:
            a = self.actor(self.features(obs_t))
        else:
            a = self.distribution(obs_t).sample()
        return a.clamp(-1.0, 1.0).numpy()

    def actor_parameters(self):
        return list(self.actor.parameters()) + [self.log_std]


def compute_gae(rewards, values, next_values, boundaries, gamma, lam):
    """Generalised advantage estimation over a (T, n_envs) rollout.

    ``next_values[t]`` is V(s_{t+1}) -- for a step that ended an episode it is
    the value of the final observation (time-limit truncation, so we still
    bootstrap) -- and ``boundaries[t]`` stops the recursion between episodes.
    """
    T = rewards.shape[0]
    adv = torch.zeros_like(rewards)
    last = torch.zeros_like(rewards[0])
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * next_values[t] - values[t]
        last = delta + gamma * lam * (1.0 - boundaries[t]) * last
        adv[t] = last
    return adv, adv + values


class PPO:
    """Rollout collection and clipped-surrogate updates for a :class:`WakeSteeringVecEnv`."""

    def __init__(self, env, cfg: PPOConfig, seed: int = 0):
        self.env, self.cfg = env, cfg
        torch.manual_seed(seed)
        self.model = ActorCritic(env.obs_dim, env.act_dim, cfg.hidden_sizes, cfg.log_std_init, cfg.direction_harmonics)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.learning_rate, eps=1e-5)
        self.batch_size = cfg.n_envs * cfg.n_steps
        self.n_iterations = max(1, cfg.total_steps // self.batch_size)
        self.iteration = 0
        self.env_steps = 0
        self.seed = seed
        self._obs = None
        self._episode_stats = []

    # ---------------------------------------------------------------- rollout
    def collect_rollout(self):
        cfg, env, n = self.cfg, self.env, self.cfg.n_envs
        T = cfg.n_steps
        obs_buf = torch.zeros(T, n, env.obs_dim)
        act_buf = torch.zeros(T, n, env.act_dim)
        logp_buf = torch.zeros(T, n)
        rew_buf = torch.zeros(T, n)
        val_buf = torch.zeros(T, n)
        next_val_buf = torch.zeros(T, n)
        bound_buf = torch.zeros(T, 1)
        stats = {"power_term": 0.0, "farm_power": 0.0, "greedy_power": 0.0, "yaw_travel": 0.0}

        if self._obs is None:
            self._obs = env.reset(seed=self.seed)
        for t in range(T):
            obs_t = torch.as_tensor(self._obs)
            with torch.no_grad():
                dist = self.model.distribution(obs_t)
                action = dist.sample()
                logp = dist.log_prob(action).sum(-1)
                value = self.model.value(obs_t)
            next_obs, reward, done, info = env.step(action.numpy())
            obs_buf[t], act_buf[t], logp_buf[t], val_buf[t] = obs_t, action, logp, value
            rew_buf[t] = torch.as_tensor(reward, dtype=torch.float32)
            stats["power_term"] += info["power_term"].mean()
            stats["farm_power"] += info["farm_power"].sum()
            stats["greedy_power"] += info["greedy_power"].sum()
            stats["yaw_travel"] += info["yaw_travel"].mean()
            if done:
                with torch.no_grad():
                    next_val_buf[t] = self.model.value(torch.as_tensor(next_obs))
                bound_buf[t] = 1.0
                next_obs = env.reset()
            self._obs = next_obs

        with torch.no_grad():
            last_value = self.model.value(torch.as_tensor(self._obs))
        for t in range(T):
            if bound_buf[t] == 0:
                next_val_buf[t] = val_buf[t + 1] if t + 1 < T else last_value

        adv, ret = compute_gae(rew_buf, val_buf, next_val_buf, bound_buf, cfg.gamma, cfg.gae_lambda)
        self.env_steps += T * n
        rollout_stats = {
            "reward": rew_buf.mean().item(),
            "power_gain_pct": 100.0 * (stats["farm_power"] - stats["greedy_power"]) / stats["greedy_power"],
            "yaw_travel_deg": stats["yaw_travel"] / T,  # per turbine per step
        }
        batch = {
            "obs": obs_buf.reshape(-1, env.obs_dim),
            "actions": act_buf.reshape(-1, env.act_dim),
            "logp": logp_buf.reshape(-1),
            "advantages": adv.reshape(-1),
            "returns": ret.reshape(-1),
            "values": val_buf.reshape(-1),
        }
        return batch, rollout_stats

    # ----------------------------------------------------------------- update
    def update(self, batch) -> dict:
        cfg = self.cfg
        if cfg.anneal_lr:
            frac = 1.0 - self.iteration / self.n_iterations
            for g in self.optimizer.param_groups:
                g["lr"] = cfg.learning_rate * max(frac, 0.05)
        N = batch["obs"].shape[0]
        mb_size = N // cfg.n_minibatches
        actor_params, critic_params = self.model.actor_parameters(), list(self.model.critic.parameters())
        clip_fracs, kls, pg_losses, v_losses, entropies = [], [], [], [], []
        stop = False
        for _ in range(cfg.update_epochs):
            perm = torch.randperm(N)
            for start in range(0, mb_size * cfg.n_minibatches, mb_size):
                idx = perm[start : start + mb_size]
                dist = self.model.distribution(batch["obs"][idx])
                new_logp = dist.log_prob(batch["actions"][idx]).sum(-1)
                log_ratio = new_logp - batch["logp"][idx]
                ratio = log_ratio.exp()
                adv = batch["advantages"][idx]
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                pg_loss = torch.max(-adv * ratio, -adv * ratio.clamp(1 - cfg.clip_coef, 1 + cfg.clip_coef)).mean()
                v_loss = 0.5 * (self.model.value(batch["obs"][idx]) - batch["returns"][idx]).pow(2).mean()
                entropy = dist.entropy().sum(-1).mean()
                loss = pg_loss - cfg.ent_coef * entropy + cfg.vf_coef * v_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(actor_params, cfg.max_grad_norm)
                nn.utils.clip_grad_norm_(critic_params, cfg.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    kls.append(((ratio - 1) - log_ratio).mean().item())
                    clip_fracs.append(((ratio - 1).abs() > cfg.clip_coef).float().mean().item())
                pg_losses.append(pg_loss.item())
                v_losses.append(v_loss.item())
                entropies.append(entropy.item())
            if cfg.target_kl and np.mean(kls[-cfg.n_minibatches :]) > cfg.target_kl:
                stop = True
                break
        self.iteration += 1
        var_y = batch["returns"].var()
        explained_var = 1.0 - (batch["returns"] - batch["values"]).var() / var_y if var_y > 0 else torch.tensor(0.0)
        return {
            "pg_loss": float(np.mean(pg_losses)),
            "v_loss": float(np.mean(v_losses)),
            "entropy": float(np.mean(entropies)),
            "approx_kl": float(np.mean(kls)),
            "clip_frac": float(np.mean(clip_fracs)),
            "explained_var": float(explained_var),
            "policy_std": float(self.model.log_std.exp().mean()),
            "lr": self.optimizer.param_groups[0]["lr"],
            "kl_stop": int(stop),
        }

    def train_iteration(self) -> dict:
        t0 = time.perf_counter()
        batch, rollout_stats = self.collect_rollout()
        t1 = time.perf_counter()
        update_stats = self.update(batch)
        t2 = time.perf_counter()
        return {
            "iteration": self.iteration,
            "env_steps": self.env_steps,
            **rollout_stats,
            **update_stats,
            "rollout_s": t1 - t0,
            "update_s": t2 - t1,
        }

    # --------------------------------------------------------------- persist
    def checkpoint(self) -> dict:
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "iteration": self.iteration,
            "env_steps": self.env_steps,
            "torch_rng": torch.get_rng_state(),
        }

    def restore(self, ckpt: dict) -> None:
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.iteration = ckpt["iteration"]
        self.env_steps = ckpt["env_steps"]
        torch.set_rng_state(ckpt["torch_rng"])


def save_policy(model: ActorCritic, path, config: dict, meta: dict | None = None) -> None:
    """Self-contained policy file: weights + architecture + the experiment config."""
    torch.save(
        {
            "state_dict": model.state_dict(),
            "obs_dim": model.obs_dim,
            "act_dim": model.act_dim,
            "hidden_sizes": list(config["ppo"]["hidden_sizes"]),
            "direction_harmonics": list(config["ppo"]["direction_harmonics"]),
            "config": config,
            "meta": meta or {},
        },
        path,
    )


def load_policy(path) -> tuple[ActorCritic, dict]:
    """Returns (model in eval mode, saved payload with 'config' and 'meta')."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model = ActorCritic(
        payload["obs_dim"], payload["act_dim"], tuple(payload["hidden_sizes"]),
        harmonics=tuple(payload["direction_harmonics"]),
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload
