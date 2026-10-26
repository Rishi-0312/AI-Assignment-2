"""PPO and baseline tests (run with ``pytest -q``)."""

import numpy as np
import torch

from wakesteer.config import config_to_dict, load_config
from wakesteer.lut import YawLUT
from wakesteer.ppo import ActorCritic, compute_gae, load_policy, save_policy


def test_gae_matches_manual_computation_with_boundary():
    gamma, lam = 0.9, 0.8
    r = torch.tensor([[1.0], [2.0], [3.0]])
    v = torch.tensor([[0.5], [0.4], [0.3]])
    nv = torch.tensor([[0.4], [0.7], [0.2]])  # step 1 ended an episode: bootstrap from V(final obs) = 0.7
    b = torch.tensor([[0.0], [1.0], [0.0]])
    adv, ret = compute_gae(r, v, nv, b, gamma, lam)
    d = (r + gamma * nv - v).squeeze(1)
    a2 = d[2]
    a1 = d[1]  # recursion stops at the boundary
    a0 = d[0] + gamma * lam * a1
    assert torch.allclose(adv.squeeze(1), torch.stack([a0, a1, a2]))
    assert torch.allclose(ret, adv + v)


def test_policy_save_load_roundtrip(tmp_path):
    cfg = config_to_dict(load_config())
    model = ActorCritic(22, 9, tuple(cfg["ppo"]["hidden_sizes"]), harmonics=tuple(cfg["ppo"]["direction_harmonics"]))
    save_policy(model, tmp_path / "p.pt", cfg, {"note": "test"})
    loaded, payload = load_policy(tmp_path / "p.pt")
    obs = np.random.default_rng(0).normal(size=(5, 22)).astype(np.float32)
    assert np.allclose(model.act(obs), loaded.act(obs))
    assert payload["meta"]["note"] == "test"


def test_policy_actions_are_bounded():
    model = ActorCritic(22, 9)
    obs = 10 * np.random.default_rng(1).normal(size=(64, 22)).astype(np.float32)
    for det in (True, False):
        a = model.act(obs, deterministic=det)
        assert a.shape == (64, 9) and np.all(np.abs(a) <= 1.0)


def test_lut_nearest_direction_and_linear_speed():
    dirs, speeds = np.arange(0.0, 360.0, 1.0), np.array([6.0, 8.0])
    offsets = np.zeros((360, 2, 2))
    offsets[270, 0], offsets[270, 1] = [10.0, 0.0], [20.0, 0.0]
    offsets[269] = -25.0
    lut = YawLUT(dirs, speeds, offsets, np.ones((360, 2)), np.ones((360, 2)))
    assert np.allclose(lut.lookup(np.array([270.3]), np.array([7.0])), [[15.0, 0.0]])
    assert np.allclose(lut.lookup(np.array([269.4]), np.array([6.0])), [[-25.0, -25.0]])  # no averaging across the jump
