"""Evaluate trained PPO agents against the baselines on held-out weather.

    python scripts/evaluate.py                       # all runs/ppo_seed*/policy.pt
    python scripts/evaluate.py --policies runs/ppo_seed0/policy.pt --episodes 128 --quick

Every controller faces exactly the same wind realisations (same seeds), so
the reported gains are paired comparisons.  Writes results/evaluation.json
and the data behind every figure in results/.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wakesteer.config import config_from_dict  # noqa: E402
from wakesteer.controllers import (  # noqa: E402
    FilteredLUTController,
    GreedyController,
    LUTController,
    PolicyController,
    RandomController,
)
from wakesteer.env import WakeSteeringVecEnv  # noqa: E402
from wakesteer.evaluation import rollout, static_wind, steady_state_sweep, summarize  # noqa: E402
from wakesteer.farm import GreedyReference, make_floris_model, make_simulator  # noqa: E402
from wakesteer.lut import load_or_build_lut, serial_refine  # noqa: E402
from wakesteer.ppo import load_policy  # noqa: E402

TEST_SEED = 2027  # held-out weather (training used seeds 0-2, validation seed 777)
VALIDATION_SEED, VALIDATION_EPISODES = 777, 128  # as in scripts/train.py
RESULTS = ROOT / "results"


def make_env(cfg, n, sim, ref, wind=None):
    env_cfg = replace(cfg.env, random_initial_offsets=False)
    return WakeSteeringVecEnv(n, cfg.farm, wind or cfg.wind, env_cfg, sim, ref)


def scored_like(main_cfg, policy_cfg):
    """Evaluation config for a policy: the main experiment's dynamics and reward, the policy's observation layout.

    Every controller is therefore scored with the same reward (e.g. the no-wear-penalty ablation is still charged
    for yaw travel at test time).
    """
    return replace(main_cfg, env=replace(main_cfg.env, observe_powers=policy_cfg.env.observe_powers))


def side_switches_per_turbine_hour(yaw: np.ndarray, threshold: float = 5.0, dt: float = 60.0) -> float:
    """How often a turbine's wake deflection flips side (offset > +thr <-> offset < -thr).

    ``yaw`` holds true yaw offsets with shape (T, n_envs, n_turbines).
    """
    side = np.sign(yaw) * (np.abs(yaw) > threshold)
    last = np.zeros(yaw.shape[1:])
    count = np.zeros(yaw.shape[1:])
    for s in side:
        count += (s != 0) & (last != 0) & (s != last)
        last = np.where(s != 0, s, last)
    return float(count.mean() / (yaw.shape[0] * dt / 3600.0))


def circular_mean_deg(angles, axis=0):
    a = np.deg2rad(angles)
    return np.mod(np.rad2deg(np.arctan2(np.sin(a).mean(axis), np.cos(a).mean(axis))), 360.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policies", nargs="*", default=None, help="policy files (default: runs/ppo_seed*/policy.pt)")
    parser.add_argument("--ablations", nargs="*", default=None, help="extra policies evaluated in the main test only")
    parser.add_argument("--episodes", type=int, default=512)
    parser.add_argument("--robust-episodes", type=int, default=256, help="episodes per robustness setting")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--quick", action="store_true", help="skip the robustness studies")
    args = parser.parse_args()
    torch.set_num_threads(4)
    RESULTS.mkdir(exist_ok=True)

    policy_paths = [Path(p) for p in args.policies] if args.policies else sorted(ROOT.glob("runs/ppo_seed*/policy.pt"))
    ablation_paths = ([Path(p) for p in args.ablations] if args.ablations else
                      sorted(ROOT.glob("runs/ablation_*/policy.pt")) + sorted(ROOT.glob("runs/gamma_*/policy.pt")))
    policies = {}
    for p in policy_paths:
        model, payload = load_policy(p)
        policies[p.parent.name] = (model, config_from_dict(payload["config"]))
    if not policies:
        raise SystemExit("no policies found - train one with scripts/train.py first")
    cfg = next(iter(policies.values()))[1]

    sim = make_simulator(cfg.farm, args.workers)
    ref = GreedyReference.load_or_build(sim)
    lut = load_or_build_lut(cfg.farm, cfg.env.max_offset, args.workers)
    report = {"test_seed": TEST_SEED, "episodes": args.episodes, "policies": list(policies)}
    keys = ("farm_power", "greedy_power", "yaw_travel", "power_term", "wind_direction", "yaw_offsets")

    # ------------------------------------- validation weather: references and baseline tuning
    t0 = time.perf_counter()
    val_env = make_env(cfg, VALIDATION_EPISODES, sim, ref)
    val_greedy = rollout(val_env, GreedyController(), VALIDATION_SEED)
    report["validation_reference"] = {
        "LUT": summarize(rollout(val_env, LUTController(lut), VALIDATION_SEED), val_greedy)["energy_gain_pct"]
    }
    # The filtered LUT gets its smoothing factor tuned on the validation weather, using the same
    # objective (mean reward) that selected the PPO checkpoints.
    tuning = {a: summarize(rollout(val_env, FilteredLUTController(lut, a), VALIDATION_SEED), val_greedy)["mean_reward"]
              for a in (0.15, 0.25, 0.4, 0.6, 1.0)}
    alpha = max(tuning, key=tuning.get)
    report["filtered_lut_tuning"] = {"alpha_to_validation_reward": {str(k): v for k, v in tuning.items()}, "alpha": alpha}
    print(f"filtered-LUT tuning ({time.perf_counter() - t0:.0f}s): {tuning} -> alpha = {alpha}")

    # ------------------------------------------------------------ main test
    t0 = time.perf_counter()
    env = make_env(cfg, args.episodes, sim, ref)
    trajs = {
        "Greedy": rollout(env, GreedyController(), TEST_SEED, keys=keys),
        "LUT": rollout(env, LUTController(lut), TEST_SEED, keys=keys),
        "LUT_filtered": rollout(env, FilteredLUTController(lut, alpha), TEST_SEED, keys=keys),
        "Random": rollout(env, RandomController(0), TEST_SEED, keys=keys),
    }
    for name, (model, pcfg) in policies.items():
        trajs[name] = rollout(make_env(scored_like(cfg, pcfg), args.episodes, sim, ref), PolicyController(model),
                              TEST_SEED, keys=keys)
    for p in ablation_paths:
        model, payload = load_policy(p)
        pcfg = config_from_dict(payload["config"])
        trajs[p.parent.name] = rollout(make_env(scored_like(cfg, pcfg), args.episodes, sim, ref),
                                       PolicyController(model), TEST_SEED, keys=keys)
    greedy = trajs["Greedy"]
    report["main"] = {name: summarize(tr, greedy) for name, tr in trajs.items()}
    for name, tr in trajs.items():
        report["main"][name]["side_switches_per_turbine_h"] = side_switches_per_turbine_hour(tr["yaw_offsets"])
    ppo_gains = [report["main"][n]["energy_gain_pct"] for n in policies]
    ppo_travel = [report["main"][n]["yaw_travel_deg_per_turbine_h"] for n in policies]
    report["ppo_aggregate"] = {
        "energy_gain_pct_mean": float(np.mean(ppo_gains)),
        "energy_gain_pct_std": float(np.std(ppo_gains, ddof=1)) if len(ppo_gains) > 1 else 0.0,
        "yaw_travel_mean": float(np.mean(ppo_travel)),
        "mean_reward_mean": float(np.mean([report["main"][n]["mean_reward"] for n in policies])),
    }
    print(f"main test done ({time.perf_counter() - t0:.0f}s)")
    for name, s in report["main"].items():
        print(f"  {name:28s} power {s['mean_power_MW']:6.3f} MW  gain {s['energy_gain_pct']:+6.2f}% "
              f"(+-{s['energy_gain_pct_sem']:.2f})  travel {s['yaw_travel_deg_per_turbine_h']:5.0f} deg/h  "
              f"switches {s['side_switches_per_turbine_h']:.2f}/h  reward {s['mean_reward']:+.3f}")

    # Gain by wind-direction sector (episodes binned by their mean direction).
    sector = 15.0
    mean_dir = circular_mean_deg(greedy["wind_direction"])
    bins = np.floor(mean_dir / sector).astype(int)
    centers = (np.arange(int(360 / sector)) + 0.5) * sector
    by_dir = {"center_deg": centers.tolist()}
    for name in ["LUT", *policies]:
        e = trajs[name]["farm_power"].sum(axis=0)
        g = greedy["farm_power"].sum(axis=0)
        by_dir[name] = [float(100 * (e[bins == b].sum() / g[bins == b].sum() - 1)) if np.any(bins == b) else None
                        for b in range(len(centers))]
    report["by_direction"] = by_dir

    # --------------------------------------------------- steady-state sweep
    t0 = time.perf_counter()
    directions = np.arange(0.0, 360.0, 1.0)
    j8 = int(np.argmin(np.abs(lut.speeds - 8.0)))
    sweep = {
        "directions": directions,
        "greedy": lut.power_greedy[:, j8],
        "lut_optimal": lut.power_opt[:, j8],
        "lut_offsets": lut.offsets[:, j8, :],
    }
    for name, (model, pcfg) in policies.items():
        p, yaw = steady_state_sweep(make_env(pcfg, 1, sim, ref), PolicyController(model), directions, speed=8.0)
        sweep[f"{name}_power"] = p.sum(axis=1)
        sweep[f"{name}_offsets"] = yaw
    np.savez_compressed(RESULTS / "steady_state.npz", **sweep)
    ss = {}
    for name in policies:
        gain = 100 * (sweep[f"{name}_power"] / sweep["greedy"] - 1)
        ss[name] = {"mean_gain_pct": float(gain.mean()),
                    "fraction_of_optimal_gain": float((sweep[f"{name}_power"] - sweep["greedy"]).sum()
                                                      / (sweep["lut_optimal"] - sweep["greedy"]).sum())}
    ss["LUT_optimal"] = {"mean_gain_pct": float((100 * (sweep["lut_optimal"] / sweep["greedy"] - 1)).mean())}
    report["steady_state_8ms"] = ss
    print(f"steady-state sweep done ({time.perf_counter() - t0:.0f}s): {json.dumps(ss)}")

    # ---------------------------------------------------- example episode
    n_ex = 64
    ex_keys = tuple(dict.fromkeys(keys + ("yaw_offsets", "measured_direction", "setpoints")))
    env_ex = make_env(cfg, n_ex, sim, ref)
    first = next(iter(policies))
    ex = {
        "Greedy": rollout(env_ex, GreedyController(), TEST_SEED + 1, keys=ex_keys),
        "LUT": rollout(env_ex, LUTController(lut), TEST_SEED + 1, keys=ex_keys),
        "PPO": rollout(make_env(policies[first][1], n_ex, sim, ref), PolicyController(policies[first][0]),
                       TEST_SEED + 1, keys=ex_keys),
    }
    np.savez_compressed(RESULTS / "example_episodes.npz",
                        **{f"{c}_{key}": ex[c][key] for c in ex for key in ex_keys + ("reward",)})

    # --------------------------------------------- hysteresis: slow veer through 270 deg
    # Noise-free wind veering at a constant 0.5 deg/min, forward (240 -> 300) and backward.
    ramp_rate, span = 0.5, (240.0, 300.0)
    steps = int((span[1] - span[0]) / ramp_rate)
    ramp_wind = replace(static_wind(cfg.wind), direction_persistence=1.0)
    hyst = {"rate_deg_per_min": ramp_rate}
    for label, start, rate in [("forward", span[0], ramp_rate), ("backward", span[1], -ramp_rate)]:
        for name, ctrl, pcfg in [("LUT", LUTController(lut), cfg)] + [
            (n, PolicyController(m), c) for n, (m, c) in policies.items()
        ]:
            env_h = WakeSteeringVecEnv(1, pcfg.farm, ramp_wind, replace(pcfg.env, episode_length=steps,
                                       random_initial_offsets=False), sim, ref)
            obs = env_h.reset(seed=0, initial_offsets=0.0, directions=start, speeds=8.0)
            env_h.wind.trend[:] = rate
            wd, yaw = [], []
            done = False
            while not done:
                obs, _, done, info = env_h.step(ctrl(obs, env_h))
                wd.append(float(info["wind_direction"][0]))
                yaw.append(info["yaw_offsets"][0].tolist())
            hyst.setdefault(name, {})[label] = {"wind_direction": wd, "yaw_offsets": yaw}
    report["hysteresis"] = hyst

    # ------------------------------------------------------ decision time
    model = policies[first][0]
    obs = torch.zeros(1, model.obs_dim)
    with torch.no_grad():
        for _ in range(50):
            model.actor(model.features(obs))
        t0 = time.perf_counter()
        for _ in range(2000):
            model.actor(model.features(obs))
    ppo_ms = (time.perf_counter() - t0) / 2000 * 1e3
    # Online alternative to the LUT: optimise the current condition with FLORIS (model already built).
    fm = make_floris_model(cfg.farm)
    fm.set(wind_directions=[270.0], wind_speeds=[8.0], turbulence_intensities=[cfg.farm.turbulence_intensity])
    sr_times = []
    for _ in range(3):
        t0 = time.perf_counter()
        serial_refine(fm, cfg.env.max_offset)
        sr_times.append(time.perf_counter() - t0)
    sr_ms = min(sr_times) * 1e3
    report["decision_time_ms"] = {"ppo_policy": ppo_ms, "serial_refine_one_condition": sr_ms}
    print(f"decision time: PPO {ppo_ms:.3f} ms vs SR optimisation {sr_ms:.0f} ms")

    if not args.quick:
        # ---------------------------------------- robustness: measurement noise
        t0 = time.perf_counter()
        n_rob = args.robust_episodes
        noise = {"sigma_deg": [0.0, 2.0, 4.0, 6.0]}
        for sigma in noise["sigma_deg"]:
            wind = replace(cfg.wind, direction_noise=sigma)
            env_n = make_env(cfg, n_rob, sim, ref, wind)
            g = rollout(env_n, GreedyController(), TEST_SEED + 2)
            res = {"LUT": summarize(rollout(env_n, LUTController(lut), TEST_SEED + 2), g),
                   "LUT_filtered": summarize(rollout(env_n, FilteredLUTController(lut, alpha), TEST_SEED + 2), g)}
            for name, (model, pcfg) in policies.items():
                res[name] = summarize(rollout(make_env(pcfg, n_rob, sim, ref, wind), PolicyController(model), TEST_SEED + 2), g)
            for name, s in res.items():
                noise.setdefault(name, {"gain": [], "travel": []})
                noise[name]["gain"].append(s["energy_gain_pct"])
                noise[name]["travel"].append(s["yaw_travel_deg_per_turbine_h"])
        report["noise_robustness"] = noise
        print(f"noise study done ({time.perf_counter() - t0:.0f}s)")

        # ----------------------------- robustness: wake-model mismatch (no retraining)
        t0 = time.perf_counter()
        mismatch = {}
        for label, farm in [("GCH (training model)", cfg.farm),
                            ("Cumulative-curl", replace(cfg.farm, wake_model="cc")),
                            ("GCH, TI = 10%", replace(cfg.farm, turbulence_intensity=0.10))]:
            sim_m = sim if farm == cfg.farm else make_simulator(farm, args.workers)
            ref_m = GreedyReference.load_or_build(sim_m)
            cfg_m = replace(cfg, farm=farm)
            env_m = make_env(cfg_m, n_rob, sim_m, ref_m)
            g = rollout(env_m, GreedyController(), TEST_SEED + 3)
            res = {"LUT": summarize(rollout(env_m, LUTController(lut), TEST_SEED + 3), g),
                   "LUT_filtered": summarize(rollout(env_m, FilteredLUTController(lut, alpha), TEST_SEED + 3), g)}
            for name, (model, pcfg) in policies.items():
                env_p = make_env(replace(pcfg, farm=farm), n_rob, sim_m, ref_m)
                res[name] = summarize(rollout(env_p, PolicyController(model), TEST_SEED + 3), g)
            mismatch[label] = {k: v["energy_gain_pct"] for k, v in res.items()}
            if sim_m is not sim:
                sim_m.close()
        report["model_mismatch"] = mismatch
        print(f"model-mismatch study done ({time.perf_counter() - t0:.0f}s): {json.dumps(mismatch, indent=1)}")

    with open(RESULTS / "evaluation.json", "w") as fh:
        json.dump(report, fh, indent=2)
    sim.close()
    print(f"wrote {RESULTS / 'evaluation.json'}")


if __name__ == "__main__":
    main()
