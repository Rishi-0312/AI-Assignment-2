"""Train a PPO wake-steering agent in the FLORIS environment.

Examples
--------
    python scripts/train.py --config configs/ppo.yaml --seed 0
    python scripts/train.py --config configs/ppo.yaml --seed 1 --set ppo.total_steps=2000000
    python scripts/train.py --config configs/ppo.yaml --seed 0 --resume

Outputs (in runs/<name>_seed<seed>/): config.yaml, progress.csv,
policy.pt (best validation reward), policy_final.pt, checkpoint.pt (resume).
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wakesteer.config import config_to_dict, load_config, save_config  # noqa: E402
from wakesteer.controllers import GreedyController, PolicyController  # noqa: E402
from wakesteer.env import WakeSteeringVecEnv  # noqa: E402
from wakesteer.evaluation import rollout, summarize  # noqa: E402
from wakesteer.farm import GreedyReference, make_simulator  # noqa: E402
from wakesteer.ppo import PPO, save_policy  # noqa: E402

VALIDATION_SEED = 777  # weather used to select checkpoints; final tests use other seeds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(ROOT / "configs" / "ppo.yaml"))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", default=None, help="output directory (default: runs/<name>_seed<seed>)")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="overrides, e.g. env.yaw_cost=0")
    parser.add_argument("--resume", action="store_true", help="continue from checkpoint.pt in the output directory")
    args = parser.parse_args()

    overrides = list(args.set) + ([f"seed={args.seed}"] if args.seed is not None else [])
    cfg = load_config(args.config, overrides)
    out = Path(args.out) if args.out else ROOT / "runs" / f"{cfg.name}_seed{cfg.seed}"
    out.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out / "config.yaml")
    cfg_dict = config_to_dict(cfg)
    torch.set_num_threads(4)

    sim = make_simulator(cfg.farm, cfg.n_workers)
    ref = GreedyReference.load_or_build(sim)
    env = WakeSteeringVecEnv(cfg.ppo.n_envs, cfg.farm, cfg.wind, cfg.env, sim, ref)
    val_env = WakeSteeringVecEnv(
        cfg.eval_episodes, cfg.farm, cfg.wind, replace(cfg.env, random_initial_offsets=False), sim, ref
    )
    greedy_val = rollout(val_env, GreedyController(), seed=VALIDATION_SEED)

    agent = PPO(env, cfg.ppo, seed=cfg.seed)
    ckpt_path, log_path = out / "checkpoint.pt", out / "progress.csv"
    best = float("-inf")
    if args.resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, weights_only=False)
        agent.restore(ckpt)
        best = ckpt["best"]
        print(f"resumed at iteration {agent.iteration} ({agent.env_steps:,} steps)")
    elif log_path.exists():
        log_path.unlink()

    print(f"{cfg.name} seed {cfg.seed}: {agent.n_iterations} iterations x {agent.batch_size} steps -> {out}")
    writer, fh = None, open(log_path, "a", newline="")
    t_start = time.perf_counter()
    try:
        while agent.iteration < agent.n_iterations:
            row = agent.train_iteration()
            row["wall_s"] = time.perf_counter() - t_start
            last = agent.iteration == agent.n_iterations
            if agent.iteration % cfg.eval_every == 0 or last or agent.iteration == 1:
                val = summarize(rollout(val_env, PolicyController(agent.model), seed=VALIDATION_SEED), greedy_val)
                row.update(
                    val_reward=val["mean_reward"], val_gain_pct=val["energy_gain_pct"],
                    val_travel=val["yaw_travel_deg_per_turbine_h"],
                )
                meta = {"iteration": agent.iteration, "env_steps": agent.env_steps, "validation": val}
                if val["mean_reward"] > best:
                    best = val["mean_reward"]
                    save_policy(agent.model, out / "policy.pt", cfg_dict, meta)
                torch.save({**agent.checkpoint(), "best": best}, ckpt_path)
                print(
                    f"  [val] it {agent.iteration:4d}: reward {val['mean_reward']:+.3f}  "
                    f"energy gain {val['energy_gain_pct']:+.2f}%  travel {val['yaw_travel_deg_per_turbine_h']:.0f} deg/h"
                )
            if writer is None:
                fields = list(row.keys()) + [k for k in ("val_reward", "val_gain_pct", "val_travel") if k not in row]
                writer = csv.DictWriter(fh, fieldnames=fields)
                if fh.tell() == 0:
                    writer.writeheader()
            writer.writerow(row)
            fh.flush()
            fps = cfg.ppo.n_envs * cfg.ppo.n_steps / (row["rollout_s"] + row["update_s"])
            print(
                f"it {agent.iteration:4d} | {agent.env_steps / 1e6:5.2f}M steps | r {row['reward']:+.3f} | "
                f"gain {row['power_gain_pct']:+.2f}% | travel {row['yaw_travel_deg']:.2f} deg | "
                f"std {row['policy_std']:.3f} | kl {row['approx_kl']:.4f} | ev {row['explained_var']:.2f} | {fps:.0f} sps",
                flush=True,
            )
    finally:
        fh.close()
        save_policy(agent.model, out / "policy_final.pt", cfg_dict, {"iteration": agent.iteration, "env_steps": agent.env_steps})
        sim.close()
    print(f"done in {(time.perf_counter() - t_start) / 60:.1f} min; best validation reward {best:+.3f}")


if __name__ == "__main__":
    main()
