"""Run the trained agent next to the baselines on a few unseen one-hour episodes.

    python scripts/demo.py                          # runs/ppo_seed0/policy.pt, 32 episodes
    python scripts/demo.py --policy runs/ppo_seed1/policy.pt --episodes 64 --workers 4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wakesteer.config import config_from_dict  # noqa: E402
from wakesteer.controllers import (  # noqa: E402
    FilteredLUTController,
    GreedyController,
    LUTController,
    PolicyController,
)
from wakesteer.env import WakeSteeringVecEnv  # noqa: E402
from wakesteer.evaluation import rollout, summarize  # noqa: E402
from wakesteer.farm import GreedyReference, make_simulator  # noqa: E402
from wakesteer.lut import load_or_build_lut  # noqa: E402
from wakesteer.ppo import load_policy  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy", default=str(ROOT / "runs" / "ppo_seed0" / "policy.pt"))
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--seed", type=int, default=31337, help="weather seed")
    parser.add_argument("--workers", type=int, default=4, help="FLORIS worker processes (0 = serial)")
    args = parser.parse_args()

    model, payload = load_policy(args.policy)
    cfg = config_from_dict(payload["config"])
    sim = make_simulator(cfg.farm, args.workers)
    ref = GreedyReference.load_or_build(sim)
    lut = load_or_build_lut(cfg.farm, cfg.env.max_offset)
    env = WakeSteeringVecEnv(args.episodes, cfg.farm, cfg.wind, replace(cfg.env, random_initial_offsets=False), sim, ref)

    evaluation = ROOT / "results" / "evaluation.json"
    alpha = json.loads(evaluation.read_text())["filtered_lut_tuning"]["alpha"] if evaluation.exists() else 0.6

    print(f"{args.episodes} one-hour episodes, 3x3 NREL 5-MW farm, FLORIS {cfg.farm.wake_model.upper()} wake model\n")
    print(f"{'controller':22s} {'power [MW]':>10s} {'gain vs greedy':>15s} {'yaw travel [deg/h]':>19s} {'reward':>8s}")
    t0 = time.perf_counter()
    greedy = rollout(env, GreedyController(), args.seed)
    for name, controller in [("Greedy", GreedyController()), ("LUT (FLORIS-SR)", LUTController(lut)),
                             (f"LUT + filter (a={alpha:g})", FilteredLUTController(lut, alpha)),
                             ("PPO agent", PolicyController(model))]:
        s = summarize(greedy if name == "Greedy" else rollout(env, controller, args.seed), greedy)
        print(f"{name:22s} {s['mean_power_MW']:10.3f} {s['energy_gain_pct']:+14.2f}% "
              f"{s['yaw_travel_deg_per_turbine_h']:19.0f} {s['mean_reward']:+8.3f}")
    print(f"\n({time.perf_counter() - t0:.0f} s)")
    sim.close()


if __name__ == "__main__":
    main()
