"""Pre-compute the cached FLORIS tables used by the environment and the baselines.

* greedy reference P_greedy(direction, speed) -- reward baseline (seconds)
* Serial-Refine optimal yaw look-up table -- model-based baseline (minutes)

    python scripts/build_lut.py --config configs/ppo.yaml
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wakesteer.config import load_config  # noqa: E402
from wakesteer.farm import GreedyReference, make_simulator  # noqa: E402
from wakesteer.lut import load_or_build_lut, lut_path  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(ROOT / "configs" / "ppo.yaml"))
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    workers = args.workers or cfg.n_workers

    t0 = time.perf_counter()
    sim = make_simulator(cfg.farm, workers)
    GreedyReference.load_or_build(sim)
    sim.close()
    print(f"greedy reference ready ({time.perf_counter() - t0:.0f}s)")

    t0 = time.perf_counter()
    lut = load_or_build_lut(cfg.farm, cfg.env.max_offset, workers)
    print(f"yaw LUT ready ({time.perf_counter() - t0:.0f}s): {lut_path(cfg.farm, cfg.env.max_offset).name}")
    gain = 100.0 * (lut.power_opt / lut.power_greedy - 1.0)
    for j, s in enumerate(lut.speeds):
        print(f"  {s:4.1f} m/s: mean steady-state gain over all directions {gain[:, j].mean():5.2f}%  (max {gain[:, j].max():5.2f}%)")


if __name__ == "__main__":
    main()
