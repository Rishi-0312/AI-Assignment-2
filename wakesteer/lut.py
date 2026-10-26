"""Model-based reference: yaw-offset look-up table from FLORIS's Serial-Refine optimiser.

This is the standard industrial approach to wake steering (optimise offline
with an engineering model, store set-points in a table, look them up online).
It is used here only as a *baseline* for the RL agent: it needs the exact
wake model and a costly offline optimisation, and it ignores yaw dynamics,
measurement noise and actuator wear.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .farm import _THREAD_VARS, ARTIFACTS, FarmConfig, make_floris_model


@dataclass
class YawLUT:
    directions: np.ndarray  # (n_dir,) uniform grid on [0, 360)
    speeds: np.ndarray  # (n_speed,) uniform grid [m/s]
    offsets: np.ndarray  # (n_dir, n_speed, n_turbines) optimal yaw offsets [deg]
    power_opt: np.ndarray  # (n_dir, n_speed) farm power with optimal offsets [W]
    power_greedy: np.ndarray  # (n_dir, n_speed) farm power with zero offsets [W]

    def lookup(self, wind_directions, wind_speeds) -> np.ndarray:
        """Offsets [deg] for the given conditions: nearest direction bin, linear in speed.

        Directions are *not* interpolated: the optimal offset flips sign as the
        wind crosses a turbine-alignment direction, and averaging across that
        jump would give a meaningless (near-zero) set-point.
        """
        dstep = self.directions[1] - self.directions[0]
        i = np.rint(np.mod(wind_directions, 360.0) / dstep).astype(int) % len(self.directions)
        s = np.clip(np.asarray(wind_speeds, dtype=float), self.speeds[0], self.speeds[-1])
        fs = (s - self.speeds[0]) / (self.speeds[1] - self.speeds[0])
        j0 = np.clip(np.floor(fs).astype(int), 0, len(self.speeds) - 2)
        w = (fs - j0)[:, None]
        return (1 - w) * self.offsets[i, j0] + w * self.offsets[i, j0 + 1]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **asdict(self))

    @classmethod
    def load(cls, path: Path) -> "YawLUT":
        data = np.load(path)
        return cls(**{k: data[k] for k in ("directions", "speeds", "offsets", "power_opt", "power_greedy")})


def serial_refine(fm, max_offset: float = 25.0, passes=(7, 6)):
    """FLORIS Serial-Refine optimisation of the conditions currently set in ``fm`` (returns a DataFrame)."""
    from floris.optimization.yaw_optimization.yaw_optimizer_sr import YawOptimizationSR

    opt = YawOptimizationSR(
        fm,
        minimum_yaw_angle=-max_offset,
        maximum_yaw_angle=max_offset,
        Ny_passes=list(passes),
        exclude_downstream_turbines=True,
    )
    return opt.optimize(print_progress=False)


def _optimize_chunk(args):
    cfg_dict, wd, ws, max_offset, passes = args
    cfg = FarmConfig(**cfg_dict)
    fm = make_floris_model(cfg)
    fm.set(wind_directions=wd, wind_speeds=ws, turbulence_intensities=np.full(len(wd), cfg.turbulence_intensity))
    df = serial_refine(fm, max_offset, passes)
    return (
        np.stack(df["yaw_angles_opt"].to_numpy()),
        df["farm_power_opt"].to_numpy(),
        df["farm_power_baseline"].to_numpy(),
    )


def optimize_offsets(cfg: FarmConfig, wind_directions, wind_speeds, max_offset=25.0, passes=(7, 6), n_workers=8, chunk=60):
    """Serial-Refine optimal offsets for arbitrary (direction, speed) pairs, in parallel."""
    wd = np.asarray(wind_directions, dtype=float)
    ws = np.asarray(wind_speeds, dtype=float)
    tasks = [
        (asdict(cfg), wd[i : i + chunk], ws[i : i + chunk], max_offset, passes) for i in range(0, len(wd), chunk)
    ]
    saved = {k: os.environ.get(k) for k in _THREAD_VARS}
    os.environ.update({k: "1" for k in _THREAD_VARS})
    try:
        with mp.get_context("spawn").Pool(n_workers) as pool:
            results = pool.map(_optimize_chunk, tasks)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    offsets = np.concatenate([r[0] for r in results])
    return offsets, np.concatenate([r[1] for r in results]), np.concatenate([r[2] for r in results])


def build_lut(cfg: FarmConfig, dir_step=1.0, speeds=np.arange(4.0, 14.01, 1.0), max_offset=25.0, n_workers=8) -> YawLUT:
    directions = np.arange(0.0, 360.0, dir_step)
    speeds = np.asarray(speeds, dtype=float)
    wd, ws = np.meshgrid(directions, speeds, indexing="ij")
    offsets, p_opt, p_base = optimize_offsets(cfg, wd.ravel(), ws.ravel(), max_offset, n_workers=n_workers)
    shape = wd.shape
    return YawLUT(directions, speeds, offsets.reshape(*shape, -1), p_opt.reshape(shape), p_base.reshape(shape))


def lut_path(cfg: FarmConfig, max_offset: float = 25.0) -> Path:
    return ARTIFACTS / f"yaw_lut_{cfg.key()}_{int(max_offset)}deg.npz"


def load_or_build_lut(cfg: FarmConfig, max_offset: float = 25.0, n_workers: int = 8) -> YawLUT:
    path = lut_path(cfg, max_offset)
    if path.exists():
        return YawLUT.load(path)
    lut = build_lut(cfg, max_offset=max_offset, n_workers=n_workers)
    lut.save(path)
    return lut
