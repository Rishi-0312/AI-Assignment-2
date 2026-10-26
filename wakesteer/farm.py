"""FLORIS-backed wind-farm model used as the simulator of the RL environment.

All wake physics come from NREL's FLORIS (v4).  This module adds what the
environment needs on top of it:

* a rectangular farm layout described by a :class:`FarmConfig`,
* batched evaluation of turbine powers for many (wind, yaw) conditions at once,
  optionally spread over worker processes.  A single FLORIS call carries a fixed
  overhead of ~20 ms, so batching all environments into one call (and splitting
  that call over CPU cores) is what makes RL training affordable,
* a cached table of the *greedy* farm power (every turbine aligned with the
  wind), used as a variance-reducing baseline in the reward.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = REPO_ROOT / "artifacts"

# Wake-model presets.  "gch" (Gauss-Curl-Hybrid, FLORIS's default) is used for
# training; the others only serve to test robustness to model error.
WAKE_MODELS = {
    "gch": {"velocity_model": "gauss", "deflection_model": "gauss", "secondary_effects": True},
    "cc": {"velocity_model": "cc", "deflection_model": "gauss", "secondary_effects": True},
    "jensen": {"velocity_model": "jensen", "deflection_model": "jimenez", "secondary_effects": False},
}

_THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")


@dataclass(frozen=True)
class FarmConfig:
    rows: int = 3
    cols: int = 3
    spacing_d: float = 5.0  # turbine spacing in rotor diameters
    turbine: str = "nrel_5MW"
    turbulence_intensity: float = 0.06
    wake_model: str = "gch"
    rotor_grid_points: int = 3  # rotor is sampled on an n x n grid

    def key(self) -> str:
        """Short hash that identifies this configuration (used for cache files)."""
        blob = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha1(blob).hexdigest()[:10]


def load_turbine(name: str) -> dict:
    """Turbine definition from FLORIS's built-in turbine library."""
    import floris

    path = Path(floris.__file__).parent / "turbine_library" / f"{name}.yaml"
    with open(path) as fh:
        return yaml.safe_load(fh)


def make_floris_model(cfg: FarmConfig):
    from floris import FlorisModel

    turbine = load_turbine(cfg.turbine)
    D = turbine["rotor_diameter"]
    xs, ys = np.meshgrid(np.arange(cfg.cols) * cfg.spacing_d * D, np.arange(cfg.rows) * cfg.spacing_d * D)

    d = FlorisModel.get_defaults()
    d["farm"]["layout_x"] = xs.ravel().tolist()
    d["farm"]["layout_y"] = ys.ravel().tolist()
    # Passing the turbine as a dict (not a library name) avoids re-parsing its
    # YAML file on every FlorisModel.set() call, which is ~25% of the cost.
    d["farm"]["turbine_type"] = [turbine]
    ff = d["flow_field"]
    ff["wind_directions"] = [270.0]
    ff["wind_speeds"] = [8.0]
    ff["turbulence_intensities"] = [cfg.turbulence_intensity]
    ff["reference_wind_height"] = turbine["hub_height"]
    d["solver"]["turbine_grid_points"] = cfg.rotor_grid_points
    d["logging"]["console"]["enable"] = False

    preset = WAKE_MODELS[cfg.wake_model]
    d["wake"]["model_strings"]["velocity_model"] = preset["velocity_model"]
    d["wake"]["model_strings"]["deflection_model"] = preset["deflection_model"]
    for flag in ("enable_secondary_steering", "enable_yaw_added_recovery", "enable_transverse_velocities"):
        d["wake"][flag] = preset["secondary_effects"]
    return FlorisModel(d)


class FarmSimulator:
    """Evaluates turbine powers for a batch of conditions in the current process."""

    max_batch = 2048  # conditions per FLORIS call (bounds memory use)

    def __init__(self, cfg: FarmConfig = FarmConfig()):
        self.cfg = cfg
        self.model = make_floris_model(cfg)
        turbine = load_turbine(cfg.turbine)
        self.rotor_diameter = float(turbine["rotor_diameter"])
        self.rated_power = 1e3 * float(np.max(turbine["power_thrust_table"]["power"]))  # kW -> W
        self.layout_x = np.asarray(self.model.layout_x, dtype=float)
        self.layout_y = np.asarray(self.model.layout_y, dtype=float)
        self.n_turbines = len(self.layout_x)

    def turbine_powers(self, wind_directions, wind_speeds, yaw_angles) -> np.ndarray:
        """Turbine powers [W] with shape (n, n_turbines).

        ``yaw_angles`` are FLORIS yaw offsets in degrees (rotor misalignment
        with respect to the wind direction), shape (n, n_turbines).
        """
        wd = np.atleast_1d(np.asarray(wind_directions, dtype=float))
        ws = np.atleast_1d(np.asarray(wind_speeds, dtype=float))
        yaw = np.asarray(yaw_angles, dtype=float).reshape(len(wd), self.n_turbines)
        out = np.empty((len(wd), self.n_turbines))
        for lo in range(0, len(wd), self.max_batch):
            sl = slice(lo, lo + self.max_batch)
            n = len(wd[sl])
            self.model.set(
                wind_directions=wd[sl],
                wind_speeds=ws[sl],
                turbulence_intensities=np.full(n, self.cfg.turbulence_intensity),
                yaw_angles=yaw[sl],
            )
            self.model.run()
            out[sl] = self.model.get_turbine_powers()
        return out

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _worker_main(conn, cfg_dict: dict) -> None:
    sim = FarmSimulator(FarmConfig(**cfg_dict))
    conn.send("ready")
    while True:
        msg = conn.recv()
        if msg is None:
            break
        conn.send(sim.turbine_powers(*msg))
    conn.close()


class ParallelFarmSimulator(FarmSimulator):
    """Same interface as :class:`FarmSimulator`; each batch is split over worker processes."""

    def __init__(self, cfg: FarmConfig = FarmConfig(), n_workers: int = 8):
        super().__init__(cfg)
        self._conns, self._procs = [], []
        # Workers run single-threaded numerics; the parallelism comes from the processes.
        saved = {k: os.environ.get(k) for k in _THREAD_VARS}
        os.environ.update({k: "1" for k in _THREAD_VARS})
        try:
            ctx = mp.get_context("spawn")
            for _ in range(n_workers):
                parent, child = ctx.Pipe()
                proc = ctx.Process(target=_worker_main, args=(child, asdict(cfg)), daemon=True)
                proc.start()
                child.close()
                self._conns.append(parent)
                self._procs.append(proc)
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        for conn in self._conns:
            conn.recv()  # wait until every worker has built its FLORIS model

    def turbine_powers(self, wind_directions, wind_speeds, yaw_angles) -> np.ndarray:
        wd = np.atleast_1d(np.asarray(wind_directions, dtype=float))
        ws = np.atleast_1d(np.asarray(wind_speeds, dtype=float))
        yaw = np.asarray(yaw_angles, dtype=float).reshape(len(wd), self.n_turbines)
        chunks = [idx for idx in np.array_split(np.arange(len(wd)), len(self._conns)) if len(idx)]
        for conn, idx in zip(self._conns, chunks):
            conn.send((wd[idx], ws[idx], yaw[idx]))
        out = np.empty((len(wd), self.n_turbines))
        for conn, idx in zip(self._conns, chunks):
            out[idx] = conn.recv()
        return out

    def close(self) -> None:
        for conn in self._conns:
            try:
                conn.send(None)
                conn.close()
            except (OSError, BrokenPipeError):
                pass
        for proc in self._procs:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()
        self._conns, self._procs = [], []

    def __del__(self):
        self.close()


def make_simulator(cfg: FarmConfig = FarmConfig(), n_workers: int = 0) -> FarmSimulator:
    """Serial simulator for ``n_workers <= 1``, otherwise a process-parallel one."""
    return ParallelFarmSimulator(cfg, n_workers) if n_workers > 1 else FarmSimulator(cfg)


class GreedyReference:
    """Farm power with every turbine aligned to the wind, P_greedy(direction, speed).

    Tabulated once on a fine grid and interpolated bilinearly (periodic in
    direction).  The wind is exogenous -- no controller can change it -- so
    subtracting this term from the reward changes every policy's return by the
    same amount: the optimal policy is unchanged, but the large reward variance
    caused by wind-speed fluctuations disappears.
    """

    def __init__(self, directions: np.ndarray, speeds: np.ndarray, power: np.ndarray):
        self.directions = np.asarray(directions, dtype=float)
        self.speeds = np.asarray(speeds, dtype=float)
        self.power = np.asarray(power, dtype=float)
        self._dir_step = self.directions[1] - self.directions[0]
        self._speed_step = self.speeds[1] - self.speeds[0]

    @classmethod
    def build(cls, sim: FarmSimulator, dir_step=0.5, speed_min=3.0, speed_max=16.0, speed_step=0.25):
        directions = np.arange(0.0, 360.0, dir_step)
        speeds = np.arange(speed_min, speed_max + 1e-9, speed_step)
        wd, ws = np.meshgrid(directions, speeds, indexing="ij")
        powers = sim.turbine_powers(wd.ravel(), ws.ravel(), np.zeros((wd.size, sim.n_turbines)))
        return cls(directions, speeds, powers.sum(axis=1).reshape(wd.shape))

    @classmethod
    def load_or_build(cls, sim: FarmSimulator, cache_dir: Path = ARTIFACTS) -> "GreedyReference":
        path = Path(cache_dir) / f"greedy_reference_{sim.cfg.key()}.npz"
        if path.exists():
            data = np.load(path)
            return cls(data["directions"], data["speeds"], data["power"])
        ref = cls.build(sim)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, directions=ref.directions, speeds=ref.speeds, power=ref.power)
        return ref

    def __call__(self, wind_directions, wind_speeds) -> np.ndarray:
        n_dir = len(self.directions)
        fd = np.mod(np.asarray(wind_directions, dtype=float), 360.0) / self._dir_step
        i0 = np.floor(fd).astype(int) % n_dir
        i1 = (i0 + 1) % n_dir
        u = fd - np.floor(fd)

        ws = np.clip(np.asarray(wind_speeds, dtype=float), self.speeds[0], self.speeds[-1])
        fs = (ws - self.speeds[0]) / self._speed_step
        j0 = np.clip(np.floor(fs).astype(int), 0, len(self.speeds) - 2)
        v = fs - j0

        P = self.power
        return (1 - u) * (1 - v) * P[i0, j0] + u * (1 - v) * P[i1, j0] + (1 - u) * v * P[i0, j0 + 1] + u * v * P[i1, j0 + 1]
