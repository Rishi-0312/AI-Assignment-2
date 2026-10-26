"""Experiment configuration: one YAML file -> nested frozen dataclasses."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass, replace
from pathlib import Path

import yaml

from .env import EnvConfig
from .farm import FarmConfig
from .ppo import PPOConfig
from .wind import WindConfig


@dataclass(frozen=True)
class ExperimentConfig:
    name: str = "ppo"
    seed: int = 0
    n_workers: int = 8  # FLORIS worker processes
    eval_every: int = 10  # iterations between deterministic evaluations
    eval_episodes: int = 128
    farm: FarmConfig = field(default_factory=FarmConfig)
    wind: WindConfig = field(default_factory=WindConfig)
    env: EnvConfig = field(default_factory=lambda: EnvConfig(random_initial_offsets=True))
    ppo: PPOConfig = field(default_factory=PPOConfig)


def _from_dict(cls, data: dict, default=None):
    base = default if default is not None else cls()
    known = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise KeyError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    kwargs = {}
    for name, value in data.items():
        current = getattr(base, name)
        if is_dataclass(current):
            value = _from_dict(type(current), value or {}, current)
        elif isinstance(value, list):
            value = tuple(value)
        kwargs[name] = value
    return replace(base, **kwargs)


def config_from_dict(data: dict) -> ExperimentConfig:
    return _from_dict(ExperimentConfig, data)


def _set_path(data: dict, dotted: str, value) -> None:
    keys = dotted.split(".")
    for k in keys[:-1]:
        data = data.setdefault(k, {})
    data[keys[-1]] = value


def _read_yaml(path: Path) -> dict:
    """Read a config file; a ``base: other.yaml`` entry is loaded first and deep-merged under it."""
    with open(path) as fh:
        data = yaml.safe_load(fh) or {}
    base = data.pop("base", None)
    if base is None:
        return data
    merged = _read_yaml(Path(path).parent / base)
    for key, value in data.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> ExperimentConfig:
    """Load a YAML config; ``overrides`` are ``section.key=value`` strings (value parsed as YAML)."""
    data = _read_yaml(Path(path)) if path is not None else {}
    for item in overrides or []:
        key, _, raw = item.partition("=")
        _set_path(data, key.strip(), yaml.safe_load(raw))
    return _from_dict(ExperimentConfig, data)


def config_to_dict(cfg) -> dict:
    def clean(obj):
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [clean(v) for v in obj]
        return obj

    return clean(asdict(cfg))


def save_config(cfg: ExperimentConfig, path: Path) -> None:
    with open(path, "w") as fh:
        yaml.safe_dump(config_to_dict(cfg), fh, sort_keys=False)
