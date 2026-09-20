# Copyright (c) 2026 Zhiqi Cai. SPDX-License-Identifier: MIT
"""Configuration loading without importing the GPU runtime."""
from pathlib import Path

METHODS = {
    "S": "standard", "B": "birkhoff", "U": "unconstrained",
    "Row-P": "row_projected", "DS-P": "ds_projected",
    "A0": "fixed_A0", "I": "fixed_I", "J": "fixed_J",
}


def method_name(value):
    value = METHODS.get(value, value)
    if value not in METHODS.values():
        raise ValueError("Unknown method: " + value)
    return value


def load_config(path, method=None, overrides=()):
    import yaml
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError("Configuration must be a YAML mapping")
    cfg = {key.replace("-", "_"): value for key, value in cfg.items()}
    for item in overrides:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise ValueError("Overrides must use KEY=VALUE")
        cfg[key.replace("-", "_")] = yaml.safe_load(value)
    route = method_name(method or cfg.get("routing", "birkhoff"))
    cfg.update(routing=route, n_streams=1 if route == "standard" else 2)
    if cfg.get("dataset") not in ("torch/cifar10", "torch/cifar100", "ncaltech101"):
        raise ValueError("Unsupported dataset")
    if cfg["time_steps"] % cfg["chunk_size"]:
        raise ValueError("time_steps must be divisible by chunk_size")
    return cfg

