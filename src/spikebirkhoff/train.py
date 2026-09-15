# Copyright (c) 2026 Zhiqi Cai. SPDX-License-Identifier: MIT
"""Run an archived training protocol with explicit method selection."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import yaml
from .config import load_config
from . import __version__


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--method", help="S, B, U, Row-P, DS-P, A0, I, J or full routing name")
    p.add_argument("--data-dir", type=Path)
    p.add_argument("--output", type=Path, default=Path("runs"))
    p.add_argument("--experiment", help="New run directory name")
    p.add_argument("--backend", choices=("cupy", "torch"), default="cupy")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--dry-run", action="store_true", help="Print resolved YAML without loading PyTorch")
    p.add_argument("--smoke", action="store_true", help="One epoch, two train/test batches; not a paper result")
    a = p.parse_args(argv)
    cfg = load_config(a.config, a.method, a.set)
    if a.data_dir is not None:
        cfg["data_dir"] = str(a.data_dir)
    name = a.experiment or "{}_{}_seed{}{}".format(
        a.config.stem, cfg["routing"], cfg["seed"], "_smoke" if a.smoke else "")
    if Path(name).name != name or name in (".", ".."):
        p.error("--experiment must be a single directory name")
    cfg.update(output=str(a.output), experiment=name, backend=a.backend)
    if a.smoke:
        cfg.update(epochs=1, cooldown_epochs=0, warmup_epochs=0,
                   max_train_batches=2, max_val_batches=2, checkpoint_hist=1)
    if a.dry_run:
        print(yaml.safe_dump(cfg, sort_keys=True), end="")
        return
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        p.error("Paper runs use one process per GPU; launch independent experiments")
    if cfg.get("resume") or cfg.get("initial_checkpoint") or cfg.get("pretrained"):
        p.error("This entry point starts fresh runs; evaluate checkpoints with spikebirkhoff.evaluate")
    if cfg.get("log_wandb"):
        p.error("External experiment tracking is not enabled by this entry point")
    from . import training_engine
    training_engine._normalize_config_defaults(cfg)
    if not training_engine.torch.cuda.is_available():
        p.error("Training requires CUDA; CPU is supported for model tests and evaluation")
    run = a.output / name
    run.mkdir(parents=True, exist_ok=False)
    resolved = run / "resolved.yaml"
    resolved.write_text(yaml.safe_dump(cfg, sort_keys=True), encoding="utf-8")
    versions = {}
    for package in ("torch", "torchvision", "timm", "spikingjelly", "numpy", "PyYAML"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not installed"
    meta = {"package_version": __version__, "method": cfg["routing"], "seed": cfg["seed"],
            "smoke_test": bool(a.smoke or cfg.get("max_train_batches") or cfg.get("max_val_batches")),
            "config_sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
            "dependencies": versions, "status": "started"}
    metadata = run / "run.json"
    metadata.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    old_argv = sys.argv
    try:
        sys.argv = ["spikebirkhoff.train", "--config", str(resolved)]
        training_engine.main()
        meta["status"] = "finished"
    except BaseException:
        meta["status"] = "interrupted_or_failed"
        raise
    finally:
        sys.argv = old_argv
        metadata.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
