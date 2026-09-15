# Copyright (c) 2026 Zhiqi Cai. SPDX-License-Identifier: MIT
"""Reload a checkpoint and evaluate the configured test split."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
from .config import load_config


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--method", required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True, help="New JSON result file")
    p.add_argument("--device", default="cuda")
    p.add_argument("--backend", choices=("cupy", "torch"), default="cupy")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--max-batches", type=int, default=0, help="Smoke testing only; 0 evaluates the complete test set")
    p.add_argument("--trusted-checkpoint", action="store_true", help="Permit pickle loading of your own legacy checkpoint")
    a = p.parse_args(argv)
    if a.output.exists():
        p.error("Output already exists; use a new filename")
    if a.device == "cpu" and a.backend != "torch":
        p.error("CPU evaluation requires --backend torch")
    if a.max_batches < 0:
        p.error("--max-batches must be nonnegative")
    import numpy as np
    import random
    import torch
    from spikingjelly.clock_driven import functional
    from timm.data import create_dataset, create_loader, resolve_data_config
    from timm.models.helpers import clean_state_dict
    from .model import build_model
    from .datasets import build_ncaltech
    cfg = load_config(a.config, a.method)
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    kwargs = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = not a.trusted_checkpoint
    elif not a.trusted_checkpoint:
        p.error("This PyTorch lacks weights_only; only use --trusted-checkpoint for a checkpoint you trust")
    checkpoint = torch.load(str(a.checkpoint), **kwargs)
    state = checkpoint.get("state_dict", checkpoint)
    model = build_model(cfg, method=a.method, backend=a.backend)
    model.load_state_dict(clean_state_dict(state), strict=True)
    model.to(a.device).eval()
    batch_size = a.batch_size or cfg.get("val_batch_size", cfg["batch_size"])
    if cfg["dataset"] == "ncaltech101":
        _, dataset = build_ncaltech(str(a.data_dir), False)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=a.workers)
    else:
        dataset = create_dataset(cfg["dataset"], root=str(a.data_dir), split=cfg.get("val_split", "validation"),
                                 is_training=False, download=False)
        data_cfg = resolve_data_config(cfg, model=model)
        loader = create_loader(dataset, input_size=data_cfg["input_size"], batch_size=batch_size,
                               is_training=False, use_prefetcher=False, interpolation=data_cfg["interpolation"],
                               mean=data_cfg["mean"], std=data_cfg["std"], crop_pct=data_cfg["crop_pct"], num_workers=a.workers,
                               persistent_workers=a.workers > 0)
    correct1 = correct5 = total = 0
    total_loss = 0.0
    functional.reset_net(model)
    with torch.no_grad():
        for index, (inputs, targets) in enumerate(loader):
            if a.max_batches and index >= a.max_batches:
                break
            inputs, targets = inputs.float().to(a.device), targets.to(a.device)
            try:
                output = model(inputs)[0]
                if cfg.get("TET", False):
                    output = output.mean(0)
                loss = torch.nn.functional.cross_entropy(output, targets, reduction="sum")
                pred = output.topk(min(5, output.shape[-1]), dim=1).indices
                correct1 += pred[:, 0].eq(targets).sum().item()
                correct5 += pred.eq(targets[:, None]).any(dim=1).sum().item()
                total_loss += loss.item()
                total += targets.numel()
            finally:
                functional.reset_net(model)
    if total == 0:
        raise ValueError("Test loader produced no samples")
    digest = hashlib.sha256()
    with a.checkpoint.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    result = {"checkpoint_sha256": digest.hexdigest(), "checkpoint_epoch": checkpoint.get("epoch"),
              "config_sha256": hashlib.sha256(a.config.read_bytes()).hexdigest(),
              "dataset": cfg["dataset"], "method": cfg["routing"], "seed": cfg["seed"],
              "backend": a.backend, "device": a.device, "samples": total, "test_set_size": len(dataset),
              "complete_test_set": total == len(dataset), "top1": 100.0 * correct1 / total,
              "top5": 100.0 * correct5 / total, "loss": total_loss / total,
              "selection_note": "Reloading does not change the split used to select the checkpoint."}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
