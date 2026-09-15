# SpikeBirkhoff

Companion code for **SpikeBirkhoff: Attributing Dual-Stream Gains in Spiking Transformers** (manuscript). Author: **Zhiqi Cai**.

Repository: [OneDecisionC/SpikeBirkhoff](https://github.com/OneDecisionC/SpikeBirkhoff).

This repository provides an STAtten spiking transformer, two-stream routing, and matched routing controls, with seven configurations on CIFAR-10, CIFAR-100, and N-Caltech101. Training uses surrogate gradients. The release includes source code, configurations, and the dataset split needed for reproduction. Experimental results and training logs are not distributed.

## Installation

Use Linux with an NVIDIA GPU for training. Python 3.8 or 3.10 and the following dependency families match the archived environments. Create a separate environment, install a matched PyTorch/torchvision pair, then install this repository from its root:

```bash
git clone https://github.com/OneDecisionC/SpikeBirkhoff.git
cd SpikeBirkhoff
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# CUDA 11.7 environment:
python -m pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
python -m pip install cupy-cuda11x
python -m pip install -e '.[test]'
```

The other archived environment uses torch 2.2.2. Choose its matching torchvision 0.17.2 and the CuPy package for your CUDA installation. Do not install multiple CuPy variants together. The package pins `timm==0.6.12` and `spikingjelly==0.0.0.0.12`; upgrading these can change training behavior. CuPy is needed for the paper's neuron backend; `--backend torch` supports CPU model tests and checkpoint evaluation.

## Data

CIFAR training uses timm/torchvision and downloads the official files into `--data-dir` if absent. Evaluation expects those files to exist already.

For N-Caltech101, obtain `Caltech101.zip` from the [dataset authors](https://www.garrickorchard.com/datasets/n-caltech101). Prepare count frames and the recorded split from the repository root:

```bash
python -m spikebirkhoff.prepare_ncaltech --archive /path/to/Caltech101.zip --data-dir ./data/ncaltech101 --workers 4
```

The preparation command verifies the archive and split hashes. The recorded split contains 7,795 training and 914 test examples across 101 classes. Frames contain 16 time bins and two polarities; the loader resizes them to 64 by 64 with nearest-neighbor interpolation. See [experiment details](docs/experiments.md).

## Training

Run commands from the repository root. Select a GPU with `CUDA_VISIBLE_DEVICES`; each experiment uses one GPU.

```bash
# Review the resolved configuration without importing PyTorch.
python -m spikebirkhoff.train --config configs/cifar100_d16_t4.yaml --method B --dry-run

# One complete experiment (200 scheduled epochs plus 10 cooldown epochs).
CUDA_VISIBLE_DEVICES=0 python -m spikebirkhoff.train --config configs/cifar100_d16_t4.yaml --method B --data-dir ./data/cifar100 --output runs

# Test the pipeline using two batches, clearly marked as a smoke run.
CUDA_VISIBLE_DEVICES=0 python -m spikebirkhoff.train --config configs/cifar100_d16_t4.yaml --method B --data-dir ./data/cifar100 --smoke
```

Methods are `S` (standard), `B` (Birkhoff), `U` (unconstrained), `Row-P` (row projection), `DS-P` (doubly stochastic projection), `A0` (fixed initial mixing), `I` (fixed identity), and `J` (fixed uniform mixing). [configs/experiments.json](configs/experiments.json) identifies which methods were run for each configuration. All configurations set the random seed to 42.

Run directories contain the resolved configuration, dependency versions, training log, `summary.csv`, checkpoints, and diagnostics. Existing run directories are rejected to prevent accidental overwriting. Use `--experiment NAME` for a different directory and `--set KEY=VALUE` for explicit overrides. Overrides create a new protocol; they should not be labeled as a reproduction of an unchanged paper configuration.

## Checkpoint evaluation

Reload a checkpoint and run the complete configured test split:

```bash
python -m spikebirkhoff.evaluate --config runs/cifar100_d16_t4_birkhoff_seed42/resolved.yaml --method B --checkpoint runs/cifar100_d16_t4_birkhoff_seed42/model_best.pth.tar --data-dir ./data/cifar100 --output runs/reloaded_test.json --trusted-checkpoint
```

Use `--trusted-checkpoint` only for a checkpoint you created or otherwise trust: legacy training checkpoints can contain Python pickle objects. Without this flag, supported PyTorch versions use restricted loading. The output records checkpoint SHA-256, sample count, loss, and top-1/top-5 accuracy. `--device cpu --backend torch` enables CPU evaluation. No model weights or datasets are redistributed here.

The paper's **Peak** is the highest test accuracy monitored during training, **Late20** is the mean over epochs 190--209, and **Final** is epoch 209. Re-evaluating the chosen weights checks their saved behavior; it does not create a separate validation-selected result or an untouched test set. See [the protocol](docs/experiments.md) for the precise definitions.

## Summarize your runs and run tests

```bash
python -m spikebirkhoff.summarize --csv runs/EXPERIMENT/summary.csv
python -m pytest -q
```

The summarizer computes metrics from your own training logs using decimal arithmetic and marks partial runs as incomplete. Tests use synthetic fixtures and do not require experimental results. The implementation preserves the archived inactive DropPath setting, initialization, final-layer freeze policy, and routing weight-decay exclusions; see [implementation notes](docs/implementation.md).

## License and citation

Original project contributions are licensed under **MIT**, Copyright (c) 2026 **Zhiqi Cai**. Inherited code retains its upstream licenses and attribution, as described in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). In particular, the external SpikingJelly version uses Open-Intelligence v1.0. See [CITATION.cff](CITATION.cff) for this software and manuscript metadata, and cite the upstream methods when using their components.
