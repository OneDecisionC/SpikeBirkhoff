# Revised-paper reproduction

## Protocol

Use Linux/CUDA and a dedicated compatible environment. Reserve GPUs; never terminate another user's process. Static templates assume physical GPUs 0–3, including smoke checks on GPU 0. Independent runs use separate GPUs, not DDP.

| Dataset/configuration | Train/validation/test | Precision | Runs |
|---|---|---|---:|
| CIFAR-100 d8/d16, T4 | 40k/5k/5k from official training set | FP32 | 40 |
| CIFAR-10 d6, T4 | 40k/5k/5k from official training set | FP32 | 20 |
| N-Caltech101 d2, T16 | 6953/878/878 recordings | FP32 | 20 |
| CIFAR10-DVS d2, T16 | 8000/1000/1000 recordings | AMP + TET | 20 |

Each uses S/B/I/A0, seeds 42–46, 210 epochs and best validation top-1 (earliest epoch on ties). CIFAR split seed: 20260916; event split seed: 20260919. Whole recordings stay together. Official CIFAR-100 evaluation reuses the 40 checkpoints.

Preserved details: two streams, softmax reads, twice-softmax writes, identity/swap mixer initialized at alpha approximately 0.878681, zero routing weight decay, frozen final mixer/write but learned final read. DropPath is configured as 0.2 but inactive in the archived forward path: deliberately not repaired.

The root dependencies describe the legacy package. The DVS archive includes historical CUDA-11 dependency pins in `reproduction/cifar10dvs/requirements.txt`; compatible PyTorch/torchvision must be installed separately. These are not a portable environment installer. Native CuPy probes require working CUDA compiler/runtime libraries. Only load trusted self-produced checkpoints: archived PyTorch loading uses pickle.

## Static datasets and N-Caltech101

Stage on the training host with its intended Python environment. Supply the dataset root expected by the adapter (cached 16-frame recordings for N-Caltech). Use a new workspace outside the data and repository.

```bash
python scripts/stage_reproduction.py --dataset cifar100 --data /data/cifar100 --workspace /runs/c100_new
```

Substitute `cifar10` or `ncaltech101` as needed. Inspect the generated protocol. **The next command runs smoke checks, freezes inputs, starts detached full training, and automatically tests the internal holdout after every selected checkpoint in the batch is frozen. It is not preparation-only.** Reserve GPUs 0–3 first:

```bash
PYTHONPATH=/runs/c100_new/source/src python /runs/c100_new/prepare.py
```

Monitor `queue.log`, `RUN_PATHS.json` and `launch_receipt.json`. Do not delete gates or alter frozen protocols. Official CIFAR-100 testing is separate:

```bash
python reproduction/evaluate_official.py --root /runs/c100_new --output /runs/c100_official_new --gpus 0,1,2,3
```

Historical static/N-Caltech experiments were frozen in two server batches (42–44 and 45–46) before batch testing. The public template groups all five seeds into one gate; this operational change does not rewrite historical provenance.

## CIFAR10-DVS

Dataset root: `frames_number_16_split_by_number/<class>/*.npz`; 10,000 recordings, float64 raw counts under `frames`, shape 16x2x128x128. Archived preprocessing uses bilinear resize to 64, antialias=False and no added normalization.

```bash
python scripts/stage_reproduction.py --dataset cifar10dvs --data /data/dvs --workspace /runs/dvs_new
python /runs/dvs_new/code/prepare_split.py --data /data/dvs --output /runs/dvs_new/split.json
python /runs/dvs_new/code/launcher.py --mode smoke --gpus 0,1,2,3,4,5,6,7 --data /data/dvs --split /runs/dvs_new/split.json --output /runs/dvs_new/smoke
python /runs/dvs_new/code/launcher.py --mode train --gpus 0,1,2,3,4,5,6,7 --data /data/dvs --split /runs/dvs_new/split.json --output /runs/dvs_new/training
python /runs/dvs_new/code/evaluate_test.py --runs /runs/dvs_new/training --gpus 0,1,2,3,4,5,6,7 --trusted-checkpoints
```

Fewer allocated GPUs are supported. Final testing is separate from training, refuses incomplete runs and verifies frozen selections. Gates are write-once. Do not modify code after freezing.

## Mechanisms

All probes use validation examples, not checkpoint selection. First create baseline records and a shared plan:

```bash
python reproduction/mechanisms/diagnose.py --root /runs/c100_new --dataset cifar100 --data /data/cifar100 --server local --gpus 0 --output /runs/diagnostics_new
python reproduction/mechanisms/membrane.py --plan /runs/diagnostics_new/plan.json --gpus 0 --output /runs/membrane_new
python reproduction/mechanisms/equal_perturbation.py --plan /runs/diagnostics_new/plan.json --gpus 0 --output /runs/equal_new --precision fp32
```

For DVS, use `--dataset dvs --root /runs/dvs_new/training --source /runs/dvs_new/code/vendor/src --adapter /runs/dvs_new/code --data /data/dvs`. The additional `layerwise.py`, `precision_control.py` (DVS), and `prefix.py` accept the baseline plan, allocated GPUs and distinct new output directories. Consult each script's help.

Probe sizes: 500 CIFAR/200 DVS, batch 8, state resets, evaluation-mode BN. Whole-probe spike disagreement and sampled downstream membrane statistics have different denominators. Equal alpha perturbations move 0.01 toward 0.5, allowing crossing; final-layer downstream values are N/A. Five-epoch prefixes retain the original 210-epoch schedule; alpha summaries exclude the frozen final mixer but read summaries include final reads.

## Reporting

`python scripts/summarize_paper.py` recomputes archived accuracy statistics with sample SD and seed-paired contrasts. These are observations from completed experiments, not a new execution of this package. Historical exposure/test-guided design remains a limitation; no pristine confirmation, equivalence or consistent routing superiority is established. Full datasets, checkpoints and raw server logs are not distributed.
