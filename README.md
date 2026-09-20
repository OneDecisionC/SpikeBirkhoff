# SpikeBirkhoff

Code for **SpikeBirkhoff: A Mechanistic Study of Residual Routing in Spiking Transformers** (manuscript, 2026).

Version 0.2.0 adds the revised paper's frozen protocols and mechanism diagnostics.

## Current paper
- `reproduction/frozen_core/`: archived CIFAR/N-Caltech model and training code.
- `reproduction/protocols/`: five-seed, validation-selected CIFAR-10, CIFAR-100 and N-Caltech101 protocols.
- `reproduction/cifar10dvs/`: archived AMP/TET implementation, independent-run GPU launcher and gated testing.
- `reproduction/mechanisms/`: interventions, membrane/layerwise/equal-perturbation probes, precision controls and training prefixes.
- `paper_data/`: de-identified numerical observations; no datasets or checkpoints.

**Read [the reproduction guide](docs/reproduction.md) before launching.** Staging does not start experiments:

```bash
python scripts/stage_reproduction.py --dataset cifar100 --data /absolute/data --workspace /absolute/new_run
python scripts/summarize_paper.py
```

The paper includes 100 full training runs: four methods, five seeds (42–46), five dataset/depth configurations. Forty official CIFAR-100 evaluations reuse selected checkpoints, not additional training. Historical test-guided development remains a limitation; resplitting does not erase it.

## Legacy workflow
Root `src/`, `configs/`, installed `spikebirkhoff-*` commands and earlier experiment/verification documents retain the v0.1 exploratory workflow. **These are not current holdout reproduction entry points.** See [the preserved README](docs/legacy_v010_README.md).

## Validation and license
CPU checks cover statistics, staging safety and existing configuration/result tests.

Original additions: MIT, Zhiqi Cai. Adapted portions retain upstream terms; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and `LICENSES/`. Paper authors are recorded separately in `CITATION.cff`.
