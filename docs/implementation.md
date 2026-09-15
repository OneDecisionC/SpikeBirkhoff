# Implementation notes

The backbone and training engine are adapted from the archived experiment implementations, with package-relative imports and a unified routing interface. The original STAtten, Spike-Driven Transformer, and timm notices are retained. The release does not contain private orchestration scripts, machine addresses, credentials, data, or checkpoint weights.

For a routed block, streams are read with a convex weight vector, the original block is evaluated once, and its residual increment is written back to both streams. Skip mixing uses the selected control. The Birkhoff mode mixes the two permutation matrices through softmax weights. The doubly stochastic property applies to this skip branch; it does not establish contraction of the full nonlinear network.

## Preserved details

- Routing initialization does not consume the backbone's random-number stream. All trainable routing logits are excluded from weight decay.
- The final block's mixing and write parameters are frozen; its read remains trainable. The output readout averages streams uniformly.
- Fixed identity and fixed initial mixing appeared with two different frozen parameter storage shapes in the archived implementations. Effective matrices agree. The builder preserves the configuration-specific shape and strict checkpoint loading accepts both frozen forms.
- The archived block constructs DropPath but never calls it in `forward`. The nominal configuration value of 0.2 is preserved without adding an operation absent from the experiments.
- Neuron state is reset between batches. The default backend is CuPy; the Torch backend enables CPU checks. Backend or dependency changes need not be bitwise equivalent to a complete archived GPU training run.
- The training engine retains the original optimizer, augmentation, scheduler, test monitoring, and checkpoint selection. New entry points add explicit configuration resolution, method selection, run metadata, and overwrite protection.
- Training is supported as one process per GPU, matching the paper. Run independent configurations on different GPUs rather than changing the batch protocol through distributed training.

## Repository map

| Path | Purpose |
| --- | --- |
| `src/spikebirkhoff/model.py` | STAtten backbone and stream orchestration |
| `src/spikebirkhoff/routing.py` | Unified routing controls |
| `src/spikebirkhoff/layers/` | Spiking patch embedding and attention blocks |
| `src/spikebirkhoff/train.py` | Safe configuration and run entry point |
| `src/spikebirkhoff/training_engine.py` | Archived optimization loop adapted to this package |
| `src/spikebirkhoff/evaluate.py` | Checkpoint reload and complete test-set evaluation |
| `src/spikebirkhoff/prepare_ncaltech.py` | Deterministic event-frame preparation |
| `configs/` | Seven protocols and the method registry |
| `data_splits/` | Exact N-Caltech101 sample partition |
| `tests/` | Routing, model, configuration, and log audit checks |

The release tests validate selected implementation properties and short executions. They do not replace rerunning full-length experiments. Experimental results are not included in this repository.

See [release verification](verification.md) for the tested environments and outcomes.
