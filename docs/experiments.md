# Experiment protocols

The release provides seven experiment configurations. Each setting uses training seed 42, an embedding width of 256, eight attention heads, and a 200-epoch main schedule followed by 10 cooldown epochs. The method list for each configuration is in [`configs/experiments.json`](../configs/experiments.json).

| Configuration | Dataset | Depth | Time steps | Chunk size | Methods |
| --- | --- | ---: | ---: | ---: | --- |
| cifar100_d8_t4 | CIFAR-100 | 8 | 4 | 2 | S, B, U, Row-P, DS-P, A0, I, J |
| cifar100_d8_t2 | CIFAR-100 | 8 | 2 | 2 | S, B, I, U |
| cifar100_d16_t2 | CIFAR-100 | 16 | 2 | 2 | S, B, I, U |
| cifar100_d16_t4 | CIFAR-100 | 16 | 4 | 2 | S, B, I, U |
| cifar10_d6_t4 | CIFAR-10 | 6 | 4 | 2 | S, B, I, A0 |
| cifar10_d6_t2 | CIFAR-10 | 6 | 2 | 2 | S, B, I, U |
| ncaltech101_d2_t16 | N-Caltech101 | 2 | 16 | 4 | S, B, A0, I |

## Optimization

| Setting | CIFAR-10 / CIFAR-100 | N-Caltech101 |
| --- | ---: | ---: |
| Optimizer | AdamW | AdamW |
| Base learning rate | 0.0003 | 0.01 |
| Minimum learning rate | 0.00001 | 0.0003 |
| Warmup learning rate | 0.000001 | 0.001 |
| Warmup epochs | 20 | 10 |
| Weight decay | 0.06 | 0.0001 |
| Training / evaluation batch size | 64 / 64 | 16 / 16 |
| Data-loader workers | 4 | 4 |
| Main / cooldown epochs | 200 / 10 | 200 / 10 |

The training engine passes these rates directly to the timm optimizer and cosine scheduler. It does not scale them with batch size. Runs use float32, disable AMP and TF32, disable cuDNN benchmarking, and enable deterministic cuDNN. Model EMA and temporal efficient training are disabled. Training starts from a fresh model without pretrained weights or a resumed checkpoint.

The configured maximum drop-path rate is 0.2. The historical backbone creates per-block DropPath objects but does not call them in its forward pass. Applying stochastic depth would change the reported protocol. The legacy `use_conv_as_linear` YAML key is ignored by the engine.

## Methods and initialization

| Label | CLI method | Effective mixing matrix |
| --- | --- | --- |
| S | `standard` | Single stream |
| B | `birkhoff` | Learned softmax mixture of identity and swap |
| U | `unconstrained` | Learned raw 2 by 2 matrix |
| Row-P | `row_projected` | Row-wise Euclidean projection of a raw 2 by 2 matrix |
| DS-P | `ds_projected` | Euclidean projection of a raw 2 by 2 matrix onto the doubly stochastic set |
| A0 | `fixed_A0` | Frozen initial B matrix |
| I | `fixed_I` | Identity |
| J | `fixed_J` | Uniform matrix with every entry equal to 0.5 |

Multi-stream methods use two residual streams and one shared block computation. With `routing_init_scale=0.01`, the fixed input expansion is `2 * softmax([-0.01, 0.01])`, read logits start at `[0, 0]`, and write logits start at `[0.01, -0.01]`. Read weights use softmax and write weights use twice softmax. The initial B matrix is `A0 = alpha * I + (1 - alpha) * swap`, with `alpha = softmax([1.99, 0.01])[0]`. U, Row-P, and DS-P initialize their raw matrix to A0.

Every multi-stream method freezes final-layer write and mixing parameters. Fixed A0, I, and J freeze mixing in every layer; read logits remain trainable. Backbone initialization is paired within each configuration. Row-P and DS-P use raw-matrix projections, while B uses softmax coefficients, so their optimization parameterizations differ.

## Dataset processing

CIFAR-10 and CIFAR-100 use their official training and test sets. The historical timm dataset interface calls the evaluation split `validation`; no separate validation holdout is used. The image size is 32 by 32. CIFAR-10 mean/std are `[0.4914, 0.4822, 0.4465]` and `[0.2470, 0.2435, 0.2616]`; CIFAR-100 mean/std are `[0.5071, 0.4867, 0.4408]` and `[0.2675, 0.2565, 0.2761]`. Training uses the configured RandAugment policy, horizontal flipping, mixup alpha 0.5, and label smoothing 0.1. Mixup is disabled from epoch 200 onward.

N-Caltech101 uses 8,709 examples in 101 classes, split into 7,795 training and 914 test examples. Classes and filenames are sorted. A single shared NumPy `RandomState(42)` generates a permutation for each sorted class; the first `floor(0.9 * class_count)` examples form its training subset. The same explicit split is used for all methods.

The exact sample membership is included in [`data_splits/ncaltech101_seed42.json`](../data_splits/ncaltech101_seed42.json). To prepare the original `Caltech101.zip` archive:

```bash
python -m spikebirkhoff.prepare_ncaltech --archive Caltech101.zip --data-dir ./data/ncaltech101
```

Preparation verifies the original archive checksum, reproduces the recorded split checksum, and writes `split_seed42.json` and `frames_number_16_split_by_number` into the dataset directory. Use that directory as the training data directory. Add `--split-only` to verify the archive and write only the split. Existing identical outputs can be reused; different outputs produce an error.

Binary events are accumulated into 16 frames by event count. The first 15 frames contain `N // 16` events each and the final frame contains the remainder. Frames contain raw float32 counts in two polarity channels at 180 by 240, then use nearest-neighbor resizing to 64 by 64. No mean/std normalization, division by 255, or per-frame count normalization is applied. Training chooses one of spatial rolling by up to three pixels per axis, rotation by up to 15 degrees, or shear by up to 15 degrees. Evaluation only resizes. The event dataset loader determines these transforms; the YAML bicubic and horizontal-flip settings do not control this path.

## Metrics for your own runs

The training engine writes a summary CSV for each new run. Use `python -m spikebirkhoff.summarize --csv runs/EXPERIMENT/summary.csv` to compute these metrics:

- **Peak-test:** maximum monitored test accuracy during training over epochs 0-209. It is not an independent validation-selected test score.
- **Late20:** mean monitored test accuracy over epochs 190-209.
- **Final:** monitored test accuracy at epoch 209.

Compute differences before display rounding. Report all evaluated methods and metrics within each setting.
