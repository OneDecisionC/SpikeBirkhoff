# Release verification

Verified on 2026-09-16 before packaging version 0.1.0.

The code-only packaging update removes distributed experimental logs and replaces the log-dependent audit test with a synthetic fixture. All 10 configuration and summarizer checks passed after this update. The model and training implementation remain unchanged.

| Check | Outcome |
| --- | --- |
| PyTorch 1.13.1+cu117 / torchvision 0.14.1+cu117 / NumPy 1.23.5 | 82 tests passed |
| PyTorch 2.2.2 / torchvision 0.17.2 / NumPy 1.26.4 | 82 tests passed |
| Exact archived-model comparison on Torch CPU | 10 cases passed |
| Short CUDA/CuPy training and checkpoint reload in both environments | Passed |
| N-Caltech101 loader on prepared data | 7,795 train / 914 test; count tensors and augmentation loaded |
| N-Caltech101 preparation comparison | Six synthetic event cases and three real event samples matched the archived integration function exactly |
| Recorded N-Caltech101 split | SHA-256 matches the original split byte for byte |
| Wheel build | Passed |
| Public text scan | No Chinese text, private server addresses, credentials, or machine-specific source paths |

Both runtime environments used timm 0.6.12 and SpikingJelly 0.0.0.0.12. The archived-model audit covered all eight routing methods plus both alternate frozen parameter representations. Initialization, random-number state, parameter freezes, optimizer exclusions, effective matrices, outputs, features, and gradients agreed exactly in the small CPU fixtures, including nonzero routing gradients.

The pipeline smoke tests used eight synthetic images, a reduced model, and two batches to exercise optimization, diagnostics, checkpoint creation, strict reload, and evaluation. They are software checks, not accuracy experiments. Full 210-epoch paper runs were not repeated for this release.
