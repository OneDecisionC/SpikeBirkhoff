# Archived manuscript observations

De-identified exports from completed experiments, not reruns of this public package.

- accuracy.json: 100 internal (H) plus 40 official CIFAR-100 (O) evaluations; identical checkpoint hashes for H/O. Five seeds; sample SD uses ddof=1. Pair seeds before contrast SD.
- mechanisms.json: validation-only interventions; whole-probe spike disagreement differs from sampled membrane statistics.
- precision.json: native AMP versus FP32 controls on existing DVS checkpoints, not retraining.
- equal_perturbation.json: layerwise matched perturbations with configuration/precision labels. Missing final downstream statistics mean N/A, not zero.

Source digests identify the local report snapshots, not independently downloadable archives. No datasets, trained checkpoints or private server paths are included.
