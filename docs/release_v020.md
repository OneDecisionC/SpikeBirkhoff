# v0.2.0 export notes

The archived scientific implementations are preserved under reproduction/frozen_core and reproduction/cifar10dvs/vendor. Deployment adapters are separate. Changes to deployment include portable paths, a five-seed CIFAR-100 completion gate, explicit trusted-checkpoint acknowledgement for DVS final evaluation and configurable evaluation GPU allocation.

New scripts stage workspaces without executing experiments and recompute archived paper accuracy summaries. Old entry points and the original README are retained and labeled legacy. Numerical snapshots are de-identified; original local archives/reports are unchanged.

Validation: 14 lightweight result/release tests passed locally. Two legacy configuration tests could not run because PyYAML is absent from the lightweight environment. GPU/model tests and end-to-end training were not run for this export. Do not interpret the historical successful runs as a fresh validation of repackaged launchers.

Use scripts/release_manifest.py to verify hashes; --write refreshes only after intentional reviewed changes. No GitHub push is performed by these scripts.
