"""Evaluate internal test only after all checkpoint selections are frozen."""
import argparse
import json
from pathlib import Path

from holdout import digest, load_partition


def main():
    import sys
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--partition", choices=("validation", "internal_test"), required=True)
    args, remaining = parser.parse_known_args()
    root = Path(__file__).resolve().parent
    if args.partition == "internal_test":
        records = json.loads((root / "selection_frozen_before_test.json").read_text())
        protocol = json.loads((root / "protocol.json").read_text())
        expected = {(config, method, seed) for config in protocol["configs"]
                    for method in protocol["methods"] for seed in protocol["seeds"]}
        if len(records) != len(expected) or {(row["config"], row["method"], row["seed"]) for row in records} != expected:
            raise ValueError("All 24 selections must be frozen before any test evaluation")
        checkpoint = Path(remaining[remaining.index("--checkpoint") + 1]).resolve()
        matches = [row for row in records if Path(row["checkpoint"]).resolve() == checkpoint]
        if len(matches) != 1 or digest(checkpoint) != matches[0]["checkpoint_sha256"]:
            raise ValueError("Checkpoint was not selected before testing")
        if "--max-batches" in remaining:
            raise ValueError("No partial internal-test evaluation is allowed")
        with (root / "receipts" / (matches[0]["name"] + ".test_started.json")).open("x") as stream:
            json.dump({"checkpoint": str(checkpoint), "partition": args.partition}, stream)
    dataset = load_partition(root, args.partition)
    import timm.data
    from spikebirkhoff import evaluate

    def create_dataset(name, root, split, is_training, **kwargs):
        if name != "torch/cifar100" or is_training or split != "validation":
            raise ValueError("Unexpected evaluation dataset request")
        return dataset

    timm.data.create_dataset = create_dataset
    evaluate.main(remaining)
    output = Path(remaining[remaining.index("--output") + 1])
    result = json.loads(output.read_text())
    result.update(evaluation_partition=args.partition, official_cifar100_test=False,
                  split_sha256=digest(root / "split.json"),
                  selection_note="Validation-best checkpoint; internal test is not the official benchmark test.")
    output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
