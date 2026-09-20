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
            raise ValueError("All supplemental selections must be frozen before any test evaluation")
        checkpoint = Path(remaining[remaining.index("--checkpoint") + 1]).resolve()
        matches = [row for row in records if Path(row["checkpoint"]).resolve() == checkpoint]
        if len(matches) != 1 or digest(checkpoint) != matches[0]["checkpoint_sha256"]:
            raise ValueError("Checkpoint was not selected before testing")
        if "--max-batches" in remaining:
            raise ValueError("No partial internal-test evaluation is allowed")
        with (root / "receipts" / (matches[0]["name"] + ".test_started.json")).open("x") as stream:
            json.dump({"checkpoint": str(checkpoint), "partition": args.partition}, stream)
    dataset = load_partition(root, args.partition)
    from spikebirkhoff import evaluate, datasets

    def build_ncaltech(data_path, transform=False, split_path=None):
        if transform or split_path is not None:
            raise ValueError("Evaluation must not augment or resplit records")
        return None, dataset

    datasets.build_ncaltech = build_ncaltech
    evaluate.main(remaining)
    output = Path(remaining[remaining.index("--output") + 1])
    result = json.loads(output.read_text())
    result.update(evaluation_partition=args.partition, historical_test_split=False,
                  split_sha256=digest(root / "split.json"),
                  selection_note="Validation-best checkpoint; internal test is not the historical N-Caltech101 test split.")
    output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
