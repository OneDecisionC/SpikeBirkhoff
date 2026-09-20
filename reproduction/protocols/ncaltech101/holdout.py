"""Stratified recording-level N-Caltech101 split with separate validation."""
import hashlib
import json
from pathlib import Path

import numpy as np
from torchvision import transforms
from spikebirkhoff.datasets import NCaltech101_aug


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def partition_records(records, seed=20260919):
    generator = np.random.RandomState(seed)
    parts = {name: [] for name in ("train", "validation", "internal_test")}
    labels = sorted(set(record["label"] for record in records))
    for label in labels:
        group = sorted((record for record in records if record["label"] == label), key=lambda record: record["relative"])
        if len(group) < 3:
            raise ValueError("Each class needs at least three complete recordings")
        order = generator.permutation(len(group))
        held = max(1, int(len(group) * 0.1 + 0.5))
        for name, indices in (("validation", order[:held]), ("internal_test", order[held:2 * held]), ("train", order[2 * held:])):
            parts[name].extend(group[int(index)] for index in indices)
    for name in parts:
        parts[name].sort(key=lambda record: record["relative"])
    identifiers = [record["relative"] for part in parts.values() for record in part]
    if len(set(identifiers)) != len(records) or len(identifiers) != len(records):
        raise ValueError("Recording leakage or missing recordings")
    return parts


def make_split(data_root, destination):
    frame_root = Path(data_root) / "frames_number_16_split_by_number"
    classes = sorted(path.name for path in frame_root.iterdir() if path.is_dir())
    if len(classes) != 101:
        raise ValueError("Expected the historical 101-class dataset")
    records = []
    for label, category in enumerate(classes):
        for path in sorted((frame_root / category).glob("*.npz")):
            records.append({"relative": category + "/" + path.name, "label": label,
                            "sha256": digest(path)})
    if len(records) != 8709:
        raise ValueError("Expected all 8709 complete recordings")
    parts = partition_records(records)
    if parts != partition_records(records):
        raise ValueError("Split is not reproducible")
    assignment = {}
    for name, group in parts.items():
        for record in group:
            previous = assignment.setdefault(record["sha256"], name)
            if previous != name:
                raise ValueError("Identical frame files cross partitions; grouping must be reviewed")
    source_hash = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
    payload = {"split_seed": 20260919, "classes": classes, "source_sha256": source_hash,
               "partitions": parts, "counts": {name: len(group) for name, group in parts.items()},
               "unit": "entire recording; all 16 frames stay together",
               "rounding": "per class holdout size=max(1,floor(0.1*n+0.5)); remainder is training"}
    with (Path(destination) / "split.json").open("x") as stream:
        json.dump(payload, stream, sort_keys=True)
        stream.write("\n")
    print("Frozen partition sizes: " + json.dumps(payload["counts"]), flush=True)


class RecordingPartition(NCaltech101_aug):
    def __init__(self, data_root, split, name):
        frame_root = Path(data_root) / "frames_number_16_split_by_number"
        records = split["partitions"][name]
        for record in records:
            relative = Path(record["relative"])
            if relative.is_absolute() or ".." in relative.parts or len(relative.parts) != 2:
                raise ValueError("Invalid recording path")
            if relative.parts[0] != split["classes"][record["label"]]:
                raise ValueError("Class label mismatch")
        self.dvs_filelist = [str(frame_root / record["relative"]) for record in records]
        self.targets = [record["label"] for record in records]
        self.data_num = len(records)
        self.transform = name == "train"
        self.data_type = "train" if self.transform else "test"
        self.classes = range(len(split["classes"]))
        self.resize = transforms.Resize((64, 64), interpolation=transforms.InterpolationMode.NEAREST)
        self.rotate = transforms.RandomRotation(degrees=15)
        self.shearx = transforms.RandomAffine(degrees=0, shear=(-15, 15))


def load_partition(root, name):
    root = Path(root)
    protocol = json.loads((root / "protocol.json").read_text())
    split = json.loads((root / "split.json").read_text())
    return RecordingPartition(protocol["data_root"], split, name)


def main():
    import sys
    from spikebirkhoff import train, training_engine

    root = Path(__file__).resolve().parent
    protocol = json.loads((root / "protocol.json").read_text())
    calls = []

    def build_ncaltech(data_path, transform=False, split_path=None):
        if str(data_path) != protocol["data_root"] or not transform or split_path is not None:
            raise ValueError("Unexpected training dataset request")
        calls.append("train_and_validation_only")
        return load_partition(root, "train"), load_partition(root, "validation")

    training_engine.dvs_utils.build_ncaltech = build_ncaltech
    train.main(sys.argv[1:])
    if calls != ["train_and_validation_only"]:
        raise ValueError("Unexpected dataset access sequence")
    print("Training used no internal-test recordings", flush=True)


if __name__ == "__main__":
    main()
