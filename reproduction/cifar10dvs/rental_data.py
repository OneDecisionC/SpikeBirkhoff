"""CIFAR10-DVS recording-level partitions; raw counts and historical resize."""
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def partition_records(records, seed):
    generator = np.random.RandomState(seed)
    result = {name: [] for name in ("train", "validation", "internal_test")}
    for label in range(10):
        group = sorted((row for row in records if row["label"] == label), key=lambda row: row["relative"])
        if len(group) != 1000:
            raise ValueError("Expected 1000 recordings per class")
        order = generator.permutation(1000)
        for name, indices in (("validation", order[:100]), ("internal_test", order[100:200]), ("train", order[200:])):
            result[name].extend(group[int(index)] for index in indices)
    for name in result:
        result[name].sort(key=lambda row: row["relative"])
    return result


def verify_split(split):
    identifiers = []
    digest_partition = {}
    for name, expected in (("train", 8000), ("validation", 1000), ("internal_test", 1000)):
        records = split["partitions"][name]
        if len(records) != expected:
            raise ValueError("Invalid partition size")
        counts = np.bincount([row["label"] for row in records], minlength=10)
        if not np.array_equal(counts, np.full(10, expected // 10)):
            raise ValueError("Class imbalance in frozen partition")
        for row in records:
            relative = Path(row["relative"])
            if relative.is_absolute() or ".." in relative.parts or len(relative.parts) != 2:
                raise ValueError("Unsafe recording path")
            if relative.parts[0] != split["classes"][row["label"]]:
                raise ValueError("Class label mismatch")
            identifiers.append(row["relative"])
            if "sha256" in row and digest_partition.setdefault(row["sha256"], name) != name:
                raise ValueError("Identical frame files cross partitions")
    if len(set(identifiers)) != 10000:
        raise ValueError("Duplicate or missing recordings across partitions")


def prepare(data, output, seed):
    frame_root = Path(data) / "frames_number_16_split_by_number"
    classes = sorted(path.name for path in frame_root.iterdir() if path.is_dir())
    if len(classes) != 10:
        raise ValueError("Expected ten class directories")
    records = []
    for label, category in enumerate(classes):
        files = sorted((frame_root / category).glob("*.npz"))
        for path in files:
            with np.load(path, allow_pickle=False) as archive:
                frames = archive["frames"]
                if frames.shape != (16, 2, 128, 128) or frames.dtype != np.float64 or not np.isfinite(frames).all() or frames.min() < 0:
                    raise ValueError("Invalid raw count frame file: " + str(path))
            records.append({"relative": category + "/" + path.name, "label": label, "sha256": sha256(path)})
    payload = {"classes": classes, "seed": seed, "partitions": partition_records(records, seed),
               "source_manifest_sha256": hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()}
    verify_split(payload)
    with Path(output).open("x") as stream:
        json.dump(payload, stream, sort_keys=True)
        stream.write("\n")
    return payload


class RecordingDataset:
    def __init__(self, data_root, split_path, partition):
        from torchvision import transforms
        split = json.loads(Path(split_path).read_text())
        verify_split(split)
        self.records = split["partitions"][partition]
        self.targets = [row["label"] for row in self.records]
        self.root = Path(data_root) / "frames_number_16_split_by_number"
        self.resize = transforms.Resize((64, 64), interpolation=transforms.InterpolationMode.BILINEAR, antialias=False)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        import torch
        row = self.records[index]
        with np.load(self.root / row["relative"], allow_pickle=False) as archive:
            frames = archive["frames"]
        if frames.shape != (16, 2, 128, 128):
            raise ValueError("Unexpected frame shape")
        resized = torch.stack([self.resize(torch.from_numpy(frame)) for frame in frames])
        return resized, row["label"]


def build_datasets(data_root, split_path):
    return RecordingDataset(data_root, split_path, "train"), RecordingDataset(data_root, split_path, "validation")
