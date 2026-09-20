"""Frozen 40k/5k/5k partitions of the original CIFAR-10 training set."""
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
from PIL import Image


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def split_indices(labels):
    if not np.array_equal(np.bincount(labels), np.full(10, 5000)):
        raise ValueError("Expected 5000 original training examples per class")
    generator = np.random.RandomState(20260916)
    groups = {name: [] for name in ("train", "validation", "internal_test")}
    for category in range(10):
        indices = generator.permutation(np.flatnonzero(labels == category))
        groups["validation"].extend(indices[:500].tolist())
        groups["internal_test"].extend(indices[500:1000].tolist())
        groups["train"].extend(indices[1000:].tolist())
    groups = {name: sorted(indices) for name, indices in groups.items()}
    if sorted(sum(groups.values(), [])) != list(range(50000)):
        raise ValueError("Partitions overlap or omit examples")
    for name, expected in (("train", 4000), ("validation", 500), ("internal_test", 500)):
        if not np.array_equal(np.bincount(labels[groups[name]], minlength=10), np.full(10, expected)):
            raise ValueError("Unbalanced partition")
    return groups


def make_split(data_root, destination):
    destination = Path(destination)
    sources = [Path(data_root) / "cifar-10-batches-py" / ("data_batch_" + str(index)) for index in range(1, 6)]
    batches = []
    for source in sources:
        with source.open("rb") as stream:
            batches.append(pickle.load(stream, encoding="latin1"))
    images = np.concatenate([batch["data"] for batch in batches]).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    labels = np.concatenate([np.asarray(batch["labels"], dtype=np.int64) for batch in batches])
    source_hash = hashlib.sha256("".join(digest(source) for source in sources).encode()).hexdigest()
    groups = split_indices(labels)
    paths = {}
    for name, indices in groups.items():
        path = destination / (name + ".npz")
        with path.open("xb") as stream:
            np.savez(stream, images=images[indices], labels=labels[indices])
        paths[path.name] = digest(path)
    with (destination / "split.json").open("x") as stream:
        json.dump({"split_seed": 20260916, "source_sha256": source_hash,
                   "indices": groups, "partition_sha256": paths,
                   "official_test_used": False}, stream, sort_keys=True)
        stream.write("\n")


class TrainingPartition:
    def __init__(self, images, labels):
        self.data = images
        self.targets = labels.tolist()
        self.transform = None

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        image = Image.fromarray(self.data[index])
        if self.transform is not None:
            image = self.transform(image)
        return image, self.targets[index]


def load_partition(root, name):
    expected = {"train": 40000, "validation": 5000, "internal_test": 5000}[name]
    root = Path(root)
    split = json.loads((root / "split.json").read_text())
    path = root / (name + ".npz")
    if digest(path) != split["partition_sha256"][path.name]:
        raise ValueError("Partition changed since preparation")
    with np.load(path, allow_pickle=False) as arrays:
        images, labels = arrays["images"], arrays["labels"]
    if images.shape != (expected, 32, 32, 3) or len(labels) != expected:
        raise ValueError("Invalid partition dimensions")
    return TrainingPartition(images, labels)


def main():
    import sys
    from spikebirkhoff import train, training_engine

    root = Path(__file__).resolve().parent
    protocol = json.loads((root / "protocol.json").read_text())
    training = load_partition(root, "train")
    validation = load_partition(root, "validation")
    calls = []

    def create_dataset(name, root, split, is_training, **kwargs):
        if name != "torch/cifar10" or str(root) != protocol["data_root"]:
            raise ValueError("Dataset differs from frozen protocol")
        if (split, is_training) == ("train", True):
            calls.append("train_40000")
            return training
        if (split, is_training) == ("validation", False):
            calls.append("validation_5000")
            return validation
        raise ValueError("Test access is forbidden during training")

    training_engine.create_dataset = create_dataset
    train.main(sys.argv[1:])
    if calls != ["train_40000", "validation_5000"]:
        raise ValueError("Unexpected dataset access sequence")
    print(json.dumps({"dataset_access": calls, "test_evaluated": False}), flush=True)


if __name__ == "__main__":
    main()
