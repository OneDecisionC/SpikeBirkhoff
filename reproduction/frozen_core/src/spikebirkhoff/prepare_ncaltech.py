# Copyright (c) 2026 Zhiqi Cai. SPDX-License-Identifier: MIT
"""Prepare the exact seed-42 N-Caltech101 split and 16 event-count frames.

Example::

    python -m spikebirkhoff.prepare_ncaltech --archive Caltech101.zip \
        --data-dir ./data/ncaltech101
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile

import numpy as np


ARCHIVE_SHA256 = "c70d387facd67f38f04a6f0afc4b1181416f2610cb924afa1f16fed99f7ebb9f"
SPLIT_SHA256 = "7b5c63a4016e65f5c5a5c9ed29f9e29c8e798d0a5b305343b51a6c53003f23f9"
FRAME_DIRECTORY = "frames_number_16_split_by_number"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def archive_samples(archive):
    """Return sorted original class/sample binary members, without extraction."""
    members = []
    for name in archive.namelist():
        path = PurePosixPath(name)
        if (len(path.parts) == 3 and path.parts[0] == "Caltech101"
                and path.suffix == ".bin"):
            if ".." in path.parts or "\\" in name:
                raise ValueError("Invalid archive member path")
            members.append(name)
    if len(members) != 8709 or len(set(members)) != len(members):
        raise ValueError("Expected 8,709 distinct N-Caltech101 binary examples")
    return sorted(members)


def make_split(members):
    """Reproduce the recorded JSON bytes using one shared RandomState."""
    classes = sorted({PurePosixPath(name).parent.name for name in members})
    if len(classes) != 101:
        raise ValueError("Expected exactly 101 classes")
    split = {
        "seed": 42,
        "classes": classes,
        "train": [],
        "test": [],
        "class_counts": {},
        "policy": ("sorted filenames then NumPy RandomState42 shared RNG "
                   "permutation per sorted class; floor(0.9*N) train"),
    }
    rng = np.random.RandomState(42)
    for label, class_name in enumerate(classes):
        names = sorted(
            class_name + "/" + PurePosixPath(name).stem + ".npz"
            for name in members if PurePosixPath(name).parent.name == class_name
        )
        order = rng.permutation(len(names))
        cut = int(len(names) * 0.9)
        for key, indices in (("train", order[:cut]), ("test", order[cut:])):
            split[key].extend({"relative": names[int(i)], "label": label}
                              for i in indices)
        split["class_counts"][class_name] = {
            "train": cut, "test": len(names) - cut
        }
    encoded = json.dumps(split, indent=2).encode("utf-8")
    if hashlib.sha256(encoded).hexdigest() != SPLIT_SHA256:
        raise ValueError("Generated split differs from the recorded protocol")
    return split, encoded


def integrate_binary_events(raw):
    """Count ordered ATIS events in equal-event bins, with remainder last.

    The input consists of five bytes per event: x, y, polarity/timestamp-high,
    timestamp-middle, timestamp-low. Timestamps are not reordered. The result
    has shape [16, 2, 180, 240] and contains unnormalized float32 event counts.
    """
    if len(raw) % 5:
        raise ValueError("ATIS data must contain five bytes per event")
    values = np.frombuffer(raw, dtype=np.uint8).astype(np.uint32)
    x = values[0::5]
    y = values[1::5]
    polarity = (values[2::5] & 128) >> 7
    count = len(x)
    if count < 16 or np.any(x >= 240) or np.any(y >= 180):
        raise ValueError("Event count or coordinates violate the dataset protocol")
    positions = (polarity * (180 * 240) + y * 240 + x).astype(np.int64)
    frames = np.empty((16, 2, 180, 240), dtype=np.float32)
    per_frame = count // 16
    for index in range(16):
        first = index * per_frame
        last = (index + 1) * per_frame if index < 15 else count
        counts = np.bincount(positions[first:last], minlength=2 * 180 * 240)
        converted = counts.astype(np.float32)
        if not np.array_equal(counts, converted):
            raise ValueError("Event counts cannot be represented exactly in float32")
        frames[index] = converted.reshape(2, 180, 240)
    if int(frames.astype(np.float64).sum()) != count:
        raise ValueError("Frame integration failed event conservation")
    return frames


def write_identical_or_new(path, data):
    """Allow reuse of an identical artifact; never replace a different file."""
    if path.exists():
        if path.read_bytes() != data:
            raise FileExistsError("Existing file differs: " + str(path))
    else:
        path.write_bytes(data)


def prepare(archive_path, data_dir, workers=4, split_only=False):
    """Prepare frames without extracting or executing archive contents."""
    archive_path = Path(archive_path)
    destination = Path(data_dir)
    if workers < 1:
        raise ValueError("workers must be positive")
    if file_sha256(archive_path) != ARCHIVE_SHA256:
        raise ValueError("Archive SHA-256 differs from the recorded dataset")
    with zipfile.ZipFile(archive_path) as archive:
        members = archive_samples(archive)
        split, encoded = make_split(members)
        destination.mkdir(parents=True, exist_ok=True)
        write_identical_or_new(destination / "split_seed42.json", encoded)
        if split_only:
            return {"split_sha256": SPLIT_SHA256,
                    "train": len(split["train"]), "test": len(split["test"])}
        frame_root = destination / FRAME_DIRECTORY
        for class_name in split["classes"]:
            (frame_root / class_name).mkdir(parents=True, exist_ok=True)

        def convert(member):
            relative = PurePosixPath(member)
            raw = archive.read(member)
            frames = integrate_binary_events(raw)
            target = frame_root / relative.parent.name / (relative.stem + ".npz")
            if target.exists():
                with np.load(target, allow_pickle=False) as existing:
                    if not np.array_equal(existing["frames"], frames):
                        raise FileExistsError("Existing frames differ: " + str(target))
            else:
                temporary = target.with_suffix(".npz.tmp")
                with temporary.open("wb") as output:
                    np.savez_compressed(output, frames=frames)
                temporary.replace(target)
            return len(raw) // 5

        events = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for index, count in enumerate(pool.map(convert, members), start=1):
                events += count
                if index % 100 == 0 or index == len(members):
                    print("Prepared {}/{} examples".format(index, len(members)),
                          flush=True)
    return {"archive_sha256": ARCHIVE_SHA256, "split_sha256": SPLIT_SHA256,
            "examples": len(members), "train": len(split["train"]),
            "test": len(split["test"]), "events": events,
            "frame_shape": [16, 2, 180, 240], "frame_dtype": "float32"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True,
                        help="Path to the original Caltech101.zip archive")
    parser.add_argument("--data-dir", required=True,
                        help="Destination dataset directory")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--split-only", action="store_true",
                        help="Verify the archive and write the exact split only")
    args = parser.parse_args(argv)
    result = prepare(args.archive, args.data_dir, args.workers, args.split_only)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
