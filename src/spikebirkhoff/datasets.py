# Adapted from the STAtten dataset loader; see LICENSES/STAtten-MIT.txt.
# Modifications (c) 2026 Zhiqi Cai: explicit split validation and package interface.
# Original project contributions are licensed under MIT.
"""N-Caltech101 frame loading and augmentation for the released protocol."""

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


DVS_DATASET = ["ncaltech101"]


class NCaltech101_aug(Dataset):
    """Load the fixed split of raw event-count frames without normalization.

    ``data_path`` contains ``frames_number_16_split_by_number`` and the
    ``split_seed42.json`` produced by :mod:`spikebirkhoff.prepare_ncaltech`.
    """

    def __init__(self, data_path=None, data_type="train", transform=True,
                 split_path=None):
        if data_type not in ("train", "test"):
            raise ValueError("data_type must be 'train' or 'test'")
        root = Path(data_path)
        frame_root = root / "frames_number_16_split_by_number"
        self.filepath = str(frame_root)
        self.clslist = sorted(path.name for path in frame_root.iterdir()
                              if path.is_dir())
        split_file = Path(split_path) if split_path else root / "split_seed42.json"
        with split_file.open(encoding="utf-8") as source:
            split = json.load(source)
        if split["seed"] != 42 or self.clslist != split["classes"]:
            raise ValueError("Frame classes do not match the recorded seed-42 split")

        self.dvs_filelist = []
        self.targets = []
        for record in split[data_type]:
            relative = Path(record["relative"])
            if (relative.is_absolute() or ".." in relative.parts
                    or len(relative.parts) != 2):
                raise ValueError("Split entries must be relative class/sample paths")
            label = record["label"]
            if not 0 <= label < len(self.clslist):
                raise ValueError("Split label is outside the class range")
            if relative.parts[0] != self.clslist[label]:
                raise ValueError("Split path and class label disagree")
            self.dvs_filelist.append(str(frame_root / relative))
            self.targets.append(label)

        self.data_num = len(self.dvs_filelist)
        self.data_type = data_type
        self.classes = range(len(self.clslist))
        if data_type != "train":
            counts = np.unique(np.asarray(self.targets), return_counts=True)[1]
            self.class_weights = torch.Tensor(counts.sum() / (counts * len(counts)))
        self.transform = transform
        self.resize = transforms.Resize(
            size=(64, 64), interpolation=transforms.InterpolationMode.NEAREST
        )
        self.rotate = transforms.RandomRotation(degrees=15)
        self.shearx = transforms.RandomAffine(degrees=0, shear=(-15, 15))

    def __len__(self):
        return self.data_num

    def __getitem__(self, index):
        with np.load(self.dvs_filelist[index], allow_pickle=False) as archive:
            data = torch.from_numpy(archive["frames"]).float()
        if tuple(data.shape) != (16, 2, 180, 240):
            raise ValueError("Expected frames with shape [16, 2, 180, 240]")
        data = self.resize(data)
        if self.transform:
            augmentation = np.random.choice(["roll", "rotate", "shear"])
            if augmentation == "roll":
                offset_y = random.randint(-3, 3)
                offset_x = random.randint(-3, 3)
                data = torch.roll(data, shifts=(offset_y, offset_x), dims=(2, 3))
            elif augmentation == "rotate":
                data = self.rotate(data)
            else:
                data = self.shearx(data)
        return data, self.targets[index]


def build_ncaltech(data_path, transform=False, split_path=None):
    """Return training and test datasets, preserving explicit split order."""
    train = NCaltech101_aug(data_path, "train", transform, split_path)
    test = NCaltech101_aug(data_path, "test", False, split_path)
    return train, test


__all__ = ["DVS_DATASET", "NCaltech101_aug", "build_ncaltech"]
