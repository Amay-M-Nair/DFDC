"""Data loading.

Evaluation: `make_loader(split, root=...)` reads one dataset's <split>/{real,fake} folders (either folder may be
missing: FFHQ test is real-only, StyleGAN test fake-only). Labels: real = 0, fake = 1.

Training (v2): `MixedTrainSet` draws from several datasets in fixed proportions per epoch:
    real          real crops of every non-synthetic dataset (FF++, Celeb-DF, FFHQ, CelebA-HQ)       label 0
    sbi           the same real crops, self-blended on the fly (needs cached landmarks)            label 1
    fake          the datasets' own fakes (FF++ methods, Celeb-DF)                                  label 1
    synthetic     fully generated faces (StyleGAN, diffusion); off by default, they train the CLIP probe
Within a group every dataset gets the same share, so a big dataset doesn't drown a small one, and an epoch is a
fixed number of samples however much data is attached. Every training image, real or fake, then goes through
the same augmentation, including the quality degradation in augment.Degrade.
"""
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import datasets, transforms

from src.classifier.augment import Degrade, RandomJPEG  # noqa: F401  (RandomJPEG re-exported for old imports)
from src.classifier.sbi import self_blend
from src.config import DATA_DIR, IMG_SIZE, MEAN, STD, SYNTHETIC

GROUP_SHARES = {"real": 0.4, "sbi": 0.35, "fake": 0.25, "synthetic": 0.0}


def train_transform():
    return transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.15, 0.15, 0.15, 0.02),
        Degrade(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
        # Hide a small random patch so the model can't rely on one region; kept rare so blending seams survive
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
    ])


def eval_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def split_dir(split: str, root=None):
    """Resolve a split folder under `root` (default DATA_DIR), accepting 'val' or 'valid'."""
    root = Path(root) if root else DATA_DIR
    aliases = {"val": ("val", "valid"), "valid": ("valid", "val")}.get(split, (split,))
    for name in aliases:
        if (root / name).is_dir():
            return root / name
    raise FileNotFoundError(f"No '{split}' folder under {root}")


class FolderLabel:
    """ImageFolder class index -> 0 (real) / 1 (fake) by folder name; works when only one class folder exists."""

    def __init__(self, classes):
        self.classes = classes

    def __call__(self, y: int) -> int:
        return int(self.classes[y] == "fake")


def make_dataset(split: str, train: bool = False, root=None):
    ds = datasets.ImageFolder(split_dir(split, root), transform=train_transform() if train else eval_transform())
    assert set(ds.classes) <= {"real", "fake"} and ds.classes, ds.class_to_idx
    ds.target_transform = FolderLabel(ds.classes)
    ds.targets = [ds.target_transform(t) for t in ds.targets]
    return ds


def make_loader(split, batch_size=32, train=False, num_workers=4, subset=None, root=None):
    ds = make_dataset(split, train, root)
    if subset:
        ds.samples = random.Random(0).sample(ds.samples, min(subset, len(ds.samples)))
        ds.imgs = ds.samples
    return DataLoader(ds, batch_size=batch_size, shuffle=train, num_workers=num_workers,
                      pin_memory=True, persistent_workers=num_workers > 0)


# ---------------------------------------------------------------------------------------------- v2 training data

def list_split(root, split: str):
    """[(key 'split/label/file', path, label)] for one dataset split; missing class folders are skipped."""
    try:
        base = split_dir(split, root)
    except FileNotFoundError:
        return []
    items = []
    for label in ("real", "fake"):
        d = base / label
        if d.is_dir():
            with os.scandir(d) as it:
                items += [(f"{split}/{label}/{e.name}", e.path, int(label == "fake"))
                          for e in it if e.name.lower().endswith((".jpg", ".jpeg", ".png"))]
    return sorted(items)


def load_rgb(path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:  # unusual formats
        return np.asarray(Image.open(path).convert("RGB"))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


class MixedTrainSet(Dataset):
    def __init__(self, roots: dict, landmarks: dict = None, transform=None, shares: dict = None, sbi: bool = True):
        """roots: {dataset name: folder with train/{real,fake}}; landmarks: {name: {'train/real/f.jpg': points}}."""
        landmarks = landmarks or {}
        shares = dict(shares or GROUP_SHARES)
        if not sbi:
            shares["sbi"] = 0.0
        self.transform = transform or train_transform()
        self.items = []  # (path, label, group, dataset, landmark points or None)
        for name, root in roots.items():
            marks = landmarks.get(name, {})
            synthetic = name in SYNTHETIC
            for key, path, label in list_split(root, "train"):
                if label:
                    self.items.append((path, 1, "synthetic" if synthetic else "fake", name, None))
                elif not synthetic:
                    self.items.append((path, 0, "real", name, None))
                    if key in marks and shares["sbi"] > 0:
                        self.items.append((path, 1, "sbi", name, marks[key]))
        counts = {}
        for _, _, g, d, _ in self.items:
            counts.setdefault(g, {}).setdefault(d, 0)
            counts[g][d] += 1
        self.counts = counts
        live = {g: s for g, s in shares.items() if s > 0 and g in counts}
        total = sum(live.values())
        if not total:
            raise ValueError(f"No training images for the requested groups {shares} under {roots}")
        self.shares = {g: s / total for g, s in live.items()}
        self.weights = torch.tensor([self.shares.get(g, 0.0) / len(counts[g]) / counts[g][d]
                                     for _, _, g, d, _ in self.items], dtype=torch.double)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label, group, _, points = self.items[i]
        img = load_rgb(path)
        if group == "sbi":
            img, _ = self_blend(img, points.astype(np.float32))
        return self.transform(Image.fromarray(img)), label

    def describe(self) -> str:
        lines = [f"{g:9} share {self.shares.get(g, 0):.2f}: " +
                 ", ".join(f"{d} {n}" for d, n in sorted(ds.items())) for g, ds in sorted(self.counts.items())]
        return "\n".join(lines)


class ListDataset(Dataset):
    """Fixed list of (key, path, label, dataset) items with an eval-style transform."""

    def __init__(self, items, transform=None):
        self.items, self.transform = items, transform or eval_transform()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        _, path, label, _ = self.items[i]
        return self.transform(Image.fromarray(load_rgb(path))), label


def split_items(roots: dict, split: str = "val", max_per_set: int = None, seed: int = 0):
    """[(key 'dataset/split/label/file', path, label, dataset)] for every dataset's split, at most `max_per_set`
    per dataset (seeded sample, so 02a's and 02b's validation lists are identical)."""
    out = []
    for name, root in sorted(roots.items()):
        items = list_split(root, split)
        if max_per_set and len(items) > max_per_set:
            items = sorted(random.Random(f"{seed}/{name}").sample(items, max_per_set))
        out += [(f"{name}/{key}", path, label, name) for key, path, label in items]
    return out


def seed_worker(_):
    seed = torch.initial_seed() % 2 ** 32
    np.random.seed(seed)
    random.seed(seed)


def mixed_loader(train_set: MixedTrainSet, samples_per_epoch: int, batch_size: int, num_workers: int = 4):
    sampler = WeightedRandomSampler(train_set.weights, samples_per_epoch, replacement=True)
    return DataLoader(train_set, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=True,
                      persistent_workers=num_workers > 0, worker_init_fn=seed_worker, drop_last=True)


def list_loader(items, batch_size: int = 64, num_workers: int = 4, transform=None):
    return DataLoader(ListDataset(items, transform), batch_size=batch_size, num_workers=num_workers, pin_memory=True)


def parse_roots(specs) -> dict:
    """['FF++=data', 'FFHQ=data_hq/ffhq'] -> {'FF++': Path('data'), ...}."""
    return {name: Path(root) for name, root in (s.split("=", 1) for s in specs)}


def load_landmarks(folder, names) -> dict:
    """{dataset: {key: points}} from <folder>/<dataset>.npz (written by src/classifier/landmarks.py)."""
    from src.classifier.landmarks import load

    found = {}
    if folder:
        for name in names:
            path = Path(folder) / f"{name}.npz"
            if path.exists():
                found[name] = load(path)
    return found
