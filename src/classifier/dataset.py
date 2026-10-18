import io
import random
from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.config import DATA_DIR, IMG_SIZE, MEAN, STD


class RandomJPEG:
    """Re-encode as JPEG at a random quality — makes the model robust to compression."""

    def __init__(self, p=0.5, quality=(40, 95)):
        self.p, self.quality = p, quality

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=random.randint(*self.quality))
        buf.seek(0)
        return Image.open(buf).convert("RGB")


def flip_label(y: int) -> int:
    return 1 - y


def train_transform():
    return transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.02),
        RandomJPEG(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
        # Hide a random patch so the model can't rely on a single facial region (reduces overfitting)
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)),
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


def make_dataset(split: str, train: bool = False, root=None):
    ds = datasets.ImageFolder(split_dir(split, root), transform=train_transform() if train else eval_transform())
    # ImageFolder sorts classes alphabetically: fake=0, real=1. Flip so fake=1.
    assert ds.class_to_idx == {"fake": 0, "real": 1}, ds.class_to_idx
    ds.target_transform = flip_label
    return ds


def make_loader(split, batch_size=32, train=False, num_workers=4, subset=None, root=None):
    ds = make_dataset(split, train, root)
    if subset:
        ds.samples = random.Random(0).sample(ds.samples, min(subset, len(ds.samples)))
        ds.imgs = ds.samples
    return DataLoader(ds, batch_size=batch_size, shuffle=train, num_workers=num_workers,
                      pin_memory=True, persistent_workers=num_workers > 0)
