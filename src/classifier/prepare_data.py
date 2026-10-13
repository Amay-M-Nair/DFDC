"""Copy a (sub-sampled) deepfake dataset into data/{train,val,test}/{real,fake}.

Default source layout is Kaggle "140k Real and Fake Faces":
    <src>/real_vs_fake/real-vs-fake/{train,valid,test}/{real,fake}/*.jpg

    python -m src.classifier.prepare_data --src path/to/140k --train 20000 --val 4000 --test 4000
"""
import argparse
import random
import shutil
from pathlib import Path

from tqdm import tqdm

from src.config import DATA_DIR

SPLITS = {"train": "train", "valid": "val", "test": "test"}


def find_root(src: Path) -> Path:
    for cand in [src, src / "real-vs-fake", src / "real_vs_fake" / "real-vs-fake"]:
        if (cand / "train").is_dir():
            return cand
    raise FileNotFoundError(f"Could not find train/ under {src}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--train", type=int, default=20000, help="images per split (half real, half fake)")
    ap.add_argument("--val", type=int, default=4000)
    ap.add_argument("--test", type=int, default=4000)
    args = ap.parse_args()

    root = find_root(args.src)
    rng = random.Random(42)
    sizes = {"train": args.train, "val": args.val, "test": args.test}
    for src_split, dst_split in SPLITS.items():
        if not (root / src_split).is_dir():
            src_split = dst_split  # dataset already uses val/
        for cls in ("real", "fake"):
            files = sorted((root / src_split / cls).glob("*.*"))
            files = rng.sample(files, min(len(files), sizes[dst_split] // 2))
            out = DATA_DIR / dst_split / cls
            out.mkdir(parents=True, exist_ok=True)
            for f in tqdm(files, desc=f"{dst_split}/{cls}"):
                shutil.copy2(f, out / f.name)


if __name__ == "__main__":
    main()
