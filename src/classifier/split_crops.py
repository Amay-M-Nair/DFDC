"""Split a test-only crop folder (e.g. Celeb-DF's official test list) into train/val/test by identity.

For a quick fine-tune when only the test crops exist. Every frame of one person lands in the same split, so the
held-out test part has no identity the model trained on. Frames are named <method>__<video stem>__<idx>.jpg
(ff_extract.extract_video); the identity is the stem's idN prefix (the person in the video for Celeb-real and
Celeb-synthesis, as in celebdf_extract.identity), and each YouTube-real video (numeric stem) is its own group.
Groups go to a split by an md5 hash of the identity, so the split is the same on every run.

    python -m src.classifier.split_crops --src path/to/celebdf_crops --out data_celebdf_split --val 0.15 --test 0.25

The output holds symlinks (copies where symlinks aren't allowed), in <out>/{train,val,test}/{real,fake}, which
train.py reads like any other dataset root. Scores on its test split are in-domain, not cross-dataset.
"""
import argparse
import hashlib
import os
import shutil
from collections import Counter
from pathlib import Path

from src.classifier.dataset import split_dir


def identity(name: str) -> str:
    parts = name.split("__")
    stem = parts[1] if len(parts) >= 3 else Path(name).stem
    return stem.split("_")[0] if stem.startswith("id") else f"yt{stem}"


def assign(ident: str, val: float, test: float) -> str:
    u = int(hashlib.md5(ident.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "test" if u < test else "val" if u < test + val else "train"


def link(src: Path, dst: Path) -> None:
    try:
        os.symlink(src.resolve(), dst)
    except OSError:
        shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True, help="folder with <split>/{real,fake} crops")
    ap.add_argument("--split", default="test", help="which split folder under --src to divide")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.25)
    args = ap.parse_args()

    base = split_dir(args.split, args.src)
    counts, groups = Counter(), {}
    for label in ("real", "fake"):
        folder = base / label
        if not folder.is_dir():
            continue
        for f in sorted(folder.iterdir()):
            if f.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            ident = identity(f.name)
            split = assign(ident, args.val, args.test)
            groups.setdefault(split, set()).add(ident)
            dst = args.out / split / label
            dst.mkdir(parents=True, exist_ok=True)
            if not (dst / f.name).exists():
                link(f, dst / f.name)
            counts[(split, label)] += 1
    if not counts:
        raise SystemExit(f"No images under {base}/{{real,fake}}")
    for split in ("train", "val", "test"):
        print(f"{split:5}: {counts[(split, 'real')]:6} real, {counts[(split, 'fake')]:6} fake, "
              f"{len(groups.get(split, ())):3} identities")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
