"""Still-image face datasets (FFHQ, CelebA-HQ, StyleGAN, diffusion faces) -> face crops, one folder per source.

Each source becomes <out>/<name lower-case>/{train,val,test}/<label>/NAME__<stem>__0.jpg plus a manifest.csv
with a 'dataset' column, so the same loaders, leakage audit and comparison code work as for FF++. Crops use the
same YuNet detector and 1.3x square crop as the video datasets, so all sources are framed alike (the HQ sets
come pre-aligned and tightly framed; keeping their own framing would let a model tell sources apart by layout).
Faces that fill the frame are padded by reflection rather than squashed.

Splits are 80/10/10, decided by a hash of the file stem (stable whatever order files are found in), or by
identity when an identity file is given (CelebA-HQ: "<file> <identity>" per line).

    python -m src.classifier.image_extract --out data_hq \
        --source "FFHQ=real:/kaggle/input/140k/real_vs_fake/real-vs-fake/*/real/*.jpg" \
        --source "StyleGAN=fake:/kaggle/input/140k/real_vs_fake/real-vs-fake/*/fake/*.jpg" \
        --max-images 20000
"""
import argparse
import csv
import glob
import hashlib
import random
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from src.faces import ensure_yunet

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}
RATIOS = (0.8, 0.1, 0.1)

_cropper = None


def _init_worker():
    global _cropper
    from src.faces import FaceCropper
    _cropper = FaceCropper(strict=True, pad=True)


def split_of(key: str, ratios=RATIOS) -> str:
    u = int(hashlib.md5(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "train" if u < ratios[0] else "val" if u < ratios[0] + ratios[1] else "test"


def crop_one(job):
    path, name, stem, label, split, out_root = job
    try:
        img = np.asarray(Image.open(path).convert("RGB"))
    except Exception:
        return None
    face = _cropper.crop(img)
    if face is None:
        return None
    file = f"{name}__{stem}__0.jpg"
    Image.fromarray(face).save(Path(out_root) / split / label / file, quality=95)
    return split, label, name, stem, 0, file, name


def read_identities(path):
    ids = {}
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2:
            ids[Path(parts[0]).stem] = parts[1]
    return ids


def extract(name, label, pattern, out: Path, max_images=None, identities=None, workers=4, seed=0):
    paths = sorted(p for p in glob.glob(pattern, recursive=True) if Path(p).suffix.lower() in IMAGE_EXT)
    if not paths:
        raise SystemExit(f"{name}: no images match {pattern}")
    if max_images and len(paths) > max_images:
        paths = sorted(random.Random(seed).sample(paths, max_images))
    root = out / name.lower()
    for split in ("train", "val", "test"):
        (root / split / label).mkdir(parents=True, exist_ok=True)

    seen, jobs = Counter(), []
    for p in paths:
        stem = Path(p).stem.replace("__", "_")
        seen[stem] += 1
        if seen[stem] > 1:  # same file name in two sub-folders: keep both, distinct names
            stem = f"{stem}-{seen[stem]}"
        group = identities.get(Path(p).stem, stem) if identities else stem
        jobs.append((p, name, stem, label, split_of(f"{name}/{group}"), str(root)))

    rows = []
    with Pool(workers, initializer=_init_worker) as pool:
        for r in tqdm(pool.imap_unordered(crop_one, jobs, chunksize=32), total=len(jobs), desc=name):
            if r:
                rows.append(r)
    with open(root / "manifest.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "label", "method", "video", "frame", "file", "dataset"])
        w.writerows(sorted(rows))
    counts = Counter(r[0] for r in rows)
    print(f"{name}: {len(rows)} of {len(jobs)} images cropped ({len(jobs) - len(rows)} without a face) "
          f"{dict(sorted(counts.items()))} -> {root}")
    return root


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", action="append", required=True, help="NAME=label:glob (label: real or fake)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-images", type=int, default=None, help="cap per source (random, seeded)")
    ap.add_argument("--identity-file", action="append", default=[], help="NAME=path: '<file> <identity>' lines")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    id_files = dict(s.split("=", 1) for s in args.identity_file)
    ensure_yunet()  # download once, before the workers start
    for spec in args.source:
        name, rest = spec.split("=", 1)
        label, pattern = rest.split(":", 1)
        if label not in ("real", "fake"):
            raise SystemExit(f"{name}: label must be real or fake, got {label!r}")
        ids = read_identities(id_files[name]) if name in id_files else None
        extract(name, label, pattern, args.out, args.max_images, ids, args.workers)


if __name__ == "__main__":
    main()
