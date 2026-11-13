"""Cache MediaPipe face-mesh landmarks for every real training/validation crop (used by self-blending).

Runs the 478-point face mesh once per image and keeps only the sbi.LANDMARKS subset (face oval + inner ring),
in pixels of the stored crop. One file per dataset:
    <out>/<NAME>.npz   keys:   "split/label/file" (str array)
                       points: N x len(LANDMARKS) x 2 (float16)
Images where MediaPipe finds no face are left out; training then uses them as real only, never blended.

    python -m src.classifier.landmarks --dataset FF++=data --dataset FFHQ=data_hq/ffhq --out landmarks
"""
import argparse
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from src.classifier.sbi import LANDMARKS

_mesh = None


def _init():
    global _mesh
    from src.faces import FaceMesh
    _mesh = FaceMesh()


def _points(path):
    try:
        pts = _mesh.points(np.asarray(Image.open(path).convert("RGB")))
    except Exception:
        return None
    return None if pts is None else pts[LANDMARKS].astype(np.float16)


def list_images(root: Path, splits, labels):
    """(key 'split/label/file', path) for every crop under root/<split>/<label>/."""
    out = []
    for split in splits:
        for label in labels:
            d = root / split / label
            if d.is_dir():
                with os.scandir(d) as it:
                    out += [(f"{split}/{label}/{e.name}", e.path) for e in it if e.name.lower().endswith((".jpg", ".png"))]
    return sorted(out)


def cache(name: str, root: Path, out: Path, splits=("train", "val"), labels=("real",), workers: int = 4):
    items = list_images(root, splits, labels)
    if not items:
        print(f"{name}: no images under {root} for {splits} x {labels}")
        return None
    from src.faces import ensure_face_mesh
    ensure_face_mesh()  # download once, before the workers start
    with Pool(workers, initializer=_init) as pool:
        pts = list(tqdm(pool.imap(_points, [p for _, p in items], chunksize=64), total=len(items), desc=name))
    keep = [i for i, p in enumerate(pts) if p is not None]
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.npz"
    np.savez(path, keys=np.array([items[i][0] for i in keep]), points=np.stack([pts[i] for i in keep]))
    print(f"{name}: landmarks for {len(keep)} of {len(items)} images ({len(keep) / len(items):.1%}) -> {path}")
    return path


def load(path) -> dict:
    """{'split/label/file': points} from a landmarks .npz."""
    z = np.load(path)
    return dict(zip(z["keys"].tolist(), z["points"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", action="append", required=True, help="NAME=root (root has <split>/<label>/)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--labels", nargs="+", default=["real"])
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    for spec in args.dataset:
        name, root = spec.split("=", 1)
        cache(name, Path(root), args.out, args.splits, args.labels, args.workers)


if __name__ == "__main__":
    main()
