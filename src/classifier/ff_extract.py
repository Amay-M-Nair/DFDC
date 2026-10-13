"""FaceForensics++ videos -> face-crop frames in data/{train,val,test}/{real,fake}.

Finds videos under --src automatically, so it works with the official download layout
    original_sequences/youtube/c23/videos/000.mp4
    manipulated_sequences/Deepfakes/c23/videos/000_003.mp4
and with the flattened Kaggle mirrors (e.g. original/000.mp4, Deepfakes/000_003.mp4).

Splitting is done per identity, never per frame: a fake "000_003" links ids 000 and 003, and every
connected group of ids goes to exactly one split, so no face appears in two splits. Pass the official
FF++ split files with --splits (dir with train.json / val.json / test.json) to use the paper's split.

Real videos get more frames than each fake video so the two classes are roughly balanced.

    python -m src.classifier.ff_extract --src /path/to/FaceForensics++ --frames 10
"""
import argparse
import csv
import json
import random
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import cv2
from PIL import Image
from tqdm import tqdm

from src.config import DATA_DIR

FAKE_METHODS = ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "FaceShifter")
REAL_MARKERS = ("original_sequences", "original", "youtube", "real")
VIDEO_EXT = {".mp4", ".avi", ".mov"}


def classify(path: Path, methods):
    """Return 'real', a fake method name, or None (skip) for a video path."""
    parts = [p.lower() for p in path.parts]
    if "actors" in parts or "deepfakedetection" in parts:  # separate Google DFD subset, different ids
        return None
    for m in methods:
        if m.lower() in parts:
            return m
    if any(m in parts for m in REAL_MARKERS):
        return "real"
    return None


def discover(src: Path, methods, compression):
    videos = []
    for p in src.rglob("*"):
        if p.suffix.lower() not in VIDEO_EXT:
            continue
        parts = {q.lower() for q in p.parts}
        levels = parts & {"raw", "c0", "c23", "c40"}
        if compression and levels and compression not in levels:
            continue
        label = classify(p.relative_to(src), methods)
        if label:
            videos.append((p, label))
    return videos


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        self.parent[self.find(a)] = self.find(b)


def ids_of(path: Path):
    return path.stem.split("_")[:2]


def assign_splits(videos, splits_dir=None, ratios=(0.72, 0.14, 0.14), seed=0):
    """Map every identity id -> 'train' | 'val' | 'test'."""
    if splits_dir:
        id_split = {}
        for name in ("train", "val", "test"):
            for a, b in json.loads((Path(splits_dir) / f"{name}.json").read_text()):
                id_split[a] = id_split[b] = name
        return id_split

    uf = UnionFind()
    for p, _ in videos:
        ids = ids_of(p)
        uf.find(ids[0])
        if len(ids) == 2:
            uf.union(ids[0], ids[1])
    groups = defaultdict(list)
    for x in list(uf.parent):
        groups[uf.find(x)].append(x)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)

    total, id_split, seen = sum(len(g) for g in groups.values()), {}, 0
    for k in keys:
        frac = seen / total
        split = "train" if frac < ratios[0] else "val" if frac < ratios[0] + ratios[1] else "test"
        for x in groups[k]:
            id_split[x] = split
        seen += len(groups[k])
    return id_split


_cropper = None


def _init_worker():
    global _cropper
    from src.faces import FaceCropper
    _cropper = FaceCropper()


def extract_video(job):
    path, label, split, n_frames, out_root = job
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        return []
    idxs = sorted({int(i * total / n_frames) for i in range(n_frames)})
    method = "original" if label == "real" else label
    cls = "real" if label == "real" else "fake"
    out_dir = Path(out_root) / split / cls
    rows = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        face = _cropper.crop(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if face is None:
            continue
        name = f"{method}__{path.stem}__{idx:05d}.jpg"
        Image.fromarray(face).save(out_dir / name, quality=95)
        rows.append((split, cls, method, path.stem, idx, name))
    cap.release()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=DATA_DIR)
    ap.add_argument("--methods", nargs="+", default=list(FAKE_METHODS))
    ap.add_argument("--compression", default="c23", help="raw/c0, c23 or c40 (ignored if layout has none)")
    ap.add_argument("--frames", type=int, default=10, help="frames per fake video")
    ap.add_argument("--real-frames", type=int, default=None, help="frames per real video (default: balanced)")
    ap.add_argument("--max-videos", type=int, default=None, help="cap videos per method (quick runs)")
    ap.add_argument("--splits", type=Path, default=None, help="dir with official train/val/test.json")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    videos = discover(args.src, args.methods, args.compression.lower())
    by_label = defaultdict(list)
    for p, label in videos:
        by_label[label].append(p)
    if args.max_videos:
        rng = random.Random(0)
        by_label = {k: sorted(rng.sample(v, min(len(v), args.max_videos))) for k, v in by_label.items()}
    videos = [(p, label) for label, ps in by_label.items() for p in ps]
    print({k: len(v) for k, v in by_label.items()})
    if "real" not in by_label or len(by_label) < 2:
        raise SystemExit(f"Need real and fake videos under {args.src}; found {sorted(by_label)}")

    n_fake_methods = len(by_label) - 1
    real_frames = args.real_frames or min(args.frames * n_fake_methods, 100)
    id_split = assign_splits(videos, args.splits)

    for split in ("train", "val", "test"):
        for cls in ("real", "fake"):
            (args.out / split / cls).mkdir(parents=True, exist_ok=True)
    jobs = []
    for p, label in videos:
        split = id_split.get(ids_of(p)[0])
        if split:
            jobs.append((p, label, split, real_frames if label == "real" else args.frames, str(args.out)))

    rows = []
    with Pool(args.workers, initializer=_init_worker) as pool:
        for r in tqdm(pool.imap_unordered(extract_video, jobs), total=len(jobs), desc="videos"):
            rows.extend(r)

    with open(args.out / "manifest.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "label", "method", "video", "frame", "file"])
        w.writerows(sorted(rows))
    if not rows:
        raise SystemExit("No faces were extracted; check the videos and that the face detector loaded.")
    counts = defaultdict(int)
    for split, cls, *_ in rows:
        counts[f"{split}/{cls}"] += 1
    print(dict(sorted(counts.items())))


if __name__ == "__main__":
    main()
