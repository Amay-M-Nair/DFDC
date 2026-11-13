"""DFDC (Deepfake Detection Challenge) videos -> face-crop frames in <out>/test/{real,fake}: the held-out
cross-dataset test. Never used for training, early stopping or calibration.

Labels come from the metadata.json next to the videos (train_sample_videos/ or the dfdc_train_part_N folders):
    {"aagfhgtpmv.mp4": {"label": "FAKE", "original": "vudstovrck.mp4", ...}, ...}
DFDC is about 80% fake; by default every real video is kept and as many fake videos are sampled (seeded).
The same face detector and crop as the other datasets are used; the largest face in each frame is taken.

    python -m src.classifier.dfdc_extract --src /kaggle/input/deepfake-detection-challenge --out data_dfdc
"""
import argparse
import csv
import json
import random
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

from tqdm import tqdm

from src.classifier.ff_extract import _init_worker, extract_video
from src.faces import ensure_yunet

METHOD = "DFDC"


def discover(src: Path):
    """[(video path, 'real' | 'fake')] from every metadata.json under src."""
    videos = []
    for meta in sorted(src.rglob("metadata.json")):
        for name, info in json.loads(meta.read_text()).items():
            path = meta.parent / name
            if path.exists() and info.get("label") in ("REAL", "FAKE"):
                videos.append((path, info["label"].lower()))
    return videos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--frames", type=int, default=10, help="frames per video")
    ap.add_argument("--max-videos", type=int, default=None, help="cap per class")
    ap.add_argument("--no-balance", action="store_true", help="keep every fake video instead of matching reals")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    videos = discover(args.src)
    if not videos:
        raise SystemExit(f"No DFDC videos with a metadata.json label under {args.src}")
    by_label = {lab: sorted(p for p, l in videos if l == lab) for lab in ("real", "fake")}
    rng = random.Random(0)
    if not args.no_balance and len(by_label["fake"]) > len(by_label["real"]):
        by_label["fake"] = sorted(rng.sample(by_label["fake"], len(by_label["real"])))
    if args.max_videos:
        by_label = {k: sorted(rng.sample(v, min(len(v), args.max_videos))) for k, v in by_label.items()}
    print({k: len(v) for k, v in by_label.items()}, "videos")

    for cls in ("real", "fake"):
        (args.out / "test" / cls).mkdir(parents=True, exist_ok=True)
    jobs = [(p, "real" if lab == "real" else METHOD, "test", args.frames, str(args.out))
            for lab, ps in by_label.items() for p in ps]

    ensure_yunet()  # download once, before the workers start
    rows = []
    with Pool(args.workers, initializer=_init_worker) as pool:
        for r in tqdm(pool.imap_unordered(extract_video, jobs), total=len(jobs), desc="videos"):
            rows.extend(r)
    if not rows:
        raise SystemExit("No faces were extracted; check the videos and that the face detector loaded.")
    with open(args.out / "manifest.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "label", "method", "video", "frame", "file", "dataset"])
        w.writerows(sorted(r + (METHOD,) for r in rows))
    print(dict(Counter(f"{r[0]}/{r[1]}" for r in rows)))


if __name__ == "__main__":
    main()
