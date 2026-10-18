"""Celeb-DF (v2) videos -> face-crop frames in <out>/test/{real,fake}, for cross-dataset testing.

Uses the official test split (List_of_testing_videos.txt: 518 videos) by default, so results are comparable
with published numbers. Expected layout (folders may be nested anywhere under --src):
    Celeb-real/id0_0000.mp4   YouTube-real/00000.mp4   Celeb-synthesis/id0_id1_0000.mp4
    List_of_testing_videos.txt

Labels come from the folder name (Celeb-synthesis = fake), not the list's label column. The same face
detector and crop as FaceForensics++ are used, so train and test crops look alike.

    python -m src.classifier.celebdf_extract --src /path/to/Celeb-DF-v2 --out data_celebdf --frames 32
"""
import argparse
import csv
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

from tqdm import tqdm

from src.classifier.ff_extract import _init_worker, extract_video
from src.faces import ensure_yunet

REAL_DIRS = {"celeb-real", "youtube-real"}
FAKE_DIR = "celeb-synthesis"
METHOD = "CelebDF"


def index_videos(src: Path):
    """(folder, filename) -> path for every Celeb-DF video under src."""
    index = {}
    for p in src.rglob("*.mp4"):
        folder = p.parent.name.lower()
        if folder in REAL_DIRS or folder == FAKE_DIR:
            index[(folder, p.name)] = p
    return index


def official_test_list(src: Path, index):
    lists = sorted(src.rglob("List_of_testing_videos.txt"))
    if not lists:
        return None
    videos = []
    for line in lists[0].read_text().splitlines():
        parts = line.strip().split()
        if len(parts) != 2:
            continue
        folder, name = parts[1].split("/")[-2:]
        path = index.get((folder.lower(), name))
        if path is None:
            print(f"listed but missing: {parts[1]}")
            continue
        videos.append(path)
    return videos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--frames", type=int, default=32, help="frames per video (real and fake)")
    ap.add_argument("--all", action="store_true", help="use every video instead of the official test list")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    index = index_videos(args.src)
    if not index:
        raise SystemExit(f"No Celeb-DF videos (Celeb-real / YouTube-real / Celeb-synthesis) under {args.src}")
    videos = None if args.all else official_test_list(args.src, index)
    if videos is None:
        print("Using all videos" + ("" if args.all else " (List_of_testing_videos.txt not found)"))
        videos = sorted(index.values())

    jobs = []
    for p in videos:
        label = "real" if p.parent.name.lower() in REAL_DIRS else METHOD
        jobs.append((p, label, "test", args.frames, str(args.out)))
    print(Counter(p.parent.name for p in videos))
    for cls in ("real", "fake"):
        (args.out / "test" / cls).mkdir(parents=True, exist_ok=True)

    ensure_yunet()  # download once, before the workers start
    rows = []
    with Pool(args.workers, initializer=_init_worker) as pool:
        for r in tqdm(pool.imap_unordered(extract_video, jobs), total=len(jobs), desc="videos"):
            rows.extend(r)
    if not rows:
        raise SystemExit("No faces were extracted; check the videos and that the face detector loaded.")

    with open(args.out / "manifest.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "label", "method", "video", "frame", "file"])
        w.writerows(sorted(rows))
    print(dict(Counter(f"{r[0]}/{r[1]}" for r in rows)))


if __name__ == "__main__":
    main()
