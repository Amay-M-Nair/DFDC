"""Data-leakage audit on the extracted face crops.

Checks (FAIL = leakage, WARN = worth a look):
  1. files on disk match manifest.csv (no stray images from an earlier run)
  2. no video (or source image) appears in more than one split of a dataset
  3. no FF++ identity (000, 003, ...) appears in more than one split; Celeb-DF identities shared between
     train/val and the official test list are a WARN (that is Celeb-DF's official protocol)
  4. no byte-identical image appears in two splits or in two datasets
  5. near-duplicate images across splits / datasets (perceptual dHash, exact match)
  6. DFDC (the cross-dataset test) is test-only
  7. class balance per split (two-class datasets only)

    python -m src.classifier.check_leakage --dataset FF++=data --dataset CelebDF=data_celebdf \
        --dataset FFHQ=data_hq/ffhq --dataset DFDC=data_dfdc
    python -m src.classifier.check_leakage --data data --celebdf data_celebdf      # v1 form
"""
import argparse
import hashlib
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from src.config import DATA_DIR, OUTPUT_DIR

results = []


def report(status, check, detail):
    results.append({"status": status, "check": check, "detail": detail})
    print(f"[{status}] {check}: {detail}")


def hashes(path):
    raw = Path(path).read_bytes()
    g = np.asarray(Image.open(path).convert("L").resize((9, 8)), dtype=np.int16)
    dhash = int("".join("1" if b else "0" for b in (g[:, 1:] > g[:, :-1]).ravel()), 2)
    return hashlib.md5(raw).hexdigest(), dhash


def load(root: Path, dataset: str):
    m = pd.read_csv(root / "manifest.csv", dtype={"video": str})
    m["dataset"] = dataset  # overrides the manifest's own 'dataset' column (v2 manifests), same value
    m["path"] = [str(root / r.split / r.label / r.file) for r in m.itertuples()]
    on_disk = {str(p) for p in root.glob("*/*/*.jpg")}
    extra, missing = on_disk - set(m.path), set(m.path) - on_disk
    if extra or missing:
        report("FAIL", f"{dataset} manifest vs disk",
               f"{len(extra)} images on disk not in manifest, {len(missing)} in manifest but missing")
    else:
        report("PASS", f"{dataset} manifest vs disk", f"{len(m)} images, all accounted for")
    return m


def cross_split(df, key, name, status="FAIL"):
    splits = df.groupby(key).split.nunique()
    bad = splits[splits > 1]
    if len(bad):
        report(status, name, f"{len(bad)} {key} values appear in more than one split, e.g. {list(bad.index[:5])}")
    else:
        report("PASS", name, f"{df[key].nunique()} {key} values, each in exactly one split")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", action="append", default=[], help="NAME=root with <split>/<label>/ + manifest.csv")
    ap.add_argument("--data", type=Path, default=None, help="v1: FF++ crops")
    ap.add_argument("--celebdf", type=Path, default=None, help="v1: Celeb-DF crops")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    specs = [tuple(s.split("=", 1)) for s in args.dataset]
    if args.data or not specs:
        specs.insert(0, ("FF++", args.data or DATA_DIR))
    if args.celebdf:
        specs.append(("CelebDF", args.celebdf))
    frames = []
    for name, root in specs:
        m = load(Path(root), name)
        frames.append(m)
        if m.split.nunique() > 1:
            cross_split(m, "video", f"{name} video overlap")
        if name == "FF++":
            ids = m[["split", "video"]].copy()
            ids["identity"] = ids.video.str.split("_")
            ids = ids.explode("identity")
            if ids.identity.str.fullmatch(r"\d{3}").all():
                cross_split(ids, "identity", "FF++ identity overlap")
            else:
                report("WARN", "FF++ identity overlap", "video names aren't FF++ ids (000 / 000_003); check skipped")
        elif name == "CelebDF":
            if set(m.split) == {"test"}:
                report("PASS", "Celeb-DF test-only", "all Celeb-DF crops are in the test split")
            else:
                ident = m.video.str.split("_").str[0]
                train_ids, test_ids = set(ident[m.split != "test"]), set(ident[m.split == "test"])
                shared = sorted(train_ids & test_ids)
                report("WARN" if shared else "PASS", "Celeb-DF identities train/val vs. test",
                       f"{len(shared)} identities in both (official protocol); Celeb-DF test is in-domain, "
                       f"DFDC is the cross-dataset test" if shared else "no shared identities")
                cross_split(m.assign(identity=ident)[m.split != "test"], "identity", "Celeb-DF train/val identity")
        elif name == "DFDC":
            ok = set(m.split) == {"test"}
            report("PASS" if ok else "FAIL", "DFDC test-only",
                   "all DFDC crops are in the test split" if ok else f"DFDC crops in splits {sorted(set(m.split))}")

    # 4-5: duplicate images
    allf = pd.concat(frames, ignore_index=True)
    allf["group"] = allf.dataset + "/" + allf.split
    with Pool(args.workers) as pool:
        hs = list(tqdm(pool.imap(hashes, allf.path, chunksize=256), total=len(allf), desc="hashing"))
    allf["md5"], allf["dhash"] = zip(*hs)

    for key, name, status in (("md5", "identical images across splits/datasets", "FAIL"),
                              ("dhash", "near-duplicate images across splits/datasets", "WARN")):
        groups = allf.groupby(key).group.nunique()
        dup = allf[allf[key].isin(groups[groups > 1].index)]
        if len(dup):
            pairs = dup.groupby(key).group.apply(lambda g: " & ".join(sorted(set(g)))).value_counts()
            pairs = {k: int(v) for k, v in pairs.head(20).items()}
            example = dup.groupby(key).file.apply(list).iloc[0][:3]
            report(status, name, f"{dup[key].nunique()} shared hashes ({dict(pairs)}); example files: {example}")
        else:
            report("PASS", name, f"none among {len(allf)} images")

    # 7: balance, for datasets that have both classes
    bal = allf.groupby(["dataset", "split", "label"]).size().unstack(fill_value=0)
    print("\n" + bal.to_string())
    two_class = allf.groupby("dataset").label.nunique()
    for (dataset, split), row in bal.iterrows():
        if two_class[dataset] < 2:
            continue
        share = row.get("fake", 0) / max(1, row.sum())
        status = "PASS" if 0.3 <= share <= 0.7 else "WARN"
        report(status, f"{dataset}/{split} class balance", f"{share:.0%} fake ({row.get('real', 0)} real, {row.get('fake', 0)} fake)")

    out = OUTPUT_DIR / "leakage_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=str))
    fails = sum(r["status"] == "FAIL" for r in results)
    print(f"\n{'NO LEAKAGE FOUND' if not fails else f'{fails} FAILED CHECK(S)'}  ->  {out}")


if __name__ == "__main__":
    main()
