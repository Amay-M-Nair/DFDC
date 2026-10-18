"""Compare trained classifiers on one or more test sets: accuracy, generalisation, cost and significance.

Finds every best.pt under --roots and evaluates each model on the test split of every dataset in
--datasets (e.g. FaceForensics++ = in-dataset, Celeb-DF = cross-dataset). Writes to outputs/compare/:
    summary.csv                      one row per model: cost + AUC (95% CI) on every dataset
    <dataset>/comparison.csv         AUC, accuracy, F1, video AUC
    <dataset>/per_method.csv         AUC per manipulation method
    <dataset>/significance.csv       paired AUC difference vs. the reference model (95% CI, p-value)
    <dataset>/predictions.csv        per-frame P(fake) of every model
    *.png                            ROC curves, AUC vs. compute, per-method bars, generalisation bars

Thresholds come from each model's FF++ validation set and are never re-tuned on a test set.
CIs use a cluster bootstrap over videos (frames of a video are not independent), stratified by class;
differences are paired (same resampled videos for both models).

    python -m src.classifier.compare --roots outputs/checkpoints --datasets ffpp=data celebdf=data_celebdf
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, roc_auc_score, roc_curve

from src.classifier.dataset import make_loader
from src.classifier.model import BASELINES, DEFAULT_BACKBONE, load_model
from src.classifier.train import predict
from src.config import CHECKPOINT, DATA_DIR, IMG_SIZE, OUTPUT_DIR


def find_checkpoints(roots):
    found = {}
    for root in roots:
        for path in sorted(Path(root).rglob("best.pt")):
            backbone = torch.load(path, map_location="cpu", weights_only=False).get("backbone", DEFAULT_BACKBONE)
            if backbone in found:
                print(f"skipping duplicate {backbone}: {path}")
                continue
            found[backbone] = path
    return found


def count_cost(model, device):
    params = sum(p.numel() for p in model.parameters()) / 1e6
    from torch.utils.flop_counter import FlopCounterMode

    x = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE, device=device)
    with FlopCounterMode(display=False) as fc, torch.no_grad():
        model(x)
    gflops = fc.get_total_flops() / 1e9

    batch = 64 if device.type == "cuda" else 8
    x = torch.randn(batch, 3, IMG_SIZE, IMG_SIZE, device=device)
    with torch.no_grad(), torch.autocast(device.type, enabled=device.type == "cuda"):
        for _ in range(5):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t, n = time.time(), 20 if device.type == "cuda" else 3
        for _ in range(n):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
    return params, gflops, batch * n / (time.time() - t)


class VideoBootstrap:
    """Resample whole videos (stratified by label) and return frame indices for each replicate."""

    def __init__(self, videos, labels, n=1000, seed=0):
        rng = np.random.default_rng(seed)
        groups = pd.Series(np.arange(len(videos))).groupby(videos).apply(np.array)
        vid_label = pd.Series(labels).groupby(videos).first()
        self.replicates = []
        for _ in range(n):
            idx = []
            for lab in (0, 1):
                vids = vid_label.index[vid_label == lab].to_numpy()
                idx.extend(groups[v] for v in rng.choice(vids, len(vids), replace=True))
            self.replicates.append(np.concatenate(idx))

    def aucs(self, labels, probs):
        return np.array([roc_auc_score(labels[i], probs[i]) for i in self.replicates])


class TestSet:
    def __init__(self, name, root, workers):
        self.name, self.root = name, Path(root)
        self.loader = make_loader("test", 64, num_workers=workers, root=self.root)
        self.files = [Path(p).name for p, _ in self.loader.dataset.samples]
        self.methods = np.array([f.split("__")[0] if "__" in f else "unknown" for f in self.files])
        self.videos = np.array(["__".join(f.split("__")[:2]) if "__" in f else f for f in self.files])
        self.labels, self.probs = None, {}


def evaluate_dataset(ds: TestSet, thresholds, reference, n_boot, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    labels, rows, per_method = ds.labels, [], {}
    boot = VideoBootstrap(ds.videos, labels, n=n_boot)
    boot_aucs = {b: boot.aucs(labels, p) for b, p in ds.probs.items()}
    real = ds.methods == "original"
    for backbone, probs in ds.probs.items():
        preds = (probs > thresholds[backbone]).astype(int)
        v_prob = pd.Series(probs).groupby(ds.videos).mean()
        v_lab = pd.Series(labels).groupby(ds.videos).first()
        lo, hi = np.percentile(boot_aucs[backbone], [2.5, 97.5])
        name = BASELINES.get(backbone, backbone)
        rows.append({"model": name, "backbone": backbone, "auc": roc_auc_score(labels, probs),
                     "auc_ci_low": lo, "auc_ci_high": hi, "accuracy": float((preds == labels).mean()),
                     "f1": f1_score(labels, preds), "video_auc": roc_auc_score(v_lab, v_prob),
                     "threshold": thresholds[backbone]})
        per_method[name] = {m: roc_auc_score(labels[real | (ds.methods == m)], probs[real | (ds.methods == m)])
                            for m in sorted(set(ds.methods) - {"original", "unknown"})}
    sig = []
    if reference in ds.probs:
        for b in ds.probs:
            if b == reference:
                continue
            diff = boot_aucs[reference] - boot_aucs[b]
            sig.append({"reference": reference, "vs": b,
                        "auc_diff": roc_auc_score(labels, ds.probs[reference]) - roc_auc_score(labels, ds.probs[b]),
                        "ci_low": np.percentile(diff, 2.5), "ci_high": np.percentile(diff, 97.5),
                        "p_value": min(1.0, 2 * min((diff <= 0).mean(), (diff >= 0).mean()))})

    table = pd.DataFrame(rows).sort_values("auc", ascending=False)
    table.to_csv(out / "comparison.csv", index=False)
    pm = pd.DataFrame(per_method)
    pm.to_csv(out / "per_method.csv")
    pd.DataFrame(sig).to_csv(out / "significance.csv", index=False)
    pd.DataFrame({"file": ds.files, "label": labels, **ds.probs}).to_csv(out / "predictions.csv", index=False)

    print(f"\n===== {ds.name} test ({len(labels)} frames, {len(set(ds.videos))} videos) =====")
    print(table.drop(columns="backbone").round(4).to_string(index=False))
    print("\nAUC per manipulation\n", pm.round(4).to_string())
    if sig:
        print(f"\nPaired bootstrap vs {reference}\n", pd.DataFrame(sig).round(4).to_string(index=False))

    plt.figure(figsize=(5.5, 5))
    for r in table.itertuples():
        fpr, tpr, _ = roc_curve(labels, ds.probs[r.backbone])
        plt.plot(fpr, tpr, label=f"{r.model} ({r.auc:.3f})", lw=2.5 if r.backbone == reference else 1.2)
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate"); plt.title(f"ROC, {ds.name} test")
    plt.legend(fontsize=8)
    plt.savefig(out / "roc_all.png", dpi=150, bbox_inches="tight"); plt.close()

    if pm.shape[0] > 1:
        pm.plot.bar(figsize=(9, 4), rot=0, ylim=(max(0.0, pm.values.min() - 0.05), 1.0),
                    title=f"{ds.name}: AUC per manipulation")
        plt.legend(fontsize=7, ncol=3); plt.tight_layout()
        plt.savefig(out / "per_method.png", dpi=150); plt.close()
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", default=[str(CHECKPOINT.parent)], help="folders searched for best.pt")
    ap.add_argument("--datasets", nargs="+", default=[f"FF++={DATA_DIR}"], help="name=folder with test/{real,fake}")
    ap.add_argument("--reference", default=DEFAULT_BACKBONE)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    ckpts = find_checkpoints(args.roots)
    print("models:", {k: str(v) for k, v in ckpts.items()})
    if not ckpts:
        raise SystemExit(f"No best.pt found under {args.roots}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sets = [TestSet(*spec.split("=", 1), args.workers) for spec in args.datasets]
    out = OUTPUT_DIR / "compare"

    cost, thresholds = [], {}
    for backbone, path in ckpts.items():
        print(f"\n== {backbone}")
        model = load_model(path, device)
        thresholds[backbone] = model.threshold
        for ds in sets:
            ds.probs[backbone], ds.labels = predict(model, ds.loader, device)
        params, gflops, ips = count_cost(model, device)
        hist = path.parent / "history.json"
        train_min = json.loads(hist.read_text())[-1].get("elapsed_min") if hist.exists() else None
        cost.append({"model": BASELINES.get(backbone, backbone), "backbone": backbone, "params_M": params,
                     "gflops": gflops, "images_per_s": ips, "train_min": train_min})
        del model
        torch.cuda.empty_cache()

    summary = pd.DataFrame(cost).set_index("backbone")
    for ds in sets:
        t = evaluate_dataset(ds, thresholds, args.reference, args.bootstrap, out / ds.name).set_index("backbone")
        summary[f"{ds.name}_auc"] = t.auc
        summary[f"{ds.name}_auc_ci"] = t.apply(lambda r: f"[{r.auc_ci_low:.3f}, {r.auc_ci_high:.3f}]", axis=1)
        summary[f"{ds.name}_video_auc"] = t.video_auc
    if len(sets) > 1:
        summary["auc_drop"] = summary[f"{sets[0].name}_auc"] - summary[f"{sets[1].name}_auc"]
    summary = summary.sort_values(f"{sets[-1].name}_auc", ascending=False)
    summary.to_csv(out / "summary.csv")
    pd.set_option("display.width", 250)
    print("\n===== SUMMARY =====\n", summary.round(4).to_string())

    # Accuracy vs. compute (on the last dataset, i.e. cross-dataset when given)
    main_ds = sets[-1].name
    plt.figure(figsize=(6.5, 4.5))
    for b, r in summary.iterrows():
        plt.scatter(r.gflops, r[f"{main_ds}_auc"], s=40 + 8 * r.params_M, alpha=0.7,
                    edgecolor="black" if b == args.reference else "none")
        plt.annotate(r.model, (r.gflops, r[f"{main_ds}_auc"]), textcoords="offset points", xytext=(6, 4), fontsize=8)
    plt.xscale("log"); plt.xlabel("GFLOPs per image (log)"); plt.ylabel(f"{main_ds} test AUC")
    plt.title("Accuracy vs. compute (bubble = parameters)")
    plt.savefig(out / "auc_vs_compute.png", dpi=150, bbox_inches="tight"); plt.close()

    if len(sets) > 1:
        summary.set_index("model")[[f"{s.name}_auc" for s in sets]].plot.bar(
            figsize=(9, 4), rot=15, ylim=(0.4, 1.0), title="In-dataset vs. cross-dataset AUC")
        plt.axhline(0.5, color="gray", ls="--", lw=0.8); plt.tight_layout()
        plt.savefig(out / "generalisation.png", dpi=150); plt.close()
    print(f"\nsaved to {out}")


if __name__ == "__main__":
    main()
