"""Robustness curves: how each model's AUC and false-alarm rate change as images get blurrier, more compressed
or lower resolution, and the blur test from the v1 diagnosis (a blurred real photo must not turn FAKE).

On a fixed sample of each two-class test set (AUC) and each real-only set (share of real faces called FAKE),
every image is degraded at increasing strength (augment.LEVELS) and re-scored. The FAKE decision uses the
calibrated t_fake from 02c when --calibration is given, else the checkpoint's threshold.

    python -m src.classifier.robustness --roots outputs/checkpoints --calibration outputs/calibration \
        --datasets FF++=data --reals FFHQ=data_hq/ffhq CelebAHQ=data_hq/celebahq --n 1000

Writes outputs/robustness/: robustness.csv, blur_test.csv and one figure per degradation.
"""
import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from src.classifier.augment import LEVELS
from src.classifier.calibrate import sigmoid
from src.classifier.compare import display_name, find_checkpoints
from src.classifier.dataset import eval_transform, list_loader, list_split
from src.classifier.model import load_model
from src.classifier.train import predict_full
from src.config import OUTPUT_DIR

BLUR_TEST_RADIUS = 1.5


class Fixed:
    """Degrade at one fixed strength, then the evaluation transform (picklable for DataLoader workers)."""

    def __init__(self, kind, level):
        self.kind, self.level, self.base = kind, level, eval_transform()

    def __call__(self, img):
        return self.base(LEVELS[self.kind][1](img, self.level))


def sample(root, name, n, seed=0):
    items = [(f"{name}/{k}", p, y, name) for k, p, y in list_split(root, "test")]
    return sorted(random.Random(f"{seed}/{name}").sample(items, min(n, len(items))))


def main():
    plt.switch_backend("Agg")
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--calibration", default=None)
    ap.add_argument("--datasets", nargs="*", default=[], help="NAME=root two-class test sets (AUC)")
    ap.add_argument("--reals", nargs="*", default=[], help="NAME=root real-only test sets (false FAKE rate)")
    ap.add_argument("--n", type=int, default=1000, help="images sampled per set")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, default=OUTPUT_DIR / "robustness")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    sets = {s.split("=", 1)[0]: sample(s.split("=", 1)[1], s.split("=", 1)[0], args.n) for s in args.datasets}
    reals = {s.split("=", 1)[0]: [it for it in sample(s.split("=", 1)[1], s.split("=", 1)[0], args.n) if it[2] == 0]
             for s in args.reals}
    cal = None
    if args.calibration:
        from src.classifier.verdict import Calibration
        cal = Calibration(args.calibration)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows = []
    for backbone, path in find_checkpoints(args.roots).items():
        model = load_model(path, device)
        calibrated = cal is not None and cal.has(backbone)

        def score(items, tf):
            z, y, _ = predict_full(model, list_loader(items, 128, args.workers, tf), device, features=False)
            if calibrated:
                p, _, t_fake = cal.cnn(backbone, z)
                return p, y, t_fake
            return sigmoid(z), y, model.threshold

        for kind, (levels, _) in LEVELS.items():
            for level in levels:
                tf = Fixed(kind, level)
                for name, items in sets.items():
                    p, y, t = score(items, tf)
                    rows.append({"model": display_name(backbone), "degradation": kind, "level": level, "set": name,
                                 "auc": roc_auc_score(y, p), "fake_rate_on_reals": float((p[y == 0] >= t).mean())})
                for name, items in reals.items():
                    p, y, t = score(items, tf)
                    rows.append({"model": display_name(backbone), "degradation": kind, "level": level, "set": name,
                                 "auc": np.nan, "fake_rate_on_reals": float((p >= t).mean())})
        print(f"{backbone}: done")
        del model
        torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_csv(args.out / "robustness.csv", index=False)
    blur = df[(df.degradation == "blur") & df.level.isin([0, BLUR_TEST_RADIUS])]
    test = blur.pivot_table(index=["model", "set"], columns="level", values="fake_rate_on_reals")
    test.columns = [f"FAKE rate, blur {c}" for c in test.columns]
    test.to_csv(args.out / "blur_test.csv")
    pd.set_option("display.width", 200)
    print(f"\nBlur test (share of real faces called FAKE, sharp vs. blur radius {BLUR_TEST_RADIUS}):\n",
          test.round(3).to_string())

    for kind in LEVELS:
        d = df[df.degradation == kind]
        panels = [("auc", n) for n in sets] + [("fake_rate_on_reals", n) for n in list(sets) + list(reals)]
        fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.6), squeeze=False)
        for ax, (metric, name) in zip(axes[0], panels):
            for model, g in d[d.set == name].groupby("model"):
                ax.plot(g.level, g[metric], marker="o", label=model)
            ax.set_title(f"{name}: {'AUC' if metric == 'auc' else 'reals called FAKE'}", fontsize=10)
            ax.set_xlabel(kind)
            if kind == "jpeg" or kind == "resample":
                ax.invert_xaxis()
        axes[0][0].legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(args.out / f"robustness_{kind}.png", dpi=130)
        plt.close()
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
