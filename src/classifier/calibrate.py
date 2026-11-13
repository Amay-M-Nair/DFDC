"""Post-training calibration from validation predictions only (never a test set, never DFDC).

For every trained CNN (val_preds.npz + train_feats.npy saved by train.py) and the CLIP probe (02b):
  temperature   one scalar T fitted on validation NLL: P(fake) = sigmoid(logit / T), so 0.9 means ~90%
  thresholds    t_fake: lowest score at which at most --max-fpr of validation reals score as high (FAKE above);
                t_real: highest score below which at most --max-fnr of validation fakes fall (REAL below);
                in between the verdict is UNCERTAIN. If the two cross, both become the Youden threshold.
  OOD           Mahalanobis distance of the pooled features to the training features (Ledoit-Wolf
                covariance); farther than the 99th percentile of validation distances = out of distribution
  fusion        logistic regression on [CNN logit / T, CLIP logit] for every CNN, with its own thresholds

CNN thresholds use the datasets the CNN was trained for (not StyleGAN/diffusion); CLIP and fused thresholds use
every validation dataset.

    python -m src.classifier.calibrate --roots outputs/checkpoints --clip outputs/clip --out outputs/calibration
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression

from src.classifier.model import DEFAULT_BACKBONE, best_threshold
from src.config import SYNTHETIC


def sigmoid(z):
    return 1 / (1 + np.exp(-np.asarray(z, dtype=np.float64)))


def fit_temperature(logits, labels) -> float:
    z = torch.as_tensor(logits, dtype=torch.float64)
    y = torch.as_tensor(labels, dtype=torch.float64)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(np.clip(log_t.exp().item(), 0.05, 20.0))  # guard against degenerate validation sets


def ece(probs, labels, bins: int = 15) -> float:
    """Expected calibration error of P(fake) (reliability over equal-width bins)."""
    probs, labels = np.asarray(probs), np.asarray(labels)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(probs, edges[1:-1]), 0, bins - 1)
    err = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            err += m.mean() * abs(probs[m].mean() - labels[m].mean())
    return float(err)


def three_way_thresholds(probs, labels, max_fpr=0.05, max_fnr=0.10):
    """(t_real, t_fake, youden): REAL below t_real, FAKE at or above t_fake, UNCERTAIN in between."""
    reals, fakes = probs[labels == 0], probs[labels == 1]
    youden = best_threshold(labels, probs)
    t_fake = float(np.quantile(reals, 1 - max_fpr)) if len(reals) else youden
    t_real = float(np.quantile(fakes, max_fnr)) if len(fakes) else youden
    if t_real >= t_fake:  # classes separate well enough: one threshold, no uncertain band
        t_real = t_fake = youden
    return t_real, t_fake, youden


def verdicts(probs, t_real, t_fake):
    """Array of 'real' / 'uncertain' / 'fake'."""
    probs = np.asarray(probs)
    return np.where(probs >= t_fake, "fake", np.where(probs < t_real, "real", "uncertain"))


def rates(probs, labels, t_real, t_fake) -> dict:
    v = verdicts(probs, t_real, t_fake)
    out = {"uncertain": float((v == "uncertain").mean())}
    if (labels == 0).any():
        out["fpr"] = float((v[labels == 0] == "fake").mean())
    if (labels == 1).any():
        out["fnr"] = float((v[labels == 1] == "real").mean())
    decided = v != "uncertain"
    if decided.any():
        out["acc_decided"] = float(((v[decided] == "fake") == (labels[decided] == 1)).mean())
    return out


class OOD:
    """Mahalanobis distance to the training feature distribution."""

    def __init__(self, mean, precision, threshold=None):
        self.mean, self.precision, self.threshold = mean, precision, threshold

    @classmethod
    def fit(cls, train_feats):
        x = np.asarray(train_feats, dtype=np.float64)
        lw = LedoitWolf().fit(x)
        return cls(lw.location_.astype(np.float32), lw.precision_.astype(np.float32))

    def distance(self, feats):
        d = np.asarray(feats, dtype=np.float32) - self.mean
        return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", d, self.precision, d), 0))

    def save(self, path):
        np.savez(path, mean=self.mean, precision=self.precision, threshold=self.threshold)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        return cls(z["mean"], z["precision"], float(z["threshold"]))


def find_val_preds(roots):
    """{backbone: folder} for every folder holding val_preds.npz + best.pt."""
    found = {}
    for root in roots:
        for p in sorted(Path(root).rglob("val_preds.npz")):
            ckpt = p.parent / "best.pt"
            if not ckpt.exists():
                continue
            backbone = torch.load(ckpt, map_location="cpu", weights_only=False).get("backbone", DEFAULT_BACKBONE)
            found.setdefault(backbone, p.parent)
    return found


def calibrate_scores(logits, labels, names, max_fpr, max_fnr, exclude=()):
    """Temperature + thresholds on the rows whose dataset is not in `exclude`."""
    m = ~np.isin(names, exclude)
    t = fit_temperature(logits[m], labels[m])
    p_raw, p_cal = sigmoid(logits[m]), sigmoid(logits[m] / t)
    t_real, t_fake, youden = three_way_thresholds(p_cal, labels[m], max_fpr, max_fnr)
    return {"temperature": t, "t_real": t_real, "t_fake": t_fake, "youden": youden,
            "ece_raw": ece(p_raw, labels[m]), "ece_calibrated": ece(p_cal, labels[m]), "val_n": int(m.sum()),
            "val_rates": rates(p_cal, labels[m], t_real, t_fake)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True, help="folders searched for val_preds.npz (02a outputs)")
    ap.add_argument("--clip", type=Path, default=None, help="02b output folder (val_preds.npz, probe.pt)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-fpr", type=float, default=0.05, help="share of real faces allowed above t_fake")
    ap.add_argument("--max-fnr", type=float, default=0.10, help="share of fakes allowed below t_real")
    ap.add_argument("--ood-quantile", type=float, default=0.99)
    ap.add_argument("--min-fusion", type=int, default=100, help="validation images shared with CLIP needed to fuse")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    clip = None
    if args.clip and (args.clip / "val_preds.npz").exists():
        z = np.load(args.clip / "val_preds.npz")
        clip = {k: z[k] for k in z.files}
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    cal = {"settings": settings, "models": {}, "fused": {}, "clip": None}

    if clip is not None:
        cal["clip"] = calibrate_scores(clip["logits"], clip["labels"], clip["datasets"], args.max_fpr, args.max_fnr)
        print(f"CLIP: {cal['clip']}")
        clip_by_key = dict(zip(clip["keys"].tolist(), clip["logits"]))

    for backbone, folder in find_val_preds(args.roots).items():
        z = np.load(folder / "val_preds.npz")
        keys, names, labels, logits, feats = z["keys"], z["datasets"], z["labels"], z["logits"], z["feats"]
        entry = calibrate_scores(logits, labels, names, args.max_fpr, args.max_fnr, exclude=SYNTHETIC)

        ood = OOD.fit(np.load(folder / "train_feats.npy"))
        core = ~np.isin(names, SYNTHETIC)
        ood.threshold = float(np.quantile(ood.distance(feats[core]), args.ood_quantile))
        ood_file = f"ood_{backbone}.npz"
        ood.save(args.out / ood_file)
        dist = ood.distance(feats)
        entry |= {"ood_file": ood_file, "ood_threshold": ood.threshold,
                  "ood_rate_val": {n: float((dist[names == n] > ood.threshold).mean()) for n in sorted(set(names))}}
        cal["models"][backbone] = entry
        print(f"{backbone}: T={entry['temperature']:.3f} t_real={entry['t_real']:.3f} t_fake={entry['t_fake']:.3f} "
              f"ECE {entry['ece_raw']:.3f}->{entry['ece_calibrated']:.3f} val {entry['val_rates']}")

        if clip is not None:
            both = [i for i, k in enumerate(keys.tolist()) if k in clip_by_key]
            if len(both) < args.min_fusion:
                print(f"  fusion skipped: only {len(both)} validation images shared with CLIP")
                continue
            x = np.stack([logits[both] / entry["temperature"], [clip_by_key[k] for k in keys[both].tolist()]], 1)
            lr = LogisticRegression(C=1.0, class_weight="balanced").fit(x, labels[both])
            fused = sigmoid(x @ lr.coef_[0] + lr.intercept_[0])
            t_real, t_fake, youden = three_way_thresholds(fused, labels[both], args.max_fpr, args.max_fnr)
            cal["fused"][backbone] = {"w_cnn": float(lr.coef_[0][0]), "w_clip": float(lr.coef_[0][1]),
                                      "bias": float(lr.intercept_[0]), "t_real": t_real, "t_fake": t_fake,
                                      "youden": youden, "ece": ece(fused, labels[both]), "val_n": len(both),
                                      "val_rates": rates(fused, labels[both], t_real, t_fake)}
            print(f"  fused with CLIP: {cal['fused'][backbone]}")

    if not cal["models"]:
        raise SystemExit(f"No val_preds.npz (with best.pt) found under {args.roots}")
    (args.out / "calibration.json").write_text(json.dumps(cal, indent=2, default=float))
    print(f"saved {args.out / 'calibration.json'}")


if __name__ == "__main__":
    main()
