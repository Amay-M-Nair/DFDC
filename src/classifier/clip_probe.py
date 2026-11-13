"""CLIP linear probe (Ojha et al., CVPR 2023 "UnivFD"): a frozen CLIP ViT-L/14 image encoder + logistic regression.

CLIP's features were never trained on deepfakes, so a linear probe on them generalises to generators it has not
seen; it is the second opinion for fully AI-generated faces (StyleGAN, diffusion), which the FF++-style CNNs
were never shown. The encoder is loaded through timm (already a dependency): no fine-tuning, fp16 on the GPU.

    python -m src.classifier.clip_probe --out outputs/clip \
        --data FF++=data --data CelebDF=data_celebdf --data FFHQ=data_hq/ffhq --data CelebAHQ=data_hq/celebahq \
        --data StyleGAN=data_hq/stylegan --data Diffusion=data_hq/diffusion

Writes to --out:
    feats_train.npz, feats_val.npz   cached features (train: a clean and a degraded view of each image)
    probe.pt                         {'model', 'weight', 'bias', 'C', 'val_auc'}
    val_preds.npz                    key, dataset, label, logit of every validation image (same list as 02a's)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from src.classifier.augment import Degrade
from src.classifier.dataset import list_loader, parse_roots, split_items

CLIP_MODEL = "vit_large_patch14_clip_224.openai"


def clip_encoder(device, name: str = CLIP_MODEL):
    import timm

    model = timm.create_model(name, pretrained=True, num_classes=0).to(device).eval()
    if device.type == "cuda":
        model = model.half()
    cfg = timm.data.resolve_data_config({}, model=model)
    return model, timm.data.create_transform(**cfg)


class DegradedView:
    """Degrade (always) then the CLIP transform: the second training view."""

    def __init__(self, transform):
        self.transform, self.degrade = transform, Degrade(p=1.0)

    def __call__(self, img: Image.Image):
        return self.transform(self.degrade(img))


@torch.no_grad()
def encode(model, loader, device):
    feats = []
    dtype = next(model.parameters()).dtype  # fp16 on the GPU (see clip_encoder), fp32 on the CPU
    for x, _ in tqdm(loader, desc="clip", leave=False):
        f = model(x.to(device, dtype=dtype, non_blocking=True)).float()
        feats.append(torch.nn.functional.normalize(f, dim=1).cpu().half())
    return torch.cat(feats).numpy()


class ClipDetector:
    """Frozen CLIP encoder + saved probe: logit of 'fake' for a batch from `self.transform` (or PIL images)."""

    def __init__(self, probe_path, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        probe = torch.load(probe_path, map_location="cpu", weights_only=False)
        self.model, self.transform = clip_encoder(self.device, probe["model"])
        self.weight = torch.as_tensor(probe["weight"], dtype=torch.float32)
        self.bias = float(probe["bias"])

    def logits_from_features(self, feats: np.ndarray) -> np.ndarray:
        return feats.astype(np.float32) @ self.weight.numpy() + self.bias

    def loader_logits(self, loader) -> np.ndarray:
        return self.logits_from_features(encode(self.model, loader, self.device))

    @torch.no_grad()
    def __call__(self, images) -> np.ndarray:
        x = torch.stack([self.transform(im.convert("RGB")) for im in images])
        return self.loader_logits([(x, None)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", action="append", required=True, help="NAME=root (repeat)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-per-set", type=int, default=15000, help="training images per dataset (seeded)")
    ap.add_argument("--val-max-per-set", type=int, default=5000, help="must match 02a so the val lists line up")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, transform = clip_encoder(device)
    roots = parse_roots(args.data)
    train = split_items(roots, "train", args.max_per_set)
    val = split_items(roots, "val", args.val_max_per_set)
    print(f"{len(train)} training images (x2 views), {len(val)} validation images")

    cache = args.out / "feats_train.npz"
    if cache.exists():
        z = np.load(cache)
        x_train, y_train = z["feats"], z["labels"]
    else:
        clean = encode(model, list_loader(train, args.batch_size, args.workers, transform), device)
        degraded = encode(model, list_loader(train, args.batch_size, args.workers, DegradedView(transform)), device)
        x_train = np.concatenate([clean, degraded])
        y_train = np.array([it[2] for it in train] * 2)
        np.savez(cache, feats=x_train, labels=y_train)
    x_val = encode(model, list_loader(val, args.batch_size, args.workers, transform), device)
    y_val = np.array([it[2] for it in val])
    np.savez(args.out / "feats_val.npz", feats=x_val, labels=y_val, keys=np.array([it[0] for it in val]))

    best = None
    for c in (0.1, 0.3, 1.0, 3.0, 10.0):
        clf = LogisticRegression(C=c, max_iter=3000, class_weight="balanced")
        clf.fit(x_train.astype(np.float32), y_train)
        auc = roc_auc_score(y_val, clf.decision_function(x_val.astype(np.float32)))
        print(f"C={c}: val AUC {auc:.4f}")
        if best is None or auc > best[0]:
            best = (auc, c, clf)
    auc, c, clf = best
    torch.save({"model": CLIP_MODEL, "weight": clf.coef_[0].astype(np.float32), "bias": float(clf.intercept_[0]),
                "C": c, "val_auc": auc}, args.out / "probe.pt")
    logits = clf.decision_function(x_val.astype(np.float32))
    np.savez(args.out / "val_preds.npz", keys=np.array([it[0] for it in val]),
             datasets=np.array([it[3] for it in val]), labels=y_val, logits=logits.astype(np.float32))
    per = {}
    names = np.array([it[3] for it in val])
    for name in sorted(set(names)):
        m = names == name
        if len(set(y_val[m])) == 2:
            per[str(name)] = round(float(roc_auc_score(y_val[m], logits[m])), 4)
    (args.out / "probe_report.json").write_text(json.dumps({"C": c, "val_auc": auc, "per_dataset_auc": per}, indent=2))
    print(f"best C={c}, val AUC {auc:.4f}; per dataset {per} -> {args.out}")


if __name__ == "__main__":
    main()
