"""Train the deepfake classifier (EfficientNet-B0 by default, or a comparison baseline).

v2 recipe: several datasets mixed in fixed proportions (MixedTrainSet), self-blended fakes made on the fly from
every real crop with cached landmarks, and the same quality degradation for real and fake images. Each epoch is
--samples-per-epoch images, so training time doesn't grow with the amount of attached data.

    python -m src.classifier.train --epochs 8 \
        --data FF++=data --data CelebDF=data_celebdf --data FFHQ=data_hq/ffhq --data CelebAHQ=data_hq/celebahq \
        --data StyleGAN=data_hq/stylegan --data Diffusion=data_hq/diffusion --landmarks landmarks
    python -m src.classifier.train --epochs 8 --backbone legacy_xception ...   # baseline -> checkpoints/legacy_xception/
    python -m src.classifier.train --epochs 1 --subset 500 ...                 # smoke test

Validation pools every dataset's val split (at most --val-max-per-set images each): early stopping uses the AUC
over the non-synthetic datasets. After training, the best checkpoint scores the whole validation list and saves
    val_preds.npz     key, dataset, label, logit and pooled features of every validation image
    train_feats.npy   pooled features of a random sample of training images (out-of-distribution statistics)
which 02c (src/classifier/calibrate.py) turns into temperatures, three-way thresholds and the OOD check.
"""
import argparse
import json
import random
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from src.classifier.dataset import (MixedTrainSet, list_loader, load_landmarks, mixed_loader, parse_roots,
                                    split_items)
from src.classifier.model import DEFAULT_BACKBONE, DeepfakeClassifier, best_threshold, checkpoint_path
from src.config import DATA_DIR, SYNTHETIC


@torch.no_grad()
def predict(model, loader, device):
    """(P(fake), labels) for every image in the loader."""
    logits, labels, _ = predict_full(model, loader, device, features=False)
    return 1 / (1 + np.exp(-logits)), labels


@torch.no_grad()
def predict_full(model, loader, device, features: bool = True):
    """(logits, labels, pooled features or None) for every image in the loader."""
    model.eval()
    logits, labels, feats = [], [], []
    for x, y in tqdm(loader, desc="eval", leave=False):
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            z, f = model.forward_features(x.to(device, non_blocking=True))
        logits.append(z.float().squeeze(1).cpu())
        labels.append(torch.as_tensor(y))
        if features:
            feats.append(f.float().flatten(1).cpu().half())
    return (torch.cat(logits).numpy(), torch.cat(labels).numpy(),
            torch.cat(feats).numpy() if features else None)


def val_report(items, logits, labels):
    """Pooled AUC over non-synthetic datasets (the early-stopping metric) + per-dataset AUC or error rate."""
    names = np.array([it[3] for it in items])
    probs = 1 / (1 + np.exp(-logits))
    core = ~np.isin(names, SYNTHETIC)
    auc = roc_auc_score(labels[core], probs[core])
    thr = best_threshold(labels[core], probs[core])
    per = {}
    for name in sorted(set(names)):
        m = names == name
        if len(set(labels[m])) == 2:
            per[f"auc_{name}"] = float(roc_auc_score(labels[m], probs[m]))
        elif labels[m][0] == 0:
            per[f"fpr_{name}"] = float((probs[m] >= thr).mean())   # reals called fake
        else:
            per[f"fnr_{name}"] = float((probs[m] < thr).mean())    # fakes called real
    return float(auc), float(thr), per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", action="append", default=None, help="NAME=root (repeat); default FF++=DATA_DIR")
    ap.add_argument("--landmarks", default=None, help="folder with <NAME>.npz landmark caches (self-blending)")
    ap.add_argument("--no-sbi", action="store_true", help="train without self-blended fakes")
    ap.add_argument("--synthetic-share", type=float, default=0.0, help="share of StyleGAN/diffusion fakes")
    ap.add_argument("--samples-per-epoch", type=int, default=60000)
    ap.add_argument("--val-max-per-set", type=int, default=5000)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--subset", type=int, default=None, help="smoke test: samples per epoch and val size")
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--train-feats", type=int, default=5000, help="training images sampled for OOD statistics")
    ap.add_argument("--backbone", default=DEFAULT_BACKBONE, help="timm model name or 'meso4'")
    args = ap.parse_args()
    ckpt_path = checkpoint_path(args.backbone)
    torch.manual_seed(0); random.seed(0); np.random.seed(0)

    roots = parse_roots(args.data or [f"FF++={DATA_DIR}"])
    marks = load_landmarks(args.landmarks, roots)
    shares = {"real": 0.4, "sbi": 0.35, "fake": 0.25 - args.synthetic_share, "synthetic": args.synthetic_share}
    train_set = MixedTrainSet(roots, marks, shares=shares, sbi=not args.no_sbi)
    print(train_set.describe())
    if not args.no_sbi and not marks:
        print("WARNING: no landmark caches found; training without self-blended fakes")
    per_epoch = min(args.samples_per_epoch, args.subset or args.samples_per_epoch)
    val_items = split_items(roots, "val", args.subset // 4 if args.subset else args.val_max_per_set)
    if not val_items:
        raise SystemExit(f"No validation images under {roots}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dl = mixed_loader(train_set, per_epoch, args.batch_size, args.workers)
    val_dl = list_loader(val_items, args.batch_size * 2, args.workers)

    model = DeepfakeClassifier(args.backbone).to(device)
    if not model.is_vit:
        model = model.to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(train_dl),
                                                pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    loss_fn = nn.BCEWithLogitsLoss()

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    best_auc, bad_epochs, history = 0.0, 0, []
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        pbar = tqdm(train_dl, desc=f"epoch {epoch}/{args.epochs}")
        for i, (x, y) in enumerate(pbar, 1):
            x = x.to(device, non_blocking=True, memory_format=torch.channels_last)
            y = y.float().to(device, non_blocking=True)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                loss = loss_fn(model(x).squeeze(1), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += loss.item()
            pbar.set_postfix(loss=f"{running / i:.4f}")

        logits, labels, _ = predict_full(model, val_dl, device, features=False)
        auc, thr, per = val_report(val_items, logits, labels)
        history.append({"epoch": epoch, "train_loss": running / len(train_dl), "val_auc": auc, "threshold": thr,
                        **per, "elapsed_min": (time.time() - start) / 60})
        print(f"epoch {epoch}: train_loss={running / len(train_dl):.4f} val_auc={auc:.4f} thr={thr:.3f} "
              + " ".join(f"{k}={v:.3f}" for k, v in per.items()))

        if auc > best_auc:
            best_auc, bad_epochs = auc, 0
            torch.save({"model": model.state_dict(), "backbone": args.backbone, "epoch": epoch, "val_auc": auc,
                        "threshold": thr, "version": 2, "datasets": sorted(roots), "train_args": vars(args)},
                       ckpt_path)
            print(f"  saved best -> {ckpt_path}")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print("early stopping")
                break
    (ckpt_path.parent / "history.json").write_text(json.dumps(history, indent=2))

    # Validation predictions + training features of the best checkpoint, for calibration (02c)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False)["model"])
    logits, labels, feats = predict_full(model, val_dl, device)
    train_items = [it for it in split_items(roots, "train") if it[3] not in SYNTHETIC]
    sample = random.Random(0).sample(train_items, min(args.train_feats, len(train_items)))
    _, _, train_feats = predict_full(model, list_loader(sample, args.batch_size * 2, args.workers), device)
    np.savez(ckpt_path.parent / "val_preds.npz", keys=np.array([it[0] for it in val_items]),
             datasets=np.array([it[3] for it in val_items]), labels=labels, logits=logits, feats=feats)
    np.save(ckpt_path.parent / "train_feats.npy", train_feats)
    print(f"saved val_preds.npz ({len(val_items)} images) and train_feats.npy ({len(sample)}) -> {ckpt_path.parent}")


if __name__ == "__main__":
    main()
