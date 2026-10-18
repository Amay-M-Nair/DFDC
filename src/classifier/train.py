"""Train the deepfake classifier (EfficientNet-B0 by default, or a comparison baseline).

    python -m src.classifier.train --epochs 8
    python -m src.classifier.train --epochs 8 --backbone legacy_xception   # baseline -> checkpoints/legacy_xception/
    python -m src.classifier.train --epochs 1 --subset 500                  # smoke test
"""
import argparse
import json
import time

import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from src.classifier.dataset import make_loader
from src.classifier.model import DEFAULT_BACKBONE, DeepfakeClassifier, best_threshold, checkpoint_path


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs, labels = [], []
    for x, y in tqdm(loader, desc="eval", leave=False):
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            logits = model(x.to(device, non_blocking=True)).squeeze(1)
        probs.append(torch.sigmoid(logits.float()).cpu())
        labels.append(y)
    return torch.cat(probs).numpy(), torch.cat(labels).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--subset", type=int, default=None)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--backbone", default=DEFAULT_BACKBONE, help="timm model name or 'meso4'")
    args = ap.parse_args()
    ckpt_path = checkpoint_path(args.backbone)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dl = make_loader("train", args.batch_size, train=True, num_workers=args.workers, subset=args.subset)
    val_dl = make_loader("val", args.batch_size * 2, num_workers=args.workers,
                         subset=args.subset // 4 if args.subset else None)

    model = DeepfakeClassifier(args.backbone).to(device)
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
            x, y = x.to(device, non_blocking=True), y.float().to(device, non_blocking=True)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                loss = loss_fn(model(x).squeeze(1), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += loss.item()
            pbar.set_postfix(loss=f"{running / i:.4f}")

        probs, labels = predict(model, val_dl, device)
        auc = roc_auc_score(labels, probs)
        acc = float(((probs > 0.5) == labels).mean())
        thr = best_threshold(labels, probs)
        acc_tuned = float(((probs > thr) == labels).mean())
        history.append({"epoch": epoch, "train_loss": running / len(train_dl), "val_auc": auc, "val_acc": acc,
                        "val_acc_tuned": acc_tuned, "threshold": thr,
                        "elapsed_min": (time.time() - start) / 60})
        print(f"epoch {epoch}: train_loss={running / len(train_dl):.4f} val_auc={auc:.4f} val_acc={acc:.4f} "
              f"val_acc@{thr:.3f}={acc_tuned:.4f}")

        if auc > best_auc:
            best_auc, bad_epochs = auc, 0
            torch.save({"model": model.state_dict(), "backbone": args.backbone, "epoch": epoch,
                        "val_auc": auc, "threshold": thr, "train_args": vars(args)}, ckpt_path)
            print(f"  saved best -> {ckpt_path}")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print("early stopping")
                break

    (ckpt_path.parent / "history.json").write_text(json.dumps(history, indent=2))


if __name__ == "__main__":
    main()
