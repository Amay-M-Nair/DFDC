"""Evaluate the best checkpoint on the test split.

    python -m src.classifier.evaluate
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, confusion_matrix, precision_recall_fscore_support,
                             roc_auc_score, roc_curve)

from src.classifier.dataset import make_loader
from src.classifier.model import load_model
from src.classifier.train import predict
from src.config import CHECKPOINT, CLASSES, OUTPUT_DIR


def breakdown(names, probs, labels) -> dict:
    """FaceForensics++ extras (files named method__video__frame.jpg): per-method and video-level scores."""
    if not names or "__" not in names[0]:
        return {}
    methods = np.array([n.split("__")[0] for n in names])
    videos = np.array(["__".join(n.split("__")[:2]) for n in names])
    real = methods == "original"
    per_method = {}
    for m in sorted(set(methods) - {"original"}):
        sel = real | (methods == m)  # each manipulation vs. all real frames
        per_method[m] = {"acc_on_fakes": float(((probs[methods == m] > 0.5) == 1).mean()),
                         "auc_vs_real": float(roc_auc_score(labels[sel], probs[sel]))}
    # Video-level: average the frame probabilities of each video
    vids = sorted(set(videos))
    v_prob = np.array([probs[videos == v].mean() for v in vids])
    v_label = np.array([labels[videos == v][0] for v in vids])
    return {"per_method": per_method,
            "video_level": {"n_videos": len(vids), "accuracy": float(((v_prob > 0.5) == v_label).mean()),
                            "roc_auc": float(roc_auc_score(v_label, v_prob))}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(CHECKPOINT, device)
    loader = make_loader(args.split, 64, num_workers=args.workers)
    probs, labels = predict(model, loader, device)
    preds = (probs > 0.5).astype(int)

    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    metrics = {"accuracy": accuracy_score(labels, preds), "precision": p, "recall": r, "f1": f1,
               "roc_auc": roc_auc_score(labels, probs), "n": int(len(labels))}
    metrics.update(breakdown([Path(p).name for p, _ in loader.dataset.samples], probs, labels))
    out = OUTPUT_DIR / "metrics"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.split}_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))

    ConfusionMatrixDisplay(confusion_matrix(labels, preds), display_labels=CLASSES).plot(cmap="Blues")
    plt.title("Confusion matrix")
    plt.savefig(out / f"{args.split}_confusion.png", dpi=150, bbox_inches="tight")
    plt.close()

    fpr, tpr, _ = roc_curve(labels, probs)
    plt.plot(fpr, tpr, label=f"AUC = {metrics['roc_auc']:.3f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate"); plt.title("ROC curve"); plt.legend()
    plt.savefig(out / f"{args.split}_roc.png", dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
