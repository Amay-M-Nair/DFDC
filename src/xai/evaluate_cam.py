"""Evaluate heatmap explanations for every trained model: faithfulness and facial-region attribution.

Faithfulness (Petsiuk et al., RISE, BMVC 2018): pixels are perturbed in the order a heatmap ranks them,
and the model's confidence in its own prediction is tracked.
    deletion  : replace the most important pixels with a blurred copy first -> confidence should drop fast
                (area under the curve, LOWER is better)
    insertion : start from the blurred image and restore the most important pixels first -> confidence
                should recover fast (HIGHER is better)
Grad-CAM, Grad-CAM++ and a random (smooth noise) map are compared on the same images; a method is only
meaningful if it beats random. Paired Wilcoxon tests compare methods per model.

Region attribution: share of Grad-CAM++ mass in each facial region (placed from YuNet landmarks),
per model, dataset and manipulation method.

The same images (one frame per test video, sampled per manipulation method) are used for every model.
Writes to outputs/gradcam/: faithfulness.csv, faithfulness_summary.csv, faithfulness_tests.csv,
curves.csv, regions.csv, region_attribution.csv, heatmaps_<model>.npz (per-image maps)

    python -m src.xai.evaluate_cam --roots outputs/checkpoints --datasets FF++=data CelebDF=data_celebdf --n 300
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.stats import wilcoxon
from torchvision.transforms.functional import gaussian_blur
from tqdm import tqdm

from src.classifier.compare import find_checkpoints
from src.classifier.dataset import eval_transform
from src.classifier.model import BASELINES, load_model
from src.config import CHECKPOINT, DATA_DIR, IMG_SIZE, OUTPUT_DIR
from src.faces import FaceCropper
from src.xai.gradcam import FACE_REGIONS, METHODS, LogitTarget, landmark_regions, region_scores

FRACTIONS = np.linspace(0, 1, 11)
trapezoid = getattr(np, "trapezoid", None) or np.trapz  # numpy 2 renamed trapz


def sample_images(root: Path, n: int, seed: int = 0):
    """One frame per test video, balanced across manipulation methods and real videos."""
    files = [(p, p.parent.name) for p in sorted((root / "test").glob("*/*.jpg"))]
    df = pd.DataFrame({"path": [str(p) for p, _ in files], "label": [l for _, l in files]})
    df["file"] = df.path.map(lambda p: Path(p).name)
    df["method"] = df.file.str.split("__").str[0]
    df["video"] = df.file.str.split("__").str[:2].str.join("__")
    one = df.groupby("video").sample(1, random_state=seed)
    groups = one.groupby("method")
    per = max(1, n // groups.ngroups)
    return pd.concat([g.sample(min(per, len(g)), random_state=seed) for _, g in groups]).reset_index(drop=True)


def random_map(seed: int) -> np.ndarray:
    """Spatially smooth random map at the same coarse resolution as a CAM (7x7, upsampled)."""
    g = torch.Generator().manual_seed(seed)
    low = torch.rand(1, 1, 7, 7, generator=g)
    return F.interpolate(low, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)[0, 0].numpy()


@torch.no_grad()
def curves(model, x, blurred, heatmap, sign):
    """Deletion and insertion curves: P(predicted class) as the top-ranked pixels are removed / restored."""
    order = torch.as_tensor(np.argsort(-heatmap.ravel(), kind="stable"), device=x.device)
    n_pix = heatmap.size
    masks = torch.zeros(len(FRACTIONS), n_pix, device=x.device)
    for i, f in enumerate(FRACTIONS):
        masks[i, order[:int(round(f * n_pix))]] = 1
    masks = masks.view(len(FRACTIONS), 1, IMG_SIZE, IMG_SIZE)
    deletion = x * (1 - masks) + blurred * masks
    insertion = blurred * (1 - masks) + x * masks
    with torch.autocast(x.device.type, enabled=x.device.type == "cuda"):
        logits = model(torch.cat([deletion, insertion])).float().squeeze(1)
    p = torch.sigmoid(sign * logits).cpu().numpy()
    return p[:len(FRACTIONS)], p[len(FRACTIONS):]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", default=[str(CHECKPOINT.parent)])
    ap.add_argument("--datasets", nargs="+", default=[f"FF++={DATA_DIR}"])
    ap.add_argument("--n", type=int, default=300, help="images per dataset (one frame per video)")
    ap.add_argument("--methods", nargs="+", default=["gradcam++", "gradcam", "random"])
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpts = find_checkpoints(args.roots)
    transform = eval_transform()
    faces = FaceCropper()

    # Same images and facial regions for every model
    images = []
    for spec in args.datasets:
        name, root = spec.split("=", 1)
        sample = sample_images(Path(root), args.n)
        for r in tqdm(sample.itertuples(), total=len(sample), desc=f"landmarks {name}"):
            img = Image.open(r.path).convert("RGB")
            rgb8 = np.asarray(img.resize((IMG_SIZE, IMG_SIZE)))
            found = faces.landmarks(rgb8)
            regions = landmark_regions(*found, IMG_SIZE, IMG_SIZE) if found is not None else dict(FACE_REGIONS)
            images.append({"dataset": name, "file": r.file, "path": r.path, "manipulation": r.method,
                           "label": r.label, "x": transform(img), "regions": regions,
                           "region_source": "landmarks" if found is not None else "grid"})
    print(f"{len(images)} images; landmarks found for {sum(i['region_source'] == 'landmarks' for i in images)}")

    out = OUTPUT_DIR / "gradcam"
    out.mkdir(parents=True, exist_ok=True)
    rows, curve_rows, region_rows = [], [], []
    for backbone, path in ckpts.items():
        model = load_model(path, device)
        cams = {m: METHODS[m](model=model, target_layers=[model.target_layer], reshape_transform=model.cam_reshape)
                for m in args.methods if m in METHODS}
        name = BASELINES.get(backbone, backbone)
        # Per-image maps (56x56, float16) for the per-model profiles in notebook 04
        saved = {"prob": [], "pred": [], **{m: [] for m in cams}}
        for k, im in enumerate(tqdm(images, desc=name)):
            x = im["x"].unsqueeze(0).to(device)
            blurred = gaussian_blur(x, kernel_size=31, sigma=10.0)
            with torch.no_grad():
                prob = torch.sigmoid(model(x)).item()
            pred = "fake" if prob >= model.threshold else "real"
            sign = 1.0 if pred == "fake" else -1.0
            saved["prob"].append(prob)
            saved["pred"].append(pred)
            for method in args.methods:
                if method == "random":
                    heat = random_map(k)
                else:
                    heat = cams[method](input_tensor=x, targets=[LogitTarget(sign)])[0]
                    saved[method].append(cv2.resize(heat, (56, 56), interpolation=cv2.INTER_AREA).astype(np.float16))
                    if heat.max() <= 1e-8:  # no positive evidence: nothing to attribute or rank
                        rows.append({"model": name, "dataset": im["dataset"], "file": im["file"], "cam": method,
                                     "label": im["label"], "pred": pred, "correct": pred == im["label"], "empty_cam": True})
                        continue
                    shares = region_scores(heat, im["regions"])
                    region_rows.append({"model": name, "dataset": im["dataset"], "file": im["file"], "manipulation": im["manipulation"],
                                        "label": im["label"], "correct": pred == im["label"], "cam": method,
                                        "region_source": im["region_source"], **shares})
                d, i = curves(model, x, blurred, heat, sign)
                rows.append({"model": name, "dataset": im["dataset"], "file": im["file"], "label": im["label"],
                             "pred": pred, "correct": pred == im["label"], "cam": method,
                             "deletion_auc": float(trapezoid(d, FRACTIONS)),
                             "insertion_auc": float(trapezoid(i, FRACTIONS)), "empty_cam": False})
                curve_rows += [{"model": name, "dataset": im["dataset"], "cam": method, "fraction": f,
                                "deletion": dv, "insertion": iv} for f, dv, iv in zip(FRACTIONS, d, i)]
        np.savez_compressed(
            out / f"heatmaps_{backbone}.npz", model=name, backbone=backbone,
            threshold=model.threshold, target_layer=type(model.target_layer).__name__,
            dataset=[i["dataset"] for i in images], files=[i["file"] for i in images],
            path=[i["path"] for i in images], manipulation=[i["manipulation"] for i in images],
            label=[i["label"] for i in images], prob=np.array(saved.pop("prob")), pred=saved.pop("pred"),
            **{f"cam_{m.replace('+', 'p')}": np.stack(v) for m, v in saved.items()})
        del model, cams
        torch.cuda.empty_cache()

    res = pd.DataFrame(rows)
    res.to_csv(out / "faithfulness.csv", index=False)
    empty = res[res.empty_cam].groupby(["model", "cam"]).size()
    if len(empty):
        print("empty heatmaps (excluded from faithfulness):\n" + empty.to_string())
    res = res[~res.empty_cam]
    pd.DataFrame(curve_rows).groupby(["model", "dataset", "cam", "fraction"]).mean().reset_index() \
        .to_csv(out / "curves.csv", index=False)
    summary = res.groupby(["model", "dataset", "cam"]).agg(
        n=("file", "count"), deletion_auc=("deletion_auc", "mean"), insertion_auc=("insertion_auc", "mean")).reset_index()
    summary.to_csv(out / "faithfulness_summary.csv", index=False)

    tests = []
    for (model, dataset), g in res.groupby(["model", "dataset"]):
        piv = g.pivot_table(index="file", columns="cam", values=["deletion_auc", "insertion_auc"])
        cams = [c for c in args.methods if c in piv["deletion_auc"]]
        for i, a in enumerate(cams):
            for b in cams[i + 1:]:
                for metric in ("deletion_auc", "insertion_auc"):
                    u, v = piv[metric][a], piv[metric][b]
                    p = wilcoxon(u, v).pvalue if (u != v).any() else 1.0
                    tests.append({"model": model, "dataset": dataset, "metric": metric, "a": a, "b": b,
                                  "mean_a": u.mean(), "mean_b": v.mean(), "p_value": p})
    pd.DataFrame(tests).to_csv(out / "faithfulness_tests.csv", index=False)

    regions = pd.DataFrame(region_rows)
    regions.to_csv(out / "regions.csv", index=False)
    region_cols = list(FACE_REGIONS) + ["background / face boundary"]
    attribution = (regions[regions.correct & (regions.cam == "gradcam++")]
                   .groupby(["model", "dataset", "manipulation"])[region_cols].mean())
    attribution.to_csv(out / "region_attribution.csv")

    pd.set_option("display.width", 250)
    print("\nFaithfulness (deletion: lower is better, insertion: higher is better)\n",
          summary.pivot_table(index=["dataset", "model"], columns="cam", values=["deletion_auc", "insertion_auc"]).round(4))
    print(f"\nsaved to {out}")


if __name__ == "__main__":
    main()
