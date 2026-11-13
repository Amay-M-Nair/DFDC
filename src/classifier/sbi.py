"""Self-blended images (Shiohara & Yamasaki, CVPR 2022): training fakes made from a single real face.

A real face crop is blended with a slightly altered copy of itself through a soft face-shaped mask. The result
has the traces every face swap leaves (a blending seam, a colour and sharpness mismatch between the inner face
and its surroundings) but no method-specific artefacts, so a model trained on it learns the blending traces
instead of the quirks of FF++'s generators.

    fake = mask * source + (1 - mask) * target      target = the real crop, source = altered copy

Landmarks come from MediaPipe's face mesh, cached once per dataset by src/classifier/landmarks.py; only the
LANDMARKS subset (face oval + an inner-face ring) is stored.

    python -m src.classifier.sbi --demo --images examples/ffpp/real --out <folder>   # visual check grid
"""
import argparse
import random
from pathlib import Path

import cv2
import numpy as np

# MediaPipe face-mesh indices. FACE_OVAL: the outline (forehead to chin); INNER: brows, cheekbones, nose wings,
# mouth and chin, whose hull is a smaller inner-face region. landmarks.npz stores FACE_OVAL + INNER in this order.
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
             176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
INNER = [70, 63, 105, 66, 107, 336, 296, 334, 293, 300, 116, 345, 98, 327, 61, 291, 17, 200]
LANDMARKS = FACE_OVAL + INNER
BLEND_RATIOS = (0.25, 0.5, 0.75, 1.0, 1.0, 1.0)


def _odd(x: float) -> int:
    return max(3, int(x) // 2 * 2 + 1)


def source_transform(img: np.ndarray) -> np.ndarray:
    """The altered copy: small colour/brightness shift, sharpness change, and a 95-105% scale + small shift."""
    out = img.astype(np.float32)
    h, w = img.shape[:2]
    if random.random() < 0.7:
        out = out * random.uniform(0.93, 1.07) + np.random.uniform(-10, 10, 3)
    if random.random() < 0.7:
        if random.random() < 0.5:
            s = random.uniform(0.5, 0.9)
            out = cv2.resize(cv2.resize(out, (max(8, int(w * s)), max(8, int(h * s)))), (w, h))
        else:
            blurred = cv2.GaussianBlur(out, (0, 0), random.uniform(0.8, 1.5))
            out = cv2.addWeighted(out, 1.6, blurred, -0.6, 0)
    out = np.clip(out, 0, 255).astype(np.uint8)
    s = random.uniform(0.95, 1.05)
    tx, ty = np.random.uniform(-0.03, 0.03, 2) * (w, h)
    m = np.float32([[s, 0, tx + (1 - s) * w / 2], [0, s, ty + (1 - s) * h / 2]])
    return cv2.warpAffine(out, m, (w, h), borderMode=cv2.BORDER_REFLECT_101)


def blend_mask(points: np.ndarray, shape) -> np.ndarray:
    """Soft face mask in [0, 1]: landmark hull (full or inner face), random erode/dilate, blurred edge, ratio."""
    h, w = shape[:2]
    pts = points[:len(FACE_OVAL)] if random.random() < 0.6 else points[len(FACE_OVAL):]
    hull = cv2.convexHull(pts.astype(np.float32).round().astype(np.int32))
    mask = np.zeros((h, w), np.float32)
    cv2.fillConvexPoly(mask, hull, 1.0)
    size = max(8.0, float(np.ptp(pts[:, 0])))
    k = _odd(size * random.uniform(0.02, 0.08))
    mask = (cv2.erode if random.random() < 0.5 else cv2.dilate)(mask, np.ones((k, k), np.uint8))
    mask = cv2.GaussianBlur(mask, (_odd(size * random.uniform(0.05, 0.15)),) * 2, 0)
    if mask.max() > 0:
        mask /= mask.max()
    return mask * random.choice(BLEND_RATIOS)


def self_blend(img: np.ndarray, points: np.ndarray):
    """(fake HxWx3 uint8, mask HxW float32) from a real RGB crop and its LANDMARKS points (pixels of the crop)."""
    mask = blend_mask(points, img.shape)
    fake = mask[..., None] * source_transform(img) + (1 - mask[..., None]) * img
    return fake.round().astype(np.uint8), mask


def demo(images: Path, out: Path, n: int = 6, seed: int = 0):
    """Grid per face: original | self-blended | |difference| x4 | mask | degraded original | degraded SBI."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    from src.classifier.augment import Degrade
    from src.faces import FaceMesh

    random.seed(seed); np.random.seed(seed)
    mesh, degrade = FaceMesh(), Degrade(p=1.0)
    paths = sorted(p for p in Path(images).iterdir() if p.suffix.lower() in {".jpg", ".png"})
    rows = []
    for p in paths:
        img = np.asarray(Image.open(p).convert("RGB"))
        pts = mesh.points(img)
        if pts is None:
            continue
        fake, mask = self_blend(img, pts[LANDMARKS])
        diff = np.clip(np.abs(fake.astype(int) - img.astype(int)) * 4, 0, 255).astype(np.uint8)
        state = random.getstate()
        real_d = np.asarray(degrade(Image.fromarray(img)))
        random.setstate(state)  # same degradation for the pair, so only the blend differs
        fake_d = np.asarray(degrade(Image.fromarray(fake)))
        rows.append((p.name, [img, fake, diff, mask, real_d, fake_d]))
        if len(rows) == n:
            break
    titles = ["real", "self-blended (fake)", "|difference| x4", "blend mask", "real, degraded", "SBI, degraded"]
    fig, axes = plt.subplots(len(rows), 6, figsize=(15, 2.6 * len(rows)), squeeze=False)
    for r, (name, ims) in enumerate(rows):
        for c, im in enumerate(ims):
            axes[r, c].imshow(im, cmap="gray" if im.ndim == 2 else None, vmin=0, vmax=1 if im.ndim == 2 else None)
            axes[r, c].axis("off")
            if r == 0:
                axes[r, c].set_title(titles[c], fontsize=10)
        axes[r, 0].text(0, -6, name, fontsize=7)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "sbi_demo.png"
    plt.tight_layout(); plt.savefig(path, dpi=110); plt.close()
    print(f"saved {path} ({len(rows)} faces)")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--images", type=Path, default=Path("examples/ffpp/real"))
    ap.add_argument("--out", type=Path, default=Path("outputs/sbi"))
    ap.add_argument("--n", type=int, default=6)
    args = ap.parse_args()
    if args.demo:
        demo(args.images, args.out, args.n)


if __name__ == "__main__":
    main()
