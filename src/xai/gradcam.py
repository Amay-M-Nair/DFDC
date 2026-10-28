"""Grad-CAM / Grad-CAM++ explanations with per-face facial-region attribution.

Grad-CAM++ is the default: it weights gradients per pixel, which gives sharper maps and handles several
suspicious regions at once (e.g. both eyes) better than Grad-CAM. Works for every model in the comparison
(CNNs hook their last conv block; ViT hooks its last block and reshapes tokens back to a 14x14 grid).

Facial regions (eyes, nose, mouth, cheeks, forehead, chin) are placed from YuNet's 5 facial landmarks on
each face, so "the mouth" really is the mouth even when the face is turned or off-centre. If no landmarks
are found, a fixed grid for centred face crops is used instead (`Explanation.region_source` says which).

    python -m src.xai.gradcam --image path/to/face.jpg [--checkpoint path/to/best.pt] [--method gradcam]
"""
import argparse
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image

from src.classifier.dataset import eval_transform
from src.classifier.model import load_model
from src.config import CHECKPOINT, IMG_SIZE, OUTPUT_DIR

METHODS = {"gradcam++": GradCAMPlusPlus, "gradcam": GradCAM}

# Fallback grid for centred face crops, normalised (x0, y0, x1, y1). Left/right from the viewer's side.
# Order matters: earlier (smaller, more specific) regions claim overlapping pixels first.
FACE_REGIONS = {
    "left eye": (0.22, 0.30, 0.48, 0.48),
    "right eye": (0.52, 0.30, 0.78, 0.48),
    "nose": (0.40, 0.42, 0.60, 0.65),
    "mouth / teeth": (0.33, 0.65, 0.67, 0.82),
    "left cheek": (0.15, 0.45, 0.40, 0.72),
    "right cheek": (0.60, 0.45, 0.85, 0.72),
    "forehead / hairline": (0.20, 0.05, 0.80, 0.30),
    "chin / jawline": (0.25, 0.80, 0.75, 0.98),
}
BACKGROUND = "background / face boundary"


def landmark_regions(box, pts, w, h) -> dict:
    """Facial-region boxes (normalised) from a YuNet face box and its 5 landmarks (pixel coordinates)."""
    bx, by, bw, bh = box
    eyes = sorted(pts[:2].tolist())                       # viewer-left eye first
    mouth = sorted(pts[3:5].tolist())
    nose = pts[2]
    d = max(1.0, float(np.hypot(*(np.subtract(eyes[1], eyes[0])))))  # inter-ocular distance
    eye_y = (eyes[0][1] + eyes[1][1]) / 2
    mouth_y = (mouth[0][1] + mouth[1][1]) / 2
    px = {
        "left eye": (eyes[0][0] - .35 * d, eyes[0][1] - .25 * d, eyes[0][0] + .35 * d, eyes[0][1] + .25 * d),
        "right eye": (eyes[1][0] - .35 * d, eyes[1][1] - .25 * d, eyes[1][0] + .35 * d, eyes[1][1] + .25 * d),
        "nose": (nose[0] - .28 * d, eye_y + .2 * d, nose[0] + .28 * d, nose[1] + .2 * d),
        "mouth / teeth": (mouth[0][0] - .15 * d, mouth_y - .3 * d, mouth[1][0] + .15 * d, mouth_y + .3 * d),
        "left cheek": (bx, eye_y + .25 * d, nose[0] - .28 * d, mouth_y),
        "right cheek": (nose[0] + .28 * d, eye_y + .25 * d, bx + bw, mouth_y),
        "forehead / hairline": (bx, by - .15 * bh, bx + bw, eye_y - .25 * d),
        "chin / jawline": (bx, mouth_y + .3 * d, bx + bw, by + 1.05 * bh),
    }
    return {name: (float(np.clip(x0 / w, 0, 1)), float(np.clip(y0 / h, 0, 1)),
                   float(np.clip(x1 / w, 0, 1)), float(np.clip(y1 / h, 0, 1)))
            for name, (x0, y0, x1, y1) in px.items()}


def region_mask(region: str, shape, regions: dict = None) -> np.ndarray:
    """Boolean mask of a named region (BACKGROUND = everything not in any face region)."""
    regions = regions or FACE_REGIONS
    h, w = shape
    mask = np.zeros(shape, dtype=bool)
    if region in regions:
        x0, y0, x1, y1 = regions[region]
        mask[int(y0 * h):int(np.ceil(y1 * h)), int(x0 * w):int(np.ceil(x1 * w))] = True
    elif region == BACKGROUND:
        mask[:] = True
        for name in regions:
            mask &= ~region_mask(name, shape, regions)
    return mask


def region_scores(heatmap: np.ndarray, regions: dict = None) -> dict:
    """Share of total heatmap mass in each region (each pixel counted once, in FACE_REGIONS order)."""
    regions = regions or FACE_REGIONS
    total = heatmap.sum() + 1e-8
    covered = np.zeros(heatmap.shape, dtype=bool)
    scores = {}
    for name in regions:
        m = region_mask(name, heatmap.shape, regions) & ~covered
        scores[name] = float(heatmap[m].sum() / total)
        covered |= m
    scores[BACKGROUND] = float(heatmap[~covered].sum() / total)
    return scores


def region_concentration(heatmap: np.ndarray, regions: dict = None) -> dict:
    """Mean heat inside each region relative to the image mean (1.0 = average, 2.0 = twice as hot).

    Unlike the share of total heat, this doesn't favour large regions (background, forehead) just for their size.
    Pixels are assigned to regions exactly as in region_scores.
    """
    regions = regions or FACE_REGIONS
    overall = float(heatmap.mean()) + 1e-8
    covered = np.zeros(heatmap.shape, dtype=bool)
    conc = {}
    for name in regions:
        m = region_mask(name, heatmap.shape, regions) & ~covered
        conc[name] = float(heatmap[m].mean() / overall) if m.any() else 0.0
        covered |= m
    conc[BACKGROUND] = float(heatmap[~covered].mean() / overall) if (~covered).any() else 0.0
    return conc


def confidence_level(prob_fake: float, threshold: float) -> str:
    """How far the score is past the decision threshold, as a fraction of the room on that side."""
    if prob_fake >= threshold:
        margin = (prob_fake - threshold) / max(1 - threshold, 1e-6)
    else:
        margin = (threshold - prob_fake) / max(threshold, 1e-6)
    return "high" if margin >= 0.5 else "moderate" if margin >= 0.15 else "borderline"


def heatmap_mask(heatmap: np.ndarray, quantile: float = 0.8) -> np.ndarray:
    return heatmap >= np.quantile(heatmap, quantile)


def hottest_bbox(heatmap: np.ndarray, quantile: float = 0.8):
    """Bounding box of the connected component containing the heatmap peak."""
    mask = heatmap_mask(heatmap, quantile).astype(np.uint8)
    _, labels = cv2.connectedComponents(mask)
    peak = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    ys, xs = np.where(labels == labels[peak])
    h, w = heatmap.shape
    return (xs.min() / w, ys.min() / h, (xs.max() + 1) / w, (ys.max() + 1) / h)


class LogitTarget:
    """Explain the predicted class: +logit for 'fake', -logit for 'real'."""

    def __init__(self, sign: float):
        self.sign = sign

    def __call__(self, model_output):
        return self.sign * model_output.reshape(-1)[0]


@dataclass
class Explanation:
    prob_fake: float
    label: str
    heatmap: np.ndarray            # HxW float in [0, 1]
    overlay: np.ndarray            # HxWx3 uint8 RGB
    bbox: tuple                    # (x0, y0, x1, y1) normalised box around the hottest region
    region_scores: dict = field(default_factory=dict)  # region -> share of heatmap mass
    regions: dict = field(default_factory=lambda: dict(FACE_REGIONS))  # region -> normalised box on this face
    region_source: str = "grid"    # "landmarks" or "grid"
    method: str = "gradcam++"
    region_concentration: dict = field(default_factory=dict)  # region -> mean heat / image mean
    rank_by: str = "concentration"  # how regions are ranked: "concentration" or "share"
    threshold: float = 0.5          # the detector's decision threshold (tuned on FF++ validation)

    @property
    def top_regions(self):
        """(region, score) ranked by `rank_by`; falls back to share if concentration wasn't computed."""
        key = self.region_concentration if self.rank_by == "concentration" and self.region_concentration \
            else self.region_scores
        return sorted(key.items(), key=lambda kv: kv[1], reverse=True)

    @property
    def confidence(self) -> str:
        return confidence_level(self.prob_fake, self.threshold)

    def describe(self, k: int = 3) -> str:
        """Plain-text evidence summary passed to the LLM: filled from fixed templates, no model involved.
        k: how many regions to list, hottest first; None lists every region (the demo)."""
        x0, y0, x1, y1 = self.bbox
        parts = []
        for name, _ in self.top_regions[:k]:
            share = f"{self.region_scores.get(name, 0):.0%} of total activation"
            if self.region_concentration:
                parts.append(f"{name} ({self.region_concentration[name]:.1f}x the image average, {share})")
            else:
                parts.append(f"{name} ({share})")
        method = {"gradcam++": "Grad-CAM++", "gradcam": "Grad-CAM"}.get(self.method, self.method)
        where = "is most concentrated in" if k is not None else "in every region, most concentrated first"
        return (f"Classifier verdict: {self.label.upper()} (P(fake) = {self.prob_fake:.3f}; decision threshold "
                f"{self.threshold:.3f}; {self.confidence} confidence). "
                f"{method} evidence {where}: {'; '.join(parts)}. "
                f"Hottest region bounding box (normalised x0,y0,x1,y1): "
                f"({x0:.2f}, {y0:.2f}, {x1:.2f}, {y1:.2f}).")


class Explainer:
    def __init__(self, checkpoint=CHECKPOINT, device=None, method: str = "gradcam++", landmarks: bool = True,
                 rank_by: str = "concentration"):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = load_model(checkpoint, self.device)
        self.method = method
        self.rank_by = rank_by
        self.cam = METHODS[method](model=self.model, target_layers=[self.model.target_layer],
                                   reshape_transform=self.model.cam_reshape)
        self.transform = eval_transform()
        self.faces = None
        if landmarks:
            from src.faces import FaceCropper

            self.faces = FaceCropper()

    def face_regions(self, rgb_uint8: np.ndarray):
        found = self.faces.landmarks(rgb_uint8) if self.faces is not None else None
        if found is None:
            return dict(FACE_REGIONS), "grid"
        h, w = rgb_uint8.shape[:2]
        return landmark_regions(*found, w, h), "landmarks"

    def heatmap(self, x: torch.Tensor, label: str) -> np.ndarray:
        return self.cam(input_tensor=x, targets=[LogitTarget(1.0 if label == "fake" else -1.0)])[0]

    def predict(self, image: Image.Image):
        """(input tensor, P(fake)) for an RGB image: the detector alone, without the heatmap."""
        x = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return x, torch.sigmoid(self.model(x)).item()

    def __call__(self, image: Image.Image) -> Explanation:
        image = image.convert("RGB")
        return self.explain(image, *self.predict(image))

    def explain(self, image: Image.Image, x: torch.Tensor, prob: float) -> Explanation:
        """Heatmap and facial-region attribution for an RGB image already scored by predict()."""
        label = "fake" if prob >= self.model.threshold else "real"
        heatmap = self.heatmap(x, label)
        rgb8 = np.asarray(image.resize((IMG_SIZE, IMG_SIZE)))
        overlay = show_cam_on_image(rgb8.astype(np.float32) / 255.0, heatmap, use_rgb=True)
        regions, source = self.face_regions(rgb8)
        return Explanation(prob, label, heatmap, overlay, hottest_bbox(heatmap),
                           region_scores(heatmap, regions), regions, source, self.method,
                           region_concentration(heatmap, regions), self.rank_by, self.model.threshold)


def draw_bbox(overlay: np.ndarray, bbox, color=(255, 255, 255)) -> np.ndarray:
    h, w = overlay.shape[:2]
    x0, y0, x1, y1 = bbox
    out = overlay.copy()
    cv2.rectangle(out, (int(x0 * w), int(y0 * h)), (int(x1 * w) - 1, int(y1 * h) - 1), color, 2)
    return out


REGION_COLORS = [(255, 80, 80), (255, 160, 0), (255, 255, 0), (0, 220, 0),
                 (0, 200, 255), (60, 100, 255), (200, 0, 255), (255, 255, 255)]


def draw_regions(rgb: np.ndarray, regions: dict) -> np.ndarray:
    """Draw every facial-region box (sanity check for the landmark-based placement)."""
    out = rgb.copy()
    h, w = out.shape[:2]
    for (name, (x0, y0, x1, y1)), color in zip(regions.items(), REGION_COLORS):
        cv2.rectangle(out, (int(x0 * w), int(y0 * h)), (int(x1 * w) - 1, int(y1 * h) - 1), color, 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--checkpoint", default=str(CHECKPOINT))
    ap.add_argument("--method", default="gradcam++", choices=sorted(METHODS))
    args = ap.parse_args()

    exp = Explainer(args.checkpoint, method=args.method)(Image.open(args.image))
    print(exp.describe(), f"[regions from {exp.region_source}]")
    out = OUTPUT_DIR / "gradcam"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{Path(args.image).stem}_{args.method.replace('+', 'pp')}.png"
    Image.fromarray(draw_bbox(exp.overlay, exp.bbox)).save(path)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
