"""Grad-CAM++ explanations for the EfficientNet classifier.

Grad-CAM++ is used instead of plain Grad-CAM because it weights gradients per-pixel, which gives
sharper maps and handles multiple suspicious regions (e.g. both eyes) better — common in face forgeries.

    python -m src.xai.gradcam --image path/to/face.jpg
"""
import argparse
from pathlib import Path
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image

from src.classifier.dataset import eval_transform
from src.classifier.model import load_model
from src.config import CHECKPOINT, IMG_SIZE, OUTPUT_DIR

# Coarse facial regions in normalised (x0, y0, x1, y1) coordinates for aligned, centred face crops
# (as in FFHQ / 140k-faces). Left/right are from the viewer's perspective.
FACE_REGIONS = {
    "forehead / hairline": (0.20, 0.05, 0.80, 0.30),
    "left eye": (0.22, 0.30, 0.48, 0.48),
    "right eye": (0.52, 0.30, 0.78, 0.48),
    "nose": (0.40, 0.42, 0.60, 0.65),
    "left cheek": (0.15, 0.45, 0.40, 0.72),
    "right cheek": (0.60, 0.45, 0.85, 0.72),
    "mouth / teeth": (0.33, 0.65, 0.67, 0.82),
    "chin / jawline": (0.25, 0.80, 0.75, 0.98),
}
BACKGROUND = "background / face boundary"


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

    @property
    def top_regions(self):
        return sorted(self.region_scores.items(), key=lambda kv: kv[1], reverse=True)

    def describe(self, k: int = 3) -> str:
        """Plain-text heatmap summary passed to the LLM."""
        x0, y0, x1, y1 = self.bbox
        parts = [f"{name} ({share:.0%} of activation)" for name, share in self.top_regions[:k]]
        return (f"Classifier verdict: {self.label.upper()} (P(fake) = {self.prob_fake:.2f}). "
                f"Grad-CAM++ activation is concentrated in: {', '.join(parts)}. "
                f"Hottest region bounding box (normalised x0,y0,x1,y1): "
                f"({x0:.2f}, {y0:.2f}, {x1:.2f}, {y1:.2f}).")


def region_box(name: str):
    return FACE_REGIONS.get(name, (0.0, 0.0, 1.0, 1.0))


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


def region_scores(heatmap: np.ndarray) -> dict:
    h, w = heatmap.shape
    total = heatmap.sum() + 1e-8
    covered = np.zeros_like(heatmap, dtype=bool)
    scores = {}
    for name, (x0, y0, x1, y1) in FACE_REGIONS.items():
        sl = (slice(int(y0 * h), int(y1 * h)), slice(int(x0 * w), int(x1 * w)))
        region = np.zeros_like(covered)
        region[sl] = True
        scores[name] = float(heatmap[region & ~covered].sum() / total)
        covered |= region
    scores[BACKGROUND] = float(heatmap[~covered].sum() / total)
    return scores


class Explainer:
    def __init__(self, checkpoint=CHECKPOINT, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = load_model(checkpoint, self.device)
        self.cam = GradCAMPlusPlus(model=self.model, target_layers=[self.model.target_layer])
        self.transform = eval_transform()

    def __call__(self, image: Image.Image) -> Explanation:
        image = image.convert("RGB")
        x = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            prob = torch.sigmoid(self.model(x)).item()
        label = "fake" if prob >= 0.5 else "real"
        heatmap = self.cam(input_tensor=x, targets=[LogitTarget(1.0 if label == "fake" else -1.0)])[0]
        rgb = np.asarray(image.resize((IMG_SIZE, IMG_SIZE)), dtype=np.float32) / 255.0
        overlay = show_cam_on_image(rgb, heatmap, use_rgb=True)
        return Explanation(prob, label, heatmap, overlay, hottest_bbox(heatmap), region_scores(heatmap))


def draw_bbox(overlay: np.ndarray, bbox, color=(255, 255, 255)) -> np.ndarray:
    h, w = overlay.shape[:2]
    x0, y0, x1, y1 = bbox
    out = overlay.copy()
    cv2.rectangle(out, (int(x0 * w), int(y0 * h)), (int(x1 * w) - 1, int(y1 * h) - 1), color, 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    args = ap.parse_args()

    exp = Explainer()(Image.open(args.image))
    print(exp.describe())
    out = OUTPUT_DIR / "gradcam"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{Path(args.image).stem}_gradcampp.png"
    Image.fromarray(draw_bbox(exp.overlay, exp.bbox)).save(path)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
