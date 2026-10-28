"""Grad-CAM / Grad-CAM++ explanations with per-face facial-region attribution.

Grad-CAM++ is the default: it weights gradients per pixel, which gives sharper maps and handles several
suspicious regions at once (e.g. both eyes) better than Grad-CAM. Works for every model in the comparison
(CNNs hook their last conv block; ViT hooks its last block and reshapes tokens back to a 14x14 grid).

Facial regions (eyes, nose, mouth, cheeks, forehead, chin) are placed from YuNet's 5 facial landmarks on
each face, so "the mouth" really is the mouth even when the face is turned or off-centre. If no landmarks
are found, a fixed grid for centred face crops is used instead (`Explanation.region_source` says which).
The demo uses finer, atomic regions (teeth, hairline, jawline, ...) placed with MediaPipe's 478-point face mesh
(`Explainer(atomic=True)`); phases 2-5 use the 9 combined regions above.

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

# Atomic regions (the demo), in claiming order. The demo places them with MediaPipe's face mesh
# (mesh_regions); without a mesh it splits the combined boxes above into their parts (landmark_regions, or this
# grid) and leaves teeth out, since YuNet's 5 points cannot show whether the mouth is open. "face boundary" is a
# band around the face that claims only pixels no face region has; the rest of the image is "background".
ATOMIC_FACE_REGIONS = {
    "left eye": (0.22, 0.30, 0.48, 0.48),
    "right eye": (0.52, 0.30, 0.78, 0.48),
    "nose": (0.40, 0.42, 0.60, 0.65),
    "teeth": (0.38, 0.70, 0.62, 0.77),
    "mouth": (0.33, 0.65, 0.67, 0.82),
    "left cheek": (0.15, 0.45, 0.40, 0.72),
    "right cheek": (0.60, 0.45, 0.85, 0.72),
    "hairline": (0.20, 0.05, 0.80, 0.15),
    "forehead": (0.20, 0.15, 0.80, 0.30),
    "chin": (0.33, 0.80, 0.67, 0.98),
    "left jawline": (0.25, 0.80, 0.33, 0.98),
    "right jawline": (0.67, 0.80, 0.75, 0.98),
    "face boundary": (0.075, 0.05, 0.925, 0.98),
}
ATOMIC_BACKGROUND = "background"
# atomic region -> the combined region it is part of
COMBINED = {**{r: r for r in FACE_REGIONS}, "teeth": "mouth / teeth", "mouth": "mouth / teeth",
            "hairline": "forehead / hairline", "forehead": "forehead / hairline", "chin": "chin / jawline",
            "left jawline": "chin / jawline", "right jawline": "chin / jawline", "face boundary": BACKGROUND,
            ATOMIC_BACKGROUND: BACKGROUND}


# MediaPipe face-mesh point groups (its FaceLandmarksConnections.FACE_LANDMARKS_*; the lips split into rings)
MESH_OVAL = (10, 21, 54, 58, 67, 93, 103, 109, 127, 132, 136, 148, 149, 150, 152, 162, 172, 176, 234, 251, 284, 288,
             297, 323, 332, 338, 356, 361, 365, 377, 378, 379, 389, 397, 400, 454)
MESH_EYES = ((7, 33, 133, 144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 163, 173, 246),
             (249, 263, 362, 373, 374, 380, 381, 382, 384, 385, 386, 387, 388, 390, 398, 466))
MESH_BROWS = (46, 52, 53, 55, 63, 65, 66, 70, 105, 107, 276, 282, 283, 285, 293, 295, 296, 300, 334, 336)
MESH_NOSE = (1, 2, 4, 5, 6, 19, 45, 48, 64, 94, 97, 98, 115, 168, 195, 197, 220, 275, 278, 294, 326, 327, 344, 440)
MESH_OUTER_LIPS = (0, 17, 37, 39, 40, 61, 84, 91, 146, 181, 185, 267, 269, 270, 291, 314, 321, 375, 405, 409)
MESH_INNER_LIPS = (13, 14, 78, 80, 81, 82, 87, 88, 95, 178, 191, 308, 310, 311, 312, 317, 318, 324, 402, 415)
MESH_TOP, MESH_CHIN = 10, 152
MOUTH_OPEN = 0.08   # inner-lip gap / inner mouth width above which the mouth counts as open (teeth may show)


def mesh_regions(pts: np.ndarray, w: int, h: int) -> dict:
    """Atomic facial-region boxes (normalised) from MediaPipe's 478 face-mesh points (pixel coordinates).
    Teeth is the inner-lip opening and is left out when the mouth is closed; the hairline is a band above the
    mesh, which stops at the upper forehead; the face boundary is a band around the face outline."""
    def bbox(idx, pad_x=0.0, pad_y=0.0):
        p = pts[list(idx)]
        return p[:, 0].min() - pad_x, p[:, 1].min() - pad_y, p[:, 0].max() + pad_x, p[:, 1].max() + pad_y

    def oval_x(y0, y1):
        """Left and right edge of the face outline between heights y0 and y1."""
        p = pts[list(MESH_OVAL)]
        band = p[(p[:, 1] >= y0) & (p[:, 1] <= y1)]
        band = band if len(band) else p
        return band[:, 0].min(), band[:, 0].max()

    top, chin = pts[MESH_TOP, 1], pts[MESH_CHIN, 1]
    face_h = chin - top
    eyes = sorted(MESH_EYES, key=lambda idx: pts[list(idx), 0].mean())   # viewer-left eye first
    eye_boxes = []
    for idx in eyes:
        x0, y0, x1, y1 = bbox(idx)
        ew = x1 - x0
        eye_boxes.append((x0 - .25 * ew, y0 - .35 * ew, x1 + .25 * ew, y1 + .35 * ew))
    nose = bbox(MESH_NOSE)
    mx0, my0, mx1, my1 = bbox(MESH_OUTER_LIPS)
    mouth_w = mx1 - mx0
    mouth_y = (pts[61, 1] + pts[291, 1]) / 2
    brow_top = pts[list(MESH_BROWS), 1].min()
    fx0, fx1 = oval_x(top, brow_top)
    lx0, _ = oval_x(eye_boxes[0][3], mouth_y)
    _, rx1 = oval_x(eye_boxes[1][3], mouth_y)
    jx0, jx1 = oval_x(mouth_y, chin)
    ox0, ox1 = oval_x(top, chin)
    px = {"left eye": eye_boxes[0], "right eye": eye_boxes[1], "nose": nose}
    gap = np.hypot(*(pts[13] - pts[14])) / max(1.0, float(np.hypot(*(pts[78] - pts[308]))))
    if gap > MOUTH_OPEN:
        px["teeth"] = bbox(MESH_INNER_LIPS)
    px.update({
        "mouth": (mx0 - .08 * mouth_w, my0 - .1 * mouth_w, mx1 + .08 * mouth_w, my1 + .1 * mouth_w),
        "left cheek": (lx0, eye_boxes[0][3], nose[0], mouth_y),
        "right cheek": (nose[2], eye_boxes[1][3], rx1, mouth_y),
        "hairline": (fx0, top - .15 * face_h, fx1, top),
        "forehead": (fx0, top, fx1, brow_top),
        "chin": (mx0, my1, mx1, chin + .03 * face_h),
        "left jawline": (jx0, mouth_y, mx0, chin + .03 * face_h),
        "right jawline": (mx1, mouth_y, jx1, chin + .03 * face_h),
        "face boundary": (ox0 - .06 * (ox1 - ox0), top - .15 * face_h, ox1 + .06 * (ox1 - ox0), chin + .06 * face_h),
    })
    return {name: (float(np.clip(x0 / w, 0, 1)), float(np.clip(y0 / h, 0, 1)),
                   float(np.clip(x1 / w, 0, 1)), float(np.clip(y1 / h, 0, 1)))
            for name, (x0, y0, x1, y1) in px.items()}


def background_name(regions: dict) -> str:
    """Name of the leftover region (every pixel outside the region boxes) for a set of regions."""
    return ATOMIC_BACKGROUND if "face boundary" in regions else BACKGROUND


def all_regions(atomic: bool = False) -> list:
    """Every region name, background last: the combined 9 (phases 2-5) or the atomic 14 (the demo)."""
    face = ATOMIC_FACE_REGIONS if atomic else FACE_REGIONS
    return list(face) + [background_name(face)]


def landmark_regions(box, pts, w, h, atomic: bool = False) -> dict:
    """Facial-region boxes (normalised) from a YuNet face box and its 5 landmarks (pixel coordinates).
    atomic: split the combined regions into their parts (ATOMIC_FACE_REGIONS order), each inside its old box,
    without teeth (5 points cannot show whether the mouth is open). The demo's fallback when MediaPipe finds no face."""
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
    if atomic:
        mx0, _, mx1, _ = px["mouth / teeth"]
        fx0, fy0, fx1, fy1 = px["forehead / hairline"]
        cx0, cy0, cx1, cy1 = px["chin / jawline"]
        hair_y = fy0 + .4 * (fy1 - fy0)
        px = {"left eye": px["left eye"], "right eye": px["right eye"], "nose": px["nose"],
              "mouth": px["mouth / teeth"], "left cheek": px["left cheek"], "right cheek": px["right cheek"],
              "hairline": (fx0, fy0, fx1, hair_y), "forehead": (fx0, hair_y, fx1, fy1),
              "chin": (mx0, cy0, mx1, cy1), "left jawline": (cx0, cy0, mx0, cy1), "right jawline": (mx1, cy0, cx1, cy1),
              "face boundary": (bx - .05 * bw, fy0, bx + 1.05 * bw, cy1)}
    return {name: (float(np.clip(x0 / w, 0, 1)), float(np.clip(y0 / h, 0, 1)),
                   float(np.clip(x1 / w, 0, 1)), float(np.clip(y1 / h, 0, 1)))
            for name, (x0, y0, x1, y1) in px.items()}


def region_mask(region: str, shape, regions: dict = None) -> np.ndarray:
    """Boolean mask of a named region (the background = everything not in any face region)."""
    regions = regions or FACE_REGIONS
    h, w = shape
    mask = np.zeros(shape, dtype=bool)
    if region in regions:
        x0, y0, x1, y1 = regions[region]
        mask[int(y0 * h):int(np.ceil(y1 * h)), int(x0 * w):int(np.ceil(x1 * w))] = True
    elif region == background_name(regions):
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
    scores[background_name(regions)] = float(heatmap[~covered].sum() / total)
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
    conc[background_name(regions)] = float(heatmap[~covered].mean() / overall) if (~covered).any() else 0.0
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

    @property
    def atomic(self) -> bool:
        """True when the regions are the atomic ones (ATOMIC_FACE_REGIONS)."""
        return background_name(self.regions) == ATOMIC_BACKGROUND

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
                 rank_by: str = "concentration", atomic: bool = False):
        """atomic: use the atomic regions (teeth, hairline, jawline, ...; the demo) instead of the combined 9,
        placed with MediaPipe's face mesh when the mediapipe package is installed."""
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = load_model(checkpoint, self.device)
        self.method = method
        self.rank_by = rank_by
        self.atomic = atomic
        self.cam = METHODS[method](model=self.model, target_layers=[self.model.target_layer],
                                   reshape_transform=self.model.cam_reshape)
        self.transform = eval_transform()
        self.faces = self.mesh = None
        if landmarks:
            from src.faces import FaceCropper, FaceMesh

            self.faces = FaceCropper()
            if atomic:
                try:
                    self.mesh = FaceMesh()
                except Exception as e:  # no mediapipe or no model: boxes from YuNet's 5 points instead
                    print(f"MediaPipe face mesh unavailable ({e}); atomic regions from YuNet landmarks, no teeth")

    def face_regions(self, rgb_uint8: np.ndarray):
        """(region boxes, source): source is "mesh" (MediaPipe, atomic only), "landmarks" (YuNet) or "grid"."""
        h, w = rgb_uint8.shape[:2]
        pts = self.mesh.points(rgb_uint8) if self.mesh is not None else None
        if pts is not None:
            return mesh_regions(pts, w, h), "mesh"
        found = self.faces.landmarks(rgb_uint8) if self.faces is not None else None
        if found is None:
            grid = dict(ATOMIC_FACE_REGIONS if self.atomic else FACE_REGIONS)
            grid.pop("teeth", None)  # a fixed grid cannot show whether the mouth is open
            return grid, "grid"
        return landmark_regions(*found, w, h, self.atomic), "landmarks"

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
