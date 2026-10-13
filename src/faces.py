"""Face detection + square cropping, shared by FaceForensics++ frame extraction and the demo UI.

Uses OpenCV's YuNet detector (downloaded once, ~230 KB) and falls back to the bundled Haar cascade
if the download is unavailable. Crops are square, centred on the face and enlarged by `scale`, so the
face sits in the middle of the crop the same way the Grad-CAM++ region grid in src/xai/gradcam.py expects.
"""
import urllib.request

import cv2
import numpy as np
from PIL import Image

from src.config import OUTPUT_DIR

YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
             "face_detection_yunet_2023mar.onnx")
YUNET_PATH = OUTPUT_DIR / "models" / "face_detection_yunet_2023mar.onnx"


def _load_yunet():
    try:
        if not YUNET_PATH.exists():
            YUNET_PATH.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(YUNET_URL, YUNET_PATH)
        return cv2.FaceDetectorYN.create(str(YUNET_PATH), "", (320, 320), score_threshold=0.7)
    except Exception as e:
        print(f"YuNet unavailable ({e}); falling back to Haar cascade")
        return None


class FaceCropper:
    def __init__(self, scale: float = 1.3, size: int = 256):
        self.scale, self.size = scale, size
        self.yunet = _load_yunet()
        self.haar = None if self.yunet else cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def detect(self, rgb: np.ndarray):
        """Return the largest face box (x, y, w, h) or None."""
        h, w = rgb.shape[:2]
        if self.yunet is not None:
            self.yunet.setInputSize((w, h))
            _, faces = self.yunet.detect(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            boxes = [] if faces is None else [f[:4] for f in faces]
        else:
            boxes = self.haar.detectMultiScale(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), 1.1, 5, minSize=(60, 60))
        if len(boxes) == 0:
            return None
        return max(boxes, key=lambda b: b[2] * b[3])

    def crop(self, rgb: np.ndarray):
        """Square face crop resized to `size`, or None if no face is found."""
        box = self.detect(rgb)
        if box is None:
            return None
        x, y, bw, bh = box
        cx, cy, side = x + bw / 2, y + bh / 2, max(bw, bh) * self.scale
        h, w = rgb.shape[:2]
        x0, y0 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
        x1, y1 = int(min(w, cx + side / 2)), int(min(h, cy + side / 2))
        if x1 - x0 < 32 or y1 - y0 < 32:
            return None
        return cv2.resize(rgb[y0:y1, x0:x1], (self.size, self.size), interpolation=cv2.INTER_AREA)

    def crop_pil(self, image: Image.Image):
        """PIL in, PIL out; returns the original image if no face is detected."""
        face = self.crop(np.asarray(image.convert("RGB")))
        return Image.fromarray(face) if face is not None else image
