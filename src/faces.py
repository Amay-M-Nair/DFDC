"""Face detection + square cropping, shared by FaceForensics++ / Celeb-DF frame extraction and the demo UI.

Uses OpenCV's YuNet detector (~230 KB ONNX, downloaded once). Crops are square, centred on the face and
enlarged by `scale`, so the face sits in the middle of the crop the way the Grad-CAM++ region grid expects.

Dataset extraction calls `ensure_yunet()` once in the main process and then builds croppers with
`strict=True`, so every frame is cropped by the same detector (a silent fallback would mix detectors).
The demo UI may fall back to OpenCV's bundled Haar cascade.
"""
import os
import urllib.request

import cv2
import numpy as np
from PIL import Image

from src.config import OUTPUT_DIR

YUNET_URLS = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/"
    "face_detection_yunet_2023mar.onnx",
)
YUNET_PATH = OUTPUT_DIR / "models" / "face_detection_yunet_2023mar.onnx"


def _create_yunet(path):
    return cv2.FaceDetectorYN.create(str(path), "", (320, 320), score_threshold=0.7)


def ensure_yunet():
    """Download YuNet atomically (temp file + rename) and check it loads. Call once, before starting workers."""
    if YUNET_PATH.exists():
        try:
            _create_yunet(YUNET_PATH)
            return YUNET_PATH
        except cv2.error:
            YUNET_PATH.unlink()  # corrupt / partial file from an earlier run
    YUNET_PATH.parent.mkdir(parents=True, exist_ok=True)
    errors = []
    for url in YUNET_URLS:
        tmp = YUNET_PATH.with_name(f"{YUNET_PATH.stem}.{os.getpid()}.tmp.onnx")  # OpenCV needs the .onnx suffix
        try:
            urllib.request.urlretrieve(url, tmp)
            _create_yunet(tmp)  # raises if it is not a valid ONNX model (e.g. an HTML or Git-LFS pointer page)
            os.replace(tmp, YUNET_PATH)
            print(f"YuNet face detector ready ({YUNET_PATH.stat().st_size // 1024} KB)")
            return YUNET_PATH
        except Exception as e:
            errors.append(f"{url}: {e}")
            tmp.unlink(missing_ok=True)
    raise RuntimeError("Could not download the YuNet face detector:\n" + "\n".join(errors))


class FaceCropper:
    def __init__(self, scale: float = 1.3, size: int = 256, strict: bool = False):
        """strict=True: YuNet must already be available (see ensure_yunet); never fall back to Haar."""
        self.scale, self.size = scale, size
        self.yunet, self.haar = None, None
        try:
            if not strict:
                ensure_yunet()
            self.yunet = _create_yunet(YUNET_PATH)
        except Exception as e:
            if strict:
                raise RuntimeError(f"YuNet face detector failed to load: {e}") from e
            print(f"YuNet unavailable ({e}); falling back to Haar cascade")
            self.haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

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

    def landmarks(self, rgb: np.ndarray):
        """Largest face as (box (x, y, w, h), 5x2 points: two eyes, nose tip, two mouth corners), or None.

        Needs YuNet (the Haar fallback has no landmarks). Coordinates are in pixels of `rgb`.
        """
        if self.yunet is None:
            return None
        h, w = rgb.shape[:2]
        self.yunet.setInputSize((w, h))
        _, faces = self.yunet.detect(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if faces is None or len(faces) == 0:
            return None
        f = max(faces, key=lambda r: r[2] * r[3])
        return f[:4], f[4:14].reshape(5, 2)

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
