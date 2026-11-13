"""Image-quality augmentation, applied identically to real and fake training images.

The v1 models learned "blurry / low resolution = fake" because FF++ fakes are softer than their real frames.
`Degrade` gives real images the same blur, resampling, compression and noise as fakes, so image quality stops
predicting the label and the model has to find manipulation traces instead.
"""
import io
import random

import numpy as np
from PIL import Image, ImageFilter


class RandomJPEG:
    """Re-encode as JPEG at a random quality — makes the model robust to compression."""

    def __init__(self, p=0.5, quality=(40, 95)):
        self.p, self.quality = p, quality

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        return jpeg(img, random.randint(*self.quality))


def jpeg(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def blur(img: Image.Image, radius: float) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius)) if radius > 0 else img


def resample(img: Image.Image, scale: float) -> Image.Image:
    """Downscale by `scale`, then back up to the original size (lost detail, like a resized upload)."""
    if scale >= 1:
        return img
    w, h = img.size
    small = img.resize((max(8, round(w * scale)), max(8, round(h * scale))), Image.BILINEAR)
    return small.resize((w, h), random.choice((Image.BILINEAR, Image.BICUBIC)))


def noise(img: Image.Image, sigma: float) -> Image.Image:
    a = np.asarray(img, dtype=np.float32) + np.random.normal(0, sigma, (img.size[1], img.size[0], 3))
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def sharpen(img: Image.Image, percent: int) -> Image.Image:
    return img.filter(ImageFilter.UnsharpMask(radius=1.5, percent=int(percent), threshold=2))


OPS = {
    "blur": lambda im: blur(im, random.uniform(0.5, 2.0)),
    "resample": lambda im: resample(im, random.uniform(0.4, 0.9)),
    "jpeg": lambda im: jpeg(im, random.randint(40, 95)),
    "noise": lambda im: noise(im, random.uniform(2, 8)),
    "sharpen": lambda im: sharpen(im, random.randint(50, 150)),
}


class Degrade:
    """With probability p, apply 1-2 random quality degradations (blur, resample, JPEG, noise, sharpen)."""

    def __init__(self, p: float = 0.5, max_ops: int = 2):
        self.p, self.max_ops = p, max_ops

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        for name in random.sample(list(OPS), random.randint(1, self.max_ops)):
            img = OPS[name](img)
        return img


# Fixed-strength versions for the robustness curves (src/classifier/robustness.py)
LEVELS = {
    "blur": ((0, 0.5, 1.0, 1.5, 2.0, 3.0), blur),
    "jpeg": ((100, 95, 75, 50, 30), lambda im, q: im if q >= 100 else jpeg(im, q)),
    "resample": ((1.0, 0.75, 0.5, 0.25), resample),
}
