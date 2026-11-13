"""The calibrated verdict: REAL / UNCERTAIN / FAKE, plus an out-of-distribution warning.

Combines a trained CNN checkpoint, calibration.json from 02c (src/classifier/calibrate.py) and, when given, the
CLIP probe from 02b. With CLIP the fused score is used (logistic regression on the calibrated CNN logit and the
CLIP logit); without it, the CNN's temperature-scaled score.

    from src.classifier.verdict import Detector
    det = Detector("outputs/checkpoints/best.pt", "outputs/calibration", clip_probe="outputs/clip/probe.pt")
    v = det(face_crop)     # Verdict(prob_fake=0.12, label='real', p_cnn=..., p_clip=..., ood=False, ...)

compare.py and robustness.py use `Calibration` directly on logits computed in bulk.
"""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from src.classifier.calibrate import OOD, sigmoid, verdicts


@dataclass
class Verdict:
    prob_fake: float          # calibrated P(fake) of the score the verdict uses (fused when CLIP is loaded)
    label: str                # 'real' | 'uncertain' | 'fake'
    t_real: float
    t_fake: float
    p_cnn: float              # temperature-scaled CNN score
    p_clip: float = None      # temperature-scaled CLIP score
    ood: bool = False         # features far from anything seen in training: treat the verdict with caution
    ood_score: float = None
    source: str = "cnn"       # 'cnn' or 'fused'


class Calibration:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.data = json.loads((self.folder / "calibration.json").read_text())
        self._ood = {}

    def has(self, backbone) -> bool:
        return backbone in self.data["models"]

    def cnn(self, backbone, logits):
        """(calibrated P(fake), t_real, t_fake)."""
        m = self.data["models"][backbone]
        return sigmoid(np.asarray(logits) / m["temperature"]), m["t_real"], m["t_fake"]

    def clip(self, logits):
        c = self.data["clip"]
        return sigmoid(np.asarray(logits) / c["temperature"]), c["t_real"], c["t_fake"]

    def fused(self, backbone, cnn_logits, clip_logits):
        f = self.data["fused"][backbone]
        t = self.data["models"][backbone]["temperature"]
        z = f["w_cnn"] * np.asarray(cnn_logits) / t + f["w_clip"] * np.asarray(clip_logits) + f["bias"]
        return sigmoid(z), f["t_real"], f["t_fake"]

    def ood(self, backbone) -> OOD:
        if backbone not in self._ood:
            self._ood[backbone] = OOD.load(self.folder / self.data["models"][backbone]["ood_file"])
        return self._ood[backbone]


class Detector:
    def __init__(self, checkpoint, calibration, clip_probe=None, device=None):
        from src.classifier.dataset import eval_transform
        from src.classifier.model import load_model

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = load_model(checkpoint, self.device)
        self.backbone = self.model.backbone_name
        self.cal = Calibration(calibration)
        if not self.cal.has(self.backbone):
            raise KeyError(f"{self.backbone} is not in {calibration}/calibration.json; run 02c with its checkpoint")
        self.transform = eval_transform()
        self.clip = None
        if clip_probe and self.cal.data.get("fused", {}).get(self.backbone):
            from src.classifier.clip_probe import ClipDetector
            self.clip = ClipDetector(clip_probe, self.device)

    @torch.no_grad()
    def __call__(self, image) -> Verdict:
        image = image.convert("RGB")
        x = self.transform(image).unsqueeze(0).to(self.device)
        z, f = self.model.forward_features(x)
        z = float(z.float().item())
        p_cnn, t_real, t_fake = self.cal.cnn(self.backbone, z)
        ood = self.cal.ood(self.backbone)
        dist = float(ood.distance(f.float().flatten(1).cpu().numpy())[0])
        p_clip, prob, source = None, float(p_cnn), "cnn"
        if self.clip is not None:
            zc = float(self.clip([image])[0])
            p_clip = float(self.cal.clip(zc)[0])
            prob, t_real, t_fake = (float(v) for v in self.cal.fused(self.backbone, z, zc))
            source = "fused"
        label = str(verdicts([prob], t_real, t_fake)[0])
        return Verdict(prob, label, float(t_real), float(t_fake), float(p_cnn), p_clip,
                       dist > ood.threshold, dist, source)
