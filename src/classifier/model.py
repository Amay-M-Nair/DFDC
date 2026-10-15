import numpy as np
import timm
import torch
import torch.nn as nn


class DeepfakeClassifier(nn.Module):
    """EfficientNet-B0 backbone with a single-logit head (logit > 0 => fake)."""

    def __init__(self, backbone: str = "efficientnet_b0", pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.backbone.num_features, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))

    @property
    def target_layer(self) -> nn.Module:
        """Last conv block — the layer Grad-CAM++ hooks into."""
        return self.backbone.conv_head


def load_model(checkpoint, device="cpu") -> DeepfakeClassifier:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model = DeepfakeClassifier(backbone=ckpt.get("backbone", "efficientnet_b0"), pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.threshold = float(ckpt.get("threshold", 0.5))  # decision threshold tuned on validation
    return model.to(device).eval()


def best_threshold(labels, probs) -> float:
    """Threshold maximising Youden's J (TPR - FPR) = best balanced accuracy on this data."""
    from sklearn.metrics import roc_curve

    fpr, tpr, thr = roc_curve(labels, probs)
    return float(np.clip(thr[np.argmax(tpr - fpr)], 1e-6, 1 - 1e-6))
