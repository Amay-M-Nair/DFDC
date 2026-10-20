import numpy as np
import timm
import torch
import torch.nn as nn

DEFAULT_BACKBONE = "efficientnet_b0"

# Baselines used in the model comparison (any other timm model name also works)
BASELINES = {
    "meso4": "MesoNet-4 (Afchar et al., 2018)",
    "legacy_xception": "Xception (FaceForensics++ baseline)",
    "resnet50": "ResNet-50",
    "efficientnet_b0": "EfficientNet-B0",
    "efficientnet_b4": "EfficientNet-B4",
    "vit_small_patch16_224": "ViT-S/16",
    "convnext_tiny": "ConvNeXt-T",
}


class Meso4(nn.Module):
    """MesoNet-4 (Afchar et al., WIFS 2018): a small CNN designed for face-forgery detection, trained from scratch."""

    num_features = 16

    def __init__(self):
        super().__init__()

        def block(cin, cout, k, pool):
            return nn.Sequential(nn.Conv2d(cin, cout, k, padding=k // 2), nn.ReLU(inplace=True),
                                 nn.BatchNorm2d(cout), nn.MaxPool2d(pool))

        self.conv1, self.conv2 = block(3, 8, 3, 2), block(8, 8, 5, 2)
        self.conv3, self.conv4 = block(8, 16, 5, 2), block(16, 16, 5, 4)
        self.fc = nn.Sequential(nn.Flatten(), nn.Dropout(0.5), nn.LazyLinear(16), nn.LeakyReLU(0.1))

    def forward(self, x):
        return self.fc(self.conv4(self.conv3(self.conv2(self.conv1(x)))))


class DeepfakeClassifier(nn.Module):
    """Backbone (EfficientNet-B0 by default) with a single-logit head (logit > 0 => fake)."""

    def __init__(self, backbone: str = DEFAULT_BACKBONE, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        self.backbone_name = backbone
        if backbone == "meso4":
            self.backbone = Meso4()
            self.backbone(torch.zeros(1, 3, 224, 224))  # materialise the lazy layer
        else:
            self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.backbone.num_features, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))

    @property
    def is_vit(self) -> bool:
        return hasattr(self.backbone, "blocks") and hasattr(self.backbone, "patch_embed")

    @property
    def target_layer(self) -> nn.Module:
        """Layer Grad-CAM hooks into: the last convolutional block, or for ViT the last block's first norm."""
        b = self.backbone
        if self.is_vit:
            return b.blocks[-1].norm1
        # EfficientNet, ResNet, Xception (activated output), MesoNet, ConvNeXt
        for attr in ("conv_head", "layer4", "act4", "conv4", "stages"):
            if hasattr(b, attr):
                layer = getattr(b, attr)
                return layer[-1] if attr == "stages" else layer
        return [m for m in b.modules() if isinstance(m, nn.Conv2d)][-1]

    @property
    def cam_reshape(self):
        """For ViT: turn the token sequence (B, prefix + H*W, C) into a (B, C, H, W) map; None for CNNs."""
        if not self.is_vit:
            return None
        prefix = getattr(self.backbone, "num_prefix_tokens", 1)

        def reshape(tokens):
            t = tokens[:, prefix:, :]
            side = int(t.shape[1] ** 0.5)
            return t.reshape(t.shape[0], side, side, t.shape[2]).permute(0, 3, 1, 2)

        return reshape


def checkpoint_path(backbone: str):
    """Main model -> outputs/checkpoints/best.pt; comparison baselines -> outputs/checkpoints/<backbone>/best.pt."""
    from src.config import CHECKPOINT

    return CHECKPOINT if backbone == DEFAULT_BACKBONE else CHECKPOINT.parent / backbone / "best.pt"


def load_model(checkpoint, device="cpu") -> DeepfakeClassifier:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model = DeepfakeClassifier(backbone=ckpt.get("backbone", DEFAULT_BACKBONE), pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.threshold = float(ckpt.get("threshold", 0.5))  # decision threshold tuned on validation
    return model.to(device).eval()


def best_threshold(labels, probs) -> float:
    """Threshold maximising Youden's J (TPR - FPR) = best balanced accuracy on this data."""
    from sklearn.metrics import roc_curve

    fpr, tpr, thr = roc_curve(labels, probs)
    return float(np.clip(thr[np.argmax(tpr - fpr)], 1e-6, 1 - 1e-6))
