"""
Step 3: Model architectures for birefringence prediction.

Architecture 1: HybridCNN
  - EfficientNet-B0 backbone (pretrained on ImageNet)
  - Scalar MLP branch for (temperature, thickness, order, reduced_temp)
  - Fusion head: concatenate image features + scalar features -> regression

Architecture 2: HybridResNet (alternative)
  - ResNet-18 backbone

Both output a single value: predicted delta_n (birefringence).
"""

import torch
import torch.nn as nn
import timm


class ScalarBranch(nn.Module):
    """Small MLP to process scalar inputs (temperature, thickness, order)."""

    def __init__(self, n_scalars: int = 4, hidden_dim: int = 64, out_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_scalars, hidden_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class FusionHead(nn.Module):
    """Regression head that fuses image + scalar features."""

    def __init__(self, img_dim: int, scalar_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(img_dim + scalar_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, img_feat, scalar_feat):
        x = torch.cat([img_feat, scalar_feat], dim=1)
        return self.net(x).squeeze(-1)


class HybridCNN(nn.Module):
    """
    Hybrid model: EfficientNet-B0 backbone + scalar branch + fusion head.

    Input:
        image: (B, 3, 224, 224)
        scalars: (B, 4) - [temperature, thickness, order, reduced_temp]
    Output:
        delta_n: (B,) - predicted birefringence
    """

    def __init__(
        self,
        backbone_name: str = "efficientnet_b0",
        pretrained: bool = True,
        n_scalars: int = 4,
        scalar_hidden: int = 64,
        scalar_out: int = 32,
        fusion_hidden: int = 256,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        # Image backbone
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,  # remove classification head, get features
        )
        img_feat_dim = self.backbone.num_features  # e.g., 1280 for efficientnet_b0

        # Scalar branch
        self.scalar_branch = ScalarBranch(n_scalars, scalar_hidden, scalar_out)

        # Fusion head
        self.fusion_head = FusionHead(img_feat_dim, scalar_out, fusion_hidden)

        # Optional: freeze backbone for initial training
        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self):
        """Freeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("  [INFO] Backbone frozen")

    def unfreeze_backbone(self, n_layers_from_end: int = 2):
        """
        Unfreeze the last n blocks of the backbone for fine-tuning.
        For EfficientNet-B0, blocks are in backbone.blocks[0..6].
        """
        # First unfreeze everything
        for param in self.backbone.parameters():
            param.requires_grad = True

        # Then freeze everything except last n blocks
        if hasattr(self.backbone, "blocks"):
            n_blocks = len(self.backbone.blocks)
            for i, block in enumerate(self.backbone.blocks):
                if i < n_blocks - n_layers_from_end:
                    for param in block.parameters():
                        param.requires_grad = False

        print(f"  [INFO] Backbone unfrozen (last {n_layers_from_end} blocks)")

    def forward(self, image, scalars):
        img_feat = self.backbone(image)       # (B, 1280)
        scalar_feat = self.scalar_branch(scalars)  # (B, 32)
        delta_n = self.fusion_head(img_feat, scalar_feat)  # (B,)
        return delta_n


class HybridResNet(HybridCNN):
    """Alternative using ResNet-18 backbone."""

    def __init__(self, pretrained=True, n_scalars=4, freeze_backbone=False):
        # Don't call HybridCNN.__init__ directly; build manually
        nn.Module.__init__(self)

        self.backbone = timm.create_model(
            "resnet18", pretrained=pretrained, num_classes=0,
        )
        img_feat_dim = self.backbone.num_features  # 512

        self.scalar_branch = ScalarBranch(n_scalars, 64, 32)
        self.fusion_head = FusionHead(img_feat_dim, 32, 128)

        if freeze_backbone:
            self.freeze_backbone()

    def unfreeze_backbone(self, n_layers_from_end=1):
        for param in self.backbone.parameters():
            param.requires_grad = True
        # Freeze everything except layer4
        for name, param in self.backbone.named_parameters():
            if "layer4" not in name and "fc" not in name:
                param.requires_grad = False
        print("  [INFO] ResNet backbone: only layer4 unfrozen")


class PhysicsInformedLoss(nn.Module):
    """
    Combined MSE loss + physics regularization.

    Physics constraint: delta_n should follow Haller approximation:
        delta_n(T) ~ delta_n_max * (1 - T/Tc)^beta
    So delta_n should decrease monotonically as T increases toward Tc,
    and be zero above Tc.

    We add a penalty when:
    1. Predicted delta_n is negative
    2. Predictions don't respect monotonicity with temperature (soft)
    """

    def __init__(self, alpha_nonneg: float = 1.0, alpha_mono: float = 0.1):
        super().__init__()
        self.mse = nn.MSELoss()
        self.alpha_nonneg = alpha_nonneg
        self.alpha_mono = alpha_mono

    def forward(self, pred, target, temperature=None):
        # Main MSE loss
        loss = self.mse(pred, target)

        # Non-negativity constraint: delta_n >= 0
        neg_penalty = torch.mean(torch.relu(-pred) ** 2)
        loss = loss + self.alpha_nonneg * neg_penalty

        # Monotonicity constraint (if temperatures provided)
        if temperature is not None and len(pred) > 2:
            # Sort by temperature
            sort_idx = torch.argsort(temperature)
            sorted_pred = pred[sort_idx]
            sorted_temp = temperature[sort_idx]

            # delta_n should decrease as temperature increases
            # Penalize cases where delta_n increases with temperature
            diff_dn = sorted_pred[1:] - sorted_pred[:-1]
            diff_temp = sorted_temp[1:] - sorted_temp[:-1]

            # Only penalize where temp increases but delta_n also increases
            violations = torch.relu(diff_dn * torch.sign(diff_temp))
            loss = loss + self.alpha_mono * torch.mean(violations ** 2)

        return loss


def count_parameters(model):
    """Count trainable and total parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def build_model(
    backbone: str = "efficientnet_b0",
    pretrained: bool = True,
    freeze_backbone: bool = True,
) -> HybridCNN:
    """Factory function to build model."""
    model = HybridCNN(
        backbone_name=backbone,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
    )
    trainable, total = count_parameters(model)
    print(f"Model: {backbone}")
    print(f"  Total params: {total:,}")
    print(f"  Trainable params: {trainable:,}")
    return model


if __name__ == "__main__":
    # Quick test
    model = build_model("efficientnet_b0", pretrained=False, freeze_backbone=True)
    dummy_img = torch.randn(4, 3, 224, 224)
    dummy_scalars = torch.randn(4, 4)
    out = model(dummy_img, dummy_scalars)
    print(f"Output shape: {out.shape}, values: {out.detach()}")
