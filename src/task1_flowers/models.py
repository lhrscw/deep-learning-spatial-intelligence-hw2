from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torchvision import models


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        weights = self.avg_pool(x).view(b, c)
        weights = self.fc(weights).view(b, c, 1, 1)
        return x * weights


class SEResNet(nn.Module):
    def __init__(self, backbone: nn.Module, num_classes: int) -> None:
        super().__init__()
        self.backbone = backbone
        channels = backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.attention = SEBlock(channels)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.attention(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


@dataclass
class ModelSpec:
    name: str
    pretrained: bool
    attention: str = "none"


def build_model(spec: ModelSpec, num_classes: int) -> nn.Module:
    name = spec.name.lower()
    if name == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if spec.pretrained else None
        backbone = models.resnet18(weights=weights)
    elif name == "resnet34":
        weights = models.ResNet34_Weights.IMAGENET1K_V1 if spec.pretrained else None
        backbone = models.resnet34(weights=weights)
    else:
        raise ValueError(f"Unsupported model: {spec.name}")

    if spec.attention.lower() == "se":
        return SEResNet(backbone, num_classes)
    if spec.attention.lower() in {"none", "baseline"}:
        backbone.fc = nn.Linear(backbone.fc.in_features, num_classes)
        return backbone
    raise ValueError(f"Unsupported attention module: {spec.attention}")


def parameter_groups(model: nn.Module, base_lr: float, head_lr: float) -> list[dict]:
    head_keywords = ("fc", "classifier")
    head_params = []
    backbone_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(k in name for k in head_keywords):
            head_params.append(param)
        else:
            backbone_params.append(param)
    return [
        {"params": backbone_params, "lr": base_lr},
        {"params": head_params, "lr": head_lr},
    ]
