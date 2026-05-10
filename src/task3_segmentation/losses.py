from __future__ import annotations

import torch
import torch.nn.functional as F

from src.task3_segmentation.dataset import IGNORE_INDEX


def dice_loss(logits: torch.Tensor, target: torch.Tensor, ignore_index: int = IGNORE_INDEX, eps: float = 1e-6) -> torch.Tensor:
    num_classes = logits.shape[1]
    valid = target != ignore_index
    safe_target = target.clone()
    safe_target[~valid] = 0
    probs = torch.softmax(logits, dim=1)
    one_hot = F.one_hot(safe_target, num_classes=num_classes).permute(0, 3, 1, 2).float()
    valid = valid.unsqueeze(1).float()
    probs = probs * valid
    one_hot = one_hot * valid
    dims = (0, 2, 3)
    intersection = torch.sum(probs * one_hot, dims)
    cardinality = torch.sum(probs + one_hot, dims)
    dice = (2 * intersection + eps) / (cardinality + eps)
    return 1 - dice.mean()


def segmentation_loss(logits: torch.Tensor, target: torch.Tensor, mode: str) -> torch.Tensor:
    ce = F.cross_entropy(logits, target, ignore_index=IGNORE_INDEX)
    dl = dice_loss(logits, target)
    if mode == "ce":
        return ce
    if mode == "dice":
        return dl
    if mode == "ce_dice":
        return ce + dl
    raise ValueError(f"Unknown loss mode: {mode}")
