from __future__ import annotations

import torch

from src.task3_segmentation.dataset import IGNORE_INDEX


class SegmentationMeter:
    def __init__(self, num_classes: int, ignore_index: int = IGNORE_INDEX) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)

    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        pred = logits.argmax(dim=1).detach().cpu()
        target = target.detach().cpu()
        valid = target != self.ignore_index
        pred = pred[valid]
        target = target[valid]
        idx = target * self.num_classes + pred
        hist = torch.bincount(idx, minlength=self.num_classes**2).reshape(self.num_classes, self.num_classes)
        self.confusion += hist

    def scores(self) -> dict[str, float]:
        hist = self.confusion.float()
        diag = torch.diag(hist)
        union = hist.sum(1) + hist.sum(0) - diag
        iou = diag / torch.clamp(union, min=1)
        acc = diag.sum() / torch.clamp(hist.sum(), min=1)
        return {
            "pixel_acc": float(acc.item()),
            "miou": float(iou.mean().item()),
        }
