from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.common.utils import device_name, ensure_dir, seed_everything
from src.task3_segmentation.dataset import CLASS_NAMES, StanfordBackground, download_stanford_background, make_split_files
from src.task3_segmentation.losses import segmentation_loss
from src.task3_segmentation.metrics import SegmentationMeter
from src.task3_segmentation.unet import UNet


def make_loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader]:
    data_dir = Path(args.data_dir)
    if args.download or not data_dir.exists():
        data_dir = download_stanford_background(args.data_root)
    split_dir = Path(args.split_dir)
    if not (split_dir / "train.txt").exists():
        make_split_files(data_dir, split_dir, val_ratio=args.val_ratio, seed=args.seed)
    image_size = tuple(args.image_size)
    train_ds = StanfordBackground(data_dir, split_dir / "train.txt", image_size=image_size, train=True)
    val_ds = StanfordBackground(data_dir, split_dir / "val.txt", image_size=image_size, train=False)
    if args.smoke_samples:
        train_ds = Subset(train_ds, range(min(args.smoke_samples, len(train_ds))))
        val_ds = Subset(val_ds, range(min(max(1, args.smoke_samples // 2), len(val_ds))))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    return train_loader, val_loader


def run_epoch(model: UNet, loader: DataLoader, optimizer: AdamW, device: torch.device, args: argparse.Namespace) -> tuple[float, dict[str, float]]:
    model.train()
    total_loss = 0.0
    total = 0
    meter = SegmentationMeter(num_classes=len(CLASS_NAMES))
    for images, masks in tqdm(loader, desc="train", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
            logits = model(images)
            loss = segmentation_loss(logits, masks, args.loss)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        total += images.size(0)
        meter.update(logits.detach(), masks.detach())
    scores = meter.scores()
    return total_loss / max(total, 1), scores


@torch.no_grad()
def evaluate(model: UNet, loader: DataLoader, device: torch.device, args: argparse.Namespace) -> tuple[float, dict[str, float]]:
    model.eval()
    total_loss = 0.0
    total = 0
    meter = SegmentationMeter(num_classes=len(CLASS_NAMES))
    for images, masks in tqdm(loader, desc="val", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        logits = model(images)
        loss = segmentation_loss(logits, masks, args.loss)
        total_loss += loss.item() * images.size(0)
        total += images.size(0)
        meter.update(logits, masks)
    return total_loss / max(total, 1), meter.scores()


def run(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    run_dir = ensure_dir(Path(args.output_dir) / f"unet_{args.loss}")
    device = device_name(args.device)
    train_loader, val_loader = make_loaders(args)
    model = UNet(num_classes=len(CLASS_NAMES), base_channels=args.base_channels).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    metrics_path = run_dir / "metrics.csv"
    best_path = run_dir / "best.pt"
    fields = ["epoch", "train_loss", "train_pixel_acc", "train_miou", "val_loss", "val_pixel_acc", "val_miou", "lr", "seconds"]
    best_miou = -1.0
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            start = time.time()
            train_loss, train_scores = run_epoch(model, train_loader, optimizer, device, args)
            val_loss, val_scores = evaluate(model, val_loader, device, args)
            scheduler.step()
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_pixel_acc": train_scores["pixel_acc"],
                "train_miou": train_scores["miou"],
                "val_loss": val_loss,
                "val_pixel_acc": val_scores["pixel_acc"],
                "val_miou": val_scores["miou"],
                "lr": optimizer.param_groups[0]["lr"],
                "seconds": time.time() - start,
            }
            writer.writerow(row)
            f.flush()
            print(row)
            if val_scores["miou"] > best_miou:
                best_miou = val_scores["miou"]
                torch.save({"model": model.state_dict(), "args": vars(args), "epoch": epoch, "val_miou": best_miou}, best_path)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train U-Net from scratch on Stanford Background.")
    parser.add_argument("--data-root", default="data/stanford_background")
    parser.add_argument("--data-dir", default="data/stanford_background/raw/iccv09Data")
    parser.add_argument("--split-dir", default="data/stanford_background/splits")
    parser.add_argument("--output-dir", default="runs/task3_segmentation")
    parser.add_argument("--loss", default="ce", choices=["ce", "dice", "ce_dice"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, nargs=2, default=[240, 320])
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--smoke-samples", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
