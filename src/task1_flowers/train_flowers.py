from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import Flowers102
from tqdm import tqdm

from src.common.utils import device_name, ensure_dir, seed_everything
from src.task1_flowers.models import ModelSpec, build_model, parameter_groups


def make_loaders(
    data_root: Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
    smoke_samples: int | None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.55, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.15, 0.15, 0.15, 0.05),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    train_ds = Flowers102(data_root, split="train", transform=train_tf, download=True)
    val_ds = Flowers102(data_root, split="val", transform=eval_tf, download=True)
    test_ds = Flowers102(data_root, split="test", transform=eval_tf, download=True)
    if smoke_samples:
        train_ds = Subset(train_ds, range(min(smoke_samples, len(train_ds))))
        val_ds = Subset(val_ds, range(min(max(1, smoke_samples // 2), len(val_ds))))
        test_ds = Subset(test_ds, range(min(max(1, smoke_samples // 2), len(test_ds))))
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
    )


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        total_loss += loss.item() * images.size(0)
        total_correct += (logits.argmax(1) == labels).sum().item()
        total += images.size(0)
    return total_loss / max(total, 1), total_correct / max(total, 1)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    amp: bool,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for images, labels in tqdm(loader, desc="train", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * images.size(0)
        total_correct += (logits.argmax(1) == labels).sum().item()
        total += images.size(0)
    return total_loss / max(total, 1), total_correct / max(total, 1)


def run(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    run_dir = ensure_dir(Path(args.output_dir) / args.run_name)
    device = device_name(args.device)
    train_loader, val_loader, test_loader = make_loaders(
        Path(args.data_root), args.image_size, args.batch_size, args.num_workers, args.smoke_samples
    )
    spec = ModelSpec(args.model, pretrained=not args.random_init, attention=args.attention)
    model = build_model(spec, num_classes=102).to(device)
    optimizer = AdamW(parameter_groups(model, args.backbone_lr, args.head_lr), weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    metrics_path = run_dir / "metrics.csv"
    best_path = run_dir / "best.pt"
    fields = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "seconds"]
    best_acc = -1.0
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            start = time.time()
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scaler, device, args.amp)
            val_loss, val_acc = evaluate(model, val_loader, device)
            scheduler.step()
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": optimizer.param_groups[-1]["lr"],
                "seconds": time.time() - start,
            }
            writer.writerow(row)
            f.flush()
            print(row)
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(
                    {
                        "model": model.state_dict(),
                        "args": vars(args),
                        "epoch": epoch,
                        "val_acc": val_acc,
                    },
                    best_path,
                )

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    test_loss, test_acc = evaluate(model, test_loader, device)
    (run_dir / "test_metrics.txt").write_text(
        f"test_loss: {test_loss:.6f}\ntest_acc: {test_acc:.6f}\n", encoding="utf-8"
    )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ResNet on Flowers102.")
    parser.add_argument("--data-root", default="data/flowers102")
    parser.add_argument("--output-dir", default="runs/task1_flowers")
    parser.add_argument("--run-name", default="resnet18_pretrained")
    parser.add_argument("--model", default="resnet18", choices=["resnet18", "resnet34"])
    parser.add_argument("--attention", default="none", choices=["none", "se"])
    parser.add_argument("--random-init", action="store_true")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--backbone-lr", type=float, default=1e-4)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--smoke-samples", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
