from __future__ import annotations

import argparse
from ultralytics import YOLO

from src.common.utils import resolve_path


def run(args: argparse.Namespace) -> None:
    model = YOLO(args.model)
    data_path = resolve_path(args.data)
    project_path = resolve_path(args.project)
    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(project_path),
        name=args.name,
        pretrained=True,
        patience=args.patience,
        cache=args.cache,
    )
    print(f"Saved YOLOv8 run to {results.save_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8 on Road Vehicle.")
    parser.add_argument("--data", default="data/road_vehicle/yolo/data.yaml")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", default="runs/task2_detection")
    parser.add_argument("--name", default="yolov8n_road_vehicle")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--cache", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
