from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np


def read_images(image_dir: Path, count: int, seed: int, list_file: Path | None = None) -> list[np.ndarray]:
    if list_file and list_file.exists():
        paths = [Path(line.strip()) for line in list_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        paths = sorted(image_dir.glob("*.jpg"))
        rng = random.Random(seed)
        rng.shuffle(paths)
    images = []
    for path in paths[:count]:
        img = cv2.imread(str(path))
        if img is None:
            continue
        images.append(img)
    if len(images) < count:
        raise RuntimeError(f"Only loaded {len(images)} images from {image_dir}")
    return images


def resize_keep(img: np.ndarray, width: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = width / w
    return cv2.resize(img, (width, int(h * scale)))


def paste(canvas: np.ndarray, crop: np.ndarray, x: int, y: int) -> None:
    h, w = crop.shape[:2]
    ch, cw = canvas.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(cw, x + w), min(ch, y + h)
    if x1 >= x2 or y1 >= y2:
        return
    sx1, sy1 = x1 - x, y1 - y
    sx2, sy2 = sx1 + (x2 - x1), sy1 + (y2 - y1)
    canvas[y1:y2, x1:x2] = crop[sy1:sy2, sx1:sx2]


def run(args: argparse.Namespace) -> None:
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    images = read_images(Path(args.image_dir), args.objects, args.seed, Path(args.list_file) if args.list_file else None)
    crops = [resize_keep(img, args.object_width) for img in images]
    width, height = args.width, args.height
    frames = int(args.seconds * args.fps)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (width, height))
    starts = [(-360, 90), (width + 40, 230), (-420, 360), (width + 100, 470)]
    ends = [(width + 80, 120), (-380, 250), (width + 100, 340), (-340, 450)]
    for frame_idx in range(frames):
        t = frame_idx / max(frames - 1, 1)
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        canvas[:, :] = (42, 48, 52)
        cv2.rectangle(canvas, (0, height // 2 - 70), (width, height // 2 + 190), (72, 76, 78), -1)
        for lane_y in [height // 2 - 10, height // 2 + 80]:
            for x in range(0, width, 80):
                cv2.line(canvas, (x, lane_y), (x + 45, lane_y), (210, 210, 210), 3)
        order = list(range(len(crops)))
        if 0.42 < t < 0.60:
            order = [1, 0, 3, 2]
        for i in order:
            sx, sy = starts[i % len(starts)]
            ex, ey = ends[i % len(ends)]
            wave = 18 * np.sin(2 * np.pi * (t + i * 0.17))
            x = int(sx + (ex - sx) * t)
            y = int(sy + (ey - sy) * t + wave)
            paste(canvas, crops[i], x, y)
        cv2.putText(canvas, "demo traffic clip", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (235, 235, 235), 2, cv2.LINE_AA)
        writer.write(canvas)
    writer.release()
    print(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a 10-30 second synthetic traffic clip from Road Vehicle images.")
    parser.add_argument("--image-dir", default="data/road_vehicle/yolo/images/val")
    parser.add_argument("--list-file", default="")
    parser.add_argument("--output", default="data/videos/road_vehicle_demo.mp4")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--seconds", type=int, default=12)
    parser.add_argument("--objects", type=int, default=4)
    parser.add_argument("--object-width", type=int, default=330)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
