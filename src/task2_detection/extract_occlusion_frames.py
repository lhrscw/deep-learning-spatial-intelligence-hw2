from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(args.video)
    wanted = set(range(args.start, args.start + args.count))
    idx = 0
    saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if idx in wanted:
            path = out_dir / f"occlusion_{idx:05d}.jpg"
            cv2.imwrite(str(path), frame)
            print(path)
            saved += 1
    cap.release()
    if saved != len(wanted):
        print(f"Saved {saved}/{len(wanted)} requested frames.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract 3-4 consecutive frames for occlusion/ID-switch analysis.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--output-dir", default="reports/figures/task2_occlusion")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
