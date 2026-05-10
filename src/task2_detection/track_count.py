from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

def parse_line(line: str, width: int, height: int) -> tuple[tuple[int, int], tuple[int, int]]:
    if line:
        vals = [float(v) for v in line.split(",")]
        if len(vals) != 4:
            raise ValueError("--line must be x1,y1,x2,y2")
        x1, y1, x2, y2 = vals
        return (int(x1), int(y1)), (int(x2), int(y2))
    return (width // 2, 0), (width // 2, height)


def side_of_line(point: tuple[float, float], a: tuple[int, int], b: tuple[int, int]) -> float:
    return (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0])


def draw_label(frame: np.ndarray, text: str, xy: tuple[int, int], color: tuple[int, int, int]) -> None:
    x, y = xy
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thick = 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    cv2.rectangle(frame, (x, y - th - 8), (x + tw + 6, y + 3), color, -1)
    cv2.putText(frame, text, (x + 3, y - 4), font, scale, (255, 255, 255), thick, cv2.LINE_AA)


def run(args: argparse.Namespace) -> None:
    model = YOLO(args.weights)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {args.video}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    line_a, line_b = parse_line(args.line, width, height)
    cap.release()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_out = output_dir / "tracked_counted.mp4"
    csv_out = output_dir / "tracking_log.csv"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_out), fourcc, fps, (width, height))

    last_side: dict[int, float] = {}
    counted_ids: set[int] = set()
    trajectories: dict[int, list[tuple[int, int]]] = defaultdict(list)

    with csv_out.open("w", newline="", encoding="utf-8") as f:
        log = csv.DictWriter(
            f,
            fieldnames=["frame", "track_id", "class_id", "class_name", "confidence", "x1", "y1", "x2", "y2", "cx", "cy", "counted"],
        )
        log.writeheader()
        for frame_idx, result in enumerate(
            model.track(
                source=args.video,
                tracker=args.tracker,
                stream=True,
                persist=True,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                device=args.device,
                verbose=False,
            ),
            start=1,
        ):
            frame = result.orig_img.copy()
            cv2.line(frame, line_a, line_b, (0, 255, 255), 2)
            boxes = result.boxes
            if boxes is not None and boxes.id is not None:
                xyxy = boxes.xyxy.cpu().numpy()
                ids = boxes.id.cpu().numpy().astype(int)
                cls = boxes.cls.cpu().numpy().astype(int)
                confs = boxes.conf.cpu().numpy()
                for box, track_id, class_id, conf in zip(xyxy, ids, cls, confs):
                    x1, y1, x2, y2 = box.tolist()
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    side = side_of_line((cx, cy), line_a, line_b)
                    counted = False
                    if track_id in last_side and track_id not in counted_ids and last_side[track_id] * side < 0:
                        counted_ids.add(track_id)
                        counted = True
                    if abs(side) > 1e-6:
                        last_side[track_id] = side
                    trajectories[track_id].append((int(cx), int(cy)))
                    trajectories[track_id] = trajectories[track_id][-30:]
                    color = (
                        int((37 * track_id) % 255),
                        int((17 * track_id + 90) % 255),
                        int((29 * track_id + 150) % 255),
                    )
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    class_name = result.names.get(class_id, str(class_id))
                    draw_label(frame, f"ID {track_id} {class_name} {conf:.2f}", (int(x1), max(20, int(y1))), color)
                    for p1, p2 in zip(trajectories[track_id][:-1], trajectories[track_id][1:]):
                        cv2.line(frame, p1, p2, color, 2)
                    log.writerow(
                        {
                            "frame": frame_idx,
                            "track_id": track_id,
                            "class_id": class_id,
                            "class_name": class_name,
                            "confidence": f"{conf:.4f}",
                            "x1": f"{x1:.1f}",
                            "y1": f"{y1:.1f}",
                            "x2": f"{x2:.1f}",
                            "y2": f"{y2:.1f}",
                            "cx": f"{cx:.1f}",
                            "cy": f"{cy:.1f}",
                            "counted": int(counted),
                        }
                    )
            draw_label(frame, f"Crossing count: {len(counted_ids)}", (18, 34), (40, 40, 40))
            writer.write(frame)
            if args.save_frames and frame_idx in args.save_frames:
                cv2.imwrite(str(output_dir / f"frame_{frame_idx:05d}.jpg"), frame)
    writer.release()
    print(f"Saved {video_out}")
    print(f"Saved {csv_out}")
    print(f"Crossing count: {len(counted_ids)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 tracking with stable IDs and virtual-line counting.")
    parser.add_argument("--weights", default="runs/task2_detection/yolov8n_road_vehicle/weights/best.pt")
    parser.add_argument("--video", required=True)
    parser.add_argument("--output-dir", default="runs/task2_detection/tracking")
    parser.add_argument("--tracker", default="botsort.yaml")
    parser.add_argument("--line", default="", help="x1,y1,x2,y2. Default is a vertical center line.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--device", default="0")
    parser.add_argument("--save-frames", type=int, nargs="*", default=[])
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
