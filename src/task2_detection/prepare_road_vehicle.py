from __future__ import annotations

import argparse
import json
import re
import shutil
import tarfile
import urllib.request
from pathlib import Path

import yaml
from tqdm import tqdm


DATASET_PAGE = "https://datasetninja.com/road-vehicle"


def download_dataset(archive_path: Path) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if archive_path.exists() and archive_path.stat().st_size > 100_000_000:
        print(f"Archive already exists: {archive_path}")
        return
    with urllib.request.urlopen(DATASET_PAGE, timeout=30) as response:
        html = response.read().decode("utf-8")
    match = re.search(r"https://assets\.supervisely\.com/remote/[^\"'<>\\s]+", html)
    if not match:
        raise RuntimeError("Could not find Dataset Ninja download link.")
    url = match.group(0).replace("&amp;", "&")
    print(f"Downloading Road Vehicle archive to {archive_path}")
    urllib.request.urlretrieve(url, archive_path)


def extract_dataset(archive_path: Path, supervisely_dir: Path) -> None:
    if (supervisely_dir / "meta.json").exists():
        print(f"Supervisely data already exists: {supervisely_dir}")
        return
    supervisely_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r") as tar:
        tar.extractall(supervisely_dir)


def link_or_copy(src: Path, dst: Path, copy_images: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if copy_images:
        shutil.copy2(src, dst)
        return
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        shutil.copy2(src, dst)


def convert_split(
    supervisely_dir: Path,
    yolo_dir: Path,
    split: str,
    out_split: str,
    class_to_idx: dict[str, int],
    copy_images: bool,
) -> int:
    img_dir = supervisely_dir / split / "img"
    ann_dir = supervisely_dir / split / "ann"
    out_img_dir = yolo_dir / "images" / out_split
    out_label_dir = yolo_dir / "labels" / out_split
    out_label_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for ann_path in tqdm(sorted(ann_dir.glob("*.json")), desc=f"convert {split}"):
        ann = json.loads(ann_path.read_text(encoding="utf-8"))
        image_name = ann_path.name.removesuffix(".json")
        image_path = img_dir / image_name
        if not image_path.exists():
            raise FileNotFoundError(image_path)
        h = float(ann["size"]["height"])
        w = float(ann["size"]["width"])
        rows = []
        for obj in ann.get("objects", []):
            title = obj["classTitle"]
            if title not in class_to_idx:
                continue
            (x1, y1), (x2, y2) = obj["points"]["exterior"]
            x1, x2 = sorted((max(0.0, float(x1)), min(w, float(x2))))
            y1, y2 = sorted((max(0.0, float(y1)), min(h, float(y2))))
            bw = x2 - x1
            bh = y2 - y1
            if bw <= 1 or bh <= 1:
                continue
            xc = (x1 + x2) / 2 / w
            yc = (y1 + y2) / 2 / h
            rows.append(f"{class_to_idx[title]} {xc:.6f} {yc:.6f} {bw / w:.6f} {bh / h:.6f}")
        (out_label_dir / f"{Path(image_name).stem}.txt").write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
        link_or_copy(image_path, out_img_dir / image_name, copy_images)
        count += 1
    return count


def convert(supervisely_dir: Path, yolo_dir: Path, copy_images: bool) -> None:
    meta = json.loads((supervisely_dir / "meta.json").read_text(encoding="utf-8"))
    names = [c["title"] for c in meta["classes"]]
    class_to_idx = {name: idx for idx, name in enumerate(names)}
    train_count = convert_split(supervisely_dir, yolo_dir, "train", "train", class_to_idx, copy_images)
    val_count = convert_split(supervisely_dir, yolo_dir, "valid", "val", class_to_idx, copy_images)
    data_yaml = {
        "path": str(yolo_dir.as_posix()),
        "train": "images/train",
        "val": "images/val",
        "nc": len(names),
        "names": names,
    }
    (yolo_dir / "data.yaml").write_text(yaml.safe_dump(data_yaml, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"Converted {train_count} train images and {val_count} val images.")
    print(f"YOLO data file: {yolo_dir / 'data.yaml'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and convert Road Vehicle Dataset Ninja archive to YOLOv8 format.")
    parser.add_argument("--archive", default="data/road_vehicle/road-vehicle-DatasetNinja.tar")
    parser.add_argument("--supervisely-dir", default="data/road_vehicle/supervisely")
    parser.add_argument("--yolo-dir", default="data/road_vehicle/yolo")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--copy-images", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    archive = Path(args.archive)
    supervisely_dir = Path(args.supervisely_dir)
    yolo_dir = Path(args.yolo_dir)
    if args.download:
        download_dataset(archive)
    if not supervisely_dir.exists():
        extract_dataset(archive, supervisely_dir)
    convert(supervisely_dir, yolo_dir, args.copy_images)
