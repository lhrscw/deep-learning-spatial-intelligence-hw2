from __future__ import annotations

import random
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


STANFORD_URL = "http://dags.stanford.edu/data/iccv09Data.tar.gz"
CLASS_NAMES = ["sky", "tree", "road", "grass", "water", "building", "mountain", "foreground"]
IGNORE_INDEX = 255


def download_stanford_background(root: str | Path = "data/stanford_background") -> Path:
    root = Path(root)
    archive = root / "iccv09Data.tar.gz"
    raw_dir = root / "raw"
    data_dir = raw_dir / "iccv09Data"
    root.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        print(f"Downloading Stanford Background to {archive}")
        urllib.request.urlretrieve(STANFORD_URL, archive)
    if not data_dir.exists():
        raw_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(raw_dir)
    return data_dir


def make_split_files(data_dir: str | Path, split_dir: str | Path, val_ratio: float = 0.2, seed: int = 42) -> None:
    data_dir = Path(data_dir)
    split_dir = Path(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)
    ids = sorted(p.stem for p in (data_dir / "images").glob("*.jpg"))
    rng = random.Random(seed)
    rng.shuffle(ids)
    val_count = int(round(len(ids) * val_ratio))
    val_ids = sorted(ids[:val_count])
    train_ids = sorted(ids[val_count:])
    (split_dir / "train.txt").write_text("\n".join(train_ids) + "\n", encoding="utf-8")
    (split_dir / "val.txt").write_text("\n".join(val_ids) + "\n", encoding="utf-8")


class StanfordBackground(Dataset):
    def __init__(
        self,
        data_dir: str | Path,
        split_file: str | Path,
        image_size: tuple[int, int] = (240, 320),
        train: bool = False,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.ids = [line.strip() for line in Path(split_file).read_text(encoding="utf-8").splitlines() if line.strip()]
        self.image_size = image_size
        self.train = train

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_id = self.ids[index]
        image = Image.open(self.data_dir / "images" / f"{image_id}.jpg").convert("RGB")
        mask = np.loadtxt(self.data_dir / "labels" / f"{image_id}.regions.txt", dtype=np.int64)
        mask[mask < 0] = IGNORE_INDEX
        image_arr = np.asarray(image).copy()
        if self.train and random.random() < 0.5:
            image_arr = image_arr[:, ::-1, :]
            mask = mask[:, ::-1].copy()
        image_arr = image_arr.copy()
        image_t = torch.from_numpy(image_arr).permute(2, 0, 1).float() / 255.0
        mask_t = torch.from_numpy(mask).long()
        if image_t.shape[1:] != self.image_size:
            image_t = F.interpolate(image_t.unsqueeze(0), size=self.image_size, mode="bilinear", align_corners=False).squeeze(0)
            resized = F.interpolate(mask_t.float().unsqueeze(0).unsqueeze(0), size=self.image_size, mode="nearest")
            mask_t = resized.squeeze(0).squeeze(0).long()
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image_t = (image_t - mean) / std
        return image_t, mask_t
