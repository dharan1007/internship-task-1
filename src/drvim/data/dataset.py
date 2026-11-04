"""Custom map-style PyTorch Dataset over a validated retinal manifest."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from .manifest import FIVE_GRADE, read_manifest
from .preprocess import RetinaTransform


class RetinaDataset(Dataset):
    def __init__(
        self,
        manifest: Path,
        *,
        source: str,
        split: str,
        image_size: int,
        training: bool = False,
        horizontal_flip: bool = False,
    ):
        frame = read_manifest(manifest)
        selected = frame.loc[(frame["source"] == source) &
                             (frame["split"] == split) &
                             (frame["included"].astype(bool))].copy()
        if selected.empty:
            raise ValueError(f"No included {source}/{split} images in {manifest}")
        if source in {"eyepacs", "messidor2"} and not (selected["label_scheme"] == FIVE_GRADE).all():
            raise ValueError(f"{source} requires five-grade ICDR labels")
        if training and split != "train":
            raise ValueError("Training transforms may only be applied to the train split")
        self.records = selected.to_dict("records")
        self.transform = RetinaTransform(image_size, training, horizontal_flip)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int | tuple[int, float]) -> dict[str, object]:
        sample_weight = 1.0
        if isinstance(index, tuple):
            index, sample_weight = index
            if sample_weight not in (0.0, 1.0):
                raise ValueError("Distributed sample weight must be zero or one")
        record = self.records[index]
        path = Path(str(record["image_path"]))
        try:
            with Image.open(path) as image:
                image.load()
                pixels = self.transform(image)
        except Exception as exc:
            raise RuntimeError(f"Cannot decode/transform {record['source']}:{record['image_id']} at {path}") from exc
        return {
            "image": pixels,
            "grade": torch.tensor(int(record["grade"]), dtype=torch.long),
            "image_id": str(record["image_id"]),
            "subject_id": str(record["subject_id"]),
            "sample_weight": sample_weight,
        }
