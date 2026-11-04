"""Documented, full-field retinal preprocessing shared by all cohorts."""

from __future__ import annotations

import random

import numpy as np
import torch
from PIL import Image, ImageOps


class RetinaTransform:
    def __init__(self, image_size: int, training: bool = False, horizontal_flip: bool = False):
        if image_size <= 0:
            raise ValueError("image_size must be positive")
        self.image_size = image_size
        self.training = training
        self.horizontal_flip = horizontal_flip

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = ImageOps.exif_transpose(image).convert("RGB")
        # Keep the complete field of view. No lesion-bearing edge is cropped.
        width, height = image.size
        scale = self.image_size / max(width, height)
        target = (max(1, round(width * scale)), max(1, round(height * scale)))
        image = image.resize(target, Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (self.image_size, self.image_size), color=(0, 0, 0))
        canvas.paste(image, ((self.image_size - target[0]) // 2,
                             (self.image_size - target[1]) // 2))
        if self.training and self.horizontal_flip and random.random() < 0.5:
            canvas = ImageOps.mirror(canvas)
        array = np.asarray(canvas, dtype=np.float32).copy() / 255.0
        return torch.from_numpy(array).permute(2, 0, 1).contiguous()
