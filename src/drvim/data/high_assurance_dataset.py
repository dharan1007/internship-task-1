"""Auditable retinal image loading for research and stricter candidate workflows.

This module is deliberately separate from the assignment's frozen training path.
It does not establish medical-device clearance, clinical validity, data rights, or
truth of external metadata. See docs/HIGH_ASSURANCE_DATA_PIPELINE.md.
"""

from __future__ import annotations

import hashlib
import io
import random
import warnings
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageCms, ImageOps
from torch.utils.data import DataLoader, Dataset, Sampler

from .manifest import (
    FIVE_GRADE,
    assert_no_exact_cross_split_duplicates,
    sha256_file,
    validate_manifest,
)

EvidenceTier = Literal["research", "clinical_candidate"]
CLINICAL_FIELDS = (
    "patient_group_token", "site_id", "device_id", "grading_reference",
    "gradability_status", "rights_reference", "color_space_attestation_ref",
)


@dataclass(frozen=True)
class RetinalDataPolicy:
    source: Literal["eyepacs", "messidor", "messidor2"]
    split: Literal["train", "validation", "test"]
    image_size: int
    source_root: Path
    recorded_path_prefix: str
    evidence_tier: EvidenceTier = "research"
    expected_manifest_sha256: str | None = None
    max_manifest_bytes: int = 128 * 1024 * 1024
    max_compressed_bytes: int = 128 * 1024 * 1024
    max_pixels: int = 100_000_000
    seed: int = 0
    horizontal_flip_probability: float = 0.0
    augmentation_protocol_ref: str | None = None

    def __post_init__(self) -> None:
        if (self.image_size <= 0 or self.max_manifest_bytes <= 0 or
                self.max_compressed_bytes <= 0 or self.max_pixels <= 0):
            raise ValueError("Image size and input bounds must be positive")
        if self.seed < 0:
            raise ValueError("Seed must be nonnegative")
        if self.source not in ("eyepacs", "messidor", "messidor2"):
            raise ValueError("Unknown retinal source")
        if self.split not in ("train", "validation", "test"):
            raise ValueError("Unknown split")
        if not 0 <= self.horizontal_flip_probability <= 1:
            raise ValueError("Flip probability must be in [0, 1]")
        if self.split != "train" and self.horizontal_flip_probability:
            raise ValueError("Validation and test images cannot be augmented")
        if self.evidence_tier not in ("research", "clinical_candidate"):
            raise ValueError("Unknown evidence tier")
        if self.evidence_tier == "clinical_candidate":
            if not self.expected_manifest_sha256:
                raise ValueError("Candidate workflow needs a pinned manifest SHA-256")
            if self.horizontal_flip_probability and not self.augmentation_protocol_ref:
                raise ValueError("Candidate augmentation needs a protocol reference")
        if self.expected_manifest_sha256 and not _is_sha256(self.expected_manifest_sha256):
            raise ValueError("Expected manifest SHA-256 is malformed")
        if (not self.recorded_path_prefix or
                not PurePosixPath(self.recorded_path_prefix).is_absolute() or
                ".." in PurePosixPath(self.recorded_path_prefix).parts):
            raise ValueError("Recorded path prefix must be an absolute POSIX directory")


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _relocate_path(recorded: str, prefix: PurePosixPath, root: Path) -> Path:
    original = PurePosixPath(recorded)
    try:
        relative = original.relative_to(prefix)
    except ValueError as exc:
        raise ValueError(f"Manifest path lies outside recorded source prefix: {recorded}") from exc
    if not relative.parts or any(part in (".", "..") for part in relative.parts):
        raise ValueError(f"Unsafe or empty relative image path: {recorded}")
    destination = root.joinpath(*relative.parts).resolve()
    if not destination.is_relative_to(root):
        raise ValueError(f"Image path escapes the approved source root: {recorded}")
    return destination


def _read_relocated_manifest(path: Path, policy: RetinalDataPolicy) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size > policy.max_manifest_bytes:
        raise ValueError("Manifest is missing or exceeds its byte bound")
    if policy.expected_manifest_sha256 and sha256_file(path) != policy.expected_manifest_sha256:
        raise ValueError("Manifest SHA-256 differs from the pinned release")
    frame = pd.read_csv(path, keep_default_na=False, dtype=str)
    if "image_path" not in frame or "included" not in frame:
        raise ValueError("Manifest lacks required path or inclusion columns")
    frame["included"] = frame["included"].map({"True": True, "False": False})
    root = policy.source_root.resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)
    prefix = PurePosixPath(policy.recorded_path_prefix)
    frame["image_path"] = frame["image_path"].map(
        lambda value: str(_relocate_path(value, prefix, root))
    )
    validate_manifest(frame)
    return frame


def _check_cohort(frame: pd.DataFrame, policy: RetinalDataPolicy) -> None:
    source = frame.loc[frame["source"] == policy.source]
    if source.empty or not source["included"].any():
        raise ValueError(f"No included images for {policy.source}")
    active = source.loc[source["included"]]
    if not active["split"].isin(["train", "validation", "test"]).all():
        raise ValueError("Included images need explicit train, validation or test splits")
    assert_no_exact_cross_split_duplicates(source)
    if active.groupby("subject_id")["split"].nunique().gt(1).any():
        raise ValueError("Manifest grouping token crosses splits")
    if policy.source in ("eyepacs", "messidor2") and not active["label_scheme"].eq(FIVE_GRADE).all():
        raise ValueError("Five-grade source has a different label scheme")
    if policy.evidence_tier == "clinical_candidate":
        missing = set(CLINICAL_FIELDS) - set(source.columns)
        if missing:
            raise ValueError(f"Candidate metadata columns missing: {sorted(missing)}")
        for key in CLINICAL_FIELDS:
            if active[key].astype(str).str.strip().eq("").any():
                raise ValueError(f"Candidate metadata is incomplete: {key}")
        if not active["gradability_status"].eq("gradable").all():
            raise ValueError("Ungradable images cannot be included as graded images")
        if active.groupby("patient_group_token")["split"].nunique().gt(1).any():
            raise ValueError("Verified patient group token crosses splits")


def _flip_for_sample(seed: int, epoch: int, image_id: str, probability: float) -> bool:
    if not probability:
        return False
    key = f"{seed}\0{epoch}\0{image_id}".encode()
    draw = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big") / 2**64
    return draw < probability


def _decode_to_srgb(data: bytes, policy: RetinalDataPolicy,
                    record: dict[str, object]) -> tuple[Image.Image, str, tuple[int, int]]:
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(io.BytesIO(data)) as opened:
            if opened.width * opened.height > policy.max_pixels:
                raise ValueError("Declared image dimensions exceed the pixel bound")
            dimensions = opened.size
            if dimensions != (int(record["width"]), int(record["height"])):
                raise ValueError("Image dimensions differ from the manifest")
            if (opened.format or "").upper() != str(record["format"]).upper():
                raise ValueError("Image format differs from the manifest")
            if opened.format not in {"JPEG", "TIFF"}:
                raise ValueError("Only audited JPEG or TIFF fundus images are supported")
            if getattr(opened, "n_frames", 1) != 1:
                raise ValueError("Multi-frame images need a separate validated adapter")
            opened.load()
            oriented = ImageOps.exif_transpose(opened)
            profile_bytes = oriented.info.get("icc_profile") or opened.info.get("icc_profile")
            if profile_bytes:
                try:
                    profile = ImageCms.ImageCmsProfile(io.BytesIO(profile_bytes))
                    srgb = ImageCms.createProfile("sRGB")
                    image = ImageCms.profileToProfile(
                        oriented, profile, srgb, renderingIntent=0, outputMode="RGB"
                    )
                except Exception as exc:
                    raise ValueError("Embedded ICC profile cannot be converted to sRGB") from exc
                color_evidence = "embedded_icc_converted_to_srgb"
            else:
                if (policy.evidence_tier == "clinical_candidate" and
                        not str(record["color_space_attestation_ref"]).strip()):
                    raise ValueError("Untagged image lacks a source color-space attestation")
                image = oriented.convert("RGB")
                color_evidence = "source_srgb_attested" if policy.evidence_tier == "clinical_candidate" \
                    else "untagged_assumed_srgb_research"
            return image, color_evidence, dimensions


def _letterbox(image: Image.Image, image_size: int, flip: bool) -> tuple[torch.Tensor, float]:
    scale = image_size / max(image.size)
    resized = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    image = image.resize(resized, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (image_size, image_size), color=(0, 0, 0))
    canvas.paste(image, ((image_size - resized[0]) // 2, (image_size - resized[1]) // 2))
    if flip:
        canvas = ImageOps.mirror(canvas)
    array = np.asarray(canvas, dtype=np.float32).copy() / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    return tensor, (resized[0] * resized[1]) / (image_size * image_size)


class HighAssuranceRetinaDataset(Dataset):
    """Verify source bytes and transform provenance at every image read.

    ``subject_id`` in the public EyePACS manifest means a filename-derived eye
    pairing token. It must not be represented as an independently verified
    patient identity. Candidate mode requires a separately sourced token.
    """

    def __init__(self, manifest: Path, policy: RetinalDataPolicy):
        self.policy = policy
        self.root = policy.source_root.resolve(strict=True)
        self.manifest_sha256 = sha256_file(manifest)
        frame = _read_relocated_manifest(manifest, policy)
        _check_cohort(frame, policy)
        selected = frame.loc[(frame["source"] == policy.source) &
                             (frame["split"] == policy.split) & frame["included"]].copy()
        if selected.empty:
            raise ValueError(f"No included {policy.source}/{policy.split} images")
        self.records = selected.to_dict("records")
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Set before creating each DataLoader iterator; workers do not persist."""
        if epoch < 0:
            raise ValueError("Epoch must be nonnegative")
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int | tuple[int, float]) -> dict[str, object]:
        sample_weight = 1.0
        if isinstance(index, tuple):
            index, sample_weight = index
            if sample_weight not in (0.0, 1.0):
                raise ValueError("Distributed contribution weight must be zero or one")
        record = self.records[index]
        path = Path(str(record["image_path"]))
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(self.root):
            raise ValueError(f"Source image escaped the approved root: {record['image_id']}")
        size = resolved.stat().st_size
        if size <= 0 or size > self.policy.max_compressed_bytes:
            raise ValueError(f"Source image violates byte bound: {record['image_id']}")
        with resolved.open("rb") as stream:
            data = stream.read(self.policy.max_compressed_bytes + 1)
        if len(data) != size or len(data) > self.policy.max_compressed_bytes:
            raise ValueError(f"Source image changed while reading: {record['image_id']}")
        if hashlib.sha256(data).hexdigest() != record["sha256"]:
            raise ValueError(f"Source image hash differs from manifest: {record['image_id']}")
        image, color_evidence, dimensions = _decode_to_srgb(data, self.policy, record)
        mirrored = _flip_for_sample(
            self.policy.seed, self.epoch, str(record["image_id"]),
            self.policy.horizontal_flip_probability,
        )
        pixels, image_area_fraction = _letterbox(image, self.policy.image_size, mirrored)
        return {
            "image": pixels,
            "grade": torch.tensor(int(record["grade"]), dtype=torch.long),
            "image_id": str(record["image_id"]),
            "group_token": (str(record["patient_group_token"])
                            if self.policy.evidence_tier == "clinical_candidate"
                            else str(record["subject_id"])),
            "group_token_basis": ("external_patient_token"
                                  if self.policy.evidence_tier == "clinical_candidate"
                                  else "filename_eye_pair"),
            "source_sha256": str(record["sha256"]),
            "color_evidence": color_evidence,
            "original_width": dimensions[0],
            "original_height": dimensions[1],
            "image_area_fraction": image_area_fraction,
            "mirrored": mirrored,
            "sample_weight": sample_weight,
        }


class EpochPermutationSampler(Sampler[int]):
    """Exact-once training order reproducible from seed and epoch."""

    def __init__(self, dataset: HighAssuranceRetinaDataset):
        if dataset.policy.split != "train":
            raise ValueError("Epoch permutation is training-only")
        self.dataset = dataset

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.dataset.policy.seed + self.dataset.epoch)
        return iter(torch.randperm(len(self.dataset), generator=generator).tolist())

    def __len__(self) -> int:
        return len(self.dataset)


def _seed_worker(worker_id: int) -> None:
    del worker_id
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def build_high_assurance_loader(
    dataset: HighAssuranceRetinaDataset,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool = False,
    sampler: Sampler | None = None,
) -> DataLoader:
    """Use PyTorch DataLoader with ordered, reproducible worker delivery."""
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("Batch size must be positive and worker count nonnegative")
    if sampler is not None and dataset.policy.split != "train":
        raise ValueError("Validation and test must use exact-once sequential traversal")
    generator = torch.Generator().manual_seed(dataset.policy.seed)
    selected_sampler = (sampler if sampler is not None else
                        EpochPermutationSampler(dataset)
                        if dataset.policy.split == "train" else None)
    options = {
        "batch_size": batch_size,
        "shuffle": False,
        "sampler": selected_sampler,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "generator": generator,
        "worker_init_fn": _seed_worker,
        "drop_last": False,
        "in_order": True,
    }
    if num_workers:
        options.update(prefetch_factor=2, persistent_workers=False)
    return DataLoader(dataset, **options)
