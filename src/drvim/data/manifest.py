"""Manifest schema and fail-closed checks for all retinal cohorts."""

from __future__ import annotations

import hashlib
import os
import tempfile
import warnings
from pathlib import Path

import pandas as pd
from PIL import Image

FIELDS = (
    "source", "image_id", "image_path", "subject_id", "eye", "grade",
    "label_scheme", "split", "included", "exclusion_reason", "sha256",
    "width", "height", "format",
)
FIVE_GRADE = "icdr_0_4"
MESSIDOR_GRADE = "messidor_0_3"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_image(path: Path) -> tuple[str, int, int, str]:
    """Hash and fully decode an actual source image before it enters a manifest."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing source image: {path}")
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(path) as image:
            image.load()
            width, height = image.size
            fmt = image.format or "unknown"
            if width <= 0 or height <= 0:
                raise ValueError(f"Invalid image dimensions: {path}")
            if len(image.getbands()) < 3:
                raise ValueError(f"Expected a colour fundus image: {path}")
    return sha256_file(path), width, height, fmt


def write_manifest(frame: pd.DataFrame, output: Path) -> None:
    validate_manifest(frame)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Manifest already exists; refusing overwrite: {output}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=".manifest-", suffix=".csv",
                                                   dir=output.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            frame.loc[:, list(FIELDS)].to_csv(stream, index=False)
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def read_manifest(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Manifest missing: {path}")
    frame = pd.read_csv(path, keep_default_na=False, dtype={"image_id": str, "subject_id": str})
    frame["included"] = frame["included"].map({"True": True, "False": False,
                                                 True: True, False: False})
    validate_manifest(frame)
    return frame


def validate_manifest(frame: pd.DataFrame) -> None:
    missing = set(FIELDS) - set(frame.columns)
    if missing:
        raise ValueError(f"Manifest lacks columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Manifest has no images")
    for key in ("source", "image_id", "image_path", "label_scheme", "sha256"):
        if frame[key].isna().any() or (frame[key].astype(str).str.strip() == "").any():
            raise ValueError(f"Manifest contains empty {key}")
    known_subjects = frame.loc[frame["source"].isin(["eyepacs", "messidor2"])]
    if known_subjects["subject_id"].isna().any() or known_subjects["subject_id"].astype(str).str.strip().eq("").any():
        raise ValueError("EyePACS and Messidor-2 need nonblank grouping tokens")
    if frame.duplicated(["source", "image_id"]).any():
        raise ValueError("Manifest repeats a source/image_id pair")
    if not frame["source"].isin(["eyepacs", "messidor", "messidor2"]).all():
        raise ValueError("Unknown retinal source")
    if frame["included"].isna().any() or not frame["included"].isin([True, False]).all():
        raise ValueError("included must be boolean")
    active = frame.loc[frame["included"].astype(bool)]
    if active.empty:
        raise ValueError("Manifest contains no included images")
    if active["grade"].isna().any() or (active["grade"].astype(str).str.strip() == "").any():
        raise ValueError("Included image lacks a grade")
    for scheme, allowed in ((FIVE_GRADE, set(range(5))), (MESSIDOR_GRADE, set(range(4)))):
        subset = active.loc[active["label_scheme"] == scheme]
        if not subset.empty:
            literals = subset["grade"].astype(str).str.strip()
            if not literals.str.fullmatch(r"[0-9]").all():
                raise ValueError(f"Noncanonical {scheme} grade")
            grades = literals.astype(int)
            if not grades.isin(allowed).all():
                raise ValueError(f"Grade outside {scheme} range")
    if not active["label_scheme"].isin([FIVE_GRADE, MESSIDOR_GRADE]).all():
        raise ValueError("Unknown grading scheme")
    expected_schemes = {"eyepacs": FIVE_GRADE, "messidor": MESSIDOR_GRADE,
                        "messidor2": FIVE_GRADE}
    if not all(active["label_scheme"].eq(active["source"].map(expected_schemes))):
        raise ValueError("Source grading scheme mismatch")
    if not frame["sha256"].astype(str).str.fullmatch(r"[0-9a-f]{64}").all():
        raise ValueError("Invalid SHA-256 digest")
    if not frame["image_path"].map(lambda value: Path(str(value)).is_absolute()).all():
        raise ValueError("Image paths must be absolute")
    if active.duplicated(["source", "sha256"]).any():
        raise ValueError("Included cohort contains exact duplicate images")
    for dimension in ("width", "height"):
        numeric = pd.to_numeric(active[dimension], errors="coerce")
        if numeric.isna().any() or numeric.le(0).any():
            raise ValueError(f"Included image has invalid {dimension}")


def assert_no_exact_cross_split_duplicates(frame: pd.DataFrame) -> None:
    active = frame.loc[frame["included"].astype(bool)]
    spans = active.groupby("sha256")["split"].nunique()
    if (spans > 1).any():
        raise ValueError(f"Exact image hashes cross splits: {(spans > 1).sum()}")


def verify_file_hashes(frame: pd.DataFrame) -> None:
    """Verify actual source bytes still match the immutable manifest."""
    for record in frame.itertuples(index=False):
        path = Path(str(record.image_path))
        if not path.is_file():
            raise FileNotFoundError(f"Manifest source image disappeared: {record.image_id}: {path}")
        if sha256_file(path) != record.sha256:
            raise ValueError(f"Manifest source image changed: {record.image_id}: {path}")
