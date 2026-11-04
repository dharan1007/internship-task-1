"""Strict adapters for the licensed source archives, without image redistribution."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from .manifest import FIVE_GRADE, MESSIDOR_GRADE, inspect_image, sha256_file, write_manifest

IMAGE_SUFFIXES = {".jpeg", ".jpg", ".png", ".tif", ".tiff"}
MESSIDOR_CORRECTIONS = {
    "20051020_63045_0100_PP.tif": 0,
    "20051020_64007_0100_PP.tif": 3,
    "20051020_63936_0100_PP.tif": 1,
    "20060523_48477_0100_PP.tif": 3,
}
# Published ADCIS Base33 erratum. Retain the first name and record the second as excluded.
MESSIDOR_DUPLICATES = {
    "20051202_54744_0400_PP.tif", "20051202_40508_0400_PP.tif",
    "20051202_41238_0400_PP.tif", "20051202_41260_0400_PP.tif",
    "20051202_54530_0400_PP.tif", "20051205_33025_0400_PP.tif",
    "20051202_55607_0400_PP.tif", "20051202_41034_0400_PP.tif",
    "20051205_35099_0400_PP.tif", "20051202_54555_0400_PP.tif",
    "20051205_35110_0400_PP.tif", "20051202_54611_0400_PP.tif",
    "20051202_55498_0400_PP.tif",
}


def _image_index(root: Path) -> dict[str, Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Image directory missing: {root}")
    images: dict[str, Path] = {}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            key = path.name.lower()
            if key in images:
                raise ValueError(f"Duplicate filename in image archive: {path.name}")
            images[key] = path.resolve(strict=True)
    if not images:
        raise ValueError(f"No supported retinal images found under {root}")
    return images


def _row(source: str, image_id: str, path: Path, subject_id: str, eye: str,
         grade: int | str, scheme: str, included: bool = True,
         reason: str = "") -> dict[str, object]:
    digest, width, height, fmt = inspect_image(path)
    return {"source": source, "image_id": image_id, "image_path": str(path),
            "subject_id": subject_id, "eye": eye, "grade": grade, "label_scheme": scheme,
            "split": "test" if source != "eyepacs" else "", "included": included,
            "exclusion_reason": reason, "sha256": digest, "width": width,
            "height": height, "format": fmt}


def _integer_grade(value: object, allowed: set[int], context: str) -> int:
    if pd.isna(value) or isinstance(value, bool):
        raise ValueError(f"Missing/noninteger grade for {context}")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Noninteger grade for {context}: {value!r}") from exc
    if str(value).strip() not in {str(number), f"{number}.0"} or number not in allowed:
        raise ValueError(f"Grade outside allowed values for {context}: {value!r}")
    return number


def prepare_eyepacs(images: Path, labels: Path, output: Path,
                    exclusions: Path | None = None) -> None:
    frame = pd.read_csv(labels)
    if list(frame.columns) != ["image", "level"]:
        raise ValueError("EyePACS labels must have exactly image,level columns")
    if frame["image"].duplicated().any() or frame.empty:
        raise ValueError("EyePACS labels are empty or repeat image IDs")
    excluded: dict[str, str] = {}
    if exclusions is not None:
        exclusion_frame = pd.read_csv(exclusions, dtype=str, keep_default_na=False)
        if list(exclusion_frame.columns) != ["image_id", "reason"]:
            raise ValueError("Exclusion ledger must have image_id,reason columns")
        if exclusion_frame["image_id"].duplicated().any():
            raise ValueError("Exclusion ledger repeats image IDs")
        if not exclusion_frame["reason"].str.fullmatch(r"[a-z][a-z0-9_]{2,63}").all():
            raise ValueError("Exclusion reasons must be explicit lowercase reason codes")
        excluded = dict(zip(exclusion_frame["image_id"], exclusion_frame["reason"]))
        if set(excluded) - set(frame["image"].astype(str)):
            raise ValueError("Exclusion ledger has unknown EyePACS IDs")
    index = _image_index(images)
    stems = {Path(name).stem: path for name, path in index.items()}
    if len(stems) != len(index):
        raise ValueError("EyePACS image stems are ambiguous across extensions")
    rows = []
    for record in frame.itertuples(index=False):
        image_id = str(record.image)
        match = re.fullmatch(r"(.+)_(left|right)", image_id, re.IGNORECASE)
        if not match:
            raise ValueError(f"EyePACS image ID does not identify subject/eye: {image_id}")
        path = stems.get(image_id.lower())
        if path is None:
            raise ValueError(f"EyePACS label has no image file: {image_id}")
        grade = _integer_grade(record.level, set(range(5)), image_id)
        if image_id in excluded:
            rows.append({"source": "eyepacs", "image_id": image_id, "image_path": str(path),
                         "subject_id": match.group(1), "eye": match.group(2).lower(),
                         "grade": grade, "label_scheme": FIVE_GRADE, "split": "", "included": False,
                         "exclusion_reason": excluded[image_id], "sha256": sha256_file(path),
                         "width": "", "height": "", "format": "excluded_uninspected"})
        else:
            rows.append(_row("eyepacs", image_id, path, match.group(1),
                             match.group(2).lower(), grade, FIVE_GRADE))
    if len(rows) != len(index):
        raise ValueError(f"EyePACS image/label reconciliation failed: {len(index)} files, {len(rows)} labels")
    write_manifest(pd.DataFrame(rows), output)


def prepare_messidor(images: Path, output: Path) -> None:
    index = _image_index(images)
    sheets = sorted(images.rglob("*.xls")) + sorted(images.rglob("*.xlsx"))
    if not sheets:
        raise FileNotFoundError("Official Messidor annotation workbooks are missing")
    if len(index) != 1200 or len(sheets) != 12:
        raise ValueError(f"Original Messidor needs 1200 images/12 workbooks; "
                         f"found {len(index)} images/{len(sheets)} workbooks")
    labels: dict[str, int] = {}
    for sheet in sheets:
        frame = pd.read_excel(sheet)
        required = {"Image name", "Retinopathy grade"}
        if not required.issubset(frame.columns):
            raise ValueError(f"Messidor workbook lacks official columns: {sheet}")
        for image_name, grade_value in frame[["Image name", "Retinopathy grade"]].itertuples(index=False, name=None):
            name = str(image_name).strip()
            if name.lower() in labels:
                raise ValueError(f"Messidor annotation repeats image: {name}")
            labels[name.lower()] = _integer_grade(grade_value, set(range(4)), name)
    if {name.lower() for name in MESSIDOR_CORRECTIONS} - set(labels):
        raise ValueError("Official Messidor correction targets missing from annotations")
    if {name.lower() for name in MESSIDOR_DUPLICATES} - set(labels):
        raise ValueError("Official Messidor duplicate targets missing from annotations")
    if set(labels) != set(index):
        raise ValueError(f"Messidor annotation/image mismatch: missing images={len(set(labels)-set(index))}, "
                         f"unlabeled images={len(set(index)-set(labels))}")
    corrections = {name.lower(): grade for name, grade in MESSIDOR_CORRECTIONS.items()}
    duplicates = {name.lower() for name in MESSIDOR_DUPLICATES}
    rows = []
    for name in sorted(labels):
        corrected = corrections.get(name, labels[name])
        excluded = name in duplicates
        rows.append(_row("messidor", name, index[name], "", "unknown",
                         corrected, MESSIDOR_GRADE, not excluded,
                         "published_duplicate_base33" if excluded else ""))
    write_manifest(pd.DataFrame(rows), output)


def prepare_messidor2(images: Path, labels: Path, output: Path, pairs: Path) -> None:
    index = _image_index(images)
    if len(index) != 1748:
        raise ValueError(f"Official Messidor-2 needs 1748 images; found {len(index)}")
    frame = pd.read_csv(labels)
    required = {"image_id", "adjudicated_dr_grade", "adjudicated_gradable"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Messidor-2 adjudicated labels lack columns {sorted(required - set(frame))}")
    if frame["image_id"].duplicated().any() or frame.empty:
        raise ValueError("Messidor-2 label IDs repeat or labels are empty")
    if len(frame) != 1748:
        raise ValueError(f"Expected 1748 Messidor-2 adjudication rows; found {len(frame)}")
    # The ADCIS pairing sheet is a separate official artifact; require a normalized
    # image_id,subject_id CSV derived from it so the pairing interpretation is auditable.
    pair_frame = pd.read_csv(pairs, dtype=str)
    if not {"image_id", "subject_id"}.issubset(pair_frame.columns):
        raise ValueError("Pair mapping needs image_id,subject_id columns")
    if pair_frame["image_id"].duplicated().any() or pair_frame["subject_id"].isna().any():
        raise ValueError("Pair mapping repeats image IDs or has missing subject IDs")
    subjects = dict(zip(pair_frame["image_id"].str.lower(), pair_frame["subject_id"]))
    if set(subjects) != set(index):
        raise ValueError("Pair mapping must cover every Messidor-2 image")
    if len(set(subjects.values())) != 874:
        raise ValueError("Messidor-2 pairing must identify 874 examinations")
    rows = []
    for record in frame.to_dict("records"):
        name = str(record["image_id"]).strip().lower()
        if name not in index:
            raise ValueError(f"Adjudicated label has no official Messidor-2 image: {name}")
        gradable = str(record["adjudicated_gradable"]).strip().lower()
        included = gradable in {"1", "1.0", "true"}
        if gradable not in {"0", "0.0", "false", "1", "1.0", "true"}:
            raise ValueError(f"Unknown gradability value for {name}: {gradable}")
        grade = (_integer_grade(record["adjudicated_dr_grade"], set(range(5)), name)
                 if included else "")
        rows.append(_row("messidor2", name, index[name], subjects[name], "unknown",
                         grade, FIVE_GRADE, included, "ungradable" if not included else ""))
    if set(index) != {str(name).strip().lower() for name in frame["image_id"]}:
        raise ValueError("Messidor-2 images/labels do not reconcile exactly")
    write_manifest(pd.DataFrame(rows), output)
