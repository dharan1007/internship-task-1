"""Stage the official byte-split EyePACS ZIP on Kaggle scratch and audit it.

The competition mount exposes the five unwrapped pieces of one ZIP file. This
reader seeks across them without first writing another 35 GB archive. Images
are extracted only to private, ephemeral scratch; manifests are saved in the
notebook's output directory. Every member is read to its CRC boundary and the
normal manifest builder then hashes and fully decodes every image.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import uuid
import zipfile
import zlib
from bisect import bisect_right
from pathlib import Path, PurePosixPath

from drvim.data.manifest import read_manifest, write_manifest
from drvim.data.prepare import prepare_eyepacs
from drvim.data.split import split_eyepacs

PART_SIZES = (8_388_608_000,) * 4 + (1_434_013_506,)
IMAGE_COUNT = 35_126


class SegmentedReader:
    """Seekable, read-only file object over the official five byte segments."""

    def __init__(self, paths: list[Path]):
        self._handles = [path.open("rb") for path in paths]
        self._sizes = [path.stat().st_size for path in paths]
        self._starts = [0]
        for size in self._sizes:
            self._starts.append(self._starts[-1] + size)
        self._position = 0

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._position

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        if whence == os.SEEK_SET:
            position = offset
        elif whence == os.SEEK_CUR:
            position = self._position + offset
        elif whence == os.SEEK_END:
            position = self._starts[-1] + offset
        else:
            raise ValueError("Invalid seek origin")
        if position < 0 or position > self._starts[-1]:
            raise ValueError("Seek outside the official EyePACS archive")
        self._position = position
        return position

    def read(self, size: int = -1) -> bytes:
        remaining = self._starts[-1] - self._position
        size = remaining if size < 0 else min(size, remaining)
        chunks: list[bytes] = []
        while size:
            index = bisect_right(self._starts, self._position) - 1
            position = self._position - self._starts[index]
            amount = min(size, self._sizes[index] - position)
            handle = self._handles[index]
            handle.seek(position)
            chunk = handle.read(amount)
            if len(chunk) != amount:
                raise OSError("Short read from an official EyePACS archive part")
            chunks.append(chunk)
            self._position += amount
            size -= amount
        return b"".join(chunks)

    def close(self) -> None:
        for handle in self._handles:
            handle.close()

    def __enter__(self):
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def _safe_image_name(name: str) -> Path:
    if "\\" in name:
        raise ValueError(f"ZIP member has a Windows path separator: {name}")
    pure = PurePosixPath(name)
    if (pure.is_absolute() or not pure.parts or
            any(part in {".", ".."} for part in pure.parts) or
            pure.suffix.lower() not in {".jpeg", ".jpg"}):
        raise ValueError(f"Unexpected EyePACS image member: {name}")
    return Path(*pure.parts)


def _matches_existing(path: Path, member: zipfile.ZipInfo) -> bool:
    if not path.is_file() or path.stat().st_size != member.file_size:
        return False
    checksum = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            checksum = zlib.crc32(block, checksum)
    return checksum == member.CRC


def _stage_images(archive_dir: Path, scratch: Path) -> dict[str, object]:
    parts = [archive_dir / f"train.zip.{number:03d}" for number in range(1, 6)]
    for path, expected in zip(parts, PART_SIZES):
        if not path.is_file() or path.stat().st_size != expected:
            raise ValueError(f"Missing or wrong-size official archive segment: {path}")
    images = scratch / "images"
    marker = scratch / "extraction-complete.json"
    if marker.exists():
        recorded = json.loads(marker.read_text(encoding="utf-8"))
        if recorded.get("part_sizes") != list(PART_SIZES):
            raise ValueError("Scratch extraction marker belongs to another source release")
    images.mkdir(parents=True, exist_ok=True)
    with SegmentedReader(parts) as segmented, zipfile.ZipFile(segmented) as source:
        members = [entry for entry in source.infolist() if not entry.is_dir()]
        if len(members) != IMAGE_COUNT:
            raise ValueError(f"Expected {IMAGE_COUNT} images, found {len(members)} members")
        relative = [_safe_image_name(entry.filename) for entry in members]
        if len({str(path).lower() for path in relative}) != IMAGE_COUNT:
            raise ValueError("EyePACS ZIP repeats an image path")
        total = sum(entry.file_size for entry in members)
        if shutil.disk_usage(scratch).free < total + 5_000_000_000:
            raise OSError("Scratch space is insufficient for the complete image archive")
        for number, (entry, name) in enumerate(zip(members, relative), start=1):
            destination = images / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                if not _matches_existing(destination, entry):
                    raise ValueError(f"Existing scratch image disagrees with ZIP: {destination}")
                continue
            temporary = destination.with_name(destination.name + f".{uuid.uuid4().hex}.partial")
            try:
                with source.open(entry) as incoming, temporary.open("xb") as outgoing:
                    shutil.copyfileobj(incoming, outgoing, length=8 * 1024 * 1024)
                if temporary.stat().st_size != entry.file_size:
                    raise ValueError(f"Extracted image has wrong byte count: {entry.filename}")
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
            if number % 1000 == 0:
                print(json.dumps({"extracted_images": number, "expected": IMAGE_COUNT}), flush=True)
    record = {"source": "official Kaggle diabetic-retinopathy-detection competition input",
              "part_sizes": list(PART_SIZES), "image_count": IMAGE_COUNT,
              "uncompressed_image_bytes": total}
    temporary_marker = marker.with_name(marker.name + ".partial")
    temporary_marker.write_text(json.dumps(record, indent=2), encoding="utf-8")
    os.replace(temporary_marker, marker)
    return record


def prepare(archive_dir: Path, scratch: Path, output: Path, seed: int) -> None:
    scratch.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    extraction = _stage_images(archive_dir, scratch)
    labels_archive = archive_dir / "trainLabels.csv.zip"
    labels = scratch / "trainLabels.csv"
    with zipfile.ZipFile(labels_archive) as wrapper:
        entries = wrapper.infolist()
        if len(entries) != 1 or entries[0].filename != "trainLabels.csv":
            raise ValueError("Unexpected official EyePACS label ZIP structure")
        if labels.exists():
            if not _matches_existing(labels, entries[0]):
                raise ValueError("Existing scratch labels disagree with official ZIP")
        else:
            with wrapper.open(entries[0]) as incoming, labels.open("xb") as outgoing:
                shutil.copyfileobj(incoming, outgoing)
    manifest = output / "eyepacs.csv"
    split = output / "eyepacs-split.csv"
    prepare_eyepacs(scratch / "images", labels, manifest)
    write_manifest(split_eyepacs(read_manifest(manifest), seed), split)
    print(json.dumps({"extraction": extraction, "manifest": str(manifest),
                      "split_manifest": str(split)}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    arguments = parser.parse_args()
    prepare(arguments.archive_dir, arguments.scratch, arguments.out, arguments.seed)
