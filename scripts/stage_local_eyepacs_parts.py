"""Privately unwrap the five verified original EyePACS ZIP segments.

Kaggle downloads each original ``train.zip.00N`` segment inside a one-member
wrapper ZIP. This command verifies recorded wrapper hashes, checks the member
CRC while streaming, and writes raw segments for the existing segmented reader.
It never publishes images or overwrites an existing segment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import zipfile
import zlib
from pathlib import Path

from verify_eyepacs_download import EXPECTED_MEMBER_BYTES, sha256_file


def _record(output: Path, record: dict[str, object]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=".segment-proof-", suffix=".json",
                                                  dir=output.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2)
            stream.write("\n")
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def _crc32_file(path: Path) -> str:
    checksum = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            checksum = zlib.crc32(chunk, checksum)
    return f"{checksum:08x}"


def stage(wrapper_dir: Path, provenance_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for number, expected_size in EXPECTED_MEMBER_BYTES.items():
        name = f"train.zip.{number:03d}"
        wrapper = wrapper_dir / f"{name}.zip"
        proof = provenance_dir / f"eyepacs-train-part-{number:03d}.json"
        if not wrapper.is_file() or not proof.is_file():
            raise FileNotFoundError(f"Official verified wrapper/proof missing: {name}")
        original = json.loads(proof.read_text(encoding="utf-8"))
        if (original.get("member_name") != name or
                original.get("member_bytes") != expected_size or
                original.get("wrapper_path") != str(wrapper.resolve(strict=True)) or
                original.get("wrapper_bytes") != wrapper.stat().st_size or
                not original.get("full_member_crc_passed")):
            raise ValueError(f"Official wrapper proof does not match {name}")
        digest = sha256_file(wrapper)
        if digest != original["wrapper_sha256"]:
            raise ValueError(f"Official wrapper changed: {name}")
        destination = output_dir / name
        stage_proof = provenance_dir / f"eyepacs-raw-part-{number:03d}.json"
        if destination.exists() or stage_proof.exists():
            if not destination.is_file():
                raise ValueError(f"Provenance exists without its raw segment: {name}")
            if (destination.stat().st_size != expected_size or
                    _crc32_file(destination) != original["member_crc32"]):
                raise ValueError(f"Pre-existing raw segment fails byte count or CRC: {name}")
            raw_digest = sha256_file(destination)
            if stage_proof.exists():
                existing = json.loads(stage_proof.read_text(encoding="utf-8"))
                if (existing.get("raw_sha256") != raw_digest or
                        existing.get("wrapper_sha256") != digest):
                    raise ValueError(f"Pre-existing raw segment changed: {name}")
            else:
                _record(stage_proof, {
                    "segment": name, "raw_path": str(destination.resolve(strict=True)),
                    "raw_bytes": expected_size, "raw_sha256": raw_digest,
                    "wrapper_sha256": digest, "member_crc32": original["member_crc32"],
                })
            print(json.dumps({"segment": name, "status": "already_verified"}), flush=True)
            continue
        if shutil.disk_usage(output_dir).free < expected_size + 1_000_000_000:
            raise OSError(f"Insufficient private output space for {name}")
        with zipfile.ZipFile(wrapper) as archive:
            members = archive.infolist()
            if (len(members) != 1 or members[0].filename != name or
                    members[0].file_size != expected_size or
                    f"{members[0].CRC:08x}" != original["member_crc32"]):
                raise ValueError(f"Unexpected official wrapper structure: {name}")
            temporary = destination.with_name(destination.name + ".partial")
            if temporary.exists():
                raise FileExistsError(f"Unreviewed incomplete output exists: {temporary}")
            raw_hash = hashlib.sha256()
            count = 0
            try:
                with archive.open(members[0]) as incoming, temporary.open("xb") as outgoing:
                    for chunk in iter(lambda: incoming.read(8 * 1024 * 1024), b""):
                        raw_hash.update(chunk)
                        count += len(chunk)
                        outgoing.write(chunk)
                    outgoing.flush()
                    os.fsync(outgoing.fileno())
                if count != expected_size:
                    raise ValueError(f"Unwrapped segment has wrong size: {name}")
                os.link(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
        _record(stage_proof, {
            "segment": name, "raw_path": str(destination.resolve(strict=True)),
            "raw_bytes": count, "raw_sha256": raw_hash.hexdigest(),
            "wrapper_sha256": digest, "member_crc32": original["member_crc32"],
        })
        print(json.dumps({"segment": name, "status": "verified",
                          "raw_bytes": count, "raw_sha256": raw_hash.hexdigest()}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wrapper-dir", required=True, type=Path)
    parser.add_argument("--provenance-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    stage(args.wrapper_dir, args.provenance_dir, args.output_dir)
