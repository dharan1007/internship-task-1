"""Verify one original EyePACS Kaggle training part and record private provenance.

Kaggle serves each ``train.zip.00N`` part inside a one-member ZIP wrapper. This
checks the complete wrapper and member CRC without extracting or publishing data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

# Sizes returned by `kaggle competitions files diabetic-retinopathy-detection`
# on 2026-11-03. They refer to the inner split-archive members, not wrappers.
EXPECTED_MEMBER_BYTES = {
    1: 8_388_608_000,
    2: 8_388_608_000,
    3: 8_388_608_000,
    4: 8_388_608_000,
    5: 1_434_013_506,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_part(archive: Path, part: int, output: Path) -> dict[str, object]:
    if part not in EXPECTED_MEMBER_BYTES:
        raise ValueError("Expected EyePACS training part 1 through 5")
    member_name = f"train.zip.{part:03d}"
    if archive.name != member_name + ".zip" or not archive.is_file():
        raise ValueError(f"Expected an existing official {member_name}.zip wrapper")
    if output.exists():
        raise FileExistsError(f"Provenance record already exists: {output}")
    with zipfile.ZipFile(archive) as wrapper:
        members = wrapper.infolist()
        if len(members) != 1 or members[0].filename != member_name or members[0].is_dir():
            raise ValueError(f"Archive must contain exactly {member_name}")
        if members[0].file_size != EXPECTED_MEMBER_BYTES[part]:
            raise ValueError(f"Unexpected {member_name} byte count: {members[0].file_size}")
        bad_member = wrapper.testzip()
        if bad_member is not None:
            raise zipfile.BadZipFile(f"Failed CRC for {bad_member}")
        member_crc32 = f"{members[0].CRC:08x}"
    record = {
        "source": f"https://www.kaggle.com/competitions/diabetic-retinopathy-detection/data?select={member_name}",
        "download_method": "official Kaggle CLI competition file download",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "wrapper_path": str(archive.resolve(strict=True)),
        "wrapper_bytes": archive.stat().st_size,
        "wrapper_sha256": sha256_file(archive),
        "member_name": member_name,
        "member_bytes": members[0].file_size,
        "member_crc32": member_crc32,
        "full_member_crc_passed": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".eyepacs-part-", suffix=".json", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2)
            stream.write("\n")
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--part", required=True, type=int)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    record = verify_part(args.archive, args.part, args.out)
    print(
        json.dumps(
            {
                "part": record["member_name"],
                "bytes": record["member_bytes"],
                "sha256": record["wrapper_sha256"],
                "crc_passed": record["full_member_crc_passed"],
            }
        )
    )


if __name__ == "__main__":
    main()
