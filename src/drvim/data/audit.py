"""Read-only reconciliation report for the immutable research cohorts."""

from __future__ import annotations

import json
import os
from pathlib import Path

from .manifest import (
    assert_no_exact_cross_split_duplicates,
    read_manifest,
    sha256_file,
    verify_file_hashes,
)


def audit(run_dir: Path, eyepacs: Path, messidor: Path, messidor2: Path | None,
          output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"Audit already exists: {output}")
    marker = run_dir / "training-complete.json"
    if not marker.is_file():
        raise RuntimeError("External cohort audit requires a completed training run")
    completed = json.loads(marker.read_text(encoding="utf-8"))
    if completed["manifest_sha256"] != sha256_file(eyepacs):
        raise ValueError("EyePACS audit manifest differs from trained manifest")
    paths = {"eyepacs": eyepacs, "messidor": messidor}
    if messidor2 is not None:
        paths["messidor2"] = messidor2
    frames = {source: read_manifest(path) for source, path in paths.items()}
    for source, frame in frames.items():
        if not (frame["source"] == source).all():
            raise ValueError(f"Wrong source in {source} manifest")
        verify_file_hashes(frame)
    eye = frames["eyepacs"]
    if set(eye.loc[eye["included"], "split"]) != {"train", "validation", "test"}:
        raise ValueError("EyePACS split assignment is incomplete")
    if eye.groupby("subject_id")["split"].nunique().gt(1).any():
        raise ValueError("EyePACS subject crossed splits")
    assert_no_exact_cross_split_duplicates(eye)
    hash_sets = {source: set(frame.loc[frame["included"], "sha256"])
                 for source, frame in frames.items()}
    external_hashes = set().union(*(hash_sets[source] for source in paths if source != "eyepacs"))
    if hash_sets["eyepacs"] & external_hashes:
        raise ValueError("Exact image overlap between development and external cohorts")
    cohorts: dict[str, object] = {}
    for source, frame in frames.items():
        active = frame.loc[frame["included"]]
        cohorts[source] = {
            "manifest_sha256": sha256_file(paths[source]),
            "rows": len(frame), "included": len(active), "excluded": len(frame) - len(active),
            "exclusion_reasons": {str(k): int(v) for k, v in
                                  frame.loc[~frame["included"], "exclusion_reason"].value_counts().items()},
            "grades": {str(k): int(v) for k, v in active["grade"].value_counts().sort_index().items()},
            "splits": {str(split): {"images": len(part),
                                    "subjects": (None if source == "messidor" else
                                                 int(part["subject_id"].nunique())),
                                    "grades": {str(k): int(v) for k, v in
                                               part["grade"].value_counts().sort_index().items()}}
                       for split, part in active.groupby("split")},
            "dimensions": {"min_width": int(active["width"].min()),
                           "max_width": int(active["width"].max()),
                           "min_height": int(active["height"].min()),
                           "max_height": int(active["height"].max())},
            "formats": {str(k): int(v) for k, v in active["format"].value_counts().items()},
        }
    overlap_names = None
    overlap_hashes = None
    if messidor2 is not None:
        original_stems = {Path(name).stem.lower() for name in frames["messidor"]["image_id"]}
        second_stems = {Path(name).stem.lower() for name in frames["messidor2"]["image_id"]}
        overlap_names = len(original_stems & second_stems)
        overlap_hashes = len(hash_sets["messidor"] & hash_sets["messidor2"])
    report = {"cohorts": cohorts,
              "eyepacs_external_exact_hash_overlap": 0,
              "original_messidor_messidor2_filename_overlap": overlap_names,
              "original_messidor_messidor2_exact_hash_overlap": overlap_hashes,
              "near_duplicate_review": "pending human/feature-level review on official images"}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".partial")
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    os.replace(temporary, output)
