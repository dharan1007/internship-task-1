"""EyePACS filename-pair-grouped, grade-stratified immutable split assignment."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .manifest import FIVE_GRADE, assert_no_exact_cross_split_duplicates, validate_manifest


def split_eyepacs(frame: pd.DataFrame, seed: int, validation_fraction: float = 0.15,
                  test_fraction: float = 0.15) -> pd.DataFrame:
    validate_manifest(frame)
    if not 0 < validation_fraction < 1 or not 0 < test_fraction < 1:
        raise ValueError("Split fractions must be in (0, 1)")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("Validation and test fractions leave no training data")
    if not (frame["source"] == "eyepacs").all():
        raise ValueError("Only EyePACS can be divided into development splits")
    if not (frame["label_scheme"] == FIVE_GRADE).all():
        raise ValueError("EyePACS manifest must use five-grade labels")
    if frame["split"].astype(str).str.strip().ne("").any():
        raise ValueError("Refusing to replace an existing split")
    rng = np.random.default_rng(seed)
    active = frame.loc[frame["included"].astype(bool)]
    grouped = active.groupby("subject_id", sort=True)["grade"].max().astype(int)
    assignments: dict[str, str] = {}
    for grade in range(5):
        subjects = grouped.index[grouped == grade].to_numpy(dtype=str)
        if len(subjects) < 3:
            raise ValueError(f"Grade {grade} has fewer than three subject groups")
        rng.shuffle(subjects)
        test_n = max(1, round(len(subjects) * test_fraction))
        val_n = max(1, round(len(subjects) * validation_fraction))
        if test_n + val_n >= len(subjects):
            raise ValueError(f"Grade {grade} has too few groups for all splits")
        for subject in subjects[:test_n]:
            assignments[subject] = "test"
        for subject in subjects[test_n:test_n + val_n]:
            assignments[subject] = "validation"
        for subject in subjects[test_n + val_n:]:
            assignments[subject] = "train"
    result = frame.copy()
    result["split"] = result["subject_id"].astype(str).map(assignments).fillna("excluded")
    if result.loc[result["included"].astype(bool), "split"].eq("excluded").any():
        raise ValueError("Some included patient groups were not assigned")
    if (result.groupby("subject_id")["split"].nunique() != 1).any():
        raise ValueError("A patient crossed splits")
    for name in ("train", "validation", "test"):
        observed = set(result.loc[(result["split"] == name) &
                                  result["included"].astype(bool), "grade"].astype(int))
        if observed != set(range(5)):
            raise ValueError(f"{name} does not contain every grade: {sorted(observed)}")
    assert_no_exact_cross_split_duplicates(result)
    return result
