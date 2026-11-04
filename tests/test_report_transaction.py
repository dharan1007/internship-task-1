"""An incomplete experiment must not leave a publishable report folder."""

from pathlib import Path

import pytest

from drvim.report import generate_report


def test_report_failure_cleans_staging(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    destination = tmp_path / "report"
    missing = tmp_path / "missing.csv"
    with pytest.raises(RuntimeError, match="genuine sealed test results"):
        generate_report(run, destination, missing, missing, missing)
    assert not destination.exists()
    assert {path.name for path in tmp_path.iterdir()} == {"run"}
