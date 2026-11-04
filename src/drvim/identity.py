"""Hash the executable model and image-input source used by a run."""

from pathlib import Path

from .data.manifest import sha256_file


def pipeline_source_hashes() -> dict[str, str]:
    root = Path(__file__).parent
    names = ("model/scan.py", "model/vim.py", "data/preprocess.py", "data/dataset.py",
             "data/manifest.py", "data/prepare.py", "data/split.py", "data/audit.py",
             "metrics.py", "train.py", "qualify_scan.py", "evaluate.py")
    return {name: sha256_file(root / name) for name in names}
