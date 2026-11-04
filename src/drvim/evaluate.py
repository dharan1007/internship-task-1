"""Sealed EyePACS and external-cohort evaluation using the selected checkpoint."""

from __future__ import annotations

import json
import os
from contextlib import nullcontext
from pathlib import Path

import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data.dataset import RetinaDataset
from .data.manifest import read_manifest, sha256_file, verify_file_hashes
from .identity import pipeline_source_hashes
from .metrics import five_class_metrics, referable_metrics
from .model.vim import VimConfig, VisionMamba


@torch.inference_mode()
def _predictions(model: VisionMamba, manifest: Path, source: str, image_size: int,
                 batch_size: int, workers: int, device: torch.device,
                 amp_dtype: torch.dtype | None) -> pd.DataFrame:
    dataset = RetinaDataset(manifest, source=source, split="test", image_size=image_size)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=workers,
                        pin_memory=device.type == "cuda", persistent_workers=workers > 0)
    rows = []
    for batch in loader:
        labels = batch["grade"]
        context = (torch.autocast(device_type="cuda", dtype=amp_dtype)
                   if device.type == "cuda" and amp_dtype else nullcontext())
        with context:
            logits = model(batch["image"].to(device, non_blocking=True)).float()
        if not torch.isfinite(logits).all():
            raise FloatingPointError(f"Nonfinite test logits in {source}")
        probabilities = F.softmax(logits, dim=-1).cpu()
        for index, image_id in enumerate(batch["image_id"]):
            record = {"source": source, "image_id": image_id,
                      "subject_id": batch["subject_id"][index],
                      "true_grade": int(labels[index]),
                      "predicted_grade": int(probabilities[index].argmax())}
            record.update({f"probability_{grade}": float(probabilities[index, grade])
                           for grade in range(5)})
            rows.append(record)
    return pd.DataFrame(rows)


def _plot_confusion(matrix: list[list[int]], title: str, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(6, 5))
    axis.imshow(matrix, cmap="Blues")
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            axis.text(j, i, str(value), ha="center", va="center", color="black")
    axis.set(title=title, xlabel="Predicted grade", ylabel="Reference grade")
    axis.set_xticks(range(len(matrix)))
    axis.set_yticks(range(len(matrix)))
    figure.tight_layout()
    figure.savefig(output, dpi=200)
    plt.close(figure)


def _plot_history(history: list[dict[str, object]], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot([row["epoch"] for row in history], [row["train_loss"] for row in history],
              label="EyePACS training")
    axis.plot([row["epoch"] for row in history],
              [row["validation"]["loss"] for row in history], label="EyePACS validation")
    axis.set(xlabel="Epoch", ylabel="Cross-entropy loss", title="Training history")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=200)
    plt.close(figure)


def evaluate(run_dir: Path, eyepacs: Path, messidor: Path, messidor2: Path | None,
             audit_path: Path,
             *, batch_size: int | None = None, workers: int | None = None) -> None:
    complete = run_dir / "training-complete.json"
    if not complete.is_file():
        raise RuntimeError("Sealed test evaluation requires a completed full training run")
    if (run_dir / "results.json").exists():
        raise FileExistsError("Test results already exist; refusing to rerun or overwrite")
    expected = json.loads(complete.read_text(encoding="utf-8"))
    if expected["manifest_sha256"] != sha256_file(eyepacs):
        raise ValueError("EyePACS evaluation manifest differs from the training manifest")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    paths = {"eyepacs": eyepacs, "messidor": messidor}
    if messidor2 is not None:
        paths["messidor2"] = messidor2
    if set(audit["cohorts"]) != set(paths):
        raise ValueError("Audit cohorts differ from requested evaluation cohorts")
    for source, path in paths.items():
        if audit["cohorts"][source]["manifest_sha256"] != sha256_file(path):
            raise ValueError(f"Evaluation {source} manifest differs from completed audit")
    frames = {source: read_manifest(path) for source, path in paths.items()}
    for source, frame in frames.items():
        if not (frame["source"] == source).all():
            raise ValueError(f"Wrong source in {source} manifest")
        verify_file_hashes(frame.loc[frame["split"] == "test"])
    eye_hashes = set(frames["eyepacs"].loc[frames["eyepacs"]["included"], "sha256"])
    for external in (source for source in paths if source != "eyepacs"):
        external_hashes = set(frames[external].loc[frames[external]["included"], "sha256"])
        if eye_hashes & external_hashes:
            raise ValueError(f"Exact EyePACS/{external} image leakage detected")
    overlap_by_name = None
    if messidor2 is not None:
        original_names = {Path(name).stem.lower() for name in frames["messidor"]["image_id"]}
        second_names = {Path(name).stem.lower() for name in frames["messidor2"]["image_id"]}
        overlap_by_name = len(original_names & second_names)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(run_dir / "checkpoint-best.pt", map_location="cpu", weights_only=True)
    if (checkpoint["manifest_sha256"] != expected["manifest_sha256"] or
            checkpoint["config_sha256"] != expected["config_sha256"] or
            checkpoint["qualification_sha256"] != expected["qualification_sha256"]):
        raise ValueError("Selected checkpoint uses a different manifest or configuration")
    if checkpoint["pipeline_source_sha256"] != pipeline_source_hashes():
        raise ValueError("Model or preprocessing source changed since training")
    if checkpoint["epoch"] + 1 != expected["best_epoch"] or checkpoint["best_qwk"] != expected["best_qwk"]:
        raise ValueError("Selected checkpoint disagrees with completed validation selection")
    config = VimConfig(**checkpoint["model_config"])
    data_config = checkpoint["experiment_config"]["data"]
    batch_size = int(data_config["batch_size"] if batch_size is None else batch_size)
    workers = int(data_config["workers"] if workers is None else workers)
    if batch_size <= 0 or workers < 0:
        raise ValueError("Evaluation batch size must be positive and workers nonnegative")
    amp_name = checkpoint["experiment_config"]["train"]["amp"]
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "none": None}[amp_name]
    if device.type == "cuda" and amp_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("Completed run requires BF16 evaluation but this GPU does not support it")
    model = VisionMamba(config)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    predictions = {source: _predictions(model, path, source, config.image_size,
                                        batch_size, workers, device, amp_dtype)
                   for source, path in paths.items()}
    results = {"checkpoint_epoch": checkpoint["epoch"] + 1,
               "validation_qwk_at_selection": checkpoint["best_qwk"],
               "experiment_config": checkpoint["experiment_config"],
               "runtime": expected["runtime"],
               "training_elapsed_seconds": expected["elapsed_seconds"],
               "eyepacs_manifest_sha256": expected["manifest_sha256"],
               "messidor_manifest_sha256": sha256_file(messidor),
               "messidor2_manifest_sha256": (sha256_file(messidor2)
                                              if messidor2 is not None else None),
               "audit_sha256": sha256_file(audit_path),
               "scan_qualification_sha256": expected["qualification_sha256"],
               "original_messidor_messidor2_filename_overlap": overlap_by_name,
               "cohorts": {}}
    for source, frame in predictions.items():
        true = frame["true_grade"].astype(int).tolist()
        predicted = frame["predicted_grade"].astype(int).tolist()
        metrics = {"referable": referable_metrics(true, predicted,
                                                     original_messidor=source == "messidor")}
        if source != "messidor":
            metrics["five_class"] = five_class_metrics(true, predicted)
        manifest_frame = frames[source]
        metrics["source_counts"] = {
            "total_labeled_rows": len(manifest_frame),
            "included": int(manifest_frame["included"].sum()),
            "excluded": int((~manifest_frame["included"]).sum()),
            "exclusion_reasons": {
                str(reason): int(count) for reason, count in
                manifest_frame.loc[~manifest_frame["included"],
                                   "exclusion_reason"].value_counts().items()
            },
        }
        results["cohorts"][source] = metrics
    current = torch.load(run_dir / "checkpoint-current.pt", map_location="cpu", weights_only=True)
    if (current["manifest_sha256"] != expected["manifest_sha256"] or
            current["config_sha256"] != expected["config_sha256"] or
            current["qualification_sha256"] != expected["qualification_sha256"]):
        raise ValueError("Last checkpoint uses a different manifest or configuration")
    if (current["epoch"] + 1 != expected["epochs"] or
            current["best_epoch"] != expected["best_epoch"] or
            current["best_qwk"] != expected["best_qwk"]):
        raise ValueError("Completed run marker disagrees with final checkpoint")
    history = current["history"]
    if not history or history[-1]["epoch"] != expected["epochs"]:
        raise ValueError("Checkpoint history is incomplete")
    plots = run_dir / "plots"
    # A failed evaluation can be retried from the same sealed checkpoint;
    # results.json is written last and blocks a second successful evaluation.
    plots.mkdir(exist_ok=True)
    _plot_history(history, plots / "loss-curves.png")
    for source, result in results["cohorts"].items():
        key = "referable" if source == "messidor" else "five_class"
        _plot_confusion(result[key]["confusion_matrix"], source, plots / f"{source}-confusion.png")
    for source, frame in predictions.items():
        frame.to_csv(run_dir / f"predictions-{source}.csv", index=False)
    destination = run_dir / "results.json"
    temporary = destination.with_name(destination.name + ".partial")
    temporary.write_text(json.dumps(results, indent=2), encoding="utf-8")
    os.replace(temporary, destination)
