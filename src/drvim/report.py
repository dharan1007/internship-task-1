"""Generate a result-bound LaTeX report; never invent absent experiments."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from .data.manifest import read_manifest, sha256_file
from .metrics import five_class_metrics, referable_metrics


def _tex(value: object) -> str:
    substitutions = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%",
                     "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
                     "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(substitutions.get(char, char) for char in str(value))


def _line(source: str, metrics: dict[str, object]) -> str:
    values = [source, metrics["count"], f'{metrics["accuracy"]:.3f}',
              f'{metrics["macro_precision"]:.3f}', f'{metrics["macro_recall"]:.3f}',
              f'{metrics["macro_f1"]:.3f}', f'{metrics["qwk"]:.3f}']
    return " & ".join(map(_tex, values)) + r" \\"


def _prediction_examples(source: str, predictions: pd.DataFrame,
                         manifest_path: Path, manifest_digest: str) -> list[dict[str, object]]:
    if sha256_file(manifest_path) != manifest_digest:
        raise ValueError(f"{source} report manifest differs from sealed evaluation")
    manifest = read_manifest(manifest_path)
    if not manifest["source"].eq(source).all():
        raise ValueError(f"Wrong source in {source} report manifest")
    eligible = manifest.loc[manifest["included"] & manifest["split"].eq("test")]
    if (predictions["image_id"].duplicated().any() or
            set(predictions["image_id"]) != set(eligible["image_id"])):
        raise ValueError(f"{source} predictions do not match the sealed test cohort")
    joined = predictions.merge(eligible, on="image_id", validate="one_to_one")
    if not joined["true_grade"].astype(int).eq(joined["grade"].astype(int)).all():
        raise ValueError(f"{source} prediction labels disagree with the manifest")
    correct = (joined["true_grade"].astype(int).ge(2).eq(
        joined["predicted_grade"].astype(int).ge(2)) if source == "messidor" else
        joined["true_grade"].astype(int).eq(joined["predicted_grade"].astype(int)))
    if source == "eyepacs":
        # Cover every reference grade using observed test cases; show failures
        # where one exists within that grade.
        examples = []
        for grade in range(5):
            grade_rows = joined.loc[joined["true_grade"].astype(int).eq(grade)]
            if grade_rows.empty:
                raise ValueError(f"EyePACS held-out examples lack reference grade {grade}")
            errors = grade_rows.loc[~correct]
            examples.append(errors.head(1) if not errors.empty else grade_rows.head(1))
        chosen = pd.concat(examples)
    else:
        chosen = pd.concat((joined.loc[~correct].head(1), joined.loc[correct].head(1)))
        if len(chosen) < 2:
            remaining = joined.loc[~joined["image_id"].isin(chosen["image_id"])]
            chosen = pd.concat((chosen, remaining.head(2 - len(chosen))))
    if chosen.empty:
        raise ValueError(f"No real {source} prediction examples")
    return chosen.to_dict("records")


def _stage_examples(output: Path, selected: dict[str, list[dict[str, object]]]) -> str:
    directory = output / "examples"
    directory.mkdir()
    panels = []
    ledger = []
    for source, rows in selected.items():
        for index, row in enumerate(rows, start=1):
            original = Path(str(row["image_path"]))
            if sha256_file(original) != row["sha256"]:
                raise ValueError(f"Source image changed before report: {source}/{row['image_id']}")
            filename = f"{source}-{index:02d}.png"
            destination = directory / filename
            with Image.open(original) as image:
                # A full-resolution, orientation-correct PNG is lossless.
                ImageOps.exif_transpose(image).convert("RGB").save(destination, format="PNG")
            ledger.append({"source": source, "image_id": row["image_id"],
                           "source_sha256": row["sha256"], "rendered_sha256": sha256_file(destination),
                           "true_grade": int(row["true_grade"]),
                           "predicted_grade": int(row["predicted_grade"])})
            reference = (f"Messidor R{int(row['true_grade'])}" if source == "messidor"
                         else f"ICDR {int(row['true_grade'])}")
            prediction = f"ICDR {int(row['predicted_grade'])}"
            panels.append(r"\begin{minipage}[t]{.48\linewidth}\centering" + "\n" +
                          rf"\includegraphics[width=\linewidth,height=45mm,keepaspectratio]{{examples/{filename}}}\par" + "\n" +
                          r"\footnotesize " + _tex(source) + ": " + _tex(row["image_id"]) +
                          r"\quad Reference " + _tex(reference) +
                          r"; prediction " + _tex(prediction) + "\n" +
                          r"\end{minipage}")
    (output / "example-provenance.json").write_text(json.dumps(ledger, indent=2), encoding="utf-8")
    rows = [r"\hfill".join(panels[index:index + 2])
            for index in range(0, len(panels), 2)]
    return "\n\\par\\medskip\n".join(rows)


def _generate_report(run_dir: Path, output: Path, eyepacs: Path,
                     messidor: Path, messidor2: Path | None) -> None:
    results_path = run_dir / "results.json"
    if not results_path.is_file():
        raise RuntimeError("Report requires genuine sealed test results")
    complete_path = run_dir / "training-complete.json"
    if not complete_path.is_file():
        raise RuntimeError("Report requires a completed full training run")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Report output is not empty; refusing overwrite")
    results = json.loads(results_path.read_text(encoding="utf-8"))
    completed = json.loads(complete_path.read_text(encoding="utf-8"))
    if (completed["manifest_sha256"] != results["eyepacs_manifest_sha256"] or
            completed["best_epoch"] != results["checkpoint_epoch"] or
            completed["best_qwk"] != results["validation_qwk_at_selection"]):
        raise ValueError("Sealed test results disagree with the full-run marker")
    sources = ["eyepacs", "messidor"] + (["messidor2"] if messidor2 is not None else [])
    if set(results["cohorts"]) != set(sources):
        raise ValueError("Sealed result cohorts differ from requested report cohorts")
    if (messidor2 is None) != (results["messidor2_manifest_sha256"] is None):
        raise ValueError("Messidor-2 manifest presence disagrees with sealed results")
    for source in sources:
        if not (run_dir / f"predictions-{source}.csv").is_file():
            raise RuntimeError(f"Missing real {source} prediction records")
    required_plots = ("loss-curves.png",) + tuple(
        f"{source}-confusion.png" for source in sources)
    if any(not (run_dir / "plots" / name).is_file() for name in required_plots):
        raise RuntimeError("A real training or confusion-matrix plot is missing")
    selected: dict[str, list[dict[str, object]]] = {}
    manifests = {"eyepacs": (eyepacs, results["eyepacs_manifest_sha256"]),
                 "messidor": (messidor, results["messidor_manifest_sha256"])}
    if messidor2 is not None:
        manifests["messidor2"] = (messidor2, results["messidor2_manifest_sha256"])
    for source in sources:
        frame = pd.read_csv(run_dir / f"predictions-{source}.csv", dtype={"image_id": str})
        columns = [f"probability_{grade}" for grade in range(5)]
        if not set(columns).issubset(frame):
            raise ValueError(f"Saved {source} predictions lack five-class probabilities")
        probabilities = frame[columns].to_numpy(dtype=float)
        if (not np.isfinite(probabilities).all() or
                (probabilities < 0).any() or (probabilities > 1).any() or
                not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5) or
                not np.array_equal(probabilities.argmax(axis=1),
                                   frame["predicted_grade"].to_numpy(dtype=int))):
            raise ValueError(f"Saved {source} probabilities disagree with predicted grades")
        true = frame["true_grade"].astype(int).tolist()
        predicted = frame["predicted_grade"].astype(int).tolist()
        expected = results["cohorts"][source]
        if referable_metrics(true, predicted, original_messidor=source == "messidor") != expected["referable"]:
            raise ValueError(f"Saved {source} binary metrics disagree with predictions")
        if source != "messidor" and five_class_metrics(true, predicted) != expected["five_class"]:
            raise ValueError(f"Saved {source} five-grade metrics disagree with predictions")
        manifest_path, digest = manifests[source]
        selected[source] = _prediction_examples(source, frame, manifest_path, digest)
    output.mkdir(parents=True, exist_ok=True)
    for name in required_plots:
        shutil.copy2(run_dir / "plots" / name, output / name)
    # Messidor source terms prohibit copying its images into a shared report.
    example_panels = _stage_examples(output, {"eyepacs": selected["eyepacs"]})
    five_grade_sources = [source for source in sources if source != "messidor"]
    rows = "\n".join(_line(source, results["cohorts"][source]["five_class"])
                     for source in five_grade_sources)
    class_rows = "\n".join(" & ".join(map(_tex, (source, item["grade"], item["support"],
        f'{item["precision"]:.3f}', f'{item["recall"]:.3f}', f'{item["f1"]:.3f}'))) + r" \\"
        for source in five_grade_sources
        for item in results["cohorts"][source]["five_class"]["per_class"])
    count_rows = "\n".join(" & ".join(map(_tex, (source,
        results["cohorts"][source]["source_counts"]["total_labeled_rows"],
        results["cohorts"][source]["source_counts"]["included"],
        results["cohorts"][source]["source_counts"]["excluded"]))) + r" \\"
        for source in sources)
    binary = "\n".join(" & ".join(map(_tex, (source,
        f'{results["cohorts"][source]["referable"]["accuracy"]:.3f}',
        f'{results["cohorts"][source]["referable"]["precision_referable"]:.3f}',
        f'{results["cohorts"][source]["referable"]["recall_referable"]:.3f}',
        f'{results["cohorts"][source]["referable"]["f1_referable"]:.3f}'))) + r" \\"
        for source in sources)
    configuration = results["experiment_config"]
    hyperparameters = [
        ("Input pixels", configuration["model"]["image_size"]),
        ("Patch pixels", configuration["model"]["patch_size"]),
        ("Blocks", configuration["model"]["depth"]),
        ("Embedding width", configuration["model"]["embed_dim"]),
        ("State dimension", configuration["model"]["state_dim"]),
        ("Microbatch per GPU", configuration["data"]["batch_size"]),
        ("Accumulation steps", configuration["train"]["grad_accum_steps"]),
        ("Nominal global batch", configuration["data"]["batch_size"] *
         configuration["train"]["grad_accum_steps"] * results["runtime"]["world_size"]),
        ("Training epochs", configuration["train"]["epochs"]),
        ("Initial learning rate", configuration["train"]["learning_rate"]),
        ("Weight decay", configuration["train"]["weight_decay"]),
        ("Warmup epochs", configuration["train"]["warmup_epochs"]),
        ("Precision", configuration["train"]["amp"]),
        ("GPU", results["runtime"]["gpu_name"]),
        ("GPU count", results["runtime"]["world_size"]),
        ("PyTorch", results["runtime"]["torch_version"]),
        ("Training hours", f'{results["training_elapsed_seconds"] / 3600:.2f}'),
    ]
    parameter_rows = "\n".join(" & ".join(map(_tex, item)) + r" \\"
                               for item in hyperparameters)
    report = r"""\documentclass[10pt,a4paper]{article}
\usepackage[margin=22mm]{geometry}
\usepackage{graphicx,booktabs}
\title{Diabetic retinopathy grading with a from-scratch Vision Mamba}
\author{Research implementation}
\date{\today}
\begin{document}
\maketitle
\begin{abstract}
This report records a five-grade EyePACS training experiment and held-out evaluation.
Metrics derive from the saved checkpoint and prediction records; no external cohort
was used for training or checkpoint selection.
\end{abstract}
\section{Method}
The network follows Vision Mamba's bidirectional image sequence design
\cite{vim,mamba}. It uses $PATCH\times PATCH$ patches, an absolute position
embedding, a middle classification token, and DEPTH bidirectional selective
state-space blocks. Each block
has independent forward and reverse causal depthwise convolutions and selective
state-space parameters, a shared input projection and gate, and a residual path.
The classifier emits five logits for ICDR grades 0--4. All weights began from
random initialization. Cross-entropy loss, AdamW, warmup followed by cosine
decay, mixed precision, and gradient clipping were used. The highest EyePACS
validation quadratic weighted kappa selected the checkpoint before test evaluation.
\begin{table}[h]\centering\small
\begin{tabular}{ll}\toprule
Parameter & Run value\\\midrule
PARAMETER_ROWS
\bottomrule\end{tabular}\caption{Recorded full-run configuration.}\end{table}
\section{Data and controls}
EyePACS left and right eyes were grouped by their filename pairing token before
stratified train, validation and held-out test assignment. This token alone
does not independently verify patient identity. Each source image was fully decoded and
SHA-256 hashed; its full field was aspect-preserving resized and padded to the
configured square resolution. Horizontal flips were limited to training.
EyePACS, Inc. is credited as the source of the 2015 Kaggle retinal images
\cite{eyepacs}.
Original Messidor's published grade corrections and duplicate exclusions were
applied \cite{messidor}. Its R0--R3 grades were not recast as five-grade ICDR
labels. MESSIDOR2_DATA
External datasets remained test-only. OVERLAP_STATEMENT
Kindly provided by the Messidor program partners (see
https://www.adcis.net/en/third-party/messidor/).
\begin{table}[h]\centering\small
\begin{tabular}{lrrr}\toprule
Cohort & Labeled rows & Included & Excluded\\\midrule
COUNT_ROWS
\bottomrule\end{tabular}\caption{Source reconciliation and exclusions.}\end{table}
\section{Results}
Selected checkpoint: epoch \textbf{EPOCH}; EyePACS validation QWK at selection:
\textbf{VQWK}.
\begin{table}[h]\centering\small
\begin{tabular}{lrrrrrr}\toprule
Cohort & $n$ & Accuracy & Macro P & Macro R & Macro F1 & QWK\\\midrule
FIVE_ROWS
\bottomrule\end{tabular}
\caption{Five-grade ICDR evaluation.}\end{table}
\begin{table}[h]\centering\small
\begin{tabular}{lrrrrr}\toprule
Cohort & Grade & Support & Precision & Recall & F1\\\midrule
PER_CLASS_ROWS
\bottomrule\end{tabular}
\caption{Per-grade results. Precision is reported as zero if no example is
predicted in that grade; support is shown explicitly.}\end{table}
\begin{table}[h]\centering\small
\begin{tabular}{lrrrr}\toprule
Cohort & Accuracy & Referable P & Referable R & Referable F1\\\midrule
BINARY_ROWS
\bottomrule\end{tabular}
\caption{Declared binary endpoint: source grade $\geq2$ versus model ICDR $\geq2$.
Original Messidor R2/R3 is a source-specific proxy, not an identical grading scale.}\end{table}
\begin{figure}[h]\centering\includegraphics[width=.75\linewidth]{loss-curves.png}
\caption{Observed training and validation cross-entropy.}\end{figure}
\clearpage
\section{Confusion matrices}
\includegraphics[width=.48\linewidth]{eyepacs-confusion.png}
MESSIDOR2_CONFUSION
\includegraphics[width=.48\linewidth]{messidor-confusion.png}
\section{Observed fundus image--prediction pairs}
These EyePACS held-out source images are rendered losslessly, with their recorded
reference grades and sealed model predictions. Messidor R grades use a
different scale from ICDR and are shown only as a source-specific binary
endpoint in the quantitative results.
\begin{figure}[h]\centering
EXAMPLES
\caption{Real held-out EyePACS images, one per reference grade, and predictions. Image hashes and IDs are
recorded in the private example provenance ledger.}
\end{figure}
\section{Interpretation and limitations}
The test estimates are tied to these exact source releases, grading definitions,
split, preprocessing and checkpoint. MESSIDOR2_LIMITATION
Disagreement can reflect domain shift, image quality, and grading differences.
These research results do not establish clinical safety or suitability for diagnosis.
\begin{thebibliography}{9}
\bibitem{eyepacs} J. Cuadros and G. Bresnick, ``EyePACS: An Adaptable
Telemedicine System for Diabetic Retinopathy Screening,'' Journal of Diabetes
Science and Technology, vol. 3, no. 3, pp. 509--516, 2009.
\bibitem{vim} L. Zhu et al., ``Vision Mamba: Efficient Visual Representation
Learning with Bidirectional State Space Model,'' arXiv:2401.09417, 2024.
\bibitem{mamba} A. Gu and T. Dao, ``Mamba: Linear-Time Sequence Modeling with
Selective State Spaces,'' arXiv:2312.00752, 2023.
\bibitem{messidor} E. Decenci\`ere et al., ``Feedback on a publicly distributed
image database: the Messidor database,'' Image Analysis \& Stereology, 2014.
\bibitem{krause} J. Krause et al., ``Grader Variability and the Importance of
Reference Standards for Evaluating Machine Learning Models for Diabetic
Retinopathy,'' Ophthalmology, 2018.
\bibitem{sources} EyePACS, Kaggle Diabetic Retinopathy Detection; ADCIS
Messidor and Messidor-2; Google Brain Messidor-2 adjudicated DR grades.
\end{thebibliography}
\end{document}
"""
    with_second = messidor2 is not None
    replacements = {
                    "MESSIDOR2_DATA": ("Messidor-2 used separately published adjudicated ICDR "
                                       r"grades and excluded ungradable images \cite{krause}."
                                       if with_second else ""),
                    "OVERLAP_STATEMENT": (
                        "The two Messidor releases share "
                        f"\\textbf{{{results['original_messidor_messidor2_filename_overlap']}}} "
                        "filename stems and are not independent validation cohorts."
                        if with_second else "Messidor-2 was not included in this experiment."),
                    "MESSIDOR2_CONFUSION": (
                        r"\includegraphics[width=.48\linewidth]{messidor2-confusion.png}"
                        if with_second else ""),
                    "MESSIDOR2_LIMITATION": (
                        "Original Messidor and Messidor-2 overlap."
                        if with_second else "Only original Messidor was used for external evaluation."),
                    "PATCH": configuration["model"]["patch_size"],
                    "DEPTH": configuration["model"]["depth"],
                    "EPOCH": results["checkpoint_epoch"],
                    "VQWK": f'{results["validation_qwk_at_selection"]:.3f}',
                    "FIVE_ROWS": rows, "PER_CLASS_ROWS": class_rows,
                    "COUNT_ROWS": count_rows, "BINARY_ROWS": binary,
                    "PARAMETER_ROWS": parameter_rows,
                    "EXAMPLES": example_panels}
    for old, new in replacements.items():
        report = report.replace(old, str(new))
    tex = output / "report.tex"
    tex.write_text(report, encoding="utf-8")
    compiler = shutil.which("tectonic") or shutil.which("pdflatex")
    if compiler is None:
        raise RuntimeError(f"LaTeX source written to {tex}; install Tectonic or pdflatex to compile")
    command = ([compiler, "-X", "compile", "report.tex"] if Path(compiler).stem == "tectonic"
               else [compiler, "-interaction=nonstopmode", "-halt-on-error", "report.tex"])
    subprocess.run(command, cwd=output, check=True, capture_output=True, text=True)
    pdf = output / "report.pdf"
    if not pdf.is_file() or pdf.stat().st_size == 0:
        raise RuntimeError(f"LaTeX exited successfully but did not produce a nonempty PDF: {pdf}")
    provenance = {
        "report_source_sha256": sha256_file(Path(__file__)),
        "training_complete_sha256": sha256_file(complete_path),
        "sealed_results_sha256": sha256_file(results_path),
        "report_tex_sha256": sha256_file(tex),
        "report_pdf_sha256": sha256_file(pdf),
        "example_provenance_sha256": sha256_file(output / "example-provenance.json"),
        "plots_sha256": {name: sha256_file(output / name) for name in required_plots},
    }
    (output / "report-artifact-provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8")


def generate_report(run_dir: Path, output: Path, eyepacs: Path,
                    messidor: Path, messidor2: Path | None) -> None:
    """Publish only a fully compiled, provenance-checked report directory."""
    if output.exists():
        raise FileExistsError(f"Report destination already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        _generate_report(run_dir, staging, eyepacs, messidor, messidor2)
        os.rename(staging, output)
    finally:
        if staging.exists():
            if staging.resolve().parent != output.parent.resolve():
                raise RuntimeError("Report staging path escaped its intended parent")
            shutil.rmtree(staging)
