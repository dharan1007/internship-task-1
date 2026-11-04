# Code handoff and evidence boundary — 2026-11-04

The governing assignment is the supplied internship task PDF.
It asks for five deliverables. The source for the first three exists; the
fourth and fifth require a completed real experiment. The PDF says code
quality, understanding of Vision Mamba, analysis, documentation and
reproducibility are evaluation criteria. It still explicitly requires model
training, EyePACS and Messidor evaluation, real figures/predictions, and a
compiled LaTeX report. A reviewer-approved code-only submission would be a
**reduced scope**, not completion of every PDF requirement.

## What exists as real code

| PDF deliverable | Implementation and evidence | Honest status |
| --- | --- | --- |
| 1. Complete PyTorch implementation | `src/drvim/model/vim.py` defines patch embedding, middle class token, position embedding, 24 bidirectional blocks, independent directional convolution/SSM, RMSNorm, and five-logit head. `src/drvim/model/scan.py` implements a mathematical reference scan and in-house parallel scan with backward. `configs/vim_base_eye.yaml` configures 1536-pixel, patch-16 Vim-Base. | Implemented from random initialization; earlier BF16/parallel source passed scan parity and two real-image optimizer steps. Current FP16/optional CUDA edits have not passed a new full-model GPU qualification. No trained model weights exist. |
| 2. Dataset loading and preprocessing | `src/drvim/data/dataset.py` is the project's custom PyTorch `Dataset`; `src/drvim/data/preprocess.py` handles EXIF orientation, full-field resize/pad, RGB tensor conversion and train-only flip. `prepare.py`, `manifest.py`, `split.py`, `audit.py` reconcile source files and labels, record hashes and exclusions, and create/audit a grouped split. `train.py` uses standard `torch.utils.data.DataLoader`, as the PDF directs. An additional `high_assurance_dataset.py` offers strict byte, color, metadata and cohort checks but is not the active trainer loader. | Implemented and exercised on original EyePACS. The private preparation job decoded, CRC-checked and hashed 35,126 labeled JPEGs, with 24,590/5,268/5,268 train/validation/test. Filename grouping does not prove independent patient identity. |
| 3. Training and evaluation scripts | `src/drvim/train.py` has the optimizer, cross-entropy loop, mixed precision, accumulation, DDP sampler, validation-only checkpoint selection, resumable checkpoints and completion marker. `src/drvim/qualify_scan.py` checks scan parity and two real-image full-model steps. `src/drvim/evaluate.py` gates test evaluation on a complete run, verifies manifests/checkpoints, saves per-image predictions and plots; `src/drvim/metrics.py` computes requested metrics. `src/drvim/cli.py` exposes commands. | Code exists. A complete training epoch, resume, full run and sealed evaluation have not been demonstrated. The latest source changes require their own qualification. |
| 4. Evaluation metrics, plots and image–prediction pairs | `evaluate.py`, `metrics.py`, and `report.py` generate these only from a complete checkpoint and actual source images. | **No measured test metrics, convergence curves, confusion plots or image–prediction pairs exist.** Two optimizer steps are diagnostics, not training results. |
| 5. Compiled LaTeX report | `src/drvim/report.py` generates TeX and compiles it after sealed results; a Tectonic smoke compile passed. | **No final experiment TeX/PDF exists.** . |

## Paper mapping and deviations

The [Vision Mamba paper](https://arxiv.org/abs/2401.09417) uses position-marked
visual tokens and bidirectional state-space processing. This source maps an
image into 16-by-16 patch tokens, inserts a class token in the middle, adds
position embeddings, applies forward and reversed selective state-space
directions in each block, and classifies the final class token. The reference
scan and its backward are authored in this project; the primary parallel
backend is also authored here. The optional pinned CUDA kernel is a low-level
efficiency dependency, **not** a ready-made Vim model. The configured 1536
resolution and 300 epochs are project choices, not mandated by the paper or
internship PDF. This implementation has not reproduced the paper's published
benchmarks, which are for other tasks and settings.

## Verified evidence and links

- [Original EyePACS preparation](https://www.kaggle.com/code/dharantejreddy/drvim-prepare-original-eyepacs): private job completed the full 35,126-image audit and split. Split manifest SHA-256: `f933f1bcff8f99688fd8a44984acaa4ce0c929d899be9bddcabdfc54ea7c2117`.
- [Full-manifest, two-step model qualification](https://www.kaggle.com/code/dharantejreddy/drvim-full-model-qualification): completed on one T4 using the earlier BF16/parallel source. Two eight-image optimizer steps took 195.71 and 197.24 seconds. This is **not** a model-training run or validation score.
- [CUDA scan comparison](https://www.kaggle.com/code/dharantejreddy/drvim-real-eyepacs-cuda-scan-comparison): a completed low-level kernel benchmark on real EyePACS inputs. It does **not** establish whole-model speed or the correctness of the current optional full-model configuration.
- Local repository validation on 2026-11-04: Ruff passed; `PYTHONPATH=src python -m pytest -q --basetemp=<writable-private-directory> tests` passed 12 tests. These checks do not establish clinical validity or classifier performance.

None of these links shows a currently running full training job. There is no
honest link to a trained checkpoint, EyePACS/Messidor test results, or final
report yet.

## Submission package boundary

For a **code review now**, submit this repository's `src/`, `configs/`,
`scripts/`, `tests/`, `pyproject.toml`, `README.md`, `docs/RESEARCH_PROTOCOL.md`,
`docs/HIGH_ASSURANCE_DATA_PIPELINE.md`, `docs/SUBMISSION_READINESS.md`, and this
handoff. State clearly that training and experimental results are pending.
Exclude private images, labels, credentials, private manifests, temporary
files, and diagnostic output from any public archive. The prior
BF16/parallel source was qualified in the linked Kaggle diagnostic; this
source snapshot includes later FP16/CUDA changes, so the earlier qualification
must not be described as validation of the current snapshot.

To meet **every PDF deliverable**, a complete five-class training run must
select a checkpoint on EyePACS validation, followed by sealed held-out EyePACS
and original Messidor evaluation, measured metrics/plots/prediction examples,
and a compiled/reviewed LaTeX report. Original Messidor's 0–3 grades need the
declared binary endpoint; its ADCIS images and annotations are still absent.
Messidor-2 adjudicated grades have been obtained, but its ADCIS images and
pairing sheet are absent. Image use and any public report distribution must
follow the applicable source terms. This repository is research software and
has no clinical validation or medical-device qualification.
