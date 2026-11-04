# Diabetic retinopathy grading with Vision Mamba

A from-scratch PyTorch research implementation of a five-grade diabetic
retinopathy classifier based on [Vision Mamba](https://arxiv.org/abs/2401.09417).
The assignment calls for training with the original
[EyePACS Kaggle release](https://www.kaggle.com/competitions/diabetic-retinopathy-detection/data)
and external evaluation on
[original Messidor](https://www.adcis.net/en/third-party/messidor/).

**Current status:** model, data pipeline, training/evaluation commands, and a
result-bound LaTeX report generator are implemented. All 35,126 labeled
EyePACS training images were decoded, reconciled, and hashed in a private
preparation job. Two real-image optimizer steps were used to qualify an
earlier source revision. **Full training has not started; there is no trained
checkpoint, measured classification score, test prediction set, or compiled
experiment report.** The 12 official original Messidor archives and annotation
workbooks have now been downloaded, integrity-checked and extracted in private
storage. Their verified manifest contains 1,200 records, with 1,187 eligible
images after 13 published duplicate exclusions. External model evaluation is
still pending. The
[18-page illustrated implementation dossier](docs/implementation_dossier.pdf)
explains the architecture, equations, data flow, measured preparation evidence,
source correspondence, and limits. Its LaTeX source is
[here](docs/implementation_dossier.tex), with
[technical appendices](docs/implementation_appendix.tex). The
[code handoff](docs/CODE_SUBMISSION_HANDOFF.md)
maps each assignment deliverable to its present evidence.

## Implementation

| Component | What this repository provides |
| --- | --- |
| Model | Own patch embedding, middle class token, learned positions, 24 bidirectional selective state-space blocks, independent forward/reverse convolution and SSM parameters, RMSNorm, residuals, stochastic depth, and a five-logit head. Random initialization; no pretrained weights or imported Vim model. |
| Selective scan | Authored recurrent reference, chunked parallel affine scan, and backward. An optional pinned low-level CUDA scan is under separate whole-model qualification. |
| Data | Own map-style PyTorch Dataset, full-field aspect-preserving preprocessing, original image/label reconciliation, cryptographic hashes, explicit exclusions, grade-stratified grouping split, and leakage audit. Standard PyTorch DataLoader batches the custom Dataset. |
| Training | Cross-entropy, AdamW, warmup/cosine schedule, mixed precision, gradient accumulation, activation checkpointing, distributed sampling, validation QWK checkpoint selection, and resumable epoch checkpoints. |
| Evaluation | Completion-gated held-out inference, image-level probabilities and predictions, accuracy, precision, recall, F1, QWK where the labels are comparable, confusion matrices, and loss curves. |
| Report | LaTeX generator for real results and hash-verified fundus image–prediction pairs; it refuses to compile a final experiment report without a completed run and sealed predictions. |

[Model code](src/drvim/model/),
[data code](src/drvim/data/),
[training](src/drvim/train.py),
[evaluation](src/drvim/evaluate.py),
[metrics](src/drvim/metrics.py),
[report](src/drvim/report.py),
[CLI](src/drvim/cli.py),
[configuration](configs/), and
[tests](tests/) are separated by responsibility.

## Architecture and decisions

The [paper](https://arxiv.org/abs/2401.09417) represents images as
position-marked patch sequences and processes them in both directions with
selective state-space blocks. Here, nonoverlapping 16 × 16 patches are embedded,
a learned class token is inserted in the middle, and learned absolute positions
are added. Every block combines independently parameterized forward and
reverse convolution/SSM paths through a shared gate, then projects back to
model width. A final normalization and linear head classify grades 0–4.

The configured candidate uses 1536 × 1536 input, depth 24, width 768, state
size 16, expansion 2, and 103,760,645 parameters. Its 9,217-token sequence
makes scan throughput a major feasibility issue. The resolution and
300-epoch schedule are project choices, not required by the paper or the
internship brief. See the [research protocol](docs/RESEARCH_PROTOCOL.md)
and [compute plan](docs/COMPUTE_PLAN.md).

The source favors auditable inputs and failures: it reconciles every image and
label, records source hashes, groups filename-paired eyes before splitting,
rejects changed manifests, blocks test evaluation before full training, and
derives reports from saved prediction records. These are verified design
properties, **not** a measured claim of better accuracy or speed than other
retinal classifiers. Clinical validity and robustness are unproven.

## Data policy

- **EyePACS:** use the original labeled Kaggle training release under its
  applicable terms. Its competition test release has no public ground-truth
  labels. The prepared split contains 24,590 train, 5,268 validation, and
  5,268 held-out test images. Filename pairing tokens keep left/right image
  pairs together but do not independently prove patient identity.
- **Preprocessing:** EXIF-corrected RGB, complete retinal field preserved by
  aspect-preserving resize and black padding; horizontal flip only in
  training. The separate
  [high-assurance loader](docs/HIGH_ASSURANCE_DATA_PIPELINE.md) adds stricter
  byte, format, color, and metadata controls, but is not the active trainer
  loader.
- **Original Messidor:** external test only. Its R0–R3 grading scheme differs
  from EyePACS ICDR 0–4; the cross-source endpoint is the declared referable
  DR proxy (Messidor R2/R3 versus model ICDR 2/3/4). The preparer accounts
  for four published grade corrections and 13 duplicate exclusions. The
  private 1,200-image manifest passed its schema validation; 1,187 images
  remain eligible for evaluation.
- **Messidor-2:** optional extra five-grade test cohort. Official images and
  pair information come from [ADCIS](https://www.adcis.net/en/third-party/messidor2/);
  adjudicated grades are a separate
  [Google Brain release](https://www.kaggle.com/datasets/google-brain/messidor2-dr-grades).
  ADCIS states that 1,058 of its images are paired images from original
  Messidor (PNG copies) and 690 are newly collected extension images (JPG).
  It cannot be presented as an independent second external dataset alongside
  original Messidor. Its official pairing spreadsheet was not in the image
  ZIP received here. The PDF's original Messidor requirement does not depend
  on Messidor-2.

No raw retinal images, labels, credentials, model weights, or image-bearing
reports are in this public repository. [Official access steps](docs/DATA_ACCESS.md)
and the [evidence ledger](docs/SUBMISSION_READINESS.md) distinguish available
from pending inputs and results. The [dated source-access timeline](docs/SOURCE_ACCESS_TIMELINE.md)
records the support correspondence and acquisition milestones without exposing
restricted ADCIS access details.

## Install and workflow

Use Python 3.10 or newer and a PyTorch CUDA build appropriate for the chosen
accelerator. Install the package and run the source checks:

~~~bash
python -m pip install -e ".[test]"
python -m ruff check src tests scripts
python -m pytest -q tests
~~~

Run these commands against official files in private storage. The paths below
illustrate the required layout; replace them with actual private mounts.
Qualify the **exact** source and configuration before starting training.

~~~bash
drvim prepare-eyepacs --images /private/eyepacs/train --labels /private/eyepacs/trainLabels.csv --out /private/manifests/eyepacs.csv
drvim split-eyepacs --manifest /private/manifests/eyepacs.csv --out /private/manifests/eyepacs-split.csv --seed 2026
drvim qualify-scan --config configs/vim_base_eye.yaml --manifest /private/manifests/eyepacs-split.csv --out /private/scan-qualification.json
drvim train --config configs/vim_base_eye.yaml --manifest /private/manifests/eyepacs-split.csv --qualification /private/scan-qualification.json --run-dir /private/runs/eyepacs-base
drvim prepare-messidor --images /private/messidor --out /private/manifests/messidor.csv
drvim audit --run-dir /private/runs/eyepacs-base --eyepacs /private/manifests/eyepacs-split.csv --messidor /private/manifests/messidor.csv --out /private/audit.json
drvim evaluate --run-dir /private/runs/eyepacs-base --eyepacs /private/manifests/eyepacs-split.csv --messidor /private/manifests/messidor.csv --audit /private/audit.json
drvim report --run-dir /private/runs/eyepacs-base --eyepacs /private/manifests/eyepacs-split.csv --messidor /private/manifests/messidor.csv --out /private/report
~~~

Training can be launched with torchrun for multiple GPUs. The
stop-after-epoch option pauses at a checkpoint boundary; resume requires the
same source, manifest, config, qualification and GPU world size. The optional
Messidor-2 manifest can be passed to audit, evaluate and report with
--messidor2. Tectonic or pdfLaTeX is required for the final report.

The earlier BF16/parallel full-model qualification is tied to an older source
digest. The current FP16 and optional CUDA changes need a new full-model GPU
qualification. At the measured earlier T4 step rate, the configured full run
does not fit the available free Kaggle session/quota. No paid compute is
authorized.

## Verified Kaggle work

These owner-controlled notebook links may require private Kaggle access.
They are **preparation and diagnostic evidence, not trained-model results**.

- [Original EyePACS preparation](https://www.kaggle.com/code/dharantejreddy/drvim-prepare-original-eyepacs):
  35,126 source images decoded, reconciled, and hashed; grouped split written.
  Split manifest SHA-256:
  f933f1bcff8f99688fd8a44984acaa4ce0c929d899be9bddcabdfc54ea7c2117.
- [Two-step full-model qualification](https://www.kaggle.com/code/dharantejreddy/drvim-full-model-qualification):
  earlier BF16/parallel source passed numerical checks and two eight-image
  optimizer steps on one T4. No trained checkpoint or validation score.
- [Pinned CUDA scan build](https://www.kaggle.com/code/dharantejreddy/drvim-pinned-cuda-scan-build)
  and [real-image scan comparison](https://www.kaggle.com/code/dharantejreddy/drvim-real-eyepacs-cuda-scan-comparison):
  low-level scan work only; no whole-model throughput result for current code.

## Credits and references

- **EyePACS, Inc.** supplied the Kaggle retinal image release and requested
  attribution in reports or publications. Cuadros J, Bresnick G. “EyePACS:
  An Adaptable Telemedicine System for Diabetic Retinopathy Screening.”
  *Journal of Diabetes Science and Technology*. 2009;3(3):509–516.
- **Vision Mamba:** Zhu L et al. “Vision Mamba: Efficient Visual
  Representation Learning with Bidirectional State Space Model.”
  [arXiv:2401.09417](https://arxiv.org/abs/2401.09417), ICML 2024.
  The [authors' repository](https://github.com/hustvl/Vim) informed
  architecture comparison; no ready-made model or weights are imported.
- **Messidor / ADCIS:** credit the
  [Messidor research program](https://www.adcis.net/en/third-party/messidor/)
  with its requested acknowledgment, “Kindly provided by the Messidor
  program partners,” and follow its access and image-use conditions. The
  original dataset citation is Decencière E et al., “Feedback on a publicly
  distributed database: the Messidor database,” *Image Analysis &
  Stereology* 33(3):231–234, 2014.
- **Messidor-2 grades:** the
  [Google Brain annotations](https://www.kaggle.com/datasets/google-brain/messidor2-dr-grades)
  are separate from the official ADCIS images. ADCIS requests attribution to
  the LaTIM laboratory and Messidor program partners for Messidor-2 use, and
  cites Abràmoff MD et al., “Automated analysis of retinal images for detection
  of referable diabetic retinopathy,” *JAMA Ophthalmology* 131(3):351–357,
  2013.
- **Optional scan kernel:** the
  [Mamba selective scan project](https://github.com/state-spaces/mamba)
  supplies a pinned low-level efficiency dependency, not model weights.

The [competition host's research-use answer](https://www.kaggle.com/c/diabetic-retinopathy-detection/discussion/141968)
supports educational research and publications with EyePACS attribution; we
infer that private training falls within that scope. It and the provider reply
do not clearly authorize sharing source images or trained weights. Confirm
those rights before distributing such outputs. This is research code, with no
clinical validation, regulatory
clearance, or diagnostic claim.
