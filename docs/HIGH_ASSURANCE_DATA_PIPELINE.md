# High-assurance retinal data path: implementation and qualification boundary

The separate implementation is
[`src/drvim/data/high_assurance_dataset.py`](../src/drvim/data/high_assurance_dataset.py).
It is an opt-in candidate for a controlled comparison with the assignment's
current `RetinaDataset`. It does not silently change the model, the sealed
EyePACS split, the ongoing Kaggle jobs, or the report. PyTorch's standard
`DataLoader` remains the batching/worker engine; the source-specific Dataset,
manifest checks, byte verification and transforms are implemented here.

## Current real-data facts and limits

The official Kaggle EyePACS preparation produced a 35,126-image manifest and
an immutable 24,590/5,268/5,268 train/validation/test split. All five grades
occur in each split. The 17,563 grouping tokens are derived from the
`<number>_left`/`<number>_right` filenames. They pair eyes, but the public
release does not independently attest that each number is a unique patient
across the entire cohort or identify site, camera, acquisition date,
gradability, adjudication, or an approved image-use scope. No assertion in
this code can manufacture those missing facts. The original Kaggle JPEGs are
untagged unless each image's embedded ICC profile says otherwise; research
mode records an untagged-sRGB assumption per sample.
The measured training split has 18,060 grade-0 and 502 grade-4 images among
24,590. Naive minority oversampling changes the exposure distribution and can
harm calibration or memorize scarce cases. This loader therefore preserves
exact-once exposure by default. Class-aware loss or sampling is a separately
predeclared experiment to compare on untouched validation data, including
calibration and false-negative effects.

The current assignment loader has full-field resizing, EXIF transpose and
train-only flip. It trusts the manifest hash recorded at preparation time but
does not rehash bytes at sample read time; it also converts to RGB without
using an embedded ICC profile. This module addresses those specific gaps.
It cannot improve diagnosis merely by changing the loader: controlled model
experiments and independent clinical validation would have to show that.

## Implemented decisions

| Concern | Behavior in this module | Remaining evidence |
| --- | --- | --- |
| Immutable dataset | Optional pinned SHA-256 of the **original** manifest; all image bytes rehashed at every read; format and pre-orientation dimensions matched | Verify official archive/release checksums and preserve the manifest, training code and checkpoint lineage |
| Relocation | Recorded Kaggle POSIX prefix is mapped to an approved local root without changing the original manifest; resolved paths must remain inside it | Audit storage permissions, retention and access logs in the eventual execution environment |
| Corrupt/hostile files | Bounded compressed bytes and declared pixel count; complete decode; unexpected formats fail; no silent skip or replacement | Measure bounds on the complete cohort and review parser/library CVEs over time |
| Color | Embedded ICC is converted to sRGB; untagged research image records its assumption; candidate mode requires a source color attestation reference | Source-specific color validation against acquisition devices and human review |
| Geometry | EXIF orientation, aspect-preserving Lanczos resize, black letterbox, original dimensions and occupied-area fraction; no lesion-cropping or contrast enhancement | A blinded review of clinically significant small lesions at the selected model resolution |
| Leakage | Exact hash and grouping-token cross-split checks; candidate mode also checks externally supplied patient group tokens | Near-duplicate search, external patient linkage and site/time leakage analysis |
| Augmentation | Off by default, training only, deterministic by seed/epoch/image ID; candidate mode requires a protocol reference | Predeclared ablation and clinical review of transformations, especially laterality-sensitive use |
| Workers and ordering | Stock PyTorch `DataLoader`, seeded workers, ordered delivery, bounded prefetch, nonpersistent workers so `set_epoch` takes effect; exact-once training permutation reproducible from seed and epoch; validation/test require sequential exact-once traversal | Throughput, CPU/RAM, restart and distributed-shard qualification on the target platform |

`clinical_candidate` is deliberately a *data-evidence gate*, not a clinical
certification flag. It requires per-image nonblank `patient_group_token`,
`site_id`, `device_id`, `grading_reference`, `gradability_status`,
`rights_reference`, and `color_space_attestation_ref` columns, a pinned manifest
digest, and `gradable` for every included image. These fields must come from
real, reviewable acquisition, grading and rights records. The code checks
presence and consistency; it cannot authenticate the truth of a written
attestation. The current public EyePACS manifest lacks those fields and will
correctly fail this mode.

The loader does not interpret DICOM. A DICOM ophthalmic-photography adapter
would require explicit handling of photometric interpretation, ICC, modality,
laterality, de-identification and acquisition metadata. Renaming a DICOM file
to JPEG or inferring laterality from a picture would be unsafe.

## Use on the actual EyePACS manifest

The example below points to the **real** preparation output inside Kaggle. It
does not create example records or synthetic fundus images. The manifest must
be attached privately and the original images must have been extracted and
verified at the specified root first.

```python
from pathlib import Path
from drvim.data.high_assurance_dataset import (
    HighAssuranceRetinaDataset, RetinalDataPolicy, build_high_assurance_loader,
)

policy = RetinalDataPolicy(
    source="eyepacs",
    split="train",
    image_size=1536,
    source_root=Path("/kaggle/temp/drvim-eyepacs/images"),
    recorded_path_prefix="/kaggle/temp/drvim-eyepacs/images",
    evidence_tier="research",
    expected_manifest_sha256=(
        "f933f1bcff8f99688fd8a44984acaa4ce0c929d899be9bddcabdfc54ea7c2117"
    ),
    seed=2026,
)
matches = list(Path("/kaggle/input").rglob("eyepacs-split.csv"))
if len(matches) != 1:
    raise RuntimeError("Expected exactly one private, audited EyePACS split")
dataset = HighAssuranceRetinaDataset(
    matches[0], policy
)
dataset.set_epoch(0)
loader = build_high_assurance_loader(dataset, batch_size=8, num_workers=4,
                                     pin_memory=True)
```

The digest is the measured SHA-256 of the completed private split manifest.
The code resolves its mount and rejects ambiguous matches. For multi-GPU use,
provide a real `DistributedSampler`,
call its `set_epoch` and this dataset's `set_epoch` before each iterator, and
reconcile padded/repeated sampler entries when reporting metrics. Validation
and test loaders use neither augmentation nor shuffle. Never fit normalization,
class weights, color correction or a quality threshold on sealed test data.
The dataset accepts the training script's `(index, contribution_weight)`
sampler entries, including zero-weight DDP padding, and returns
`sample_weight`. The trainer must use that weight in its loss reduction.
The default sampler keeps every included training image exactly once per epoch;
it does not silently oversample rare grades or change the observed prevalence.

## Controlled comparison before adoption

1. Freeze the official archive hashes, manifest digest, source commit, split,
   training recipe, model initialization seeds and compute budget. Review
   duplicate and grouping audits before a test result is read.
2. Run the existing and candidate data paths on the same **real** images and
   compare decoded tensors, orientation, color evidence, per-image exceptions,
   I/O time and RAM. Investigate every unexpected difference rather than
   silently skipping a source image.
3. If color handling or augmentation differs, train independent models from
   scratch under otherwise matched conditions. Select only on training and
   validation. Keep all test images sealed until the final locked comparison.
4. Report grade confusion, per-class precision/recall/F1, quadratic weighted
   kappa, calibration and interval estimates. A referral-use case also needs a
   prespecified referable-DR threshold, sensitivity/specificity, false-negative
   review, gradability/abstention policy and prevalence-sensitive workload.
   Report site/device/patient-subgroup behavior only where verified metadata
   actually exists; mark unavailable strata as unavailable.
5. Repeat on the *original* Messidor external cohort under its native 0–3
   grading, with an explicit, reviewed endpoint mapping if one is needed.
   Do not relabel Messidor 0–3 as EyePACS 0–4. A separate Messidor-2 result
   cannot replace the original Messidor deliverable. ADCIS documents 13
   duplicate Base33 pairs, including one pair with conflicting DR grades,
   plus four published DR grade corrections. The current preparer corrects
   those four grades and excludes one copy of each duplicate pair. The
   conflicting retained pair must be called out and handled in a prespecified
   sensitivity analysis or separately adjudicated before making a clean-label
   performance claim. The official release leaves its errata uncorrected.

No comparative accuracy result is claimed here. The pending CUDA scan kernel
comparison is a **numerical and speed** gate for computation, not a measured
clinical improvement from this loader.

## Product and licensing boundary

Potentially licensable code needs a documented bill of materials, dependency
licences, security maintenance, reproducible builds, versioned schemas,
traceable requirements/risks/tests, human review, release and rollback process,
and a contract stating intended use. An open or commercial *code* licence
does not convey EyePACS or Messidor image/label rights. The EyePACS reply asks
for attribution and points to the Kaggle dataset; the requested permission
for this exact internship/report-image/weight-sharing scope remains a separate
rights question. Keep raw images and identifiable metadata in approved private
storage until the scope is documented.

Clinical deployment requires more than passing these software checks:
prospective intended-use definition, analytical and clinical validation,
representative external cohorts, clinician workflow/human factors, subgroup
and drift monitoring, incident handling, and a maintained quality-management
system under the applicable jurisdiction. This repository is a research
implementation and has no medical-device clearance or clinical-use claim.

Primary references: [PyTorch data loading](https://docs.pytorch.org/docs/stable/data.html),
[Pillow color management](https://pillow.readthedocs.io/en/latest/reference/ImageCms.html),
[FDA/IMDRF Good Machine Learning Practice](https://www.fda.gov/medical-devices/artificial-intelligence-enabled-medical-devices/good-machine-learning-practice-medical-device-development-guiding-principles),
[IMDRF SaMD clinical evaluation](https://www.imdrf.org/documents/software-medical-device-samd-clinical-evaluation),
[FDA quality system regulation](https://www.fda.gov/medical-devices/device-advice-comprehensive-regulatory-assistance/quality-and-compliance-medical-devices),
[DICOM ophthalmic photography](https://dicom.nema.org/medical/dicom/2022c/output/chtml/part03/sect_A.41.3.html).
The source-specific [ADCIS Messidor errata and access terms](https://www.adcis.net/en/third-party/messidor/)
govern the external evaluation design and image-sharing boundary.
