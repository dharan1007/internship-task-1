# Submission readiness — 2026-11-04

This is an evidence ledger for `IntershipTask-1.pdf`, not a claim that the
experiment is complete. The PDF requires a from-scratch PyTorch Vim classifier,
custom EyePACS loading and preprocessing, full five-grade training, held-out
EyePACS and Messidor evaluation, metrics and visual results, and a compiled
LaTeX report with real fundus image--prediction pairs.

| Required evidence | Current verified state |
| --- | --- |
| Vim model, custom Dataset/DataLoader workflow, training/evaluation/report scripts | Source present in `src/drvim`. Diagnostics on real original EyePACS images and a separate full-manifest qualification passed CUDA scan parity and two full-model optimizer steps at 1536 pixels. The complete training run remains unrun. A separate opt-in high-assurance data module and design review are in `src/drvim/data/high_assurance_dataset.py` and `docs/HIGH_ASSURANCE_DATA_PIPELINE.md`; they are not yet adopted into the training run. |
| Original EyePACS images and labels | Authenticated access is verified with the official Kaggle CLI. `trainLabels.csv.zip` and all five original wrapper parts `train.zip.001.zip`–`.005.zip` are downloaded, CRC-checked and SHA-256 recorded privately. The private Kaggle mount supplied all five raw parts to the completed full-data preparation job. A separate local raw-segment reconstruction was stopped after two verified segments; it is not needed to establish the Kaggle preparation result. Earlier Chrome `.crdownload` files are incomplete and are not treated as source data. |
| EyePACS data audit, grouping split and complete five-grade training | A private Kaggle job fully extracted, decoded, hashed and reconciled all 35,126 original JPEGs with labels. Its split manifest SHA-256 is `f933f1bcff8f99688fd8a44984acaa4ce0c929d899be9bddcabdfc54ea7c2117`. The split contains 24,590 train, 5,268 validation and 5,268 test images, with all five grades in each. Filename-derived left/right pairing groups do not cross splits, but unique patient identity is not independently verified. The full-manifest model qualification passed numerical and two-step gates, but there is no completed training checkpoint. |
| Original Messidor evaluation | ADCIS supplied restricted alternative official access after the form messages failed. All 12 original image archives and 12 canonical annotation workbooks passed integrity checks. The preparer decoded and hashed all 1,200 TIFFs and wrote a schema-valid manifest: 13 published duplicate exclusions, 1,187 eligible images. No external inference or metrics exist. |
| Messidor-2 adjudicated five-grade evaluation | The Google Brain adjudicated grade CSV is checked and staged privately. ADCIS supplied restricted official access; all four image archive parts passed individual byte-size and SHA-256 checks. The reconstructed ZIP passed complete CRC; all 1,748 images were extracted, decoded, hashed and found byte-unique. The official pairing metadata is not present in the ZIP; no optional external metrics exist. |
| Accuracy, precision, recall, F1, QWK, confusion matrices, curves and image examples | Generators exist; none of these result artifacts can be produced honestly before a completed run and sealed evaluation. |
| Concise compiled LaTeX report | Result-bound generator exists and now embeds real, hash-checked EyePACS fundus image pairs after a completed run. The separate [illustrated implementation dossier](implementation_dossier.pdf) has been compiled and visually reviewed. No final experiment `.tex` or PDF result artifact exists because there are no model results. |

The latest training source adds an epoch-boundary pause option for resumable
Kaggle sessions. It has not run on GPU yet. The completed full-manifest scan
qualification is pinned to the preceding training source hash; a new
qualification is required after the final training/scan source is frozen.

The downloaded EyePACS label archive contains 35,126 original image labels:
25,810 grade 0; 2,443 grade 1; 5,292 grade 2; 873 grade 3; and 708 grade 4.
Its SHA-256 is
`981e7b444fd4613591c0141d55dedb8e6804fce5fae0c6bf9932e68a3726ac1d`.
The archive and its private provenance record are excluded from Git.
The official Kaggle CLI download of `train.zip.005.zip` contains exactly one
`train.zip.005` member of 1,434,013,506 bytes and passes a complete member CRC
check. Its wrapper SHA-256 is
`ec7b7c439e4ce774e8a0c1e7ec302fc179b538d5fdae2331f853879f86b9f5a1`.
Parts `.001` and `.002` were fully resumed and verified on 2026-11-04: each
contains one archive member of the official 8,388,608,000 byte size and passes
full CRC. Their wrapper SHA-256 values are, respectively,
`f6de0539655a0b84d8e4404098229fb4a061d5bb3b7f4334ab7d51d429aed360`
and `0c89268dd58fd40763e64a315bda906ace1b1fac8ac9c912b387429143cea997`.
Part `.003` passed full CRC with wrapper SHA-256
`5baed49539c8fffe48f735aac220cdda4b5e6c383b51d46962aedd9d5795775e`.
Part `.004` passed full CRC on 2026-11-04 with wrapper SHA-256
`69493d173c802fccb1b478f2c98cd2b9100078f7abaa3c773bb7d05fd29e3f88`.
All five local official wrappers are now verified. Raw local segment staging
was stopped after `.001` and `.002`; an incomplete `.003.partial` remains
private and is not counted as a valid raw segment. The private Kaggle source
mount independently supplied all five complete segments to the completed
preparation job.
The private Kaggle input mount independently exposes all five unwrapped
original training parts and `trainLabels.csv.zip`. Its byte-split ZIP index
contains 35,126 JPEG images totaling 37,944,223,074 uncompressed bytes. The
completed preparation job extracted, CRC-checked, decoded and SHA-256 hashed
each image, then wrote the manifest and immutable grade-stratified split. The
locally downloaded output was checked to contain 35,126 included images, zero
excluded, all five grades in each split, and zero filename-pair group overlaps.
These are source and split checks, not
trained-model measurements. A private Kaggle T4 x2 session loaded the reviewed
103,760,645-parameter source and reported PyTorch 2.11.0+cu130 and two CUDA
devices with 15,636,037,632 bytes each. A subsequent diagnostic on one T4
and 20 real EyePACS images passed the numerical scan gate and two configured
optimizer steps. Those steps took 191.35 and 191.24 seconds and peaked at
4,643,165,184 and 5,471,968,768 allocated CUDA bytes. This is model and
hardware qualification on a small real-image set, not a training checkpoint
or validation metric. The later full-manifest qualification passed on the
measured 24,590-image training split and recorded two eight-image optimizer
steps at 195.71 and 197.24 seconds on one T4. It is still a two-step
diagnostic, not a trained model or measured epoch. At the observed rate, the
configured 300-epoch full run is incompatible with Kaggle's roughly 30-hour
weekly GPU quota; see `docs/COMPUTE_PLAN.md` for the conditional estimate.
The label-only audit finds 17,563 filename pairing IDs, each with one left and one right
image; no label ID is duplicated or malformed. Its subject-level maximum-grade
counts are 12,155; 1,442; 3,022; 499; and 445 for grades 0--4. Image-to-label
reconciliation and the split completed in the private Kaggle preparation job.

The independently published Messidor-2 grades CSV has 1,748 unique image IDs;
1,744 are gradable (grades 0--4: 1,017; 270; 347; 75; 35), and four are
ungradable. Its SHA-256 is
`bb8bbd82eb4bfddc8a0eb9aa45d84274d814e913534f8027f6a4ff0a310fda72`.
The CSV and its private provenance record are excluded from Git.
The original Messidor manifest contains 1,200 unique TIFF records, all assigned
to external test: 1,187 eligible and 13 excluded under the published Base33
duplicate policy. The four published grade-correction targets were found and
applied. Eligible R0/R1/R2/R3 counts are 547/149/240/251. The manifest passed
`read_manifest` schema validation and has private SHA-256
`e1e186d911b0d1daca82e3100cb01b1b0ba4f07e7019346d8d6882051e1ba778`.
Its absolute source paths make that file digest specific to this private
storage layout. Comparing included image SHA-256 sets against the verified
35,126-row EyePACS split found zero exact cross-cohort duplicates. This does
not rule out near duplicates or shared patients and records data readiness,
not model generalization.
ADCIS describes Messidor-2 as 1,058 paired images from original Messidor
reencoded as PNG, plus 690 extension JPGs. It overlaps the required original
Messidor external cohort by design. A later optional Messidor-2 analysis must
state this relationship and must not count both cohorts as independent
replications. The verified ZIP has exactly those extension counts; all 1,058
PNG filename stems match names in the original Messidor TIFF manifest, and
none of the 690 JPG stems do. This filename audit does not itself prove pixel
identity. The official Messidor-2 page also requests LaTIM laboratory and
Messidor program acknowledgment.
Both verified label sources are staged in access-controlled private research
storage. Partial browser transfers cannot be extracted or counted as acquired
data. The official Kaggle CLI authenticated with a private token outside the
repository. No token or raw source image is included here.

The [official EyePACS competition rules](https://www.kaggle.com/competitions/diabetic-retinopathy-detection/rules)
state that, unless otherwise permitted by the competition website, the data
may be used solely for the competition's purpose and duration and must not be
provided to nonparticipants. A [competition host's answer on the official
forum](https://www.kaggle.com/c/diabetic-retinopathy-detection/discussion/141968)
allows research and publications with EyePACS credit and the
Cuadros--Bresnick (2009) citation. We infer that private educational training
falls within research use; the answer does not expressly grant source-image
redistribution or model-weight sharing. An exact-scope request was sent to
`contact@eyepacs.org` on 2026-11-03. The 2026-11-04 provider reply directed
the user to the public Kaggle dataset and repeated the attribution request,
without expressly resolving those output-sharing scopes.
ADCIS technical support responded on 2026-11-04 with restricted alternative
access after the request forms failed to deliver personal links. The support
message asked that its access link not be shared. The files and access details
remain outside this public repository. See the
[dated correspondence and acquisition timeline](SOURCE_ACCESS_TIMELINE.md).

## Remaining gates, in order

1. Keep research use within the competition host's public permission and
   document attribution. Confirm separate rights before distributing source
   images in the report or sharing model weights; the competition rules
   restrict redistribution.
2. Requalify the exact current FP16 training source and optional CUDA scan
   configuration with real EyePACS images. The prior BF16/parallel full-model
   qualification and low-level CUDA comparison are complete, but neither
   proves current whole-model throughput or distributed training. Determine
   whether a verified zero-cost allocation can finish the full run.
3. Run the exact-source and cross-cohort audit using the finished original
   Messidor manifest before sealed evaluation. The optional Messidor-2 image
   ZIP passed CRC; locate and verify the official pairing sheet before
   attempting its five-grade manifest. Do not substitute a mirrored or
   relabeled dataset.
4. Finish real-image quality review, distributed throughput qualification and
   an epoch-boundary resume check, then freeze the run configuration. The full
   EyePACS manifest and split are already prepared.
5. Train through the declared epochs; select the checkpoint using EyePACS
   validation only. Run sealed tests once, produce metrics/plots, compile the
   report, review the PDF, and check that sharing its images is authorized.

The existing short Kaggle notebook run is not evidence of model training: its
visible version ran for 8.2 seconds without a GPU or output files. No published
number or unit check is represented here as a study result.
