# Research protocol and exact deliverables

This is the design protocol for the supplied internship task PDF.
The reference is [Vision Mamba](https://arxiv.org/abs/2401.09417), the authors'
[model code](https://github.com/hustvl/Vim/blob/main/vim/models_mamba.py), and
[BiMamba-v2 mixer](https://github.com/hustvl/Vim/blob/main/mamba-1p1p1/mamba_ssm/modules/mamba_simple.py).

## Submission inventory

| Artifact | Required purpose | Current verified state |
| --- | --- | --- |
| PyTorch Vim source | Build the architecture from scratch | From-scratch model and scan implemented; the earlier BF16/parallel source passed two real-image full-model optimizer steps, while current FP16/optional CUDA edits remain unqualified as a full model |
| Custom Dataset/DataLoader | Load and preprocess EyePACS in PyTorch | Own Dataset and preprocessing implemented; PyTorch DataLoader used for batching; all 35,126 original EyePACS images audited in private Kaggle preparation |
| Full EyePACS training and held-out evaluation | Produce real five-grade results | All five official EyePACS image parts and labels verified; immutable split prepared; full training has not started; zero-cost throughput and output-sharing rights remain unresolved |
| Original Messidor external evaluation | Evaluate the separate test cohort named in the PDF | Official images and annotations acquired and reconciled in private storage; the declared binary endpoint is implemented, but model evaluation is pending |
| Optional Messidor-2 adjudicated five-grade evaluation | Add a five-grade external comparison beyond the PDF's original Messidor requirement | Official images and independent grades acquired and checked privately; the official pairing sheet and model evaluation are pending |
| Metrics, confusion matrices, curves, fundus image--prediction pairs | Supply the PDF's quantitative and visual evidence | Generated only from actual checkpoint, manifests, and predictions |
| Concise compiled LaTeX report | Explain method, data, training, results, conclusions, references | Generator implemented; final PDF requires real results |

The brief does not ask for a clinical service, mobile app, web dashboard,
model API, lesion segmentation, heatmaps, fundus image redistribution, or a
pretrained baseline. None substitutes for the listed experiment. A source-only
build or metrics from unrelated papers is not a final submission.

## Fixed scientific choices

1. **Labels.** EyePACS uses ICDR 0--4. Original Messidor uses R0--R3 under
   different criteria, so numerical five-class QWK against it would be
   invalid. Original Messidor reports R2/R3 versus model ICDR >=2 as a
   source-specific binary proxy. Messidor-2 uses separately released Google
   Brain adjudicated ICDR 0--4 labels. Ungradable images have a recorded
   exclusion and are not assigned a fabricated grade.
2. **Split.** Group EyePACS left and right eyes by their filename pairing token before seeded
   grade-stratified 70/15/15 training/validation/test assignment. The manifest
   and split are immutable. Model decisions use training and validation only.
   The public filename token is not independent proof of a unique patient;
   avoid calling this a clinically verified patient-level split.
   External test cohorts are prepared and evaluated only after the full-run
   marker exists. Their archives can be obtained earlier but must stay sealed.
3. **Images.** Preserve full field, EXIF orientation and aspect ratio; pad to
   square instead of cropping lesions. The initial primary input is 1536
   square with 16-square patches, subject to real-image review and accelerator
   qualification before the full run. Do not create lossy JPEG derivatives.
   Horizontal flips apply only during training. Inspect real examples for
   border, laterality, black space and small-lesion visibility before freezing.
4. **Architecture.** 24 blocks, width 768, state dimension 16, expansion 2,
   causal depthwise convolution width 4 in each direction, separate reverse
   convolution/SSM parameters, shared input gate, directional sum as in the
   paper's Algorithm 1, middle class token, absolute position embedding,
   RMSNorm, FP32 residual and 0--0.1
   stochastic depth. The affine selective scan and gradients are written here.
   The parallel implementation retains only chunk-boundary states and
   recomputes each chunk during backward, following the paper's state
   recomputation principle. It is a PyTorch implementation, not the paper's
   fused SRAM kernel; measured CUDA memory and speed remain qualification gates.
   No pretrained weights or ready-made Vim model enter training. The primary
   parallel scan is implemented here. An optional, pinned low-level CUDA scan
   kernel is under separate qualification and is not a trained result.
5. **Training.** Cross-entropy, AdamW with excluded norm/SSM/position decay,
   warmup then cosine learning-rate schedule, configured bf16 or fp16 after
   accelerator qualification, gradient
   clipping, accumulation, activation checkpointing and `torchrun` DDP.
   Epoch checkpoints save model, optimizer, scheduler, scaler, RNG, config and
   manifest identity. Highest EyePACS validation QWK chooses the checkpoint.
   Training must complete the declared epochs before test evaluation.
6. **Evaluation.** Run the selected checkpoint once on all three test
   cohorts. Save each included image's true/predicted grade and five
   probabilities. Derive metrics and plots from those records, with source
   counts, exclusions, per-class support, exact confusion counts and observed
   correct/incorrect pairs. Original Messidor and Messidor-2 overlap; do not
   interpret them as independent tests.

## Mandatory preflight on the chosen training environment

- Obtain official sources, preserve archive checksums and access terms, and
  keep raw images out of Git and public artifacts.
- Reconcile every label to exactly one file and fully decode every included
  image. Apply ADCIS's four published grade corrections and account for all
  13 published Base33 duplicate pairs. The implementation excludes one copy
  of each pair, yielding at most 1,187 included images, but one retained pair
  has conflicting retinopathy grades. That pair needs a declared sensitivity
  analysis or adjudication before a clean-label primary estimate is claimed.
  Review extra duplicates or suspicious near matches.
- If an EyePACS image is genuinely undecodable, record its image ID and a
  reviewed reason code in a private exclusion ledger. Its source file and label
  remain accounted for; the preparer never omits it silently.
- Inspect real image quality and grades. Verify five classes in each EyePACS
  split and no subject or exact-hash cross-split overlap.
- Compare reference and parallel scan outputs/gradients under FP32 and bf16
  on CUDA real batches. Measure peak memory, step time and throughput. Adjust
  microbatch and GPU count only before the full run, then freeze the config.
- Resume an epoch-boundary checkpoint on the same world size. Verify sealed
  evaluation rejects changed manifests, incomplete runs and existing results.
  Review generated report against the saved prediction rows.

## Final acceptance

The final handoff includes source, locked environment, source provenance,
manifest audit, immutable split, completed checkpoints, training history,
sealed predictions, metrics, plots, LaTeX source and compiled PDF. Record the
container image, CUDA driver, GPU model, world size, archive/manifest hashes,
selected epoch and run duration. A reviewer should be able to recompute each
metric from the prediction CSVs. The report must contain real fundus images
tied to those rows and a private source-hash ledger. Do not distribute retinal
images in the report until applicable source terms and consent are confirmed.
The visual example panels use EyePACS only; original Messidor's image-copy
prohibition rules out embedding its images in a shared PDF.
