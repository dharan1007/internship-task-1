# Implementation and acceptance plan

The assignment source is the supplied internship task PDF.
Architecture reference: Zhu et al., *Vision Mamba*, arXiv:2401.09417v3.

## Fixed design decisions

- Implement the paper's bidirectional SSM plus separate reverse Conv1d, 16x16
  patch embedding, absolute position embedding, middle class token, stochastic
  depth and five
  logits. The primary full-size configuration is Vim-Base (24 blocks,
  embedding dimension 768, state dimension 16, expansion 2).
- No pretrained weights, ready-made Vim/Mamba models, dataset loaders, or
  synthetic retinal training/evaluation data.
- EyePACS filename left/right pairing tokens define split groups; those tokens
  do not independently verify patient identity. The final EyePACS test set is
  sealed until checkpoint and hyperparameters are fixed on validation.
- Original Messidor is evaluated with its own source-grade interpretation and
  a declared binary endpoint. Messidor-2 plus adjudicated five-grade labels is
  the five-class external benchmark. Both are test-only. Overlapping original
  Messidor images in Messidor-2 are reported as overlap.
- Image source files are retained. The input resolution and preprocessing are
  selected after real-image review and fixed before external evaluation.

## Build work

1. **Data provenance:** obtain official images/labels, verify source and
   licensing, create manifests, apply original Messidor errata, check exact
   duplicates and image decoding, record all exclusions.
2. **Splits:** group both eyes by the EyePACS filename pairing token; create immutable train,
   validation, and held-out test assignments; audit leakage and class counts.
3. **Model:** implement patch embedding, positional/class tokens, independent
   forward/reverse selective SSMs, stable discretization, gating, residual
   stack, final norm and classifier.
4. **Scan:** implement a PyTorch mathematical reference and a from-scratch
   affine associative scan with explicit reverse recurrence for gradients.
   Compile its parallel scan on CUDA. Verify forward, backward, precision,
   memory and throughput against the reference on the actual accelerator.
5. **Training:** run the full dataset from random initialization, save
   resumable complete checkpoints, select one checkpoint by the declared
   validation rule. Handle failed batches and nonfinite values explicitly.
6. **Evaluation:** run sealed EyePACS and both external cohorts once after
   selection; generate per-class metrics, QWK where valid, confusion matrices,
   predictions, loss curves and failure analysis.
7. **Report:** compile a short LaTeX report with exact method, data counts,
   hyperparameters, plots, real examples, references and limitations.

## Completion gates

- A source review confirms every paper block component and parameterization.
- Mathematical and parallel scans agree on real inputs and gradients; the
  parallel scan meets the measured throughput and memory budget on the
  selected training accelerator. The configured model, optimizer and data
  loader also complete two real-image optimizer steps on that accelerator.
- Data manifests reconcile exactly with label rows and available files;
  duplicate and patient overlap audits are clean or explicitly accounted for.
- Training completes and the checkpoint can resume with optimizer, scheduler,
  scaler, RNG and configuration state.
- Validation selects the checkpoint before any test dataset is opened by the
  evaluation command.
- Test metrics, plots and example predictions are generated from saved
  predictions. No missing or invalid metric is silently replaced by zero.
- LaTeX source compiles; final PDF is visually reviewed. Dataset and image
  usage in the shared report comply with the applicable terms.

No result gate can pass until the official data and full training environment
are available. A source-only build or a unit test is not a finished submission.
