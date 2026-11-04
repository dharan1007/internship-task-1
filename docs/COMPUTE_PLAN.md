# Full-scale training environment and run handoff

This plan deliberately does not use the developer laptop. The user requires
zero spending, so no paid provider, machine, storage purchase, or trial that
can charge after credits expire is authorized. A private Kaggle T4 x2 session
has been verified with the official EyePACS competition input and reviewed
model source. A diagnostic on 20 real EyePACS images passed the numerical scan
gate and two full-model optimizer steps at the configured image size and
microbatch. The complete 35,126-image manifest and immutable split are now
prepared. A separate full-manifest qualification on one T4 passed the
numerical gate and two optimizer steps on 2026-11-04; it has not measured a
complete training epoch or distributed throughput.
The larger candidate below describes the compute class to seek from a no-cost
institutional or research allocation if Kaggle proves insufficient; no such
larger allocation has been verified.

## Measured Kaggle feasibility, 2026-11-04

The private [real-image diagnostic](https://www.kaggle.com/code/dharantejreddy/drvim-full-model-feasibility-on-real-eyepacs)
ran reviewed source commit `10b4139` on one Tesla T4 with original EyePACS
images (four unique images from each grade). The exact 1536-pixel, 24-block,
103,760,645-parameter configuration passed FP32 and bf16 scan parity and two
8-image optimizer steps. The two forward/backward/optimizer times were 191.35
and 191.24 seconds; peak CUDA allocations were 4.64 and 5.47 GB of the
15.64 GB device. These are diagnostic step measurements, not training results.
The 300-epoch schedule is this project's configuration; the internship brief
does not mandate that epoch count.
The later full-manifest qualification used the exact pinned split digest
`f933f1bcff8f99688fd8a44984acaa4ce0c929d899be9bddcabdfc54ea7c2117`.
It passed FP32 and bf16 numerical gates and two eight-image optimizer steps
at 195.71 and 197.24 seconds, with 4.64 and 5.47 GB peak CUDA allocation.
The second batch contained only grade-0 images; its low loss is not evidence
of multi-grade learning. The result file is retained in the private
qualification notebook output.

At the measured 24,590-image training split and eight images per step, the
full-manifest diagnostic step rate projects to about 168 GPU-hours per epoch.
Even with an unverified perfect 2x gain from both T4s, it would take about
84 wall-hours per epoch and 25,200 hours for 300 epochs, before validation,
I/O and checkpoints. This estimate does not claim measured
distributed throughput. Kaggle's
[GPU guidance](https://www.kaggle.com/docs/efficient-gpu-usage) describes an
approximately 30-hour weekly quota, so the current implementation and free
allocation do not support completing this run. A faster exact scan and its
measured qualification, or a substantially larger verified no-cost allocation,
are required before committing to the full experiment.

## Candidate provision

- A single Linux multi-GPU node with eight accelerator GPUs in the 80 GB
  memory class (H100 or equivalent). This is a **candidate**, not a statement
  that the 1536-pixel Vim-Base run fits or reaches a particular speed.
- Fast local NVMe scratch for extracted source images and manifests; budget at
  least 2 TB before provisioning and confirm with the actual archive sizes.
  Set `--run-dir` on a private durable mounted volume so each epoch checkpoint
  persists independently of the ephemeral compute node.
- A pinned CUDA/PyTorch 2.11+ container image. Record its digest, GPU driver,
  CUDA runtime, PyTorch and dependency versions before the run. Keep data and
  result buckets private; use short-lived credentials and delete them from
  logs and job scripts.
- The initial configuration uses 1536-pixel full-field inputs, 24-block
  Vim-Base, two images per GPU, four accumulation steps and eight GPUs for a
  nominal global batch of 64. This is adjustable **before** full training
  after real-image and scan qualification; a changed config gets a new hash
  and a new run directory.

## Provisioning decision gate

1. Obtain official EyePACS access and load the source archives into a
   permitted private environment. Do not copy licensed data into Git,
   public storage or an unapproved third-party service.
2. Run the EyePACS manifest preparation and filename-pair-grouped split. This
   is complete for all 35,126 images, with zero excluded. Independently verify
   patient identity if a patient-level claim will be made; the public filename
   tokens alone do not prove it. Freeze the input policy.
3. Provision one time-limited GPU qualification session, not the entire
   multi-week training allocation. Run `drvim qualify-scan` on an actual
   EyePACS training image at the intended resolution. Record full-sequence
   FP32 parity, short-sequence FP32/bf16 gradient parity, peak memory and
   measured forward/backward time.
4. The same `drvim qualify-scan` command now runs two complete full-model
   optimizer steps on distinct real EyePACS images with the configured
   microbatch, accumulation, loader workers, mixed precision and checkpointing.
   Inspect both recorded losses, gradient norms, data-loading times, GPU step
   times and peak memory. This single-GPU qualification does not measure DDP
   communication; validate multi-GPU throughput separately before estimating
   the full run from **measured** images/second. Confirm the no-cost allocation
   covers the complete run and retained storage.
5. Only within that confirmed zero-cost allocation, run the full training job
   with its run directory on durable private storage. Resume only with the same
   world size, source hashes, model source, config and qualification artifact.
6. After the full-run marker exists, prepare external cohorts, audit, evaluate
   once and generate the LaTeX/PDF report. Preserve test predictions and
   source manifests with the final checkpoint.

## Operations and failure handling

The training command logs each epoch's loss, validation QWK and duration.
Epoch-boundary checkpoints include optimizer, scheduler, scaler and RNG state.
The training CLI now supports `--stop-after-epoch N` as an absolute,
operational session boundary. It writes the usual resumable checkpoint and
history at that epoch, then exits without a `training-complete.json` marker;
the next session uses `--resume` with the identical config, manifest, source,
world size and qualification artifact. This option does not make an epoch
shorter: a measured epoch must fit within the provider's session limit with
time left for validation and checkpoint upload before using it.
The best validation checkpoint is separate from the latest resumable one.
The run aborts on changed files, hash mismatches, nonfinite loss or gradients,
unsupported bf16 and missing scan qualification. A failed epoch resumes from
the preceding completed checkpoint after its cause is fixed; a changed data
or model policy starts a new experiment. The final result must be based on the
sealed held-out EyePACS and external cohorts, with no metric fabricated from
logs or paper baselines.
