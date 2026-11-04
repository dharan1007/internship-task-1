"""CUDA qualification on actual EyePACS training pixels, not generated inputs."""

from __future__ import annotations

import json
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import yaml
from torch import nn
from torch.nn import functional as F

from .data.dataset import RetinaDataset
from .data.manifest import sha256_file
from .identity import pipeline_source_hashes
from .model.scan import selective_scan
from .model.vim import VimConfig, VisionMamba
from .train import _loader, _optimizer


def _copy_inputs(values: tuple[torch.Tensor, ...], tokens: int | None = None
                 ) -> list[torch.Tensor]:
    result = []
    for value in values:
        sliced = value[:, :tokens] if tokens is not None and value.ndim == 3 else value
        result.append(sliced.detach().clone().requires_grad_())
    return result


def _qualify_full_model_step(config: dict, model_config: VimConfig,
                             manifest_path: Path, device: torch.device) -> dict:
    """Measure two real-image optimizer steps with the exact training microbatch."""
    data = config["data"]
    training = config["train"]
    batch_size = int(data["batch_size"])
    accumulation = int(training["grad_accum_steps"])
    workers = int(data["workers"])
    if batch_size <= 0 or accumulation <= 0 or workers < 0:
        raise ValueError("Invalid training batch, accumulation or worker count")
    amp = training["amp"]
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "none": None}[amp]
    if amp_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError("Configured bf16 is not native on the qualification GPU")
    dataset = RetinaDataset(manifest_path, source="eyepacs", split="train",
                            image_size=model_config.image_size, training=True,
                            horizontal_flip=True)
    if len(dataset) < 2 * batch_size * accumulation:
        raise ValueError("Two complete real-image optimizer steps require more training images")
    loader = _loader(dataset, batch_size, workers, bool(data["pin_memory"]))
    iterator = iter(loader)
    model = VisionMamba(model_config).to(device).train()
    optimizer = _optimizer(model, float(training["learning_rate"]),
                           float(training["weight_decay"]))
    initial_scale = float(training.get("grad_scaler_init_scale", 65536.0))
    if not 0 < initial_scale < float("inf"):
        raise ValueError("grad_scaler_init_scale must be finite and positive")
    scaler = torch.amp.GradScaler("cuda", init_scale=initial_scale, enabled=amp == "fp16")
    measured: list[dict] = []
    for step in range(2):
        batches = []
        load_start = time.perf_counter()
        for _ in range(accumulation):
            batch = next(iterator)
            if len(batch["image_id"]) != batch_size:
                raise RuntimeError("Qualification received a partial microbatch")
            batches.append(batch)
        load_seconds = time.perf_counter() - load_start
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        losses = []
        for batch in batches:
            images = batch["image"].to(device, non_blocking=True)
            grades = batch["grade"].to(device, non_blocking=True)
            context = (torch.autocast(device_type="cuda", dtype=amp_dtype)
                       if amp_dtype else nullcontext())
            with context:
                logits = model(images)
                loss = F.cross_entropy(logits.float(), grades) / accumulation
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite full-model qualification loss")
            scaler.scale(loss).backward()
            losses.append(loss.detach())
        scaler.unscale_(optimizer)
        gradient_norm = nn.utils.clip_grad_norm_(
            model.parameters(), float(training["max_grad_norm"]), error_if_nonfinite=True)
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError("Nonfinite full-model qualification gradient")
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize(device)
        step_seconds = time.perf_counter() - start
        measured.append({
            "optimizer_step": step + 1,
            "image_ids": [image_id for batch in batches for image_id in batch["image_id"]],
            "loss": float(torch.stack(losses).sum()),
            "gradient_norm": float(gradient_norm),
            "loss_scale": scaler.get_scale() if amp == "fp16" else None,
            "data_loader_seconds": load_seconds,
            "forward_backward_optimizer_seconds": step_seconds,
            "peak_cuda_bytes": torch.cuda.max_memory_allocated(device),
        })
    return {
        "passed_full_model_step": True,
        "microbatch_size": batch_size,
        "gradient_accumulation_steps": accumulation,
        "amp": amp,
        "grad_scaler_init_scale": initial_scale if amp == "fp16" else None,
        "activation_checkpointing": model_config.activation_checkpointing,
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()
                                    if parameter.requires_grad),
        "gpu_total_bytes": torch.cuda.get_device_properties(device).total_memory,
        "full_model_steps": measured,
    }


def qualify_scan(config_path: Path, manifest_path: Path, output: Path,
                 *, gradient_tokens: int = 128) -> None:
    if output.exists():
        raise FileExistsError(f"Scan qualification already exists: {output}")
    if not torch.cuda.is_available():
        raise RuntimeError("Scan qualification requires the selected CUDA training accelerator")
    if gradient_tokens <= 0:
        raise ValueError("gradient_tokens must be positive")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model_config = VimConfig(**config["model"])
    if model_config.scan_backend not in {"parallel", "cuda_kernel"}:
        raise ValueError("Scan qualification requires an explicit accelerated backend")
    dataset = RetinaDataset(manifest_path, source="eyepacs", split="train",
                            image_size=model_config.image_size)
    record = dataset[0]
    device = torch.device("cuda")
    torch.manual_seed(int(config["seed"]))
    torch.cuda.manual_seed_all(int(config["seed"]))
    model = VisionMamba(model_config).to(device).eval()
    image = record["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        tokens = model.patch_embed(image).flatten(2).transpose(1, 2)
        position = model.cls_position
        tokens = torch.cat((tokens[:, :position], model.cls_token,
                            tokens[:, position:]), dim=1) + model.pos_embed
        projected, _ = model.blocks[0].in_proj(model.blocks[0].norm(tokens)).chunk(2, dim=-1)
        values = model.blocks[0].forward_direction.scan_inputs(projected)
    short_reference = _copy_inputs(values, min(gradient_tokens, values[0].shape[1]))
    short_candidate = _copy_inputs(values, min(gradient_tokens, values[0].shape[1]))
    reference = selective_scan(*short_reference, backend="reference")
    candidate = selective_scan(*short_candidate, backend=model_config.scan_backend)
    forward_error = float((reference.float() - candidate.float()).abs().max())
    reference_grads = torch.autograd.grad(reference.float().sum(), short_reference)
    candidate_grads = torch.autograd.grad(candidate.float().sum(), short_candidate)
    gradient_error = max(float((left.float() - right.float()).abs().max())
                         for left, right in zip(reference_grads, candidate_grads))
    mixed_forward_error = None
    mixed_gradient_error = None
    mixed_kind = config["train"]["amp"]
    mixed_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "none": None}[mixed_kind]
    if mixed_dtype is not None:
        if mixed_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported(including_emulation=False):
            raise RuntimeError("Configured bf16 is not native on the qualification GPU")
        mixed_values = (values[0].to(mixed_dtype), values[1], values[2],
                        values[3].to(mixed_dtype), values[4].to(mixed_dtype), values[5])
        mixed_reference = _copy_inputs(mixed_values, min(gradient_tokens, values[0].shape[1]))
        mixed_candidate = _copy_inputs(mixed_values, min(gradient_tokens, values[0].shape[1]))
        mixed_ref_output = selective_scan(*mixed_reference, backend="reference")
        mixed_candidate_output = selective_scan(*mixed_candidate, backend=model_config.scan_backend)
        mixed_forward_error = float((mixed_ref_output.float() - mixed_candidate_output.float()).abs().max())
        mixed_ref_grads = torch.autograd.grad(mixed_ref_output.float().sum(), mixed_reference)
        mixed_candidate_grads = torch.autograd.grad(
            mixed_candidate_output.float().sum(), mixed_candidate)
        mixed_gradient_error = max(float((left.float() - right.float()).abs().max())
                                   for left, right in zip(mixed_ref_grads, mixed_candidate_grads))
    with torch.no_grad():
        full_reference = selective_scan(*values, backend="reference")
        full_candidate = selective_scan(*values, backend=model_config.scan_backend)
        full_forward_error = float((full_reference.float() - full_candidate.float()).abs().max())
    del full_reference, full_candidate, reference, candidate
    torch.cuda.empty_cache()
    # Compilation has already occurred above. Measure a complete steady-state
    # forward/backward of one real-direction sequence at the intended resolution.
    full_inputs = _copy_inputs(values)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    output_values = selective_scan(*full_inputs, backend=model_config.scan_backend)
    output_values.float().sum().backward()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    tolerance = 2e-3
    mixed_tolerance = {"bf16": 5e-2, "fp16": 1e-2, "none": None}[mixed_kind]
    report = {
        "training_image_id": record["image_id"],
        "seed": int(config["seed"]),
        "manifest_sha256": sha256_file(manifest_path),
        "config_sha256": sha256_file(config_path),
        "pipeline_source_sha256": pipeline_source_hashes(),
        "gpu": torch.cuda.get_device_name(device),
        "torch_version": str(torch.__version__),
        "scan_backend": model_config.scan_backend,
        "tokens": int(values[0].shape[1]),
        "gradient_comparison_tokens": int(short_reference[0].shape[1]),
        "short_forward_max_abs_error": forward_error,
        "short_gradient_max_abs_error": gradient_error,
        "full_forward_max_abs_error": full_forward_error,
        "mixed_precision_dtype": mixed_kind,
        "mixed_short_forward_max_abs_error": mixed_forward_error,
        "mixed_short_gradient_max_abs_error": mixed_gradient_error,
        "steady_state_direction_forward_backward_seconds": elapsed,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
        "absolute_tolerance": tolerance,
        "mixed_absolute_tolerance": mixed_tolerance,
        "passed_numerical_gate": (
            max(forward_error, gradient_error, full_forward_error) <= tolerance and
            (mixed_forward_error is None or
             max(mixed_forward_error, mixed_gradient_error) <= mixed_tolerance)
        ),
    }
    if report["passed_numerical_gate"]:
        del model, image, tokens, projected, values, short_reference, short_candidate
        del reference_grads, candidate_grads, full_inputs, output_values
        if mixed_forward_error is not None:
            del mixed_values, mixed_reference, mixed_candidate, mixed_ref_output
            del mixed_candidate_output, mixed_ref_grads, mixed_candidate_grads
        torch.cuda.empty_cache()
        report.update(_qualify_full_model_step(config, model_config, manifest_path, device))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not report["passed_numerical_gate"]:
        raise AssertionError(f"Accelerated scan failed real-data numerical gate: {output}")
