"""Full EyePACS training, validation-only selection, and epoch checkpoints."""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from .data.dataset import RetinaDataset
from .data.manifest import read_manifest, sha256_file, verify_file_hashes
from .identity import pipeline_source_hashes
from .metrics import five_class_metrics
from .model.vim import VimConfig, VisionMamba


class ExactContributionDistributedSampler(DistributedSampler):
    """Pad ranks for DDP synchronization without counting repeated images as training data."""

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        indices = (torch.randperm(len(self.dataset), generator=generator).tolist()
                   if self.shuffle else list(range(len(self.dataset))))
        real_count = len(indices)
        padding = self.total_size - real_count
        if padding:
            indices.extend((indices * ((padding + real_count - 1) // real_count))[:padding])
        weights = [1.0] * real_count + [0.0] * padding
        return iter(zip(indices[self.rank:self.total_size:self.num_replicas],
                        weights[self.rank:self.total_size:self.num_replicas]))


def _atomic_torch_save(value: object, destination: Path) -> None:
    temporary = destination.with_name(destination.name + ".partial")
    torch.save(value, temporary)
    os.replace(temporary, destination)


def _rng_state() -> dict[str, object]:
    numpy_state = np.random.get_state()
    return {"python": random.getstate(),
            "numpy": (numpy_state[0], numpy_state[1].tolist(), int(numpy_state[2]),
                      int(numpy_state[3]), float(numpy_state[4])),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def _restore_rng(state: dict[str, object]) -> None:
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32),
                         int(numpy_state[2]), int(numpy_state[3]), float(numpy_state[4])))
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def _optimizer(model: nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (no_decay if parameter.ndim < 2 or name.endswith(("a_log", "skip")) or
         name in {"cls_token", "pos_embed"} else decay).append(parameter)
    return torch.optim.AdamW([{"params": decay, "weight_decay": weight_decay},
                              {"params": no_decay, "weight_decay": 0.0}], lr=lr)


def _loader(dataset: RetinaDataset, batch_size: int, workers: int, pin_memory: bool,
            *, sampler: DistributedSampler | None = None, shuffle: bool = False) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, num_workers=workers,
                      pin_memory=pin_memory, persistent_workers=False,
                      sampler=sampler, shuffle=shuffle and sampler is None, drop_last=False)


@torch.inference_mode()
def _validate(model: nn.Module, loader: DataLoader, device: torch.device,
              amp_dtype: torch.dtype | None) -> dict[str, object]:
    model.eval()
    truth, predicted = [], []
    loss_sum = 0.0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["grade"].to(device, non_blocking=True)
        context = torch.autocast(device_type=device.type, dtype=amp_dtype) if amp_dtype else nullcontext()
        with context:
            logits = model(images)
            loss = F.cross_entropy(logits.float(), labels)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite validation loss")
        loss_sum += float(loss) * len(labels)
        truth.extend(labels.cpu().tolist())
        predicted.extend(logits.argmax(-1).cpu().tolist())
    metrics = five_class_metrics(truth, predicted)
    metrics["loss"] = loss_sum / len(truth)
    return metrics


def train(config_path: Path, manifest_path: Path, qualification_path: Path,
          run_dir: Path, *, resume: bool = False,
          stop_after_epoch: int | None = None) -> None:
    config_bytes = config_path.read_bytes()
    config = yaml.safe_load(config_bytes)
    if config["data"]["image_policy"] != "preserve_full_field":
        raise ValueError("Unreviewed image preprocessing policy")
    if config["train"]["validation_metric"] != "qwk":
        raise ValueError("Checkpoint selection is fixed to validation QWK")
    model_config = VimConfig(**config["model"])
    frame = read_manifest(manifest_path)
    if not (frame["source"] == "eyepacs").all():
        raise ValueError("Training manifest must contain EyePACS only")
    if set(frame.loc[frame["included"].astype(bool), "split"]) != {"train", "validation", "test"}:
        raise ValueError("EyePACS manifest must contain sealed train, validation and test splits")
    rank = int(os.environ.get("RANK", "0"))
    if rank == 0:
        verify_file_hashes(frame.loc[frame["split"].isin(["train", "validation"])])
    if not torch.cuda.is_available():
        raise RuntimeError("Full training requires a CUDA training environment")
    qualification = json.loads(qualification_path.read_text(encoding="utf-8"))
    if not qualification["passed_numerical_gate"]:
        raise ValueError("Parallel scan has not passed its numerical gate")
    if not qualification.get("passed_full_model_step"):
        raise ValueError("The configured full-model real-image optimizer step has not passed")
    if (qualification["config_sha256"] != sha256_file(config_path) or
            qualification["manifest_sha256"] != sha256_file(manifest_path)):
        raise ValueError("Scan qualification used another config or manifest")
    if qualification["pipeline_source_sha256"] != pipeline_source_hashes():
        raise ValueError("Model or preprocessing source changed after scan qualification")
    if qualification["tokens"] != (model_config.image_size // model_config.patch_size) ** 2 + 1:
        raise ValueError("Scan qualification used a different token length")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    if distributed:
        dist.init_process_group("nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("cuda", torch.cuda.current_device())
    runtime = {"torch_version": str(torch.__version__),
               "gpu_name": torch.cuda.get_device_name(device), "world_size": world_size}
    if runtime["gpu_name"] != qualification["gpu"] or runtime["torch_version"] != qualification["torch_version"]:
        raise ValueError("Training GPU or PyTorch version differs from scan qualification")
    if rank == 0:
        run_dir.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()
    current_path = run_dir / "checkpoint-current.pt"
    best_path = run_dir / "checkpoint-best.pt"
    if not resume and (current_path.exists() or best_path.exists()):
        raise FileExistsError("Run already has checkpoints; use --resume or a new run directory")
    if resume and (run_dir / "training-complete.json").exists():
        raise RuntimeError("Run is already complete; do not resume a sealed experiment")
    seed = int(config["seed"])
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)
    settings = config["data"]
    if int(settings["batch_size"]) <= 0 or int(settings["workers"]) < 0:
        raise ValueError("Invalid batch size or worker count")
    train_set = RetinaDataset(manifest_path, source="eyepacs", split="train",
                              image_size=model_config.image_size, training=True,
                              horizontal_flip=True)
    val_set = (RetinaDataset(manifest_path, source="eyepacs", split="validation",
                             image_size=model_config.image_size) if rank == 0 else None)
    sampler = ExactContributionDistributedSampler(train_set, num_replicas=world_size,
                                                   rank=rank, shuffle=True, seed=seed)
    if not distributed:
        sampler = None
    train_loader = _loader(train_set, int(settings["batch_size"]), int(settings["workers"]),
                           bool(settings["pin_memory"]), sampler=sampler, shuffle=True)
    val_loader = (_loader(val_set, int(settings["batch_size"]), int(settings["workers"]),
                          bool(settings["pin_memory"])) if val_set is not None else None)
    model = VisionMamba(model_config).to(device)
    training = config["train"]
    optimizer = _optimizer(model, float(training["learning_rate"]), float(training["weight_decay"]))
    total_epochs = int(training["epochs"])
    warmup = int(training["warmup_epochs"])
    if total_epochs <= 0 or warmup < 0 or warmup >= total_epochs:
        raise ValueError("Invalid epoch/warmup schedule")
    if stop_after_epoch is not None and not 1 <= stop_after_epoch <= total_epochs:
        raise ValueError("stop_after_epoch must be within the configured training schedule")
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda epoch: (epoch + 1) / max(1, warmup) if epoch < warmup else
        0.5 * (1 + np.cos(np.pi * (epoch - warmup) / (total_epochs - warmup)))
    )
    amp = training["amp"]
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "none": None}[amp]
    if amp_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError("Requested bf16 is not native on this accelerator")
    initial_scale = float(training.get("grad_scaler_init_scale", 65536.0))
    if not np.isfinite(initial_scale) or initial_scale <= 0:
        raise ValueError("grad_scaler_init_scale must be finite and positive")
    max_consecutive_overflows = int(training.get("max_consecutive_overflows", 8))
    if max_consecutive_overflows < 1:
        raise ValueError("max_consecutive_overflows must be positive")
    scaler = torch.amp.GradScaler("cuda", init_scale=initial_scale, enabled=amp == "fp16")
    start_epoch, best_qwk, best_epoch, history, previous_elapsed = 0, float("-inf"), 0, [], 0.0
    total_overflows, consecutive_overflows = 0, 0
    config_sha = hashlib.sha256(config_bytes).hexdigest()
    manifest_sha = sha256_file(manifest_path)
    if resume:
        if not current_path.is_file():
            raise FileNotFoundError("No checkpoint-current.pt to resume")
        checkpoint = torch.load(current_path, map_location="cpu", weights_only=True)
        if (checkpoint["config_sha256"] != config_sha or
                checkpoint["manifest_sha256"] != manifest_sha or
                checkpoint["qualification_sha256"] != sha256_file(qualification_path)):
            raise ValueError("Configuration, manifest or scan qualification changed")
        if checkpoint["world_size"] != world_size:
            raise ValueError("Exact resume requires the original world size")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = checkpoint["epoch"] + 1
        best_qwk = checkpoint["best_qwk"]
        best_epoch = checkpoint["best_epoch"]
        history = checkpoint["history"]
        previous_elapsed = float(checkpoint.get("elapsed_seconds", 0.0))
        total_overflows = int(checkpoint.get("total_gradient_overflows", 0))
        consecutive_overflows = int(checkpoint.get("consecutive_gradient_overflows", 0))
        _restore_rng(checkpoint["rng_by_rank"][rank])
    if stop_after_epoch is not None and stop_after_epoch <= start_epoch:
        raise ValueError("Requested session boundary was already completed")
    if distributed:
        model = DistributedDataParallel(model, device_ids=[device.index])
    raw_model = model.module if distributed else model
    accumulation = int(training["grad_accum_steps"])
    if accumulation <= 0:
        raise ValueError("grad_accum_steps must be positive")
    run_start = time.monotonic()
    paused = False
    for epoch in range(start_epoch, total_epochs):
        epoch_start = time.monotonic()
        model.train()
        if sampler is not None:
            sampler.set_epoch(epoch)
        optimizer.zero_grad(set_to_none=True)
        local_loss_sum, local_count = 0.0, 0
        window_loss_sum, window_count = 0.0, 0
        epoch_overflows = 0
        for step, batch in enumerate(train_loader):
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["grade"].to(device, non_blocking=True)
            final_microbatch = (step + 1) % accumulation == 0 or step + 1 == len(train_loader)
            sync_context = (model.no_sync() if distributed and not final_microbatch else nullcontext())
            with sync_context:
                context = torch.autocast(device_type="cuda", dtype=amp_dtype) if amp_dtype else nullcontext()
                with context:
                    logits = model(images)
                    per_image_loss = F.cross_entropy(logits.float(), labels, reduction="none")
                if not torch.isfinite(per_image_loss).all():
                    raise FloatingPointError(f"Nonfinite train loss at epoch={epoch}, step={step}")
                weights = batch["sample_weight"].to(device, dtype=per_image_loss.dtype)
                weighted_loss = (per_image_loss * weights).sum()
                # DDP averages rank gradients. Restore the global mean over
                # real images in this accumulation window; padded slots weigh zero.
                window_first = (step // accumulation) * accumulation
                window_last = min(window_first + accumulation, len(train_loader))
                first_global = window_first * train_loader.batch_size * world_size
                window_capacity = (window_last - window_first) * train_loader.batch_size * world_size
                window_examples = min(window_capacity, len(train_set) - first_global)
                if window_examples <= 0:
                    raise RuntimeError("Distributed accumulation window has no real images")
                scaler.scale(weighted_loss * (world_size / window_examples)).backward()
            window_loss_sum += float(weighted_loss.detach())
            window_count += int(weights.sum().item())
            if final_microbatch:
                scaler.unscale_(optimizer)
                gradient_norm = nn.utils.clip_grad_norm_(raw_model.parameters(),
                                                           float(training["max_grad_norm"]),
                                                           error_if_nonfinite=False)
                if not torch.isfinite(gradient_norm):
                    if amp != "fp16":
                        raise FloatingPointError("Nonfinite gradient norm")
                    old_scale = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    new_scale = scaler.get_scale()
                    if new_scale >= old_scale:
                        raise FloatingPointError("Nonfinite FP16 gradient did not reduce loss scale")
                    total_overflows += 1
                    epoch_overflows += 1
                    consecutive_overflows += 1
                    optimizer.zero_grad(set_to_none=True)
                    window_loss_sum, window_count = 0.0, 0
                    if consecutive_overflows >= max_consecutive_overflows:
                        raise FloatingPointError(
                            f"FP16 gradient overflowed {consecutive_overflows} consecutive "
                            f"optimizer windows at epoch={epoch}, step={step}; "
                            f"loss scale fell to {new_scale}")
                    if rank == 0:
                        print(json.dumps({"event": "gradient_overflow", "epoch": epoch + 1,
                                          "step": step, "old_scale": old_scale,
                                          "new_scale": new_scale}), flush=True)
                    continue
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                consecutive_overflows = 0
                local_loss_sum += window_loss_sum
                local_count += window_count
                window_loss_sum, window_count = 0.0, 0
        scheduler.step()
        totals = torch.tensor([local_loss_sum, local_count], dtype=torch.float64, device=device)
        if distributed:
            dist.all_reduce(totals)
        if totals[1] <= 0:
            raise FloatingPointError("No training examples contributed to an optimizer update")
        train_loss = float(totals[0] / totals[1])
        if rank == 0:
            assert val_loader is not None
            validation = _validate(raw_model, val_loader, device, amp_dtype)
            previous_best_epoch = best_epoch
            improved = validation["qwk"] > best_qwk
            if improved:
                best_qwk = validation["qwk"]
                best_epoch = epoch + 1
            history.append({"epoch": epoch + 1, "train_loss": train_loss,
                            "validation": validation, "learning_rate": scheduler.get_last_lr()[0],
                            "gradient_overflows": epoch_overflows,
                            "contributing_training_images": int(totals[1]),
                            "epoch_seconds": time.monotonic() - epoch_start})
        if distributed:
            dist.barrier()
        states = [None] * world_size
        if distributed:
            dist.all_gather_object(states, _rng_state())
        else:
            states[0] = _rng_state()
        if rank == 0:
            snapshot = {"epoch": epoch, "model": raw_model.state_dict(),
                        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                        "scaler": scaler.state_dict(), "rng_by_rank": states,
                        "config_sha256": config_sha, "manifest_sha256": manifest_sha,
                        "qualification_sha256": sha256_file(qualification_path),
                        "model_config": asdict(model_config), "best_qwk": best_qwk,
                        "best_epoch": best_epoch,
                        "history": history, "world_size": world_size,
                        "total_gradient_overflows": total_overflows,
                        "consecutive_gradient_overflows": consecutive_overflows,
                        "experiment_config": config, "runtime": runtime,
                        "pipeline_source_sha256": pipeline_source_hashes(),
                        "elapsed_seconds": previous_elapsed + time.monotonic() - run_start}
            if improved:
                _atomic_torch_save(snapshot, run_dir / f"checkpoint-best-epoch-{best_epoch:04d}.pt")
            _atomic_torch_save(snapshot, current_path)
            if improved and previous_best_epoch:
                (run_dir / f"checkpoint-best-epoch-{previous_best_epoch:04d}.pt").unlink(
                    missing_ok=True)
            (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
            print(json.dumps({"epoch": epoch + 1, "train_loss": train_loss,
                              "validation_loss": validation["loss"],
                              "validation_qwk": validation["qwk"], "best_qwk": best_qwk,
                              "gradient_overflows": epoch_overflows,
                              "contributing_training_images": int(totals[1]),
                              "epoch_seconds": history[-1]["epoch_seconds"]}), flush=True)
        if distributed:
            dist.barrier()
        if stop_after_epoch is not None and epoch + 1 == stop_after_epoch < total_epochs:
            paused = True
            if rank == 0:
                print(json.dumps({"status": "paused_at_epoch_boundary", "completed_epochs": epoch + 1,
                                  "configured_epochs": total_epochs}), flush=True)
            break
    if rank == 0 and not paused:
        selected = run_dir / f"checkpoint-best-epoch-{best_epoch:04d}.pt"
        if not selected.is_file():
            raise FileNotFoundError(f"Selected validation checkpoint missing: {selected}")
        temporary_best = best_path.with_name(best_path.name + ".partial")
        shutil.copyfile(selected, temporary_best)
        os.replace(temporary_best, best_path)
        marker = run_dir / "training-complete.json"
        temporary_marker = marker.with_name(marker.name + ".partial")
        temporary_marker.write_text(
            json.dumps({"epochs": total_epochs, "config_sha256": config_sha,
                        "manifest_sha256": manifest_sha, "best_qwk": best_qwk,
                        "best_epoch": best_epoch,
                        "qualification_sha256": sha256_file(qualification_path),
                        "runtime": runtime,
                        "elapsed_seconds": previous_elapsed + time.monotonic() - run_start},
                       indent=2),
            encoding="utf-8")
        os.replace(temporary_marker, marker)
    if distributed:
        dist.barrier()
        dist.destroy_process_group()
