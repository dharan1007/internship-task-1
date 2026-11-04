"""Selective state-space scan, independent of ready-made Mamba implementations.

The reference is intentionally simple and mathematically explicit. The
parallel affine scan compiles on CUDA, processes bounded token chunks, and
recomputes chunk states in its hand-derived reverse recurrence. Its throughput
and precision on the target GPU still require qualification. Backend selection
is explicit.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch
from torch import Tensor
from torch._higher_order_ops.associative_scan import associative_scan

_SCAN_CHUNK_TOKENS = 256
_CUDA_SCAN_VERSION = "2.3.2.post1"
_CUDA_SCAN_EXTENSION_SHA256 = "7d2620a53cddccbc2fbba943a6cb394f569954d50a674d6b35d96d97acd64f48"


@lru_cache(maxsize=1)
def _pinned_cuda_scan():
    """Load only the pinned low-level kernel, never a ready-made Vim model."""
    try:
        installed = version("mamba-ssm")
    except PackageNotFoundError as exc:
        raise RuntimeError("Pinned low-level CUDA scan wheel is not installed") from exc
    if installed != _CUDA_SCAN_VERSION:
        raise RuntimeError("Installed CUDA scan package differs from the qualified version")
    try:
        import selective_scan_cuda
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    except ImportError as exc:
        raise RuntimeError("Pinned low-level CUDA scan dependencies are missing") from exc

    binary = Path(selective_scan_cuda.__file__)
    digest = hashlib.sha256()
    with binary.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != _CUDA_SCAN_EXTENSION_SHA256:
        raise RuntimeError("Installed CUDA scan binary differs from the reviewed wheel")
    return selective_scan_fn


def _cuda_selective_scan(x: Tensor, delta: Tensor, a: Tensor, b: Tensor,
                         c: Tensor, skip: Tensor) -> Tensor:
    if not x.is_cuda:
        raise RuntimeError("Pinned CUDA selective scan requires a CUDA tensor")
    kernel = _pinned_cuda_scan()
    # The existing from-scratch scan maintains FP32 state and discretization.
    # The external kernel requires input and delta to have identical dtypes.
    result = kernel(x.float().transpose(1, 2).contiguous(),
                    delta.float().transpose(1, 2).contiguous(), a.float().contiguous(),
                    b.float().transpose(1, 2).contiguous(),
                    c.float().transpose(1, 2).contiguous(), skip.float().contiguous(),
                    delta_softplus=False)
    return result.transpose(1, 2).to(x.dtype)


def _compose(first: tuple[Tensor, Tensor], second: tuple[Tensor, Tensor]
             ) -> tuple[Tensor, Tensor]:
    # (a,b) represents h -> a*h+b. Apply first, then second.
    return first[0] * second[0], second[0] * first[1] + second[1]


def _affine_scan(multiplier: Tensor, drive: Tensor) -> Tensor:
    mode = "pointwise" if multiplier.is_cuda else "generic"
    return associative_scan(_compose, (multiplier, drive), dim=1,
                            combine_mode=mode)[1]


# The retinal pipeline fixes image size and scan chunk size. Forcing static
# specialization prevents Inductor from promoting a changed chunk length to
# symbolic dimensions, which its associative_scan lowering cannot codegen.
_compiled_affine_scan = torch.compile(_affine_scan, fullgraph=True, dynamic=False)


def _dispatch_affine_scan(multiplier: Tensor, drive: Tensor) -> Tensor:
    return (_compiled_affine_scan if multiplier.is_cuda else _affine_scan)(multiplier, drive)


def _scan_chunk(multiplier: Tensor, drive: Tensor, initial: Tensor) -> Tensor:
    # Fold the preceding chunk's final state into the first affine transform.
    first_drive = drive[:, :1] + multiplier[:, :1] * initial.unsqueeze(1)
    return _dispatch_affine_scan(multiplier, torch.cat((first_drive, drive[:, 1:]), dim=1))


class _ParallelSelectiveScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, delta: Tensor, a: Tensor, b: Tensor,
                c: Tensor, skip: Tensor) -> Tensor:
        x32, dt32, a32, b32, c32, skip32 = (tensor.float() for tensor in
                                                (x, delta, a, b, c, skip))
        initial = torch.zeros((x.shape[0], x.shape[2], a.shape[1]),
                              device=x.device, dtype=torch.float32)
        boundaries = [initial]
        outputs = []
        for start in range(0, x.shape[1], _SCAN_CHUNK_TOKENS):
            end = min(start + _SCAN_CHUNK_TOKENS, x.shape[1])
            x_part, dt_part = x32[:, start:end], dt32[:, start:end]
            alpha = torch.exp(dt_part.unsqueeze(-1) * a32)
            drive = dt_part.unsqueeze(-1) * b32[:, start:end].unsqueeze(2) * x_part.unsqueeze(-1)
            state = _scan_chunk(alpha, drive, boundaries[-1])
            outputs.append(((state * c32[:, start:end].unsqueeze(2)).sum(-1) +
                            skip32 * x_part).to(x.dtype))
            # Clone avoids retaining the whole chunk through a view of its last state.
            boundaries.append(state[:, -1].clone())
        ctx.save_for_backward(x, delta, a, b, c, skip, torch.stack(boundaries, dim=1))
        return torch.cat(outputs, dim=1)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, ...]:
        x, delta, a, b, c, skip, boundaries = ctx.saved_tensors
        x32, dt32, a32, b32, c32, skip32 = (tensor.float() for tensor in
                                                (x, delta, a, b, c, skip))
        gy = grad_output.float()
        carry = torch.zeros_like(boundaries[:, 0])
        x_parts, dt_parts, b_parts, c_parts = [], [], [], []
        grad_a = torch.zeros_like(a32)
        grad_skip = torch.zeros_like(skip32)
        chunks = boundaries.shape[1] - 1
        for chunk in range(chunks - 1, -1, -1):
            start = chunk * _SCAN_CHUNK_TOKENS
            end = min(start + _SCAN_CHUNK_TOKENS, x.shape[1])
            xp, dtp = x32[:, start:end], dt32[:, start:end]
            bp, cp, gyp = b32[:, start:end], c32[:, start:end], gy[:, start:end]
            alpha = torch.exp(dtp.unsqueeze(-1) * a32)
            drive = dtp.unsqueeze(-1) * bp.unsqueeze(2) * xp.unsqueeze(-1)
            state = _scan_chunk(alpha, drive, boundaries[:, chunk])
            previous = torch.cat((boundaries[:, chunk].unsqueeze(1), state[:, :-1]), dim=1)
            next_alpha = torch.cat((alpha[:, 1:], torch.zeros_like(alpha[:, :1])), dim=1)
            reverse_drive = (gyp.unsqueeze(-1) * cp.unsqueeze(2)).flip(1)
            reverse_drive = torch.cat((reverse_drive[:, :1] + carry.unsqueeze(1),
                                       reverse_drive[:, 1:]), dim=1)
            # Adjoint at the final token includes all later chunks through carry.
            adjoint = _dispatch_affine_scan(next_alpha.flip(1), reverse_drive).flip(1)
            x_parts.append(gyp * skip32 +
                           (adjoint * dtp.unsqueeze(-1) * bp.unsqueeze(2)).sum(-1))
            dt_parts.append((adjoint * (a32 * alpha * previous +
                                        bp.unsqueeze(2) * xp.unsqueeze(-1))).sum(-1))
            grad_a += (adjoint * dtp.unsqueeze(-1) * alpha * previous).sum((0, 1))
            b_parts.append((adjoint * dtp.unsqueeze(-1) * xp.unsqueeze(-1)).sum(2))
            c_parts.append((gyp.unsqueeze(-1) * state).sum(2))
            grad_skip += (gyp * xp).sum((0, 1))
            carry = alpha[:, 0] * adjoint[:, 0]
        return (torch.cat(x_parts[::-1], dim=1).to(x.dtype),
                torch.cat(dt_parts[::-1], dim=1).to(delta.dtype),
                grad_a.to(a.dtype), torch.cat(b_parts[::-1], dim=1).to(b.dtype),
                torch.cat(c_parts[::-1], dim=1).to(c.dtype), grad_skip.to(skip.dtype))


def selective_scan_reference(
    x: Tensor,
    delta: Tensor,
    a: Tensor,
    b: Tensor,
    c: Tensor,
    skip: Tensor,
) -> Tensor:
    """Compute the input-dependent diagonal SSM recurrence.

    Args:
        x, delta: [batch, sequence, expanded_channels]
        a: [expanded_channels, state_dim], with strictly negative values
        b, c: [batch, sequence, state_dim]
        skip: [expanded_channels]

    Returns:
        [batch, sequence, expanded_channels]

    The scan state and discretization run in float32 even under autocast.
    The input-dependent B discretization follows the released Vim/Mamba
    selective-scan implementation (delta * B), not a fabricated stationary
    convolution. No state tensor is expanded over the whole sequence.
    """
    if x.ndim != 3 or delta.shape != x.shape:
        raise ValueError("x and delta must have matching [B, L, E] shapes")
    batch, length, channels = x.shape
    if a.ndim != 2 or a.shape[0] != channels:
        raise ValueError("a must have shape [E, N]")
    state_dim = a.shape[1]
    if b.shape != (batch, length, state_dim) or c.shape != b.shape:
        raise ValueError("b and c must have shape [B, L, N]")
    if skip.shape != (channels,):
        raise ValueError("skip must have shape [E]")
    if length == 0:
        raise ValueError("A class-token sequence cannot be empty")

    state = torch.zeros((batch, channels, state_dim), device=x.device, dtype=torch.float32)
    a32 = a.float()
    skip32 = skip.float()
    outputs: list[Tensor] = []
    for index in range(length):
        x_t = x[:, index].float()
        dt_t = delta[:, index].float()
        transition = torch.exp(dt_t.unsqueeze(-1) * a32)
        drive = dt_t.unsqueeze(-1) * b[:, index].float().unsqueeze(1) * x_t.unsqueeze(-1)
        state = transition * state + drive
        y_t = (state * c[:, index].float().unsqueeze(1)).sum(-1) + skip32 * x_t
        outputs.append(y_t)
    return torch.stack(outputs, dim=1).to(x.dtype)


def selective_scan(
    x: Tensor,
    delta: Tensor,
    a: Tensor,
    b: Tensor,
    c: Tensor,
    skip: Tensor,
    *,
    backend: str,
) -> Tensor:
    if backend == "reference":
        return selective_scan_reference(x, delta, a, b, c, skip)
    if backend in {"parallel", "cuda_kernel"}:
        if (x.ndim != 3 or delta.shape != x.shape or a.ndim != 2 or
                a.shape[0] != x.shape[2]):
            raise ValueError("Invalid selective scan input shapes")
        if x.shape[1] == 0:
            raise ValueError("A class-token sequence cannot be empty")
        if b.shape != (x.shape[0], x.shape[1], a.shape[1]) or c.shape != b.shape:
            raise ValueError("Invalid selective scan B/C shapes")
        if skip.shape != (x.shape[2],):
            raise ValueError("Invalid selective scan skip shape")
        return (_ParallelSelectiveScan.apply(x, delta, a, b, c, skip)
                if backend == "parallel" else _cuda_selective_scan(x, delta, a, b, c, skip))
    raise ValueError(f"Scan backend {backend!r} is unavailable; refusing an implicit fallback")
