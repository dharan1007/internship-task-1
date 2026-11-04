"""From-scratch bidirectional Vision Mamba image classifier."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .scan import selective_scan


@dataclass(frozen=True)
class VimConfig:
    image_size: int = 768
    patch_size: int = 16
    depth: int = 24
    embed_dim: int = 768
    state_dim: int = 16
    expansion: int = 2
    conv_kernel: int = 4
    num_classes: int = 5
    scan_backend: str = "parallel"
    activation_checkpointing: bool = True
    drop_path_rate: float = 0.1

    def __post_init__(self) -> None:
        if self.image_size <= 0 or self.patch_size <= 0 or self.image_size % self.patch_size:
            raise ValueError("image_size must be positive and divisible by patch_size")
        if min(self.depth, self.embed_dim, self.state_dim, self.expansion, self.conv_kernel) <= 0:
            raise ValueError("All Vim dimensions must be positive")
        if self.num_classes != 5:
            raise ValueError("The assignment requires exactly five EyePACS grades")
        if self.scan_backend not in {"reference", "parallel", "cuda_kernel"}:
            raise ValueError("scan_backend must be reference, parallel or cuda_kernel")
        if not 0 <= self.drop_path_rate < 1:
            raise ValueError("drop_path_rate must be in [0, 1)")


class DropPath(nn.Module):
    def __init__(self, probability: float):
        super().__init__()
        self.probability = probability

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.probability == 0:
            return x
        keep = 1 - self.probability
        mask = torch.rand((x.shape[0],) + (1,) * (x.ndim - 1), device=x.device) < keep
        return x * mask / keep


class SelectiveDirection(nn.Module):
    """One direction's depthwise convolution and selective state-space model."""

    def __init__(self, embed_dim: int, inner_dim: int, state_dim: int, kernel: int, backend: str):
        super().__init__()
        self.backend = backend
        self.state_dim = state_dim
        rank = math.ceil(embed_dim / 16)
        self.conv = nn.Conv1d(inner_dim, inner_dim, kernel, groups=inner_dim,
                              padding=kernel - 1, bias=True)
        self.x_proj = nn.Linear(inner_dim, rank + 2 * state_dim, bias=False)
        self.dt_proj = nn.Linear(rank, inner_dim, bias=True)
        self.a_log = nn.Parameter(torch.log(torch.arange(1, state_dim + 1,
                                                   dtype=torch.float32)).repeat(inner_dim, 1))
        self.skip = nn.Parameter(torch.ones(inner_dim, dtype=torch.float32))
        self._init_delta(rank)

    def _init_delta(self, rank: int) -> None:
        bound = rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -bound, bound)
        dt = torch.exp(torch.empty(self.dt_proj.out_features).uniform_(
            math.log(0.001), math.log(0.1)))
        inverse_softplus = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inverse_softplus)

    def scan_inputs(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        length = x.shape[1]
        conv_x = F.silu(self.conv(x.transpose(1, 2))[..., :length].transpose(1, 2))
        dt_low, b, c = torch.split(
            self.x_proj(conv_x), [self.dt_proj.in_features, self.state_dim, self.state_dim], dim=-1
        )
        delta = F.softplus(self.dt_proj(dt_low).float())
        a = -torch.exp(self.a_log.float())
        return conv_x, delta, a, b, c, self.skip

    def forward(self, x: Tensor) -> Tensor:
        return selective_scan(*self.scan_inputs(x), backend=self.backend)


class BidirectionalVimBlock(nn.Module):
    """Paper-aligned forward and reverse SSM plus Conv1d with a shared gate."""

    def __init__(self, config: VimConfig, drop_path_rate: float):
        super().__init__()
        d = config.embed_dim
        e = d * config.expansion
        self.norm = nn.RMSNorm(d, eps=1e-5)
        self.in_proj = nn.Linear(d, 2 * e, bias=False)
        self.forward_direction = SelectiveDirection(
            d, e, config.state_dim, config.conv_kernel, config.scan_backend
        )
        self.backward_direction = SelectiveDirection(
            d, e, config.state_dim, config.conv_kernel, config.scan_backend
        )
        self.out_proj = nn.Linear(e, d, bias=False)
        self.drop_path = DropPath(drop_path_rate)
        nn.init.kaiming_uniform_(self.out_proj.weight, a=math.sqrt(5))
        with torch.no_grad():
            self.out_proj.weight.div_(math.sqrt(config.depth))

    def forward(self, tokens: Tensor) -> Tensor:
        x, gate = self.in_proj(self.norm(tokens)).chunk(2, dim=-1)
        forward = self.forward_direction(x)
        backward = self.backward_direction(x.flip(1)).flip(1)
        mixed = (forward + backward) * 0.5 * F.silu(gate)
        return tokens.float() + self.drop_path(self.out_proj(mixed)).float()


class VisionMamba(nn.Module):
    """Image-to-five-logit Vim classifier trained with random initialization."""

    def __init__(self, config: VimConfig):
        super().__init__()
        self.config = config
        self.patch_embed = nn.Conv2d(3, config.embed_dim, config.patch_size,
                                     stride=config.patch_size)
        patches = (config.image_size // config.patch_size) ** 2
        self.cls_position = patches // 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, patches + 1, config.embed_dim))
        rates = [0.0] + torch.linspace(0, config.drop_path_rate, config.depth).tolist()
        self.blocks = nn.ModuleList(BidirectionalVimBlock(config, rates[i])
                                    for i in range(config.depth))
        self.norm = nn.RMSNorm(config.embed_dim, eps=1e-5)
        self.head = nn.Linear(config.embed_dim, config.num_classes)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.patch_embed.weight,
                        std=math.sqrt(1 / (3 * config.patch_size * config.patch_size)))
        nn.init.zeros_(self.patch_embed.bias)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, images: Tensor) -> Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must have shape [B, 3, H, W]")
        expected = self.config.image_size
        if images.shape[2:] != (expected, expected):
            raise ValueError(f"Expected {expected}x{expected} images, got {images.shape[2:]}")
        tokens = self.patch_embed(images).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(tokens.shape[0], -1, -1)
        p = self.cls_position
        tokens = torch.cat((tokens[:, :p], cls, tokens[:, p:]), dim=1) + self.pos_embed
        for block in self.blocks:
            tokens = (checkpoint(block, tokens, use_reentrant=False)
                      if self.training and self.config.activation_checkpointing else block(tokens))
        return self.head(self.norm(tokens[:, p]))
