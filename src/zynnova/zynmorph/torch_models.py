"""Trainable conditional 3-D rectified-flow model for discrete microstructures.

This module is optional and imported only when PyTorch training/inference is
requested. The dependency-light spectral backend remains available otherwise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

try:
    import torch
    from torch import Tensor, nn
    from torch.nn import functional as F
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError("ZynMorph flow models require PyTorch; install zynnova[zynnova-ml]") from exc


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        if dimension < 4 or dimension % 2:
            raise ValueError("dimension must be an even integer of at least four")
        self.dimension = dimension

    def forward(self, time: Tensor) -> Tensor:
        half = self.dimension // 2
        frequencies = torch.exp(
            -math.log(10_000.0)
            * torch.arange(half, device=time.device, dtype=time.dtype)
            / max(half - 1, 1)
        )
        angles = time.reshape(-1, 1) * frequencies.reshape(1, -1)
        return torch.cat((angles.sin(), angles.cos()), dim=-1)


class FiLMResidualBlock3D(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, condition_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(input_channels), input_channels)
        self.conv1 = nn.Conv3d(input_channels, output_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(_groups(output_channels), output_channels)
        self.conv2 = nn.Conv3d(output_channels, output_channels, kernel_size=3, padding=1)
        self.condition = nn.Sequential(
            nn.SiLU(),
            nn.Linear(condition_dim, 2 * output_channels),
        )
        self.skip = (
            nn.Identity()
            if input_channels == output_channels
            else nn.Conv3d(input_channels, output_channels, kernel_size=1)
        )

    def forward(self, values: Tensor, condition: Tensor) -> Tensor:
        hidden = self.conv1(F.silu(self.norm1(values)))
        scale, shift = self.condition(condition).chunk(2, dim=-1)
        hidden = self.norm2(hidden)
        hidden = hidden * (1.0 + scale[:, :, None, None, None])
        hidden = hidden + shift[:, :, None, None, None]
        hidden = self.conv2(F.silu(hidden))
        return hidden + self.skip(values)


class ConditionalRectifiedFlow3D(nn.Module):
    """Compact 3-D U-Net velocity field with time and descriptor FiLM conditioning."""

    def __init__(
        self,
        phases: int,
        condition_dim: int,
        *,
        base_channels: int = 48,
        channel_multipliers: Sequence[int] = (1, 2, 4),
        embedding_dim: int = 256,
    ) -> None:
        super().__init__()
        if phases < 2:
            raise ValueError("phases must be at least two")
        self.phases = int(phases)
        self.condition_dim = int(condition_dim)
        self.time_embedding = SinusoidalEmbedding(embedding_dim)
        self.condition_embedding = nn.Sequential(
            nn.Linear(embedding_dim + condition_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        channels = [base_channels * int(multiplier) for multiplier in channel_multipliers]
        self.input = nn.Conv3d(phases, channels[0], kernel_size=3, padding=1)
        self.down_blocks = nn.ModuleList()
        self.downsample = nn.ModuleList()
        current = channels[0]
        for output in channels:
            self.down_blocks.append(FiLMResidualBlock3D(current, output, embedding_dim))
            current = output
            if output != channels[-1]:
                self.downsample.append(nn.Conv3d(current, current, kernel_size=4, stride=2, padding=1))
        self.middle = nn.ModuleList(
            [
                FiLMResidualBlock3D(current, current, embedding_dim),
                FiLMResidualBlock3D(current, current, embedding_dim),
            ]
        )
        self.up_blocks = nn.ModuleList()
        self.upsample = nn.ModuleList()
        for skip_channels in reversed(channels[:-1]):
            self.upsample.append(nn.ConvTranspose3d(current, skip_channels, kernel_size=4, stride=2, padding=1))
            self.up_blocks.append(
                FiLMResidualBlock3D(skip_channels * 2, skip_channels, embedding_dim)
            )
            current = skip_channels
        self.output = nn.Sequential(
            nn.GroupNorm(_groups(current), current),
            nn.SiLU(),
            nn.Conv3d(current, phases, kernel_size=3, padding=1),
        )

    def forward(self, values: Tensor, time: Tensor, condition: Tensor) -> Tensor:
        if values.ndim != 5 or values.shape[1] != self.phases:
            raise ValueError("values must have shape (B, phases, D, H, W)")
        if condition.shape != (values.shape[0], self.condition_dim):
            raise ValueError(
                f"condition must have shape {(values.shape[0], self.condition_dim)}"
            )
        embedded = self.condition_embedding(
            torch.cat((self.time_embedding(time), condition), dim=-1)
        )
        hidden = self.input(values)
        skips: list[Tensor] = []
        down_index = 0
        for block_index, block in enumerate(self.down_blocks):
            hidden = block(hidden, embedded)
            skips.append(hidden)
            if block_index < len(self.down_blocks) - 1:
                hidden = self.downsample[down_index](hidden)
                down_index += 1
        for block in self.middle:
            hidden = block(hidden, embedded)
        for upsample, block, skip in zip(
            self.upsample,
            self.up_blocks,
            reversed(skips[:-1]),
            strict=True,
        ):
            hidden = upsample(hidden)
            if hidden.shape[2:] != skip.shape[2:]:
                hidden = F.interpolate(
                    hidden,
                    size=skip.shape[2:],
                    mode="trilinear",
                    align_corners=False,
                )
            hidden = block(torch.cat((hidden, skip), dim=1), embedded)
        return self.output(hidden)


@dataclass(frozen=True, slots=True)
class FlowBatch:
    labels: Tensor
    condition: Tensor


def rectified_flow_loss(
    model: ConditionalRectifiedFlow3D,
    batch: FlowBatch,
    *,
    generator: torch.Generator | None = None,
) -> Tensor:
    """One-hot discrete rectified-flow objective with uniform time sampling."""

    labels = batch.labels.long()
    if labels.ndim != 4:
        raise ValueError("labels must have shape (B, D, H, W)")
    target = F.one_hot(labels, num_classes=model.phases).movedim(-1, 1).float()
    noise = torch.randn(
        target.shape,
        dtype=target.dtype,
        device=target.device,
        generator=generator,
    )
    time = torch.rand(
        target.shape[0],
        dtype=target.dtype,
        device=target.device,
        generator=generator,
    )
    mixed = (1.0 - time[:, None, None, None, None]) * noise
    mixed = mixed + time[:, None, None, None, None] * target
    velocity = target - noise
    prediction = model(mixed, time, batch.condition)
    return F.mse_loss(prediction, velocity)


@torch.inference_mode()
def sample_flow(
    model: ConditionalRectifiedFlow3D,
    condition: Tensor,
    shape: tuple[int, int, int],
    *,
    steps: int = 32,
    seed: int = 0,
    exact_counts: Mapping[int, int] | None = None,
    phase_ids: Sequence[int] | None = None,
) -> Tensor:
    """Heun integration followed by optional exact-count discrete projection."""

    if steps < 1:
        raise ValueError("steps must be positive")
    generator = torch.Generator(device=condition.device)
    generator.manual_seed(int(seed))
    values = torch.randn(
        (condition.shape[0], model.phases, *shape),
        device=condition.device,
        dtype=next(model.parameters()).dtype,
        generator=generator,
    )
    time_grid = torch.linspace(0.0, 1.0, steps + 1, device=values.device, dtype=values.dtype)
    for start, stop in zip(time_grid[:-1], time_grid[1:], strict=True):
        dt = stop - start
        current_time = torch.full((len(values),), start, device=values.device, dtype=values.dtype)
        first = model(values, current_time, condition)
        proposal = values + dt * first
        next_time = torch.full((len(values),), stop, device=values.device, dtype=values.dtype)
        second = model(proposal, next_time, condition)
        values = values + 0.5 * dt * (first + second)
    if exact_counts is None:
        return values.argmax(dim=1)
    if len(values) != 1:
        raise ValueError("exact-count projection currently expects a batch of one")
    ids = tuple(range(model.phases)) if phase_ids is None else tuple(int(item) for item in phase_ids)
    projected = project_exact_counts(values[0].detach().cpu().numpy(), ids, exact_counts)
    return torch.from_numpy(projected).to(values.device).unsqueeze(0)


def project_exact_counts(
    logits: np.ndarray,
    phase_ids: Sequence[int],
    counts: Mapping[int, int],
) -> np.ndarray:
    """Greedy regret projection that satisfies every requested phase count exactly."""

    scores = np.asarray(logits, dtype=np.float64).reshape(len(phase_ids), -1)
    phase_ids_array = np.asarray(tuple(phase_ids), dtype=np.int32)
    target = np.asarray([counts[int(phase)] for phase in phase_ids_array], dtype=np.int64)
    if int(target.sum()) != scores.shape[1]:
        raise ValueError("exact counts must sum to the number of voxels")
    # Assign highest-confidence phase opportunities first while honoring quotas.
    choices = np.argsort(scores, axis=0)[::-1]
    margins = scores[choices[0], np.arange(scores.shape[1])] - scores[
        choices[1], np.arange(scores.shape[1])
    ]
    order = np.argsort(-margins, kind="stable")
    remaining = target.copy()
    assigned = np.full(scores.shape[1], -1, dtype=np.int64)
    for voxel in order:
        for candidate in choices[:, voxel]:
            if remaining[candidate] > 0:
                assigned[voxel] = candidate
                remaining[candidate] -= 1
                break
    if np.any(assigned < 0) or np.any(remaining):
        raise RuntimeError("exact-count projection failed")
    return phase_ids_array[assigned].reshape(logits.shape[1:])


def _groups(channels: int) -> int:
    for groups in (16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


__all__ = [
    "ConditionalRectifiedFlow3D",
    "FlowBatch",
    "project_exact_counts",
    "rectified_flow_loss",
    "sample_flow",
]
