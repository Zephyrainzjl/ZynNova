from __future__ import annotations

import math

from ...common import require_torch
from .config import QM9FlowModelConfig
from .data import center_coordinates


torch = require_torch()
nn = torch.nn


def sinusoidal_time_embedding(time, dimension: int):
    half = dimension // 2
    frequencies = torch.exp(
        torch.arange(half, device=time.device, dtype=time.dtype)
        * (-math.log(10000.0) / max(half - 1, 1))
    )
    angles = time[:, None] * frequencies[None, :]
    embedding = torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1)
    if dimension % 2:
        embedding = torch.nn.functional.pad(embedding, (0, 1))
    return embedding


class DenseRadialBasis(nn.Module):
    def __init__(self, num_rbf: int, cutoff: float) -> None:
        super().__init__()
        self.cutoff = float(cutoff)
        centers = torch.linspace(0.0, cutoff, num_rbf)
        self.register_buffer("centers", centers)
        spacing = float(centers[1] - centers[0]) if num_rbf > 1 else cutoff
        self.gamma = 1.0 / max(spacing * spacing, 1.0e-8)

    def forward(self, distance):
        radial = torch.exp(-self.gamma * (distance[..., None] - self.centers).square())
        envelope = 0.5 * (torch.cos(math.pi * distance / self.cutoff) + 1.0)
        envelope = envelope * (distance < self.cutoff).to(distance.dtype)
        return radial * envelope[..., None]


class EquivariantFlowLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_rbf: int) -> None:
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + num_rbf, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.coordinate_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, hidden, positions, radial, pair_mask):
        batch, count, hidden_dim = hidden.shape
        receiver = hidden[:, :, None, :].expand(batch, count, count, hidden_dim)
        sender = hidden[:, None, :, :].expand(batch, count, count, hidden_dim)
        edge = self.edge_mlp(torch.cat((receiver, sender, radial), dim=-1))
        edge = edge * pair_mask[..., None]
        aggregate = edge.sum(dim=2)
        hidden = self.norm(hidden + self.node_mlp(torch.cat((hidden, aggregate), dim=-1)))
        displacement = positions[:, :, None, :] - positions[:, None, :, :]
        distance = torch.linalg.vector_norm(displacement, dim=-1).clamp_min(1.0e-6)
        direction = displacement / distance[..., None]
        coefficient = self.coordinate_head(edge).squeeze(-1) * pair_mask
        velocity = (direction * coefficient[..., None]).sum(dim=2)
        return hidden, velocity


class QM9EquivariantFlow(nn.Module):
    """Composition-conditioned E(3)-equivariant flow for QM9 coordinates."""

    def __init__(self, config: QM9FlowModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or QM9FlowModelConfig()
        self.atom_embedding = nn.Embedding(
            self.config.max_atomic_number + 1,
            self.config.hidden_dim,
            padding_idx=0,
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(self.config.time_embedding_dim, self.config.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
        )
        self.rbf = DenseRadialBasis(self.config.num_rbf, self.config.cutoff_A)
        self.layers = nn.ModuleList(
            EquivariantFlowLayer(self.config.hidden_dim, self.config.num_rbf)
            for _ in range(self.config.num_layers)
        )
        self.velocity_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, z, positions, time, mask):
        if time.ndim == 0:
            time = time.expand(z.shape[0])
        hidden = self.atom_embedding(z.long())
        hidden = hidden + self.time_mlp(
            sinusoidal_time_embedding(time, self.config.time_embedding_dim)
        )[:, None, :]
        hidden = hidden * mask[..., None]
        displacement = positions[:, :, None, :] - positions[:, None, :, :]
        distance = torch.linalg.vector_norm(displacement, dim=-1)
        pair_mask = mask[:, :, None] & mask[:, None, :]
        eye = torch.eye(z.shape[1], device=z.device, dtype=torch.bool)[None, :, :]
        pair_mask = pair_mask & (~eye) & (distance < self.config.cutoff_A)
        radial = self.rbf(distance)
        total_velocity = torch.zeros_like(positions)
        for layer in self.layers:
            hidden, velocity = layer(hidden, positions, radial, pair_mask)
            hidden = hidden * mask[..., None]
            total_velocity = total_velocity + velocity
        total_velocity = total_velocity * self.velocity_scale
        return center_coordinates(total_velocity, mask)


__all__ = ["QM9EquivariantFlow", "sinusoidal_time_embedding"]
