from __future__ import annotations

import math

from ...common import require_torch
from .config import QM9GeneratorModelConfig
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
        radial = torch.exp(
            -self.gamma * (distance[..., None] - self.centers).square()
        )
        envelope = 0.5 * (torch.cos(math.pi * distance / self.cutoff) + 1.0)
        envelope = envelope * (distance < self.cutoff).to(distance.dtype)
        return radial * envelope[..., None]


class ConditionalEquivariantLayer(nn.Module):
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
        hidden = self.norm(
            hidden + self.node_mlp(torch.cat((hidden, aggregate), dim=-1))
        )
        displacement = positions[:, :, None, :] - positions[:, None, :, :]
        distance = torch.linalg.vector_norm(displacement, dim=-1).clamp_min(1.0e-6)
        direction = displacement / distance[..., None]
        coefficient = self.coordinate_head(edge).squeeze(-1) * pair_mask
        velocity = (direction * coefficient[..., None]).sum(dim=2)
        return hidden, velocity


class InvariantPropertyLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_rbf: int) -> None:
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + num_rbf, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, hidden, radial, pair_mask):
        batch, count, hidden_dim = hidden.shape
        receiver = hidden[:, :, None, :].expand(batch, count, count, hidden_dim)
        sender = hidden[:, None, :, :].expand(batch, count, count, hidden_dim)
        edge = self.edge_mlp(torch.cat((receiver, sender, radial), dim=-1))
        edge = edge * pair_mask[..., None]
        aggregate = edge.sum(dim=2)
        return self.norm(
            hidden + self.node_mlp(torch.cat((hidden, aggregate), dim=-1))
        )


def _geometry(positions, mask, *, cutoff: float):
    displacement = positions[:, :, None, :] - positions[:, None, :, :]
    distance = torch.linalg.vector_norm(displacement, dim=-1)
    pair_mask = mask[:, :, None] & mask[:, None, :]
    eye = torch.eye(
        positions.shape[1],
        device=positions.device,
        dtype=torch.bool,
    )[None, :, :]
    pair_mask = pair_mask & (~eye) & (distance < cutoff)
    return distance, pair_mask


class QM9ConditionalGenerator(nn.Module):
    """Property-conditioned E(3)-equivariant coordinate flow for QM9.

    The flow receives a fixed atomic composition and normalized molecular
    properties.  A separate invariant property head predicts properties from
    coordinates without seeing the requested condition, enabling honest ranking
    and differentiable property guidance during sampling.
    """

    def __init__(self, config: QM9GeneratorModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or QM9GeneratorModelConfig()
        self.config.__post_init__()
        property_count = len(self.config.property_names)
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
        self.condition_mlp = nn.Sequential(
            nn.Linear(property_count * 2, self.config.condition_hidden_dim),
            nn.SiLU(),
            nn.Linear(self.config.condition_hidden_dim, self.config.hidden_dim),
        )
        self.rbf = DenseRadialBasis(self.config.num_rbf, self.config.cutoff_A)
        self.flow_layers = nn.ModuleList(
            ConditionalEquivariantLayer(
                self.config.hidden_dim,
                self.config.num_rbf,
            )
            for _ in range(self.config.num_flow_layers)
        )
        self.velocity_scale = nn.Parameter(torch.tensor(0.1))

        self.property_atom_embedding = nn.Embedding(
            self.config.max_atomic_number + 1,
            self.config.hidden_dim,
            padding_idx=0,
        )
        self.property_layers = nn.ModuleList(
            InvariantPropertyLayer(
                self.config.hidden_dim,
                self.config.num_rbf,
            )
            for _ in range(self.config.num_property_layers)
        )
        self.property_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim * 2, self.config.property_head_hidden_dim),
            nn.SiLU(),
            nn.Linear(self.config.property_head_hidden_dim, property_count),
        )
        self.property_normalizer = None

    def _condition_embedding(self, properties, property_mask):
        mask_float = property_mask.to(properties.dtype)
        encoded = torch.cat((properties * mask_float, mask_float), dim=-1)
        embedding = self.condition_mlp(encoded)
        present = property_mask.any(dim=-1, keepdim=True).to(properties.dtype)
        return embedding * present

    def forward(
        self,
        z,
        positions,
        time,
        mask,
        properties,
        property_mask,
    ):
        if time.ndim == 0:
            time = time.expand(z.shape[0])
        if properties.shape[-1] != len(self.config.property_names):
            raise ValueError("properties has the wrong final dimension")
        hidden = self.atom_embedding(z.long())
        hidden = hidden + self.time_mlp(
            sinusoidal_time_embedding(time, self.config.time_embedding_dim)
        )[:, None, :]
        hidden = hidden + self._condition_embedding(
            properties,
            property_mask,
        )[:, None, :]
        hidden = hidden * mask[..., None]
        distance, pair_mask = _geometry(
            positions,
            mask,
            cutoff=self.config.cutoff_A,
        )
        radial = self.rbf(distance)
        velocity = torch.zeros_like(positions)
        for layer in self.flow_layers:
            hidden, increment = layer(hidden, positions, radial, pair_mask)
            hidden = hidden * mask[..., None]
            velocity = velocity + increment
        velocity = center_coordinates(
            velocity * self.velocity_scale,
            mask,
        )
        return velocity

    def predict_properties(self, z, positions, mask):
        hidden = self.property_atom_embedding(z.long()) * mask[..., None]
        distance, pair_mask = _geometry(
            positions,
            mask,
            cutoff=self.config.cutoff_A,
        )
        radial = self.rbf(distance)
        for layer in self.property_layers:
            hidden = layer(hidden, radial, pair_mask)
            hidden = hidden * mask[..., None]
        weights = mask.to(hidden.dtype)[..., None]
        count = weights.sum(dim=1).clamp_min(1.0)
        pooled_mean = (hidden * weights).sum(dim=1) / count
        pooled_sum = (hidden * weights).sum(dim=1) / count.sqrt()
        return self.property_head(torch.cat((pooled_mean, pooled_sum), dim=-1))


__all__ = [
    "QM9ConditionalGenerator",
    "sinusoidal_time_embedding",
]
