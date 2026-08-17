from __future__ import annotations

import math

from ...common import require_torch
from .config import CrystalGNNModelConfig


torch = require_torch()
nn = torch.nn


def scatter_sum(values, index, dim_size: int):
    result = values.new_zeros((dim_size, *values.shape[1:]))
    if values.numel():
        result.index_add_(0, index, values)
    return result


class RadialBasis(nn.Module):
    def __init__(self, num_rbf: int, cutoff_A: float) -> None:
        super().__init__()
        self.cutoff_A = float(cutoff_A)
        centers = torch.linspace(0.0, cutoff_A, num_rbf)
        self.register_buffer("centers", centers)
        spacing = float(centers[1] - centers[0]) if num_rbf > 1 else cutoff_A
        self.gamma = 1.0 / max(spacing * spacing, 1.0e-8)

    def forward(self, distance):
        basis = torch.exp(-self.gamma * (distance[:, None] - self.centers[None, :]).square())
        envelope = 0.5 * (torch.cos(math.pi * distance / self.cutoff_A) + 1.0)
        envelope = envelope * (distance < self.cutoff_A).to(distance.dtype)
        return basis * envelope[:, None]


class CrystalMessageLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_rbf: int) -> None:
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(hidden_dim * 2 + num_rbf, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, hidden, receiver, sender, radial):
        edge_message = self.message(torch.cat((hidden[receiver], hidden[sender], radial), dim=-1))
        aggregate = scatter_sum(edge_message, receiver, hidden.shape[0])
        return self.norm(hidden + self.update(torch.cat((hidden, aggregate), dim=-1)))


class CrystalGNN(nn.Module):
    """Periodic crystal graph network for Matbench formation-energy prediction."""

    def __init__(self, config: CrystalGNNModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or CrystalGNNModelConfig()
        self.embedding = nn.Embedding(
            self.config.max_atomic_number + 1,
            self.config.hidden_dim,
            padding_idx=0,
        )
        self.rbf = RadialBasis(self.config.num_rbf, self.config.cutoff_A)
        self.layers = nn.ModuleList(
            CrystalMessageLayer(self.config.hidden_dim, self.config.num_rbf)
            for _ in range(self.config.num_layers)
        )
        self.readout = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.config.hidden_dim, 1),
        )
        self.register_buffer("target_mean", torch.tensor(float(self.config.target_mean)))
        self.register_buffer("target_std", torch.tensor(float(self.config.target_std)))

    def set_target_normalization(self, mean: float, std: float) -> None:
        self.target_mean.fill_(float(mean))
        self.target_std.fill_(max(float(std), 1.0e-8))
        self.config.target_mean = float(mean)
        self.config.target_std = max(float(std), 1.0e-8)

    def forward(self, batch):
        z = batch["z"].long()
        edge_index = batch["edge_index"].long()
        edge_distance = batch["edge_distance"]
        graph_index = batch["batch"].long()
        hidden = self.embedding(z)
        radial = self.rbf(edge_distance)
        receiver, sender = edge_index
        for layer in self.layers:
            hidden = layer(hidden, receiver, sender, radial)
        graph_count = int(graph_index.max().item()) + 1 if graph_index.numel() else 1
        pooled = scatter_sum(hidden, graph_index, graph_count)
        counts = scatter_sum(
            torch.ones((hidden.shape[0], 1), device=hidden.device, dtype=hidden.dtype),
            graph_index,
            graph_count,
        ).clamp_min(1.0)
        pooled = pooled / counts
        normalized = self.readout(pooled).squeeze(-1)
        return normalized * self.target_std + self.target_mean


__all__ = ["CrystalGNN"]
