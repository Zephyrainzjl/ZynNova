"""Strictly E(3)-invariant continuous-filter convolution baseline.

The model uses only species and pair distances. Its scalar energy is therefore
invariant under rotations, reflections and translations, while forces obtained
from the energy gradient are exactly equivariant. This is the intentionally
ordinary-convolution option; no angular tensor product is hidden inside it.
"""

from __future__ import annotations

import math
from typing import Any

from .._deps import require_torch
from .base import BackboneAdapter, BackboneCapabilities, register_backbone

torch = require_torch()
nn = torch.nn


def _scatter_sum(values: Any, index: Any, size: int) -> Any:
    output = values.new_zeros((size, *values.shape[1:]))
    output.index_add_(0, index, values)
    return output


class GaussianRadialBasis(nn.Module):
    def __init__(self, cutoff_A: float, count: int) -> None:
        super().__init__()
        centers = torch.linspace(0.0, float(cutoff_A), int(count))
        spacing = float(cutoff_A) / max(1, int(count) - 1)
        self.register_buffer("centers", centers)
        self.register_buffer("gamma", torch.tensor(1.0 / max(spacing * spacing, 1e-8)))
        self.register_buffer("cutoff", torch.tensor(float(cutoff_A)))

    def forward(self, distance: Any) -> Any:
        radius = distance.reshape(-1, 1)
        basis = torch.exp(-self.gamma * (radius - self.centers).square())
        x = (radius / self.cutoff).clamp(0.0, 1.0)
        envelope = 1.0 - 10.0 * x.pow(3) + 15.0 * x.pow(4) - 6.0 * x.pow(5)
        return basis * envelope * (radius < self.cutoff).to(radius.dtype)


class ContinuousFilterBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        radial_count: int,
        radial_hidden: tuple[int, ...],
        average_neighbors: float,
    ) -> None:
        super().__init__()
        filter_layers: list[Any] = []
        current = radial_count
        for width in radial_hidden:
            filter_layers.extend((nn.Linear(current, int(width)), nn.SiLU()))
            current = int(width)
        filter_layers.append(nn.Linear(current, channels))
        self.filter_network = nn.Sequential(*filter_layers)
        self.source = nn.Linear(channels, channels, bias=False)
        self.update = nn.Sequential(
            nn.Linear(channels, 2 * channels),
            nn.SiLU(),
            nn.Linear(2 * channels, channels),
        )
        self.normalization = math.sqrt(float(average_neighbors))

    def forward(self, features: Any, edge_index: Any, radial: Any) -> Any:
        source, target = edge_index[0], edge_index[1]
        filters = self.filter_network(radial)
        messages = filters * self.source(features[source])
        aggregate = _scatter_sum(messages, target, features.shape[0])
        return features + self.update(aggregate / self.normalization)


class ConvolutionCore(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        if config.max_ell != 0:
            raise ValueError("convolution backbone requires max_ell=0")
        self.embedding = nn.Embedding(119, int(config.channels), padding_idx=0)
        self.radial = GaussianRadialBasis(config.cutoff_A, config.num_bessel)
        self.interactions = nn.ModuleList(
            ContinuousFilterBlock(
                int(config.channels),
                int(config.num_bessel),
                tuple(config.radial_mlp),
                float(config.average_num_neighbors),
            )
            for _ in range(config.num_interactions)
        )
        self.readouts = nn.ModuleList(
            nn.Sequential(
                nn.Linear(config.channels, config.channels),
                nn.SiLU(),
                nn.Linear(config.channels, 1),
            )
            for _ in range(config.num_interactions)
        )
        atomic_energies = (
            config.atomic_energies_eV
            if config.atomic_energies_eV is not None
            else (0.0,) * len(config.atomic_numbers)
        )
        table = torch.zeros(119)
        for number, energy in zip(config.atomic_numbers, atomic_energies, strict=True):
            table[int(number)] = float(energy)
        self.register_buffer("atomic_energy_table", table)
        self.register_buffer("atomic_numbers", torch.tensor(config.atomic_numbers))
        self.register_buffer("r_max", torch.tensor(float(config.cutoff_A)))
        self.num_interactions = int(config.num_interactions)
        self.pair_repulsion = bool(config.pair_repulsion)

    @staticmethod
    def _zbl_pair_energy(numbers: Any, edge_index: Any, distance: Any) -> Any:
        """Universal screened nuclear repulsion for short-range robustness."""

        source, target = edge_index[0], edge_index[1]
        z_source = numbers[source].to(dtype=distance.dtype)
        z_target = numbers[target].to(dtype=distance.dtype)
        screening_length = 0.8854 * 0.529177210903 / (
            z_source.pow(0.23) + z_target.pow(0.23)
        )
        reduced = distance.clamp_min(1.0e-4) / screening_length
        coefficients = distance.new_tensor((0.1818, 0.5099, 0.2802, 0.02817))
        exponents = distance.new_tensor((3.2, 0.9423, 0.4029, 0.2016))
        screening = (
            coefficients[None] * torch.exp(-reduced[:, None] * exponents[None])
        ).sum(-1)
        return 14.3996454784255 * z_source * z_target * screening / distance.clamp_min(
            1.0e-4
        )

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        positions = data["positions"]
        edge_index = data["edge_index"]
        source, target = edge_index[0], edge_index[1]
        vectors = positions[target] - positions[source]
        shifts = data.get("shifts")
        if shifts is not None:
            vectors = vectors + shifts
        distance = torch.linalg.vector_norm(vectors, dim=-1)
        radial = self.radial(distance)
        numbers = data["atomic_numbers"]
        features = self.embedding(numbers)
        invariant_layers = []
        learned = positions.new_zeros(positions.shape[0])
        for interaction, readout in zip(
            self.interactions, self.readouts, strict=True
        ):
            features = interaction(features, edge_index, radial)
            invariant_layers.append(features)
            learned = learned + readout(features).reshape(-1)
        reference = self.atomic_energy_table[numbers].to(dtype=positions.dtype)
        node_energy = reference + learned
        if self.pair_repulsion and edge_index.numel():
            x = (distance / self.r_max).clamp(0.0, 1.0)
            envelope = 1.0 - 10.0 * x.pow(3) + 15.0 * x.pow(4) - 6.0 * x.pow(5)
            pair = self._zbl_pair_energy(numbers, edge_index, distance)
            pair = pair * envelope * (distance < self.r_max).to(distance.dtype)
            # Complete radius graphs contain both directed orientations.
            node_energy = node_energy + _scatter_sum(
                0.5 * pair, target, positions.shape[0]
            )
        batch = data["batch"]
        graph_count = int(batch.max().item()) + 1 if batch.numel() else 0
        energy = _scatter_sum(node_energy, batch, graph_count)
        return {
            "energy": energy,
            "node_energy": node_energy,
            "invariant_features": torch.cat(invariant_layers, dim=-1),
            "raw_output": {
                "node_feats": features,
                "node_energy": node_energy,
                "energy": energy,
            },
        }


class ConvolutionBackboneAdapter(BackboneAdapter):
    def __init__(self, core: ConvolutionCore, config: Any) -> None:
        super().__init__(
            core,
            kind="convolution",
            architecture="continuous-filter-distance-convolution",
            implementation="zivar-native-convolution-0.1.0",
            invariant_dim=int(config.channels) * int(config.num_interactions),
            atomic_numbers=tuple(config.atomic_numbers),
            cutoff_A=float(config.cutoff_A),
            capabilities=BackboneCapabilities(maximum_ell=0, local_mliap=False),
        )

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        return self.model(data)


def build_convolution_backbone(
    config: Any, *, device: Any = "cpu"
) -> ConvolutionBackboneAdapter:
    if config.backend not in {"auto", "e3nn"}:
        raise ValueError("convolution backbone has no tensor-product converter")
    core = ConvolutionCore(config).to(device)
    core.zivar_acceleration_backend = "torch-native"
    return ConvolutionBackboneAdapter(core, config)


def register() -> None:
    register_backbone(
        "convolution",
        build_convolution_backbone,
        description="Strict E(3)-invariant continuous-filter convolution baseline",
        provenance="Native distance-only message passing implementation",
    )


__all__ = [
    "ContinuousFilterBlock",
    "ConvolutionBackboneAdapter",
    "ConvolutionCore",
    "GaussianRadialBasis",
    "build_convolution_backbone",
    "register",
]
