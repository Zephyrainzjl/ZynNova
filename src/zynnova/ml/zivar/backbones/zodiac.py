"""Zodiac: native E(3)-equivariant tensor-field message-passing backbone.

This is an independent implementation of the tensor-field convolution family:
edge spherical harmonics are coupled to node irreps with Clebsch--Gordan tensor
products, radial networks generate per-edge tensor-product weights, and only
even scalar channels are exposed to the ZIVAR electronic functional.  It does
not use symmetric ACE contraction and is therefore architecturally independent
of the default backbone family.
"""

from __future__ import annotations

import math
from typing import Any

from .._deps import require_e3nn, require_torch, upstream_warning_guard
from .base import BackboneAdapter, BackboneCapabilities, register_backbone

torch = require_torch()
nn = torch.nn
functional = torch.nn.functional


def _scatter_sum(values: Any, index: Any, size: int) -> Any:
    output = values.new_zeros((size, *values.shape[1:]))
    output.index_add_(0, index, values)
    return output


def _options(config: Any) -> dict[str, Any]:
    return dict(getattr(config, "options", {}) or {})


def _complete_hidden_irreps(config: Any, o3: Any) -> Any:
    options = _options(config)
    explicit = options.get("zodiac_hidden_irreps")
    if explicit is not None:
        irreps = o3.Irreps(str(explicit))
        if irreps.count(o3.Irrep(0, 1)) < 1 or irreps.lmax < config.max_ell:
            raise ValueError(
                "zodiac_hidden_irreps requires 0e channels and every requested ell"
            )
        return irreps
    scalar_channels = int(config.channels)
    equivariant_channels = int(
        options.get("equivariant_channels", max(2, scalar_channels // 12))
    )
    if equivariant_channels < 1:
        raise ValueError("equivariant_channels must be positive")
    families: list[tuple[int, tuple[int, int]]] = [
        (scalar_channels, (0, 1)),
        (equivariant_channels, (0, -1)),
    ]
    for ell in range(1, int(config.max_ell) + 1):
        # Both tensor parities retain polar and axial coupling paths.
        families.extend(
            (
                (equivariant_channels, (ell, (-1) ** ell)),
                (equivariant_channels, (ell, -((-1) ** ell))),
            )
        )
    return o3.Irreps(families)


class SmoothRadialBasis(nn.Module):
    """Bessel basis times a compact C2 polynomial cutoff."""

    def __init__(self, cutoff_A: float, count: int, order: int) -> None:
        super().__init__()
        frequencies = math.pi / cutoff_A * torch.arange(1, count + 1)
        self.register_buffer("frequencies", frequencies)
        self.register_buffer("cutoff", torch.tensor(float(cutoff_A)))
        self.register_buffer("prefactor", torch.tensor(math.sqrt(2.0 / cutoff_A)))
        self.order = int(order)

    def forward(self, distances: Any) -> Any:
        radius = distances.reshape(-1, 1)
        numerator = torch.sin(radius * self.frequencies)
        limit = self.frequencies.expand_as(numerator)
        basis = torch.where(radius.abs() > 1.0e-12, numerator / radius, limit)
        x = radius / self.cutoff
        p = float(self.order)
        envelope = (
            1.0
            - ((p + 1.0) * (p + 2.0) / 2.0) * x.pow(self.order)
            + p * (p + 2.0) * x.pow(self.order + 1)
            - (p * (p + 1.0) / 2.0) * x.pow(self.order + 2)
        )
        envelope = envelope * (radius < self.cutoff).to(radius.dtype)
        return self.prefactor * basis * envelope


class EquivariantNormGate(nn.Module):
    """Parity-safe scalar activations and norm gates for higher irreps."""

    def __init__(self, irreps: Any) -> None:
        super().__init__()
        self.irreps = irreps
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        for multiplicity, irrep in irreps:
            if irrep.l == 0:
                self.weights.append(nn.Parameter(torch.empty(0), requires_grad=False))
                self.biases.append(nn.Parameter(torch.empty(0), requires_grad=False))
            else:
                self.weights.append(nn.Parameter(torch.ones(multiplicity)))
                self.biases.append(nn.Parameter(torch.zeros(multiplicity)))

    def forward(self, features: Any) -> Any:
        blocks = []
        for index, ((multiplicity, irrep), block_slice) in enumerate(
            zip(self.irreps, self.irreps.slices(), strict=True)
        ):
            block = features[:, block_slice].reshape(
                features.shape[0], multiplicity, irrep.dim
            )
            if irrep.l == 0:
                activated = functional.silu(block) if irrep.p == 1 else torch.tanh(block)
            else:
                norm = block.square().mean(-1).add(1.0e-12).sqrt()
                gate = torch.sigmoid(
                    norm * self.weights[index][None] + self.biases[index][None]
                )
                activated = block * gate[..., None]
            blocks.append(activated.reshape(features.shape[0], -1))
        return torch.cat(blocks, dim=-1)


class TensorFieldInteraction(nn.Module):
    def __init__(
        self,
        irreps: Any,
        spherical_irreps: Any,
        radial_dim: int,
        radial_hidden: tuple[int, ...],
        average_neighbors: float,
        o3: Any,
    ) -> None:
        super().__init__()
        self.self_connection = o3.Linear(irreps, irreps)
        self.tensor_product = o3.FullyConnectedTensorProduct(
            irreps,
            spherical_irreps,
            irreps,
            shared_weights=False,
            internal_weights=False,
        )
        layers: list[Any] = []
        current = radial_dim
        for width in radial_hidden:
            layers.extend((nn.Linear(current, int(width)), nn.SiLU()))
            current = int(width)
        layers.append(nn.Linear(current, self.tensor_product.weight_numel))
        self.radial_network = nn.Sequential(*layers)
        self.gate = EquivariantNormGate(irreps)
        self.normalization = math.sqrt(float(average_neighbors))

    def forward(
        self,
        features: Any,
        edge_index: Any,
        spherical: Any,
        radial: Any,
    ) -> Any:
        source, target = edge_index[0], edge_index[1]
        weights = self.radial_network(radial)
        messages = self.tensor_product(features[source], spherical, weights)
        aggregate = _scatter_sum(messages, target, features.shape[0])
        update = self.self_connection(features) + aggregate / self.normalization
        return features + self.gate(update)


class ZodiacCore(nn.Module):
    """Short-range conservative energy model implementing the common contract."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        require_e3nn()
        from e3nn import o3

        self.irreps = _complete_hidden_irreps(config, o3)
        self.spherical_irreps = o3.Irreps.spherical_harmonics(config.max_ell)
        self.spherical_harmonics = o3.SphericalHarmonics(
            self.spherical_irreps,
            normalize=True,
            normalization="component",
        )
        self.scalar_channels = self.irreps.count(o3.Irrep(0, 1))
        self.scalar_slice = next(
            block_slice
            for (_, irrep), block_slice in zip(
                self.irreps, self.irreps.slices(), strict=True
            )
            if irrep == o3.Irrep(0, 1)
        )
        self.embedding = nn.Linear(len(config.atomic_numbers), self.scalar_channels)
        self.radial_basis = SmoothRadialBasis(
            config.cutoff_A,
            config.num_bessel,
            config.cutoff_polynomial_order,
        )
        self.interactions = nn.ModuleList(
            TensorFieldInteraction(
                self.irreps,
                self.spherical_irreps,
                config.num_bessel,
                tuple(config.radial_mlp),
                config.average_num_neighbors,
                o3,
            )
            for _ in range(config.num_interactions)
        )
        self.readouts = nn.ModuleList(
            o3.Linear(self.irreps, o3.Irreps("1x0e"))
            for _ in range(config.num_interactions)
        )
        atomic_energies = (
            config.atomic_energies_eV
            if config.atomic_energies_eV is not None
            else (0.0,) * len(config.atomic_numbers)
        )
        self.register_buffer("atomic_numbers", torch.tensor(config.atomic_numbers))
        self.register_buffer("atomic_energies", torch.tensor(atomic_energies))
        self.register_buffer("r_max", torch.tensor(float(config.cutoff_A)))
        self.num_interactions = int(config.num_interactions)
        self.cutoff_order = int(config.cutoff_polynomial_order)
        self.pair_repulsion = bool(config.pair_repulsion)

    def _initial_features(self, node_attrs: Any) -> Any:
        scalars = self.embedding(node_attrs)
        prefix = self.scalar_slice.start
        suffix = self.irreps.dim - self.scalar_slice.stop
        blocks = []
        if prefix:
            blocks.append(scalars.new_zeros((scalars.shape[0], prefix)))
        blocks.append(scalars)
        if suffix:
            blocks.append(scalars.new_zeros((scalars.shape[0], suffix)))
        return torch.cat(blocks, dim=-1)

    def _zbl_node_energy(
        self, node_attrs: Any, edge_index: Any, distances: Any
    ) -> Any:
        if not self.pair_repulsion or edge_index.numel() == 0:
            return node_attrs.new_zeros(node_attrs.shape[0])
        source, target = edge_index[0], edge_index[1]
        atomic_numbers = self.atomic_numbers.to(dtype=node_attrs.dtype)
        z = node_attrs @ atomic_numbers
        zi, zj = z[source], z[target]
        radius = distances.reshape(-1).clamp_min(1.0e-9)
        screening_length = 0.8854 * 0.529177210903 / (
            zi.pow(0.23) + zj.pow(0.23)
        )
        scaled = radius / screening_length
        coefficients = radius.new_tensor((0.1818, 0.5099, 0.2802, 0.02817))
        exponents = radius.new_tensor((3.2, 0.9423, 0.4029, 0.2016))
        screening = (
            coefficients[None] * torch.exp(-scaled[:, None] * exponents[None])
        ).sum(-1)
        x = radius / self.r_max
        p = float(self.cutoff_order)
        cutoff = (
            1.0
            - ((p + 1.0) * (p + 2.0) / 2.0) * x.pow(self.cutoff_order)
            + p * (p + 2.0) * x.pow(self.cutoff_order + 1)
            - (p * (p + 1.0) / 2.0) * x.pow(self.cutoff_order + 2)
        )
        cutoff = cutoff * (radius < self.r_max).to(radius.dtype)
        directed = 0.5 * 14.3996454784255 * zi * zj * screening * cutoff / radius
        return _scatter_sum(directed, target, node_attrs.shape[0])

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        positions = data["positions"]
        edge_index = data["edge_index"]
        source, target = edge_index[0], edge_index[1]
        shifts = data.get("shifts")
        vectors = positions[target] - positions[source]
        if shifts is not None:
            vectors = vectors + shifts
        distances = torch.linalg.vector_norm(vectors, dim=-1, keepdim=True)
        spherical = self.spherical_harmonics(vectors)
        radial = self.radial_basis(distances)
        features = self._initial_features(data["node_attrs"])
        layer_scalars = []
        learned_node_energy = positions.new_zeros(positions.shape[0])
        for interaction, readout in zip(
            self.interactions, self.readouts, strict=True
        ):
            features = interaction(features, edge_index, spherical, radial)
            layer_scalars.append(features[:, self.scalar_slice])
            learned_node_energy = learned_node_energy + readout(features).reshape(-1)
        reference = data["node_attrs"] @ self.atomic_energies.to(
            device=positions.device, dtype=positions.dtype
        )
        repulsion = self._zbl_node_energy(
            data["node_attrs"], edge_index, distances
        )
        node_energy = reference + learned_node_energy + repulsion
        batch = data["batch"]
        graph_count = int(batch.max().item()) + 1 if batch.numel() else 0
        energy = _scatter_sum(node_energy, batch, graph_count)
        return {
            "energy": energy,
            "node_energy": node_energy,
            "invariant_features": torch.cat(layer_scalars, dim=-1),
            "raw_output": {
                "node_feats": features,
                "node_energy": node_energy,
                "energy": energy,
            },
        }


class ZodiacBackboneAdapter(BackboneAdapter):
    def __init__(self, core: ZodiacCore, config: Any) -> None:
        super().__init__(
            core,
            kind="zodiac",
            architecture="e3-tensor-field-message-passing",
            implementation="zivar-native-zodiac-0.1.0",
            invariant_dim=core.scalar_channels * core.num_interactions,
            atomic_numbers=tuple(config.atomic_numbers),
            cutoff_A=float(config.cutoff_A),
            capabilities=BackboneCapabilities(local_mliap=False),
        )

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        return self.model(data)


def build_zodiac_backbone(
    config: Any, *, device: Any = "cpu"
) -> ZodiacBackboneAdapter:
    if config.backend not in {"auto", "e3nn"}:
        raise ValueError(
            "Zodiac uses its native e3nn tensor-field path; cueq/oeq/hybrid "
            "are MACE converter formats and cannot be requested for Zodiac"
        )
    with upstream_warning_guard():
        core = ZodiacCore(config).to(device)
    core.zivar_acceleration_backend = "e3nn-native"
    return ZodiacBackboneAdapter(core, config)


def register() -> None:
    register_backbone(
        "zodiac",
        build_zodiac_backbone,
        description="Native tensor-field E(3)-equivariant message-passing potential",
        provenance="Independent NequIP-family implementation using public e3nn primitives",
    )


__all__ = [
    "EquivariantNormGate",
    "SmoothRadialBasis",
    "TensorFieldInteraction",
    "ZodiacBackboneAdapter",
    "ZodiacCore",
    "build_zodiac_backbone",
    "register",
]
