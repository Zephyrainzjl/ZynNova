"""Time-reversal-equivariant non-collinear spin-lattice Hamiltonian.

The production energy is an explicit function ``E(R, S)``.  It contains
longitudinal Landau terms, isotropic and biquadratic exchange, environment
dependent anisotropic exchange, centrosymmetry-safe Dzyaloshinskii--Moriya
coupling, local crystal anisotropy and a symmetry-complete neural residual.
Effective magnetic fields and torques are obtained by differentiating this
single scalar energy with respect to the input spin vectors in ``model.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ._deps import require_torch
from .config import SpinConfig
from .polar import graph_count, scatter_sum

torch = require_torch()
nn = torch.nn
functional = torch.nn.functional


def _mlp(input_dim: int, hidden: tuple[int, ...], output_dim: int) -> Any:
    layers: list[Any] = []
    current = input_dim
    for width in hidden:
        layers.extend((nn.Linear(current, width), nn.SiLU()))
        current = width
    layers.append(nn.Linear(current, output_dim))
    return nn.Sequential(*layers)


def _zero_last(module: Any) -> None:
    for item in reversed(list(module.modules())):
        if isinstance(item, nn.Linear):
            nn.init.zeros_(item.weight)
            if item.bias is not None:
                nn.init.zeros_(item.bias)
            return


def _safe_norm(vector: Any, floor: float) -> Any:
    return torch.sqrt(vector.square().sum(-1) + float(floor) ** 2)


def _smooth_cutoff(x: Any) -> Any:
    clipped = x.clamp(0.0, 1.0)
    return torch.where(
        x < 1.0,
        1.0 - 10.0 * clipped.pow(3) + 15.0 * clipped.pow(4) - 6.0 * clipped.pow(5),
        torch.zeros_like(x),
    )


def _edge_geometry(
    positions: Any,
    edge_index: Any,
    shifts: Any | None,
    cutoff_A: float,
    floor: float,
) -> tuple[Any, Any, Any]:
    source, target = edge_index[0], edge_index[1]
    vector = positions[target] - positions[source]
    if shifts is not None:
        vector = vector + shifts
    distance = _safe_norm(vector, floor)
    direction = vector / distance[:, None]
    envelope = _smooth_cutoff(distance / float(cutoff_A))
    return distance, direction, envelope


def _radial_basis(distance: Any, cutoff_A: float, count: int) -> Any:
    x = (distance / float(cutoff_A)).clamp(0.0, 1.0)
    frequencies = torch.arange(
        1, count + 1, device=distance.device, dtype=distance.dtype
    )
    # sinc-like Bessel basis with an analytic x=0 limit.
    argument = torch.pi * x[:, None] * frequencies[None]
    basis = torch.sinc(argument / torch.pi)
    return basis * _smooth_cutoff(x)[:, None]


def local_chirality_and_quadrupole(
    positions: Any,
    edge_index: Any,
    shifts: Any | None,
    cutoff_A: float,
    floor: float,
) -> tuple[Any, Any]:
    """Return parity-odd chirality and parity-even local STF rank-two field."""

    atom_count = positions.shape[0]
    source = edge_index[0]
    distance, direction, envelope = _edge_geometry(
        positions, edge_index, shifts, cutoff_A, floor
    )
    x = distance / float(cutoff_A)
    moments = []
    for power in (0, 1, 2):
        weight = envelope * x.pow(power)
        moments.append(scatter_sum(weight[:, None] * direction, source, atom_count))
    chirality = (moments[0] * torch.cross(moments[1], moments[2], dim=-1)).sum(-1)
    identity = torch.eye(3, device=positions.device, dtype=positions.dtype)
    dyadic = direction[:, :, None] * direction[:, None, :] - identity[None] / 3.0
    quadrupole = scatter_sum(envelope[:, None, None] * dyadic, source, atom_count)
    return chirality, quadrupole


@dataclass(slots=True)
class MagneticPrediction:
    energy: Any
    atomic_energy: Any
    spin_vectors: Any
    magnitudes: Any
    chirality: Any
    onsite_energy: Any
    exchange_energy: Any
    biquadratic_energy: Any
    anisotropy_energy: Any
    dmi_energy: Any
    neural_energy: Any
    external_energy: Any


class SpinLatticeHamiltonian(nn.Module):
    """Backbone-neutral O(3), parity and time-reversal respecting energy."""

    def __init__(self, feature_dim: int, config: SpinConfig, radial_count: int = 8) -> None:
        super().__init__()
        self.config = config
        self.radial_count = int(radial_count)
        if self.radial_count < 2:
            raise ValueError("radial_count must be at least two")
        self.onsite = _mlp(feature_dim, config.hidden, 4)
        pair_input = 2 * feature_dim + radial_count
        self.pair = _mlp(pair_input, config.hidden, 4)
        # time-even invariant inputs: m_i^2, m_j^2, dot, dot^2,
        # longitudinal product, DMI pseudoscalar and distance basis.
        self.high_order = _mlp(pair_input + 6, config.hidden, 1)
        # The scalar-magnitude predictor is an explicit compatibility route.
        # It is not instantiated in the non-collinear production model, so it
        # cannot run in parallel with the physical spin Hamiltonian.
        self.auxiliary_magnitude = (
            _mlp(feature_dim, config.hidden, 1)
            if config.mode == "magnitude_auxiliary"
            else None
        )
        _zero_last(self.onsite)
        _zero_last(self.pair)
        _zero_last(self.high_order)
        if self.auxiliary_magnitude is not None:
            _zero_last(self.auxiliary_magnitude)

    def resolve_spins(
        self,
        features: Any,
        conditions: dict[str, Any],
    ) -> tuple[Any, Any]:
        candidate = conditions.get("spin_vectors")
        if candidate is None:
            candidate = conditions.get("initial_magnetic_moments")
        if candidate is not None:
            candidate = candidate.to(device=features.device, dtype=features.dtype)
            if candidate.shape != (features.shape[0], 3):
                raise ValueError("spin_vectors must have shape [N,3]")
            return candidate, torch.linalg.vector_norm(candidate, dim=-1)
        if self.config.mode == "spin_lattice" and self.config.require_spin_input:
            raise ValueError(
                "spin_lattice mode requires conditions['spin_vectors']; "
                "use ZIVARConfig.chgnet_compatible() for scalar moment labels"
            )
        vectors = features.new_zeros((features.shape[0], 3))
        magnitude = features.new_zeros(features.shape[0])
        if self.config.mode == "magnitude_auxiliary":
            if self.auxiliary_magnitude is None:
                raise RuntimeError("magnitude auxiliary head is unavailable")
            magnitude = functional.softplus(
                self.auxiliary_magnitude(features).reshape(-1)
            )
        if self.config.mode == "collinear_density":
            scalar = conditions.get("collinear_spins")
            if scalar is not None:
                scalar = scalar.to(device=features.device, dtype=features.dtype)
                if scalar.shape != (features.shape[0],):
                    raise ValueError("collinear_spins must have shape [N]")
                vectors[:, 2] = scalar
                magnitude = scalar.abs()
        return vectors, magnitude

    def forward(
        self,
        features: Any,
        *,
        positions: Any,
        batch: Any,
        edge_index: Any,
        shifts: Any | None,
        cutoff_A: float,
        conditions: dict[str, Any] | None = None,
        spin_vectors: Any | None = None,
    ) -> MagneticPrediction:
        conditions = dict(conditions or {})
        if spin_vectors is None:
            spin_vectors, magnitude = self.resolve_spins(features, conditions)
        else:
            if spin_vectors.shape != (features.shape[0], 3):
                raise ValueError("spin_vectors must have shape [N,3]")
            magnitude = torch.linalg.vector_norm(spin_vectors, dim=-1)
        count = graph_count(batch)
        zero_atomic = positions.new_zeros(positions.shape[0])
        if self.config.mode != "spin_lattice":
            zero_graph = positions.new_zeros(count)
            return MagneticPrediction(
                zero_graph, zero_atomic, spin_vectors, magnitude,
                zero_atomic, zero_graph, zero_graph, zero_graph, zero_graph,
                zero_graph, zero_graph, zero_graph,
            )
        source, target = edge_index[0], edge_index[1]
        distance, direction, envelope = _edge_geometry(
            positions, edge_index, shifts, cutoff_A, self.config.minimum_moment
        )
        radial = _radial_basis(distance, cutoff_A, self.radial_count)
        symmetric_features = torch.cat(
            (
                features[source] + features[target],
                (features[source] - features[target]).abs(),
                radial,
            ),
            dim=-1,
        )
        coefficients = self.pair(symmetric_features)
        spin_i, spin_j = spin_vectors[source], spin_vectors[target]
        dot = (spin_i * spin_j).sum(-1)
        longitudinal = (spin_i * direction).sum(-1) * (spin_j * direction).sum(-1)
        cross = torch.cross(spin_i, spin_j, dim=-1)
        chirality, quadrupole = local_chirality_and_quadrupole(
            positions, edge_index, shifts, cutoff_A, self.config.minimum_moment
        )
        edge_chirality = 0.5 * (chirality[source] + chirality[target])
        dmi_invariant = edge_chirality * (direction * cross).sum(-1)
        exchange_edge = (
            self.config.energy_scale_eV
            * torch.tanh(coefficients[:, 0])
            * dot
            * envelope
            if self.config.exchange else torch.zeros_like(dot)
        )
        biquadratic_edge = (
            self.config.energy_scale_eV
            * torch.tanh(coefficients[:, 1])
            * dot.square()
            * envelope
            if self.config.biquadratic_exchange else torch.zeros_like(dot)
        )
        anisotropic_edge = (
            self.config.anisotropy_scale_eV
            * torch.tanh(coefficients[:, 2])
            * longitudinal
            * envelope
            if self.config.anisotropy else torch.zeros_like(dot)
        )
        dmi_edge = (
            self.config.dmi_scale_eV
            * torch.tanh(coefficients[:, 3])
            * dmi_invariant
            * envelope
            if self.config.dmi else torch.zeros_like(dot)
        )
        high_invariants = torch.stack(
            (
                spin_i.square().sum(-1), spin_j.square().sum(-1), dot,
                dot.square(), longitudinal, dmi_invariant,
            ),
            dim=-1,
        )
        neural_edge = (
            self.config.energy_scale_eV
            * self.high_order(torch.cat((symmetric_features, high_invariants), dim=-1)).reshape(-1)
            * envelope
            if self.config.neural_high_order else torch.zeros_like(dot)
        )
        # Directed reciprocal graphs contain every physical interaction twice.
        def pair_graph(value: Any) -> Any:
            return 0.5 * scatter_sum(value, batch[source], count)

        pair_total = exchange_edge + biquadratic_edge + anisotropic_edge + dmi_edge + neural_edge
        pair_atomic = zero_atomic.clone()
        pair_atomic.index_add_(0, source, 0.25 * pair_total)
        pair_atomic.index_add_(0, target, 0.25 * pair_total)
        onsite_coefficients = self.onsite(features)
        m2 = spin_vectors.square().sum(-1)
        onsite_atomic = zero_atomic
        if self.config.onsite_landau:
            a2 = self.config.energy_scale_eV * torch.tanh(onsite_coefficients[:, 0])
            a4 = self.config.energy_scale_eV * torch.tanh(onsite_coefficients[:, 1])
            a6 = self.config.energy_scale_eV * (
                1.0e-6 + functional.softplus(onsite_coefficients[:, 2] - 10.0)
            )
            onsite_atomic = a2 * m2 + a4 * m2.square() + a6 * m2.pow(3)
        local_anisotropy = zero_atomic
        if self.config.anisotropy:
            projected = torch.einsum("ni,nij,nj->n", spin_vectors, quadrupole, spin_vectors)
            local_anisotropy = (
                self.config.anisotropy_scale_eV
                * torch.tanh(onsite_coefficients[:, 3])
                * projected
            )
        external_atomic = zero_atomic
        magnetic_field = conditions.get("external_magnetic_field")
        if magnetic_field is not None and self.config.external_field:
            magnetic_field = magnetic_field.to(device=positions.device, dtype=positions.dtype)
            if magnetic_field.shape != (count, 3):
                raise ValueError("external_magnetic_field must have shape [B,3]")
            external_atomic = -self.config.bohr_magneton_eV_per_T * (
                spin_vectors * magnetic_field[batch]
            ).sum(-1)
        atomic = pair_atomic + onsite_atomic + local_anisotropy + external_atomic
        return MagneticPrediction(
            energy=scatter_sum(atomic, batch, count),
            atomic_energy=atomic,
            spin_vectors=spin_vectors,
            magnitudes=magnitude,
            chirality=chirality,
            onsite_energy=scatter_sum(onsite_atomic, batch, count),
            exchange_energy=pair_graph(exchange_edge),
            biquadratic_energy=pair_graph(biquadratic_edge),
            anisotropy_energy=pair_graph(anisotropic_edge)
            + scatter_sum(local_anisotropy, batch, count),
            dmi_energy=pair_graph(dmi_edge),
            neural_energy=pair_graph(neural_edge),
            external_energy=scatter_sum(external_atomic, batch, count),
        )


def magnetic_torque(spin_vectors: Any, effective_field: Any) -> Any:
    if spin_vectors.shape != effective_field.shape or spin_vectors.shape[-1] != 3:
        raise ValueError("spin_vectors and effective_field must have shape [N,3]")
    return torch.cross(spin_vectors, effective_field, dim=-1)


__all__ = [
    "MagneticPrediction", "SpinLatticeHamiltonian", "local_chirality_and_quadrupole",
    "magnetic_torque",
]
