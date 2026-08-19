"""Production spin-resolved polar Gaussian-multipole density.

The model is non-self-consistent by construction: a fixed number of
equivariant field updates refines a complete STF multipole family.  Charge and
spin monopoles are re-normalised after every update with positive Fukui
weights.  The final energy always contains the explicit Coulomb functional and
a restricted invariant non-local correction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ._deps import require_torch
from .config import ElectronicConfig
from .electrostatics import (
    ElectrostaticResult,
    electrostatic_energy,
    electrostatic_potential_features,
)
from .multipoles import (
    SpinMultipoleState,
    multipole_slice,
    solid_harmonic_features,
)

torch = require_torch()
nn = torch.nn
functional = torch.nn.functional


def scatter_sum(values: Any, index: Any, size: int) -> Any:
    result = values.new_zeros((size, *values.shape[1:]))
    result.index_add_(0, index, values)
    return result


def graph_count(batch: Any) -> int:
    return int(batch.max().item()) + 1 if batch.numel() else 0


def graph_target(
    value: Any | None,
    count: int,
    reference: Any,
    *,
    default: float = 0.0,
) -> Any:
    if value is None:
        return reference.new_full((count,), float(default))
    value = value.to(device=reference.device, dtype=reference.dtype)
    if value.ndim == 0 or value.shape == (1,):
        return value.reshape(1).expand(count)
    if value.shape != (count,):
        raise ValueError("condition must be scalar or have shape [B]")
    return value


def positive_fukui_projection(
    values: Any,
    logits: Any,
    batch: Any,
    target: Any,
    *,
    floor: float,
    mobility: Any | None = None,
) -> tuple[Any, Any]:
    """Minimum positive-weight correction satisfying exact graph totals."""

    if values.shape != batch.shape or logits.shape != values.shape:
        raise ValueError("Fukui projection requires [N] values and logits")
    weights = functional.softplus(logits) + float(floor)
    if mobility is not None:
        if mobility.shape != values.shape:
            raise ValueError("mobility must have shape [N]")
        weights = weights * mobility.to(device=values.device, dtype=values.dtype)
    count = int(target.shape[0])
    denominator = scatter_sum(weights, batch, count)
    current = scatter_sum(values, batch, count)
    tolerance = 100.0 * torch.finfo(values.dtype).eps
    impossible = (denominator <= tolerance) & ((target - current).abs() > tolerance)
    if bool(torch.any(impossible).detach()):
        raise ValueError("a constrained graph has no mobile electronic degrees of freedom")
    safe = denominator.clamp_min(tolerance)
    corrected = values + weights * (target - current)[batch] / safe[batch]
    return corrected, weights


def _weighted_region_projection(
    values: Any,
    logits: Any,
    batch: Any,
    target: Any,
    region: Any,
    *,
    floor: float,
) -> tuple[Any, Any]:
    """Constrain a weighted closed region without constraining its complement."""

    region = region.to(device=values.device, dtype=values.dtype)
    if region.shape != values.shape or bool(torch.any((region < 0) | (region > 1)).detach()):
        raise ValueError("constraint region must have shape [N] and values in [0,1]")
    weights = (functional.softplus(logits) + float(floor)) * region
    count = int(target.shape[0])
    current = scatter_sum(values * region, batch, count)
    denominator = scatter_sum(weights, batch, count)
    tolerance = 100.0 * torch.finfo(values.dtype).eps
    impossible = (denominator <= tolerance) & ((target - current).abs() > tolerance)
    if bool(torch.any(impossible).detach()):
        raise ValueError("a constrained region has no mobile charge degrees of freedom")
    corrected = (
        values
        + weights
        * (target - current)[batch]
        / denominator.clamp_min(tolerance)[batch]
    )
    return corrected, weights


def _require_binary_mask(value: Any, shape: tuple[int, ...], name: str) -> Any:
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {list(shape)}")
    tolerance = 100.0 * torch.finfo(value.dtype).eps
    binary = (value - value.round()).abs() <= tolerance
    if bool(torch.any(~binary).detach()) or bool(torch.any((value < 0) | (value > 1)).detach()):
        raise ValueError(f"{name} must contain only zero/one values")
    return value


def constrain_charge_monopoles(
    values: Any,
    logits: Any,
    batch: Any,
    conditions: dict[str, Any],
    config: ElectronicConfig,
) -> tuple[Any, Any, Any | None]:
    """Apply fixed-charge, fixed-potential, mixed and fragment constraints."""

    count = graph_count(batch)
    weights = functional.softplus(logits) + config.fukui_floor
    fragment_membership = conditions.get("fragment_membership")
    fragment_target = conditions.get("fragment_charge")
    occupied = values.new_zeros(values.shape)
    if fragment_target is not None and fragment_membership is None:
        raise ValueError("fragment_charge requires fragment_membership")
    if fragment_membership is not None:
        membership = fragment_membership.to(device=values.device, dtype=values.dtype)
        if membership.ndim != 2 or membership.shape[0] != values.shape[0]:
            raise ValueError("fragment_membership must have shape [N,F]")
        membership = _require_binary_mask(
            membership, tuple(membership.shape), "fragment_membership"
        )
        if bool(torch.any(membership.sum(-1) > 1.0).detach()):
            raise ValueError("fragment memberships must be disjoint")
        if fragment_target is None:
            raise ValueError("fragment constraints require fragment_charge")
        target = fragment_target.to(device=values.device, dtype=values.dtype)
        if target.shape != (membership.shape[1],):
            raise ValueError("fragment_charge must have shape [F]")
        graph_presence = scatter_sum(
            membership, batch, count
        ) > 0
        if bool(torch.any(graph_presence.sum(0) > 1).detach()):
            raise ValueError("each fragment must be confined to one graph")
        weighted = weights[:, None] * membership
        denominator = weighted.sum(0)
        if bool(torch.any(denominator <= 0).detach()):
            raise ValueError("every constrained fragment must contain mobile atoms")
        current = (membership * values[:, None]).sum(0)
        values = values + (weighted * ((target - current) / denominator)[None]).sum(1)
        occupied = membership.sum(-1).clamp(0.0, 1.0)
    fixed_mask = conditions.get("fixed_charge_mask")
    fixed_values = conditions.get("fixed_charges")
    if fixed_values is not None and fixed_mask is None:
        raise ValueError("fixed_charges requires fixed_charge_mask")
    if fixed_mask is not None:
        mask = fixed_mask.to(device=values.device, dtype=values.dtype)
        mask = _require_binary_mask(mask, tuple(values.shape), "fixed_charge_mask")
        if fragment_membership is not None and bool(
            torch.any(mask[:, None] * membership > 0).detach()
        ):
            raise ValueError("fixed-charge atoms cannot overlap constrained fragments")
        if fixed_values is None:
            raise ValueError("fixed_charge_mask requires fixed_charges")
        prescribed = fixed_values.to(device=values.device, dtype=values.dtype)
        if prescribed.shape != values.shape:
            raise ValueError("fixed_charges must have shape [N]")
        values = values * (1.0 - mask) + prescribed * mask
        occupied = torch.maximum(occupied, mask)
    free = (1.0 - occupied).clamp(0.0, 1.0)
    if config.boundary_mode == "fixed_potential":
        return values, weights, None
    if config.boundary_mode == "mixed":
        reservoir = conditions.get("reservoir_mask")
        closed_target = conditions.get("closed_region_charge")
        if reservoir is None or closed_target is None:
            raise ValueError("mixed boundary requires reservoir_mask and closed_region_charge")
        reservoir = reservoir.to(device=values.device, dtype=values.dtype)
        reservoir = _require_binary_mask(
            reservoir, tuple(values.shape), "reservoir_mask"
        )
        closed_all = 1.0 - reservoir
        closed = closed_all * free
        target = graph_target(closed_target, count, values)
        fixed_in_closed = scatter_sum(
            values * closed_all * (1.0 - free), batch, count
        )
        values, weights = _weighted_region_projection(
            values, logits, batch, target - fixed_in_closed, closed,
            floor=config.fukui_floor
        )
        return values, weights, None
    target = graph_target(conditions.get("total_charge"), count, values)
    values, weights = positive_fukui_projection(
        values, logits, batch, target, floor=config.fukui_floor, mobility=free
    )
    return values, weights, target


def project_vector_totals(
    vectors: Any,
    logits: Any,
    batch: Any,
    target: Any,
    *,
    floor: float,
) -> tuple[Any, Any]:
    """O(3)-covariant and time-odd vector Fukui equilibration."""

    if vectors.ndim != 2 or vectors.shape[1] != 3:
        raise ValueError("vector constraint requires shape [N,3]")
    if target.shape != (graph_count(batch), 3):
        raise ValueError("vector target must have shape [B,3]")
    components = []
    weights = None
    for axis in range(3):
        component, weights = positive_fukui_projection(
            vectors[:, axis], logits, batch, target[:, axis], floor=floor
        )
        components.append(component)
    return torch.stack(components, dim=-1), weights


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


def _smooth_cutoff(x: Any) -> Any:
    clipped = x.clamp(0.0, 1.0)
    return torch.where(
        x < 1.0,
        1.0 - 10.0 * clipped.pow(3) + 15.0 * clipped.pow(4) - 6.0 * clipped.pow(5),
        torch.zeros_like(x),
    )


def _safe_direction(vector: Any) -> tuple[Any, Any]:
    epsilon = 32.0 * torch.finfo(vector.dtype).eps
    distance = torch.sqrt(vector.square().sum(-1) + epsilon * epsilon)
    return distance, vector / distance[:, None]


def _squash_equivariant(block: Any, scale: float) -> Any:
    scaled = block / float(scale)
    denominator = torch.sqrt(1.0 + scaled.square().sum(-1, keepdim=True))
    return scaled / denominator


def _squash_spin_multipole(block: Any, scale: float) -> Any:
    """Bound a complete axial-vector x spatial-irrep tensor invariantly."""

    scaled = block / float(scale)
    denominator = torch.sqrt(
        1.0 + scaled.square().sum((-2, -1), keepdim=True)
    )
    return scaled / denominator


class LocalDensityScalars(nn.Module):
    names = (
        "charge", "charge_fukui", "spin_fukui", "moment_logit",
        "collinear_spin", "electronegativity", "hardness",
    )

    def __init__(self, feature_dim: int, config: ElectronicConfig) -> None:
        super().__init__()
        self.config = config
        self.network = _mlp(feature_dim, config.hidden, len(self.names))
        _zero_last(self.network)

    def forward(self, features: Any) -> dict[str, Any]:
        raw = self.network(features)
        output = {name: raw[:, index] for index, name in enumerate(self.names)}
        output["hardness"] = (
            functional.softplus(output["hardness"]) + self.config.hardness_floor_eV
        )
        return output


class EquivariantGeometryDensity(nn.Module):
    """Local O(3)-covariant charge and time-odd spin multipoles.

    The spin density is not restricted to ``S_i`` times one spatial shape.
    Independent central-spin and neighbour-spin channels are accumulated on
    every directed edge.  Their sum spans general environment-dependent
    axial-vector x rank-``l`` spatial tensors within the selected finite
    Gaussian/STF basis.
    """

    def __init__(self, feature_dim: int, config: ElectronicConfig) -> None:
        super().__init__()
        self.config = config
        self.register_buffer(
            "radial_centers",
            torch.linspace(0.0, 1.0, config.radial_basis),
            persistent=True,
        )
        output = 3 * config.density_lmax
        self.edge_network = _mlp(
            2 * feature_dim + config.radial_basis,
            config.hidden,
            output,
        ) if output else None
        if self.edge_network is not None:
            _zero_last(self.edge_network)

    def forward(
        self,
        features: Any,
        positions: Any,
        edge_index: Any,
        shifts: Any | None,
        cutoff_A: float,
        spin_vectors: Any | None,
    ) -> tuple[Any, Any]:
        dimension = self.config.multipole_dim
        charge = positions.new_zeros((positions.shape[0], dimension))
        spin_density = positions.new_zeros((positions.shape[0], 3, dimension))
        if self.config.density_lmax == 0 or edge_index.numel() == 0:
            return charge, spin_density
        source, target = edge_index[0], edge_index[1]
        vector = positions[target] - positions[source]
        if shifts is not None:
            vector = vector + shifts
        distance, direction = _safe_direction(vector)
        x = distance / float(cutoff_A)
        spacing = 1.0 / max(1, self.config.radial_basis - 1)
        radial = torch.exp(-((x[:, None] - self.radial_centers) / spacing).square())
        weight = self.edge_network(
            torch.cat((features[source], features[target], radial), dim=-1)
        ) * _smooth_cutoff(x)[:, None]
        harmonics = solid_harmonic_features(direction, self.config.density_lmax)
        for ell in range(1, self.config.density_lmax + 1):
            block = multipole_slice(ell)
            charge[:, block] = scatter_sum(
                weight[:, ell - 1, None] * harmonics[ell], target, positions.shape[0]
            )
            if spin_vectors is not None:
                central = spin_vectors[target]
                neighbour = spin_vectors[source]
                central_weight = weight[:, self.config.density_lmax + ell - 1]
                neighbour_weight = weight[:, 2 * self.config.density_lmax + ell - 1]
                edge_spin = (
                    central_weight[:, None] * central
                    + neighbour_weight[:, None] * neighbour
                )[:, :, None] * harmonics[ell][:, None, :]
                spin_density[:, :, block] = scatter_sum(
                    edge_spin, target, positions.shape[0]
                )
        return charge, spin_density


class EquivariantPolarUpdate(nn.Module):
    """Scalar gates acting on complete equivariant multipole blocks."""

    def __init__(self, feature_dim: int, config: ElectronicConfig) -> None:
        super().__init__()
        self.config = config
        invariant_dim = feature_dim + 4 * (config.density_lmax + 1) + 2
        # density/potential/geometry gates for charge and spin, plus two Fukui deltas
        output_dim = 6 * (config.density_lmax + 1) + 2
        self.network = _mlp(invariant_dim, config.hidden, output_dim)
        _zero_last(self.network)

    def forward(
        self,
        features: Any,
        state: SpinMultipoleState,
        geometry_charge: Any,
        geometry_spin: Any,
        potential: Any,
        charge_fukui: Any,
        spin_fukui: Any,
    ) -> tuple[SpinMultipoleState, Any, Any]:
        invariants = [features, state.charges[:, None],
                      state.magnetic_moments.square().sum(-1, keepdim=True)]
        for ell in range(self.config.density_lmax + 1):
            block = multipole_slice(ell)
            density = state.charge[:, block]
            field = potential[:, block]
            geometry = geometry_charge[:, block]
            invariants.extend((
                density.square().sum(-1, keepdim=True),
                field.square().sum(-1, keepdim=True),
                geometry.square().sum(-1, keepdim=True),
                (density * field).sum(-1, keepdim=True),
            ))
        gates = self.network(torch.cat(invariants, dim=-1))
        rank_count = self.config.density_lmax + 1
        charge_gates = gates[:, : 3 * rank_count].reshape(-1, rank_count, 3)
        spin_gates = gates[:, 3 * rank_count : 6 * rank_count].reshape(
            -1, rank_count, 3
        )
        next_charge = state.charge.clone()
        next_spin = state.spin.clone()
        moment = state.magnetic_moments
        moment_scale = torch.sqrt(1.0 + moment.square().sum(-1, keepdim=True))
        unit_like = moment / moment_scale
        for ell in range(rank_count):
            block = multipole_slice(ell)
            field = _squash_equivariant(
                potential[:, block], self.config.potential_scale_eV
            )
            density = _squash_equivariant(state.charge[:, block], 1.0)
            geometry = _squash_equivariant(geometry_charge[:, block], 1.0)
            cg = torch.tanh(charge_gates[:, ell])
            delta = (
                cg[:, 0, None] * field
                + cg[:, 1, None] * density
                + cg[:, 2, None] * geometry
            )
            next_charge[:, block] = (
                next_charge[:, block] + self.config.multipole_update_scale * delta
            )
            sg = torch.tanh(spin_gates[:, ell])
            if ell > 0:
                spin_geometry = _squash_spin_multipole(
                    geometry_spin[:, :, block], 1.0
                )
                spin_density = _squash_spin_multipole(
                    state.spin[:, :, block], 1.0
                )
                moment_field = unit_like[:, :, None] * field[:, None, :]
                spin_delta = (
                    sg[:, 0, None, None] * moment_field
                    + sg[:, 1, None, None] * spin_geometry
                    + sg[:, 2, None, None] * spin_density
                )
                next_spin[:, :, block] = (
                    next_spin[:, :, block]
                    + self.config.multipole_update_scale
                    * spin_delta
                )
        return (
            SpinMultipoleState(next_charge, next_spin),
            charge_fukui + gates[:, -2],
            spin_fukui + gates[:, -1],
        )


@dataclass(slots=True)
class PolarPrediction:
    state: SpinMultipoleState
    energy: Any
    atomic_energy: Any
    learned_energy: Any
    coulomb_energy: Any
    external_energy: Any
    charge_fukui: Any
    spin_fukui: Any
    potential_coefficients: Any
    residual: Any
    spin_residual: Any
    electrostatic_backend: tuple[str, ...]
    electronegativity: Any
    hardness: Any
    auxiliary_magnitudes: Any


class PolarDensityModel(nn.Module):
    """Backbone-neutral production polar spin-charge density."""

    def __init__(self, feature_dim: int, config: ElectronicConfig) -> None:
        super().__init__()
        self.config = config
        self.local = LocalDensityScalars(feature_dim, config)
        self.geometry = EquivariantGeometryDensity(feature_dim, config)
        self.updates = nn.ModuleList(
            EquivariantPolarUpdate(feature_dim, config)
            for _ in range(config.polarization_updates)
        )
        descriptor_dim = feature_dim + 4 * (config.density_lmax + 1) + 2
        self.nonlocal_energy = _mlp(descriptor_dim, config.hidden, 1)
        _zero_last(self.nonlocal_energy)

    def _initial_state(
        self,
        local: dict[str, Any],
        geometry_charge: Any,
        geometry_spin: Any,
        batch: Any,
        conditions: dict[str, Any],
        spin_vectors: Any | None,
    ) -> tuple[SpinMultipoleState, Any, Any, Any, Any]:
        count = graph_count(batch)
        charge, charge_fukui, target_charge = constrain_charge_monopoles(
            local["charge"], local["charge_fukui"], batch, conditions,
            self.config,
        )
        charge_density = geometry_charge.clone()
        charge_density[:, 0] = charge
        if spin_vectors is None:
            spin_vectors = charge.new_zeros((charge.shape[0], 3))
            spin_vectors[:, 2] = local["collinear_spin"]
            requested = conditions.get("total_spin")
            if requested is not None:
                scalar, spin_fukui = positive_fukui_projection(
                    spin_vectors[:, 2], local["spin_fukui"], batch,
                    graph_target(requested, count, charge),
                    floor=self.config.fukui_floor,
                )
                spin_vectors[:, 2] = scalar
            else:
                spin_fukui = functional.softplus(local["spin_fukui"]) + self.config.fukui_floor
        else:
            spin_vectors = spin_vectors.to(device=charge.device, dtype=charge.dtype)
            if spin_vectors.shape != (charge.shape[0], 3):
                raise ValueError("spin_vectors must have shape [N,3]")
            spin_fukui = functional.softplus(local["spin_fukui"]) + self.config.fukui_floor
        spin_density = charge.new_zeros((charge.shape[0], 3, self.config.multipole_dim))
        spin_density[:, :, 0] = spin_vectors
        if self.config.density_lmax:
            spin_density[:, :, 1:] = geometry_spin[:, :, 1:]
        state = SpinMultipoleState(charge_density, spin_density)
        target_spin = conditions.get("total_magnetization")
        if target_spin is None:
            target_spin = scatter_sum(spin_vectors, batch, count)
        else:
            target_spin = target_spin.to(device=charge.device, dtype=charge.dtype)
        return state, charge_fukui, spin_fukui, target_charge, target_spin

    def _constrain_state(
        self,
        state: SpinMultipoleState,
        charge_logits: Any,
        spin_logits: Any,
        batch: Any,
        charge_target: Any,
        spin_target: Any,
        conditions: dict[str, Any],
    ) -> tuple[SpinMultipoleState, Any, Any, Any | None]:
        charge, charge_weights, charge_target = constrain_charge_monopoles(
            state.charges, charge_logits, batch, conditions,
            self.config,
        )
        charge_density = state.charge.clone()
        charge_density[:, 0] = charge
        spin_density = state.spin
        spin_weights = functional.softplus(spin_logits) + self.config.fukui_floor
        if self.config.enforce_spin_each_update:
            moment, spin_weights = project_vector_totals(
                state.magnetic_moments, spin_logits, batch, spin_target,
                floor=self.config.fukui_floor,
            )
            spin_density = spin_density.clone()
            spin_density[:, :, 0] = moment
        return (
            SpinMultipoleState(charge_density, spin_density),
            charge_weights,
            spin_weights,
            charge_target,
        )

    def _energy_descriptors(
        self,
        features: Any,
        state: SpinMultipoleState,
        potential: Any,
    ) -> Any:
        descriptor = [features, state.charges[:, None],
                      state.magnetic_moments.square().sum(-1, keepdim=True)]
        for ell in range(self.config.density_lmax + 1):
            block = multipole_slice(ell)
            density = state.charge[:, block]
            field = potential[:, block]
            descriptor.extend((
                density.square().sum(-1, keepdim=True),
                field.square().sum(-1, keepdim=True),
                (density * field).sum(-1, keepdim=True),
                state.spin[:, :, block].square().sum((1, 2), keepdim=False)[:, None],
            ))
        return torch.cat(descriptor, dim=-1)

    def _external_energy(
        self,
        state: SpinMultipoleState,
        positions: Any,
        batch: Any,
        conditions: dict[str, Any],
    ) -> Any:
        count = graph_count(batch)
        atomic = positions.new_zeros(positions.shape[0])
        electrode = conditions.get("electrode_potential")
        if electrode is not None:
            electrode = electrode.to(device=positions.device, dtype=positions.dtype)
            if electrode.ndim == 0 or electrode.shape == (1,):
                electrode = electrode.reshape(1).expand(count)[batch]
            elif electrode.shape == (count,):
                electrode = electrode[batch]
            elif electrode.shape != state.charges.shape:
                raise ValueError("electrode_potential must be scalar, per graph or per atom")
            atomic = atomic - electrode * state.charges
        electric = conditions.get("external_electric_field")
        if electric is not None:
            electric = electric.to(device=positions.device, dtype=positions.dtype)
            if electric.shape != (count, 3):
                raise ValueError("external_electric_field must have shape [B,3]")
            origin = conditions.get("electric_field_origin")
            if origin is None:
                origin = positions.new_zeros((count, 3))
            origin = origin.to(device=positions.device, dtype=positions.dtype)
            atomic = atomic - state.charges * (
                (positions - origin[batch]) * electric[batch]
            ).sum(-1)
            atomic = atomic - (state.dipoles * electric[batch]).sum(-1)
        return atomic

    def forward(
        self,
        features: Any,
        *,
        positions: Any,
        batch: Any,
        edge_index: Any,
        shifts: Any | None,
        cell: Any | None,
        pbc: Any | None,
        cutoff_A: float,
        conditions: dict[str, Any] | None = None,
        spin_vectors: Any | None = None,
    ) -> PolarPrediction:
        conditions = dict(conditions or {})
        local = self.local(features)
        geometry_charge, geometry_spin = self.geometry(
            features, positions, edge_index, shifts, cutoff_A, spin_vectors
        )
        state, charge_fukui, spin_fukui, charge_target, spin_target = self._initial_state(
            local, geometry_charge, geometry_spin, batch, conditions, spin_vectors
        )
        potential = positions.new_zeros(
            (positions.shape[0], (self.config.potential_lmax + 1) ** 2)
        )
        backends: tuple[str, ...] = ()
        for update in self.updates:
            field = electrostatic_potential_features(
                state, positions, batch, cell, pbc, self.config,
                conditions=conditions,
            )
            potential, backends = field.coefficients, field.backend
            state, charge_fukui, spin_fukui = update(
                features, state, geometry_charge, geometry_spin, potential,
                charge_fukui, spin_fukui,
            )
            state, charge_fukui, spin_fukui, charge_target = self._constrain_state(
                state, charge_fukui, spin_fukui, batch, charge_target, spin_target,
                conditions,
            )
        field = electrostatic_potential_features(
            state, positions, batch, cell, pbc, self.config, conditions=conditions
        )
        potential, backends = field.coefficients, field.backend
        electrostatic: ElectrostaticResult = electrostatic_energy(
            state, positions, batch, cell, pbc, self.config
        )
        learned_atomic = self.config.learned_energy_scale_eV * self.nonlocal_energy(
            self._energy_descriptors(features, state, potential)
        ).reshape(-1)
        external_atomic = self._external_energy(state, positions, batch, conditions)
        atomic = electrostatic.atomic_energy + learned_atomic + external_atomic
        count = graph_count(batch)
        energy = scatter_sum(atomic, batch, count)
        charge_residual = (
            state.charges.new_zeros(count)
            if charge_target is None
            else scatter_sum(state.charges, batch, count) - charge_target
        )
        spin_residual = scatter_sum(state.magnetic_moments, batch, count) - spin_target
        return PolarPrediction(
            state=state,
            energy=energy,
            atomic_energy=atomic,
            learned_energy=scatter_sum(learned_atomic, batch, count),
            coulomb_energy=electrostatic.energy,
            external_energy=scatter_sum(external_atomic, batch, count),
            charge_fukui=charge_fukui,
            spin_fukui=spin_fukui,
            potential_coefficients=potential,
            residual=charge_residual,
            spin_residual=spin_residual,
            electrostatic_backend=backends,
            electronegativity=local["electronegativity"],
            hardness=local["hardness"],
            auxiliary_magnitudes=auxiliary_magnitude(state, local),
        )


def auxiliary_magnitude(state: SpinMultipoleState, local: dict[str, Any]) -> Any:
    physical = torch.sqrt(
        state.magnetic_moments.square().sum(-1) + torch.finfo(state.charge.dtype).eps
    )
    predicted = functional.softplus(local["moment_logit"])
    has_spin = state.magnetic_moments.square().sum(-1) > torch.finfo(state.charge.dtype).eps
    return torch.where(has_spin, physical, predicted)


__all__ = [
    "PolarDensityModel", "PolarPrediction", "constrain_charge_monopoles",
    "graph_count", "graph_target",
    "positive_fukui_projection", "project_vector_totals", "scatter_sum",
]
