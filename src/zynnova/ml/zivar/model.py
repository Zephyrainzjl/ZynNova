"""Unified conservative polar-electronic and spin-lattice ZIVAR potential."""

from __future__ import annotations

from typing import Any

from ._deps import require_torch
from .backbones import BackboneAdapter, build_backbone
from .backbones.mace import MaceBackboneAdapter
from .config import ZIVARConfig
from .electronic import StableElectronicModel
from .magnetism import SpinLatticeHamiltonian, magnetic_torque
from .multipoles import SpinMultipoleState, cartesian_to_coefficients, multipole_slice
from .polar import graph_count, scatter_sum
from .types import Conditions, ElectronicState, EnergyBreakdown, ZIVARBatch, ZIVARPrediction
from .variational import VariationalElectroSpinModel

torch = require_torch()
nn = torch.nn


def _normalise_graph_energy(value: Any, count: int) -> Any:
    return value.expand(count) if value.ndim == 0 else value.reshape(count)


def _graph_condition(value: Any, count: int, reference: Any) -> Any:
    value = value.to(device=reference.device, dtype=reference.dtype)
    if value.ndim == 0 or value.shape == (1,):
        return value.reshape(1).expand(count)
    if value.shape != (count,):
        raise ValueError("condition must be scalar or have shape [B]")
    return value


def _batched_cell(cell: Any | None, count: int) -> Any | None:
    if cell is None:
        return None
    if cell.numel() != count * 9:
        raise ValueError("cell must contain one 3x3 matrix per graph")
    return cell.reshape(count, 3, 3)


def _symmetric_voigt(tensor: Any) -> Any:
    tensor = 0.5 * (tensor + tensor.transpose(-1, -2))
    return torch.stack(
        (
            tensor[:, 0, 0], tensor[:, 1, 1], tensor[:, 2, 2],
            tensor[:, 1, 2], tensor[:, 0, 2], tensor[:, 0, 1],
        ),
        dim=-1,
    )


class ZIVAR(nn.Module):
    """Pluggable local backbone plus production electro-spin Hamiltonian."""

    # The current full-model LAMMPS bridge is a Python reference callback.  A
    # model is not advertised as native/Kokkos deployable until its compiled
    # runtime manifest proves that capability.
    global_lammps_deployable = False

    def __init__(self, backbone: Any, config: ZIVARConfig) -> None:
        super().__init__()
        self._backbone_sealed = False
        self.config = config
        if isinstance(backbone, BackboneAdapter):
            adapter = backbone
        elif config.backbone.kind in {"mace", "zephyr"}:
            adapter = MaceBackboneAdapter(
                backbone,
                kind=config.backbone.kind,
                architecture=(
                    "scale-shift-symmetric-contraction"
                    if config.backbone.kind == "mace"
                    else "zephyr-source-parity-symmetric-contraction"
                ),
            )
        else:
            raise TypeError("raw backbones must be wrapped in a registered BackboneAdapter")
        if adapter.kind != config.backbone.kind:
            raise ValueError("backbone adapter kind differs from configuration")
        if adapter.atomic_numbers != tuple(config.backbone.atomic_numbers):
            raise ValueError("backbone element table differs from configuration")
        if abs(adapter.cutoff_A - float(config.backbone.cutoff_A)) > 1.0e-12:
            raise ValueError("backbone cutoff differs from configuration")
        self.backbone = adapter
        self._built_backbone_fingerprint = adapter.manifest.fingerprint
        self.magnetic = SpinLatticeHamiltonian(
            adapter.invariant_dim, config.spin, radial_count=config.electronic.radial_basis
        )
        if config.electronic.method == "variational":
            if config.electronic.oxidation.enabled:
                raise ValueError(
                    "formal oxidation inference must be enabled as an explicit "
                    "post-SCF registered head; legacy inline oxidation heads cannot "
                    "be attached to the variational production core"
                )
            self.variational = VariationalElectroSpinModel(
                adapter.invariant_dim,
                config.electronic,
                config.scf,
                config.electrostatics,
                bohr_magneton_eV_per_T=config.spin.bohr_magneton_eV_per_T,
                constrain_total_magnetization=config.spin.constrain_total_magnetization,
            )
            self.electronic = None
        else:
            self.variational = None
            self.electronic = StableElectronicModel(
                adapter.invariant_dim,
                tuple(config.backbone.atomic_numbers),
                config.electronic,
            )
        self.register_buffer(
            "zivar_atomic_numbers",
            torch.as_tensor(config.backbone.atomic_numbers, dtype=torch.long),
            persistent=True,
        )

    def __setattr__(self, name: str, value: Any) -> None:
        if (
            name == "backbone"
            and getattr(self, "_backbone_sealed", False)
            and hasattr(self, "backbone")
        ):
            raise RuntimeError("backbone architecture is sealed; build a new model")
        super().__setattr__(name, value)

    @property
    def backbone_manifest(self) -> dict[str, Any]:
        return self.backbone.manifest.to_dict()

    @property
    def backbone_sealed(self) -> bool:
        return bool(self._backbone_sealed)

    def seal_backbone(self) -> ZIVAR:
        self._verify_backbone_identity()
        self._backbone_sealed = True
        return self

    def _verify_backbone_identity(self) -> None:
        if self.backbone.manifest.fingerprint != self._built_backbone_fingerprint:
            raise RuntimeError("backbone identity changed after model construction")

    @property
    def cutoff_A(self) -> float:
        return float(self.config.backbone.cutoff_A)

    @property
    def atomic_numbers(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.zivar_atomic_numbers.tolist())

    @property
    def execution_backend(self) -> str:
        return self.backbone.execution_backend

    def compile_backbone(self) -> ZIVAR:
        self.backbone.compile_model(self.config.backbone.compile_mode)
        return self

    def _resolve_atomic_numbers(self, data: dict[str, Any]) -> Any:
        numbers = data.get("atomic_numbers")
        if numbers is not None:
            return numbers.to(device=data["positions"].device, dtype=torch.long)
        attrs = data.get("node_attrs")
        if attrs is None or attrs.ndim != 2 or attrs.shape[1] != len(self.atomic_numbers):
            raise ValueError("data requires atomic_numbers or one-hot node_attrs")
        return self.zivar_atomic_numbers.to(attrs.device)[attrs.argmax(-1)]

    def _forward_variational(
        self,
        model_data: dict[str, Any],
        local: dict[str, Any],
        short_energy: Any,
        *,
        batch: Any,
        count: int,
        conditions: dict[str, Any],
        spin_vectors: Any,
    ) -> dict[str, Any]:
        """Evaluate the one default SCF functional and expose legacy-safe keys."""

        if self.variational is None:
            raise RuntimeError("variational core is unavailable")
        # The unified functional owns the Zeeman coupling of both the
        # dynamical spin S and induced moment m.  Do not also evaluate the
        # SpinLatticeHamiltonian's legacy external-field term.
        spin_conditions = dict(conditions)
        spin_conditions.pop("external_magnetic_field", None)
        magnetic = self.magnetic(
            local["invariant_features"],
            positions=model_data["positions"],
            batch=batch,
            edge_index=model_data["edge_index"],
            shifts=model_data.get("shifts"),
            cutoff_A=self.cutoff_A,
            conditions=spin_conditions,
            spin_vectors=spin_vectors,
        )
        result = self.variational(
            local["invariant_features"],
            positions=model_data["positions"],
            batch=batch,
            edge_index=model_data["edge_index"],
            shifts=model_data.get("shifts"),
            cell=model_data.get("cell"),
            pbc=model_data.get("pbc"),
            cutoff_A=self.cutoff_A,
            conditions=conditions,
            external_spins=spin_vectors,
            short_range_energy=short_energy,
            external_spin_energy=magnetic.energy,
        )
        state = result.state
        breakdown = result.breakdown
        total_energy = breakdown.total
        counts = torch.bincount(batch, minlength=count).to(total_energy.dtype)
        if bool(torch.any(counts == 0).detach()):
            raise ValueError("batch contains an empty structure")
        short_atomic = local.get("node_energy")
        if short_atomic is None:
            short_atomic = short_energy[batch] / counts[batch]
        else:
            short_atomic = short_atomic.reshape(-1)
        electronic_graph = (
            breakdown.charge
            + breakdown.polarization
            + breakdown.quadrupole
            + breakdown.magnetic
            + breakdown.coupling
            + breakdown.external
        )
        electronic_atomic = electronic_graph[batch] / counts[batch]
        atomic_energy = short_atomic + magnetic.atomic_energy + electronic_atomic

        multipole_dim = self.config.electronic.multipole_dim
        charge_coefficients = state.q.new_zeros((state.atom_count, multipole_dim))
        spin_coefficients = state.q.new_zeros((state.atom_count, 3, multipole_dim))
        charge_coefficients[:, 0] = state.q
        spin_coefficients[:, :, 0] = state.m
        if self.config.electronic.density_lmax >= 1:
            charge_coefficients[:, multipole_slice(1)] = cartesian_to_coefficients(
                state.p, 1
            )
        if self.config.electronic.density_lmax >= 2:
            charge_coefficients[:, multipole_slice(2)] = cartesian_to_coefficients(
                state.Q, 2
            )
        legacy_state = SpinMultipoleState(charge_coefficients, spin_coefficients)
        response_matrix = legacy_state.spin_density_matrix()

        target_charge = _graph_condition(
            conditions.get("total_charge", total_energy.new_zeros(count)),
            count,
            total_energy,
        )
        total_charge = scatter_sum(state.q, batch, count)
        charge_residual = total_charge - target_charge
        constraint_residual = result.constraints.residual(state)
        spin_residual = state.m.new_zeros((count, 3))
        if constraint_residual.numel() > count:
            spin_residual = constraint_residual[count:].reshape(count, 3)

        dipole_origin = conditions.get(
            "dipole_origin", conditions.get("electric_field_origin")
        )
        if dipole_origin is None:
            dipole_origin = model_data["positions"].new_zeros((count, 3))
        dipole_origin = torch.as_tensor(
            dipole_origin,
            device=model_data["positions"].device,
            dtype=model_data["positions"].dtype,
        ).reshape(count, 3)
        total_dipole = scatter_sum(
            state.q[:, None]
            * (model_data["positions"] - dipole_origin[batch])
            + state.p,
            batch,
            count,
        )
        inverse_hardness = result.parameters.hardness.reciprocal()
        normalizer = scatter_sum(inverse_hardness, batch, count).clamp_min(
            torch.finfo(inverse_hardness.dtype).tiny
        )
        fukui = inverse_hardness / normalizer[batch]
        coulomb_atomic = 0.5 * state.q * result.electrostatic_potential
        coulomb_energy = scatter_sum(coulomb_atomic, batch, count)
        induced_magnetic_energy = (
            breakdown.magnetic + breakdown.coupling
        )
        external_magnetic_energy = total_energy.new_zeros(count)
        magnetic_field = conditions.get("external_magnetic_field")
        if magnetic_field is not None:
            magnetic_field = torch.as_tensor(
                magnetic_field,
                device=state.q.device,
                dtype=state.q.dtype,
            ).reshape(count, 3)
            external_magnetic_energy = scatter_sum(
                -self.config.spin.bohr_magneton_eV_per_T
                * ((magnetic.spin_vectors + state.m) * magnetic_field[batch]).sum(-1),
                batch,
                count,
            )
        electronic_residual = total_energy.new_full(
            (count,), float(result.report.final_residual)
        )
        electric_field = state.p.new_zeros(state.p.shape)
        finite = (
            total_energy,
            atomic_energy,
            state.q,
            state.p,
            state.Q,
            state.m,
            electronic_residual,
            charge_residual,
            spin_residual,
        )
        if not all(bool(torch.isfinite(value).all().detach()) for value in finite):
            raise FloatingPointError("non-finite variational electro-spin prediction")
        prediction = ZIVARPrediction(
            energy=total_energy,
            state=state,
            breakdown=breakdown,
            atomic_energy=atomic_energy,
            scf_report=result.report,
        )
        output = {
            "prediction": prediction,
            "electronic_state": state,
            "energy_breakdown": breakdown,
            "energy": total_energy,
            "atomic_energy": atomic_energy,
            "short_range_energy": breakdown.short_range,
            "electronic_energy": electronic_graph,
            "magnetic_energy": (
                magnetic.energy + induced_magnetic_energy + external_magnetic_energy
            ),
            "learned_nonlocal_energy": total_energy.new_zeros(count),
            "coulomb_energy": coulomb_energy,
            "polarization_energy": breakdown.polarization,
            "quadrupole_energy": breakdown.quadrupole,
            "induced_magnetic_energy": induced_magnetic_energy,
            "external_electric_energy": breakdown.external - external_magnetic_energy,
            "external_magnetic_energy": external_magnetic_energy,
            "charge_multipoles": charge_coefficients,
            "spin_multipoles": spin_coefficients,
            "charges": state.q,
            "spin_vectors": magnetic.spin_vectors,
            "dynamical_spin_vectors": magnetic.spin_vectors,
            "spin_magnitudes": magnetic.magnitudes,
            "magmom_vectors": state.m,
            "electronic_magmom_vectors": state.m,
            "magmoms": torch.linalg.vector_norm(state.m, dim=-1),
            "collinear_spins": magnetic.spin_vectors[:, 2],
            "total_charge": total_charge,
            "charge_constraint_residual": charge_residual,
            "spin_constraint_residual": spin_residual,
            "total_spin_vector": scatter_sum(magnetic.spin_vectors, batch, count),
            "total_magnetic_moment": scatter_sum(state.m, batch, count),
            "dipoles": state.p,
            "total_dipole": total_dipole,
            "quadrupoles": state.Q,
            "spin_charge_response_matrix": response_matrix,
            "spin_density_matrix": response_matrix,
            "fukui_weights": fukui,
            "spin_fukui_weights": fukui,
            "electrostatic_potential": result.electrostatic_potential,
            "electrostatic_potential_coefficients": torch.cat(
                (
                    result.electrostatic_potential[:, None],
                    state.q.new_zeros((state.atom_count, multipole_dim - 1)),
                ),
                dim=-1,
            ),
            "electric_field": electric_field,
            "electronegativity": result.parameters.electronegativity,
            "hardness": result.parameters.hardness,
            "electronic_method": "variational",
            "electronic_residual": electronic_residual,
            "electronic_converged": result.report.converged,
            "scf_report": result.report,
            "scf_iterations": result.report.iterations,
            "scf_energy_change": result.report.energy_change,
            "scf_energy_error": result.report.energy_error,
            "scf_constraint_residual": result.report.constraint_residual,
            "electrostatic_backend": result.electrostatic_backend,
            "magnetic_chirality": magnetic.chirality,
            "magnetic_onsite_energy": magnetic.onsite_energy,
            "exchange_energy": magnetic.exchange_energy,
            "biquadratic_exchange_energy": magnetic.biquadratic_energy,
            "magnetic_anisotropy_energy": magnetic.anisotropy_energy,
            "dmi_energy": magnetic.dmi_energy,
            "magnetic_neural_energy": magnetic.neural_energy,
        }
        return output

    def forward(
        self,
        data: dict[str, Any] | ZIVARBatch,
        *,
        conditions: dict[str, Any] | Conditions | None = None,
    ) -> dict[str, Any]:
        self._verify_backbone_identity()
        if isinstance(data, ZIVARBatch):
            data = {
                key: value for key, value in data.as_dict().items() if value is not None
            }
        conditions = (
            conditions.as_dict() if isinstance(conditions, Conditions) else dict(conditions or {})
        )
        batch = data["batch"]
        count = graph_count(batch)
        model_data = dict(data)
        model_data["atomic_numbers"] = self._resolve_atomic_numbers(data)
        # MACE's multi-head surface requires one graph-level head index.  The
        # typed public batch makes this field optional because non-MACE
        # backbones need not expose heads; the canonical single-head default is
        # therefore materialised only at the adapter boundary.
        if model_data.get("head") is None:
            model_data["head"] = torch.zeros(
                count, device=batch.device, dtype=torch.long
            )
        if data.get("cell") is not None:
            model_data["cell"] = _batched_cell(data["cell"], count)
        local = self.backbone(model_data)
        short_energy = _normalise_graph_energy(local["energy"], count)
        spin_vectors, auxiliary_magnitude = self.magnetic.resolve_spins(
            local["invariant_features"], conditions
        )
        physical_spin_input = (
            spin_vectors if self.config.spin.mode in {"spin_lattice", "collinear_density"} else None
        )
        if self.variational is not None:
            return self._forward_variational(
                model_data,
                local,
                short_energy,
                batch=batch,
                count=count,
                conditions=conditions,
                spin_vectors=spin_vectors,
            )
        if self.electronic is None:
            raise RuntimeError("legacy electronic path is unavailable")
        electronic = self.electronic(
            local["invariant_features"],
            positions=model_data["positions"],
            atomic_numbers=model_data["atomic_numbers"],
            batch=batch,
            edge_index=model_data["edge_index"],
            shifts=model_data.get("shifts"),
            cell=model_data.get("cell"),
            pbc=model_data.get("pbc"),
            cutoff_A=self.cutoff_A,
            conditions=conditions,
            spin_vectors=physical_spin_input,
        )
        magnetic = self.magnetic(
            local["invariant_features"],
            positions=model_data["positions"],
            batch=batch,
            edge_index=model_data["edge_index"],
            shifts=model_data.get("shifts"),
            cutoff_A=self.cutoff_A,
            conditions=conditions,
            spin_vectors=(
                spin_vectors
                if self.config.spin.mode in {"spin_lattice", "collinear_density"}
                else None
            ),
        )
        total_energy = short_energy + electronic.energy + magnetic.energy
        short_atomic = local.get("node_energy")
        if short_atomic is None:
            counts = torch.bincount(batch, minlength=count).to(total_energy.dtype)
            short_atomic = short_energy[batch] / counts[batch]
        else:
            short_atomic = short_atomic.reshape(-1)
        atomic_energy = short_atomic + electronic.atomic_energy + magnetic.atomic_energy
        state = electronic.state
        finite = (
            total_energy, atomic_energy, state.charge, state.spin,
            electronic.residual, electronic.spin_residual, magnetic.spin_vectors,
            electronic.energy, electronic.learned_energy, electronic.coulomb_energy,
            electronic.potential_coefficients, magnetic.energy, magnetic.magnitudes,
            auxiliary_magnitude,
        )
        if not all(bool(torch.isfinite(value).all().detach()) for value in finite):
            raise FloatingPointError(f"non-finite {electronic.method} electro-spin prediction")
        dipole_origin = conditions.get("dipole_origin", conditions.get("electric_field_origin"))
        if dipole_origin is None:
            dipole_origin = model_data["positions"].new_zeros((count, 3))
        dipole_origin = dipole_origin.to(
            device=model_data["positions"].device,
            dtype=model_data["positions"].dtype,
        )
        charge_position = state.charges[:, None] * (
            model_data["positions"] - dipole_origin[batch]
        )
        total_dipole = scatter_sum(charge_position + state.dipoles, batch, count)
        response_matrix = state.spin_density_matrix()
        if self.config.electronic.boundary_mode == "fixed_charge":
            target_charge = _graph_condition(
                conditions.get("total_charge", total_energy.new_zeros(count)),
                count,
                total_energy,
            )
            charge_constraint_residual = scatter_sum(state.charges, batch, count) - target_charge
        else:
            charge_constraint_residual = total_energy.new_zeros(count)
        electronic_magmom_vectors = state.magnetic_moments
        if self.config.spin.mode == "magnitude_auxiliary":
            magmoms = auxiliary_magnitude
        elif self.config.spin.mode == "disabled":
            magmoms = electronic.magmoms
        else:
            magmoms = torch.linalg.vector_norm(electronic_magmom_vectors, dim=-1)
        output = {
            "energy": total_energy,
            "atomic_energy": atomic_energy,
            "short_range_energy": short_energy,
            "electronic_energy": electronic.energy,
            "magnetic_energy": magnetic.energy,
            "learned_nonlocal_energy": electronic.learned_energy,
            "coulomb_energy": electronic.coulomb_energy,
            "external_electric_energy": electronic.external_energy,
            "external_magnetic_energy": magnetic.external_energy,
            "charge_multipoles": state.charge,
            "spin_multipoles": state.spin,
            "charges": state.charges,
            "spin_vectors": magnetic.spin_vectors,
            "dynamical_spin_vectors": magnetic.spin_vectors,
            "spin_magnitudes": magnetic.magnitudes,
            "magmom_vectors": electronic_magmom_vectors,
            "electronic_magmom_vectors": electronic_magmom_vectors,
            "magmoms": magmoms,
            "collinear_spins": magnetic.spin_vectors[:, 2],
            "total_charge": scatter_sum(state.charges, batch, count),
            "charge_constraint_residual": charge_constraint_residual,
            "spin_constraint_residual": electronic.spin_residual,
            "total_spin_vector": scatter_sum(magnetic.spin_vectors, batch, count),
            "total_magnetic_moment": scatter_sum(
                electronic_magmom_vectors, batch, count
            ),
            "backup_magnitude_head": auxiliary_magnitude,
            "dipoles": state.dipoles,
            "total_dipole": total_dipole,
            "quadrupoles": state.quadrupoles,
            "spin_charge_response_matrix": response_matrix,
            "spin_density_matrix": response_matrix,
            "fukui_weights": electronic.fukui_weights,
            "spin_fukui_weights": electronic.spin_fukui_weights,
            "electrostatic_potential": electronic.potential,
            "electrostatic_potential_coefficients": electronic.potential_coefficients,
            "electric_field": electronic.electric_field,
            "electronegativity": electronic.electronegativity,
            "hardness": electronic.hardness,
            "electronic_method": electronic.method,
            "electronic_residual": electronic.residual,
            "electronic_converged": electronic.converged,
            "electrostatic_backend": electronic.electrostatic_backend,
            "magnetic_chirality": magnetic.chirality,
            "magnetic_onsite_energy": magnetic.onsite_energy,
            "exchange_energy": magnetic.exchange_energy,
            "biquadratic_exchange_energy": magnetic.biquadratic_energy,
            "magnetic_anisotropy_energy": magnetic.anisotropy_energy,
            "dmi_energy": magnetic.dmi_energy,
            "magnetic_neural_energy": magnetic.neural_energy,
        }
        if electronic.oxidation is not None:
            output.update(
                {
                    "oxidation_logits": electronic.oxidation.logits,
                    "oxidation_probabilities": electronic.oxidation.probabilities,
                    "oxidation_confidence": electronic.oxidation.confidence,
                    "oxidation_entropy": electronic.oxidation.entropy,
                    "oxidation_expectation": electronic.oxidation.expectation,
                    "oxidation_states": electronic.oxidation.states,
                    "oxidation_allowed_mask": electronic.oxidation.allowed_mask,
                    "oxidation_state_values": self.electronic.oxidation.state_values,
                }
            )
        return output

    def predict_typed(
        self,
        data: dict[str, Any] | ZIVARBatch,
        *,
        conditions: dict[str, Any] | Conditions | None = None,
    ) -> ZIVARPrediction:
        """Return the authoritative typed prediction for the variational core."""

        output = self.forward(data, conditions=conditions)
        prediction = output.get("prediction")
        if not isinstance(prediction, ZIVARPrediction):
            raise RuntimeError(
                "typed predictions are available only for the variational architecture"
            )
        return prediction

    def energy_forces_stress(
        self,
        data: dict[str, Any] | ZIVARBatch,
        *,
        conditions: dict[str, Any] | Conditions | None = None,
        create_graph: bool = False,
        compute_stress: bool = True,
        compute_spin_fields: bool | None = None,
    ) -> dict[str, Any]:
        """Differentiate one total energy for forces, stress and spin fields."""

        if isinstance(data, ZIVARBatch):
            data = {
                key: value for key, value in data.as_dict().items() if value is not None
            }
        positions = data["positions"].detach().requires_grad_(True)
        batch = data["batch"]
        count = graph_count(batch)
        strain = positions.new_zeros((count, 3, 3), requires_grad=compute_stress)
        identity = torch.eye(3, device=positions.device, dtype=positions.dtype)
        deformation = identity[None] + strain
        transformed = dict(data)
        transformed["positions"] = torch.einsum("ni,nji->nj", positions, deformation[batch])
        edge_batch = batch[data["edge_index"][0]]
        if data.get("shifts") is not None:
            transformed["shifts"] = torch.einsum(
                "ei,eji->ej", data["shifts"], deformation[edge_batch]
            )
        cell = _batched_cell(data.get("cell"), count)
        if cell is not None:
            transformed["cell"] = torch.einsum("bni,bji->bnj", cell, deformation)
        resolved_conditions = (
            conditions.as_dict() if isinstance(conditions, Conditions) else dict(conditions or {})
        )
        if compute_spin_fields is None:
            compute_spin_fields = self.config.spin.mode == "spin_lattice"
        spin_variable = None
        if compute_spin_fields:
            source = resolved_conditions.get("spin_vectors")
            if source is None:
                source = resolved_conditions.get("initial_magnetic_moments")
            if source is None:
                raise ValueError("spin-field evaluation requires spin_vectors")
            spin_variable = source.detach().to(
                device=positions.device, dtype=positions.dtype
            ).requires_grad_(True)
            resolved_conditions.pop("initial_magnetic_moments", None)
            resolved_conditions["spin_vectors"] = spin_variable
        output = self.forward(transformed, conditions=resolved_conditions)
        variables: list[Any] = [positions]
        if compute_stress:
            variables.append(strain)
        if spin_variable is not None:
            variables.append(spin_variable)
        gradients = torch.autograd.grad(
            output["energy"].sum(), tuple(variables), create_graph=create_graph,
            retain_graph=create_graph or self.training, allow_unused=False,
        )
        result = dict(output)
        cursor = 1
        result["forces"] = -gradients[0]
        if compute_stress:
            if cell is None:
                raise ValueError("stress requires a cell")
            volume = torch.linalg.det(cell).abs()
            if bool(torch.any(volume <= 1.0e-12).detach()):
                raise ValueError("stress requires positive cell volumes")
            result["virial"] = gradients[cursor]
            result["stress"] = _symmetric_voigt(
                gradients[cursor] / volume[:, None, None]
            )
            cursor += 1
        if spin_variable is not None:
            spin_gradient = gradients[cursor]
            effective_eV = -spin_gradient
            result["spin_energy_gradient_eV_per_muB"] = spin_gradient
            result["effective_field_eV_per_muB"] = effective_eV
            result["effective_field_T"] = (
                effective_eV / self.config.spin.bohr_magneton_eV_per_T
            )
            result["magnetic_torque_eV"] = magnetic_torque(
                spin_variable, effective_eV
            )
        state = result.get("electronic_state")
        breakdown = result.get("energy_breakdown")
        if isinstance(state, ElectronicState) and isinstance(breakdown, EnergyBreakdown):
            result["prediction"] = ZIVARPrediction(
                energy=result["energy"],
                state=state,
                breakdown=breakdown,
                atomic_energy=result.get("atomic_energy"),
                forces=result.get("forces"),
                stress=result.get("stress"),
                virial=result.get("virial"),
                effective_field=result.get("effective_field_eV_per_muB"),
                torque=result.get("magnetic_torque_eV"),
                scf_report=result.get("scf_report"),
            )
        return result


def build_zivar(config: ZIVARConfig | None = None, *, device: Any = "cpu") -> ZIVAR:
    config = config or ZIVARConfig.production()
    model = ZIVAR(build_backbone(config.backbone, device=device), config)
    return model.to(device).compile_backbone()


__all__ = ["ZIVAR", "build_zivar"]
