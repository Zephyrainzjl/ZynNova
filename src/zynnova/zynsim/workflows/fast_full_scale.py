"""Multi-rate fast atomistic-to-cell workflow with resumable checkpoints."""

from __future__ import annotations

import json
import os
import hashlib
import tempfile
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from ..battery.mechanics import CathodeMechanicalMultiscaleModel, CathodeScaleFeedback
from ..battery.p2d import P2DParameters
from ..digital_twin import BatteryDigitalTwin, SensorPacket
from ..multiphysics import CoupledBatteryState, CoupledTrajectory, CouplingDiagnostics, FullyCoupledBatterySolver
from ..multiscale import CrossScaleRecord, MultiscaleCoordinator
from .full_scale import ScaleCoupling


SensorSource = Callable[[CoupledBatteryState], SensorPacket | None]
ControlSource = Callable[[CoupledBatteryState], Mapping[str, float]]
ElectrochemicalDump = Callable[[Any, Path], Mapping[str, object] | None]
ElectrochemicalLoad = Callable[[Path, Mapping[str, object]], Any]


@dataclass(frozen=True, slots=True)
class FastFullScaleConfig:
    electrochemical_time_step_s: float = 1.0
    property_refresh_interval_s: float = 30.0
    mechanics_interval_s: float = 30.0
    digital_twin_interval_s: float = 1.0
    record_interval_s: float = 1.0
    checkpoint_interval_s: float = 600.0
    fail_on_nonfinite_state: bool = True
    force_final_mechanics_update: bool = True

    def __post_init__(self) -> None:
        values = (
            self.electrochemical_time_step_s,
            self.property_refresh_interval_s,
            self.mechanics_interval_s,
            self.digital_twin_interval_s,
            self.record_interval_s,
            self.checkpoint_interval_s,
        )
        if min(values) <= 0.0:
            raise ValueError("all multi-rate intervals must be positive")


@dataclass(frozen=True, slots=True)
class FastScaleEvent:
    time_s: float
    kind: str
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class FastFullScaleResult:
    trajectory: CoupledTrajectory
    events: list[FastScaleEvent]
    cathode_feedback: list[CathodeScaleFeedback]
    property_records: Mapping[str, tuple[CrossScaleRecord, ...]]
    digital_twin_records: list[Any]
    checkpoint_paths: list[Path]


class FastFullScaleWorkflow:
    """Advance fast cell physics and expensive RVE physics on separate clocks.

    Expensive atomistic/property and 3-D cathode mechanics updates are event
    driven.  The latest constitutive closure is held between updates, while the
    electrochemical/thermal state advances with a short stable step.  This is
    the standard multi-rate strategy needed to make whole-cycle cross-scale
    calculations practical without discarding microstructure feedback.
    """

    def __init__(
        self,
        solver: FullyCoupledBatterySolver,
        *,
        p2d_parameters: P2DParameters | None = None,
        cathode_mechanics: CathodeMechanicalMultiscaleModel | None = None,
        scale_couplings: Sequence[ScaleCoupling] = (),
        digital_twin: BatteryDigitalTwin | None = None,
        sensor_source: SensorSource | None = None,
        control_source: ControlSource | None = None,
        config: FastFullScaleConfig | None = None,
    ) -> None:
        self.solver = solver
        self.p2d_parameters = p2d_parameters
        self.cathode_mechanics = cathode_mechanics
        self.scale_couplings = tuple(scale_couplings)
        self.digital_twin = digital_twin
        self.sensor_source = sensor_source
        self.control_source = control_source
        self.config = config or FastFullScaleConfig()
        if cathode_mechanics is not None and p2d_parameters is None:
            raise ValueError("cathode mechanics feedback requires p2d_parameters")

    def run(
        self,
        initial_state: CoupledBatteryState,
        time_s: np.ndarray,
        current_A: np.ndarray,
        *,
        checkpoint_directory: str | Path | None = None,
        electrochemical_dump: ElectrochemicalDump | None = None,
    ) -> FastFullScaleResult:
        times = np.asarray(time_s, dtype=float)
        currents = np.asarray(current_A, dtype=float)
        if times.ndim != 1 or currents.shape != times.shape or len(times) < 2:
            raise ValueError("time/current arrays must be aligned and one-dimensional")
        if np.any(np.diff(times) <= 0.0):
            raise ValueError("time grid must be strictly increasing")
        state = initial_state.copy()
        states = [state.copy()]
        diagnostics: list[CouplingDiagnostics] = []
        events: list[FastScaleEvent] = []
        feedback_history: list[CathodeScaleFeedback] = []
        checkpoints: list[Path] = []
        twin_start = 0 if self.digital_twin is None else len(self.digital_twin.records)
        last = {
            "property": -np.inf,
            "mechanics": -np.inf,
            "twin": -np.inf,
            "record": float(state.time_s),
            "checkpoint": float(state.time_s),
        }
        for left, right in zip(times[:-1], times[1:], strict=True):
            interval = float(right - left)
            substeps = max(1, int(np.ceil(interval / self.config.electrochemical_time_step_s)))
            dt = interval / substeps
            for local in range(substeps):
                target = float(left + (local + 1) * dt)
                current = float(np.interp(target, times, currents))
                if target - last["property"] >= self.config.property_refresh_interval_s - 1.0e-12:
                    uncertainty = self._refresh_properties(state)
                    state.metadata["parameter_relative_uncertainty"] = uncertainty
                    last["property"] = target
                    events.append(FastScaleEvent(target, "property_refresh", {"maximum_relative_uncertainty": uncertainty}))
                state, diagnostic = self.solver.step(state, current, dt)
                diagnostics.append(diagnostic)
                if self.config.fail_on_nonfinite_state:
                    _validate_coupled_state(state)
                if (
                    self.cathode_mechanics is not None
                    and target - last["mechanics"] >= self.config.mechanics_interval_s - 1.0e-12
                ):
                    feedback = self._advance_mechanics(state, current, target)
                    feedback_history.append(feedback)
                    last["mechanics"] = target
                    events.append(FastScaleEvent(target, "cathode_mechanics", _feedback_summary(feedback)))
                if self.digital_twin is not None and target - last["twin"] >= self.config.digital_twin_interval_s - 1.0e-12:
                    self._advance_twin(state, dt)
                    last["twin"] = target
                    events.append(FastScaleEvent(target, "digital_twin"))
                if target - last["record"] >= self.config.record_interval_s - 1.0e-12:
                    states.append(state.copy())
                    last["record"] = target
                if checkpoint_directory is not None and target - last["checkpoint"] >= self.config.checkpoint_interval_s - 1.0e-12:
                    checkpoint = save_fast_full_scale_checkpoint(
                        Path(checkpoint_directory) / f"checkpoint-{target:014.6f}",
                        state,
                        cathode_mechanics=self.cathode_mechanics,
                        electrochemical_dump=electrochemical_dump,
                    )
                    checkpoints.append(checkpoint)
                    last["checkpoint"] = target
                    events.append(FastScaleEvent(target, "checkpoint", {"path": str(checkpoint)}))
        if self.cathode_mechanics is not None and self.config.force_final_mechanics_update:
            final = self._advance_mechanics(state, float(currents[-1]), float(times[-1]), force=True)
            if not feedback_history or final.time_s != feedback_history[-1].time_s:
                feedback_history.append(final)
        if not states or states[-1].time_s != state.time_s:
            states.append(state.copy())
        return FastFullScaleResult(
            trajectory=CoupledTrajectory(states, diagnostics),
            events=events,
            cathode_feedback=feedback_history,
            property_records={coupling.name: tuple(coupling.coordinator.records) for coupling in self.scale_couplings},
            digital_twin_records=[] if self.digital_twin is None else list(self.digital_twin.records[twin_start:]),
            checkpoint_paths=checkpoints,
        )

    def _refresh_properties(self, state: CoupledBatteryState) -> float:
        uncertainty = 0.0
        for coupling in self.scale_couplings:
            values = coupling.coordinator.update(
                coupling.resolve_target(state),
                float(state.soc),
                float(np.mean(state.temperature_K)),
            )
            uncertainty = max(
                uncertainty,
                max((value.relative_uncertainty() or 0.0 for value in values.values()), default=0.0),
            )
        return float(uncertainty)

    def _advance_mechanics(
        self,
        state: CoupledBatteryState,
        current_A: float,
        time_s: float,
        *,
        force: bool = False,
    ) -> CathodeScaleFeedback:
        assert self.cathode_mechanics is not None and self.p2d_parameters is not None
        theta = self.p2d_parameters.positive.stoichiometry(float(state.soc))
        capacity = max(self.p2d_parameters.theoretical_capacity_Ah(), 1.0e-12)
        # P2D convention: positive current discharges the cell, so the positive
        # electrode lithiates (positive occupancy rate); charging is negative.
        c_rate = float(current_A) / capacity
        feedback = self.cathode_mechanics.advance(
            time_s=time_s,
            mean_theta=theta,
            mean_c_rate=c_rate,
            temperature_K=float(np.mean(state.temperature_K)),
            force=force,
        )
        self.cathode_mechanics.apply_to_p2d(self.p2d_parameters, feedback)
        state.metadata["cathode_mechanics"] = _feedback_summary(feedback)
        state.active_material_fraction = np.minimum(
            np.asarray(state.active_material_fraction, dtype=float),
            feedback.active_material_multiplier,
        )
        return feedback

    def _advance_twin(self, state: CoupledBatteryState, dt_s: float) -> None:
        assert self.digital_twin is not None
        controls = dict(self.control_source(state)) if self.control_source is not None else {
            "current_A": float(state.current_A),
            "cooling_command": 0.0,
        }
        packet = None if self.sensor_source is None else self.sensor_source(state)
        self.digital_twin.advance(controls, dt_s, sensor_packet=packet)


def save_fast_full_scale_checkpoint(
    directory: str | Path,
    state: CoupledBatteryState,
    *,
    cathode_mechanics: CathodeMechanicalMultiscaleModel | None = None,
    electrochemical_dump: ElectrochemicalDump | None = None,
) -> Path:
    """Atomically save all built-in array states without pickle."""

    target = Path(directory)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    arrays = {
        "temperature_K": state.temperature_K,
        "displacement_m": state.displacement_m,
        "stress_Pa": state.stress_Pa,
        "damage": state.damage,
        "sei_thickness_m": state.sei_thickness_m,
        "cei_thickness_m": state.cei_thickness_m,
        "plated_lithium_mol": state.plated_lithium_mol,
        "active_material_fraction": state.active_material_fraction,
    }
    np.savez_compressed(temporary / "coupled_state.npz", **arrays)
    manifest: dict[str, Any] = {
        "schema": "zynsim-fast-full-scale-v3",
        "time_s": float(state.time_s),
        "current_A": float(state.current_A),
        "terminal_voltage_V": float(state.terminal_voltage_V),
        "soc": float(state.soc),
        "lithium_inventory_fraction": float(state.lithium_inventory_fraction),
        "metadata": _json_safe(state.metadata),
    }
    if electrochemical_dump is not None:
        manifest["electrochemical"] = _json_safe(electrochemical_dump(state.electrochemical, temporary))
    elif hasattr(state.electrochemical, "to_dict"):
        manifest["electrochemical"] = _json_safe(state.electrochemical.to_dict())
    else:
        manifest["electrochemical"] = {
            "requires_loader": True,
            "type": f"{type(state.electrochemical).__module__}.{type(state.electrochemical).__qualname__}",
        }
    if cathode_mechanics is not None:
        manifest["cathode"] = {
            "last_update_time_s": cathode_mechanics.last_update_time_s,
            "rve_count": len(cathode_mechanics.states),
        }
        for index, rve in enumerate(cathode_mechanics.states):
            zeros = np.zeros_like(rve.theta)
            np.savez_compressed(
                temporary / f"cathode_rve_{index:04d}.npz",
                theta=rve.theta,
                damage=rve.damage,
                history_energy_J_m3=rve.history_energy_J_m3,
                fatigue_energy_J_m3=rve.fatigue_energy_J_m3,
                plastic_shear=rve.plastic_shear,
                oxygen_deficiency=rve.oxygen_deficiency,
                minimum_theta_history=rve.minimum_theta_history,
                previous_tensile_energy_J_m3=rve.previous_tensile_energy_J_m3,
                wetting_fraction=rve.wetting_fraction,
                strain=rve.strain,
                stress_Pa=rve.stress_Pa,
                chemical_potential_J_mol=rve.chemical_potential_J_mol,
                transformed_phase_fraction=(
                    zeros
                    if rve.transformed_phase_fraction is None
                    else rve.transformed_phase_fraction
                ),
                mobile_oxygen_fraction=(
                    zeros if rve.mobile_oxygen_fraction is None else rve.mobile_oxygen_fraction
                ),
                trapped_oxygen_fraction=(
                    zeros if rve.trapped_oxygen_fraction is None else rve.trapped_oxygen_fraction
                ),
                time_s=np.asarray(rve.time_s),
            )
    (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    checksums = {
        item.name: _sha256_file(item)
        for item in sorted(temporary.iterdir())
        if item.is_file()
    }
    (temporary / "checksums.json").write_text(
        json.dumps(checksums, indent=2, sort_keys=True), encoding="utf-8"
    )
    if target.exists():
        import shutil
        shutil.rmtree(target)
    os.replace(temporary, target)
    return target



def load_fast_full_scale_checkpoint(
    directory: str | Path,
    *,
    electrochemical_load: ElectrochemicalLoad,
    cathode_mechanics: CathodeMechanicalMultiscaleModel | None = None,
    verify_checksums: bool = True,
) -> CoupledBatteryState:
    """Restore an atomic full-scale checkpoint and optionally its RVE states.

    The electrochemical state is intentionally delegated to a typed loader;
    ZynSim never reconstructs arbitrary Python objects through pickle.
    """

    source = Path(directory)
    if not source.is_dir():
        raise FileNotFoundError(source)
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") not in {
        "zynsim-fast-full-scale-v1",
        "zynsim-fast-full-scale-v2",
        "zynsim-fast-full-scale-v3",
    }:
        raise ValueError(f"unsupported checkpoint schema {manifest.get('schema')!r}")
    checksum_path = source / "checksums.json"
    if verify_checksums and checksum_path.is_file():
        expected = json.loads(checksum_path.read_text(encoding="utf-8"))
        for name, digest in expected.items():
            path = source / name
            if not path.is_file() or _sha256_file(path) != digest:
                raise IOError(f"checkpoint integrity verification failed for {name}")
    with np.load(source / "coupled_state.npz", allow_pickle=False) as values:
        arrays = {name: values[name].copy() for name in values.files}
    electrochemical = electrochemical_load(source, manifest.get("electrochemical", {}))
    state = CoupledBatteryState(
        time_s=float(manifest["time_s"]),
        electrochemical=electrochemical,
        temperature_K=arrays["temperature_K"],
        displacement_m=arrays["displacement_m"],
        stress_Pa=arrays["stress_Pa"],
        damage=arrays["damage"],
        sei_thickness_m=arrays["sei_thickness_m"],
        cei_thickness_m=arrays["cei_thickness_m"],
        plated_lithium_mol=arrays["plated_lithium_mol"],
        active_material_fraction=arrays["active_material_fraction"],
        lithium_inventory_fraction=float(manifest["lithium_inventory_fraction"]),
        current_A=float(manifest["current_A"]),
        terminal_voltage_V=float(manifest["terminal_voltage_V"]),
        soc=float(manifest["soc"]),
        metadata=dict(manifest.get("metadata", {})),
    )
    if cathode_mechanics is not None and "cathode" in manifest:
        expected_count = int(manifest["cathode"]["rve_count"])
        if expected_count != len(cathode_mechanics.states):
            raise ValueError(
                f"checkpoint contains {expected_count} RVE states but model has "
                f"{len(cathode_mechanics.states)}"
            )
        for index, template in enumerate(cathode_mechanics.states):
            with np.load(source / f"cathode_rve_{index:04d}.npz", allow_pickle=False) as values:
                zeros = np.zeros_like(values["theta"])
                restored = type(template)(
                    time_s=float(values["time_s"]),
                    theta=values["theta"].copy(),
                    damage=values["damage"].copy(),
                    history_energy_J_m3=values["history_energy_J_m3"].copy(),
                    fatigue_energy_J_m3=values["fatigue_energy_J_m3"].copy(),
                    plastic_shear=values["plastic_shear"].copy(),
                    oxygen_deficiency=values["oxygen_deficiency"].copy(),
                    minimum_theta_history=values["minimum_theta_history"].copy(),
                    previous_tensile_energy_J_m3=(
                        values["previous_tensile_energy_J_m3"].copy()
                        if "previous_tensile_energy_J_m3" in values.files else zeros.copy()
                    ),
                    wetting_fraction=(
                        values["wetting_fraction"].copy()
                        if "wetting_fraction" in values.files else zeros.copy()
                    ),
                    strain=values["strain"].copy(),
                    stress_Pa=values["stress_Pa"].copy(),
                    chemical_potential_J_mol=values["chemical_potential_J_mol"].copy(),
                    metadata={"restored_from": str(source)},
                    transformed_phase_fraction=(
                        values["transformed_phase_fraction"].copy()
                        if "transformed_phase_fraction" in values.files else zeros.copy()
                    ),
                    mobile_oxygen_fraction=(
                        values["mobile_oxygen_fraction"].copy()
                        if "mobile_oxygen_fraction" in values.files else zeros.copy()
                    ),
                    trapped_oxygen_fraction=(
                        values["trapped_oxygen_fraction"].copy()
                        if "trapped_oxygen_fraction" in values.files else zeros.copy()
                    ),
                )
            cathode_mechanics.states[index] = restored
        cathode_mechanics.last_update_time_s = float(manifest["cathode"]["last_update_time_s"])
    _validate_coupled_state(state)
    return state


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def _feedback_summary(feedback: CathodeScaleFeedback) -> dict[str, float]:
    return {
        "damage_fraction": feedback.damage_fraction,
        "connected_active_fraction": feedback.connected_active_fraction,
        "crack_surface_density_m_inv": feedback.crack_surface_density_m_inv,
        "capacity_multiplier": feedback.capacity_multiplier,
        "diffusivity_multiplier": feedback.diffusivity_multiplier,
        "reaction_area_multiplier": feedback.reaction_area_multiplier,
        "electronic_conductivity_multiplier": feedback.electronic_conductivity_multiplier,
        "transformed_phase_fraction": feedback.transformed_phase_fraction,
        "trapped_oxygen_fraction": feedback.trapped_oxygen_fraction,
        "maximum_principal_stress_Pa": feedback.maximum_principal_stress_Pa,
        "porosity_multiplier": feedback.porosity_multiplier,
        "electrolyte_transport_multiplier": feedback.electrolyte_transport_multiplier,
        "plating_risk_multiplier": feedback.plating_risk_multiplier,
    }


def _validate_coupled_state(state: CoupledBatteryState) -> None:
    for name in (
        "temperature_K", "displacement_m", "stress_Pa", "damage",
        "sei_thickness_m", "cei_thickness_m", "plated_lithium_mol",
        "active_material_fraction",
    ):
        if not np.isfinite(np.asarray(getattr(state, name), dtype=float)).all():
            raise FloatingPointError(f"non-finite values detected in {name}")
    if np.any(np.asarray(state.temperature_K) <= 0.0):
        raise FloatingPointError("temperature must remain positive")
    if not 0.0 <= state.soc <= 1.0:
        raise FloatingPointError("SOC left [0,1]")


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, np.ndarray):
        return {"array_shape": list(value.shape), "dtype": str(value.dtype)}
    return repr(value)


__all__ = [
    "FastFullScaleConfig",
    "FastFullScaleResult",
    "FastFullScaleWorkflow",
    "FastScaleEvent",
    "load_fast_full_scale_checkpoint",
    "save_fast_full_scale_checkpoint",
]
