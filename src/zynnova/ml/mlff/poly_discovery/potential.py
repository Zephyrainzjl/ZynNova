from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .config import PolymerPotentialConfig


@dataclass(slots=True)
class LoadedPolymerPotential:
    checkpoint: Path
    model: Any
    calculator: Any
    config: PolymerPotentialConfig | None = None


@dataclass(frozen=True, slots=True)
class CommitteePrediction:
    energy_mean_eV: float
    energy_std_eV: float
    forces_mean_eV_A: np.ndarray
    forces_std_eV_A: np.ndarray
    stress_mean_eV_A3: np.ndarray | None
    stress_std_eV_A3: np.ndarray | None
    member_count: int

    @property
    def maximum_force_uncertainty_eV_A(self) -> float:
        if self.forces_std_eV_A.size == 0:
            return 0.0
        return float(np.linalg.norm(self.forces_std_eV_A, axis=-1).max(initial=0.0))


@dataclass(frozen=True, slots=True)
class SelectedFrame:
    index: int
    uncertainty_score: float
    energy_std_eV_per_atom: float
    maximum_force_std_eV_A: float
    novelty_score: float


def build_polymer_jouleweave_config(
    config: PolymerPotentialConfig | None = None,
) -> Any:
    """Translate a polymer preset into the existing JouleWeave configuration."""

    config = config or PolymerPotentialConfig()
    config.__post_init__()
    from ..jouleweave import (
        JouleWeaveConfig,
        JouleWeaveDataConfig,
        JouleWeaveModelConfig,
        JouleWeaveTrainConfig,
    )

    preset_values: dict[str, Any] = {
        "mode": "specialist",
        "hidden_dim": config.hidden_dim,
        "num_layers": config.num_layers,
        "num_radial": 24,
        "num_attention_heads": 8,
        "num_experts": 4,
        "interaction_cutoff_A": config.cutoff_A,
        "max_neighbors": config.max_neighbors,
        "use_zbl": True,
        "use_dispersion": True,
        "dispersion_cutoff_A": max(8.0, config.cutoff_A),
        "use_qeq": False,
        "use_charge_head": False,
        "use_oxidation_states": False,
        "use_magmoms": False,
    }
    charge_scheme = str(
        config.data_overrides.get("charge_label_scheme", "unspecified")
    )
    if config.preset == "electrostatic":
        preset_values.update(
            {
                "use_qeq": True,
                "qeq_max_atoms": 1024,
                "use_charge_head": bool(config.data_overrides.get("charges_source")),
                "charge_label_scheme": charge_scheme,
            }
        )
    elif config.preset == "reactive":
        preset_values.update(
            {
                "zbl_outer_A": 2.0,
                "learnable_zbl_scale": True,
                "use_qeq": bool(config.data_overrides.get("total_charge_source")),
            }
        )
    preset_values.update(config.model_overrides)
    model = JouleWeaveModelConfig(**preset_values)

    dataset_kwargs = dict(config.dataset_kwargs)
    if config.dataset == "trajectory":
        dataset_kwargs.setdefault("material_type", "polymer")
    data_values: dict[str, Any] = {
        "dataset": config.dataset,
        "dataset_kwargs": dataset_kwargs,
        "dataset_root": config.dataset_root,
        "material_types": ("polymer",),
        "batch_size": config.batch_size,
        "seed": config.seed,
    }
    data_values.update(config.data_overrides)
    data = JouleWeaveDataConfig(**data_values)

    energy_weight = 0.1 if config.force_centric else config.energy_weight
    force_weight = max(config.force_weight, 100.0) if config.force_centric else config.force_weight
    train_values: dict[str, Any] = {
        "epochs": config.epochs,
        "learning_rate": config.learning_rate,
        "energy_weight": energy_weight,
        "force_weight": force_weight,
        "stress_weight": config.stress_weight,
        "device": config.device,
        "dtype": config.dtype,
        "workspace_root": config.workspace_root,
        "run_name": config.run_name,
        "seed": config.seed,
        "fine_tune_checkpoint": config.fine_tune_checkpoint,
        "freeze_backbone_epochs": (
            config.freeze_backbone_epochs if config.fine_tune_checkpoint else 0
        ),
    }
    train_values.update(config.train_overrides)
    train = JouleWeaveTrainConfig(**train_values)

    if train.charge_weight > 0 and not data.charges_source:
        raise ValueError(
            "charge supervision requested without data_overrides['charges_source']; "
            "a partition convention such as DDEC6 must also be recorded"
        )
    if train.charge_weight > 0 and not model.use_charge_head:
        raise ValueError("charge supervision requires model use_charge_head=True")
    if train.charge_weight > 0 and data.charge_label_scheme == "unspecified":
        raise ValueError(
            "charge supervision requires a recorded charge_label_scheme"
        )
    if (
        model.use_charge_head
        and model.charge_label_scheme != data.charge_label_scheme
    ):
        raise ValueError(
            "model and data charge_label_scheme values must be identical"
        )
    if train.dipole_weight > 0 and not data.dipole_source:
        raise ValueError(
            "dipole supervision requested without data_overrides['dipole_source']"
        )
    if train.dipole_weight > 0 and not (model.use_qeq or model.use_charge_head):
        raise ValueError(
            "dipole supervision requires use_qeq or use_charge_head in the model"
        )
    return JouleWeaveConfig(model=model, data=data, train=train)


def train_polymer_potential(
    config: PolymerPotentialConfig | None = None,
    *,
    source: Any | None = None,
) -> Any:
    """Train or fine-tune a polymer JouleWeave potential."""

    from ..jouleweave import train_jouleweave

    resolved = config or PolymerPotentialConfig()
    return train_jouleweave(build_polymer_jouleweave_config(resolved), source=source)


def load_polymer_potential(
    checkpoint: str | Path,
    *,
    device: str = "auto",
    dtype: str = "float32",
    analytic_stress: bool = False,
    use_ema: bool = True,
) -> LoadedPolymerPotential:
    from ..jouleweave import load_jouleweave, load_jouleweave_calculator
    from ...common import resolve_device

    path = Path(checkpoint).expanduser().resolve()
    resolved_device = resolve_device(device)
    model = load_jouleweave(
        path,
        device=str(resolved_device),
        dtype=dtype,
        use_ema=use_ema,
    )
    calculator = load_jouleweave_calculator(
        path,
        device=str(resolved_device),
        dtype=dtype,
        analytic_stress=analytic_stress,
        use_ema=use_ema,
    )
    return LoadedPolymerPotential(
        checkpoint=path,
        model=model,
        calculator=calculator,
    )


class PotentialCommittee:
    """Checkpoint committee for epistemic uncertainty and active learning."""

    def __init__(
        self,
        checkpoints: Sequence[str | Path],
        *,
        device: str = "auto",
        dtype: str = "float32",
        analytic_stress: bool = False,
        loader: Callable[..., Any] | None = None,
    ) -> None:
        if len(checkpoints) < 2:
            raise ValueError("a potential committee requires at least two checkpoints")
        self.checkpoints = tuple(Path(path).expanduser().resolve() for path in checkpoints)
        if loader is None:
            from ..jouleweave import load_jouleweave_calculator

            loader = load_jouleweave_calculator
        self.calculators = tuple(
            loader(
                path,
                device=device,
                dtype=dtype,
                analytic_stress=analytic_stress,
            )
            for path in self.checkpoints
        )
        self.analytic_stress = bool(analytic_stress)

    def predict(self, atoms: Any) -> CommitteePrediction:
        energies: list[float] = []
        forces: list[np.ndarray] = []
        stresses: list[np.ndarray] = []
        for calculator in self.calculators:
            member_atoms = atoms.copy()
            member_atoms.calc = calculator
            energies.append(float(member_atoms.get_potential_energy()))
            forces.append(np.asarray(member_atoms.get_forces(), dtype=float))
            if self.analytic_stress and bool(np.any(member_atoms.pbc)):
                stresses.append(np.asarray(member_atoms.get_stress(), dtype=float))
        energy_array = np.asarray(energies, dtype=float)
        force_array = np.stack(forces, axis=0)
        stress_array = np.stack(stresses, axis=0) if stresses else None
        return CommitteePrediction(
            energy_mean_eV=float(np.mean(energy_array)),
            energy_std_eV=float(np.std(energy_array, ddof=1)),
            forces_mean_eV_A=np.mean(force_array, axis=0),
            forces_std_eV_A=np.std(force_array, axis=0, ddof=1),
            stress_mean_eV_A3=(
                None if stress_array is None else np.mean(stress_array, axis=0)
            ),
            stress_std_eV_A3=(
                None if stress_array is None else np.std(stress_array, axis=0, ddof=1)
            ),
            member_count=len(self.calculators),
        )

    def select_uncertain_frames(
        self,
        frames: Sequence[Any],
        *,
        count: int,
        force_weight: float = 1.0,
        energy_weight: float = 0.2,
        diversity_weight: float = 0.25,
        minimum_force_std_eV_A: float = 0.0,
    ) -> tuple[SelectedFrame, ...]:
        if count < 1:
            raise ValueError("count must be positive")
        if not frames:
            return ()
        predictions = [self.predict(frame) for frame in frames]
        descriptors = np.stack([_frame_descriptor(frame) for frame in frames])
        descriptors = _standardize(descriptors)
        base_scores = []
        for frame, prediction in zip(frames, predictions, strict=True):
            natoms = max(len(frame), 1)
            force_std = prediction.maximum_force_uncertainty_eV_A
            energy_std = prediction.energy_std_eV / natoms
            base_scores.append(force_weight * force_std + energy_weight * energy_std)

        eligible = [
            index
            for index, prediction in enumerate(predictions)
            if prediction.maximum_force_uncertainty_eV_A >= minimum_force_std_eV_A
        ]
        selected: list[int] = []
        selected_rows: list[SelectedFrame] = []
        while eligible and len(selected) < min(count, len(frames)):
            scored: list[tuple[float, int, float]] = []
            for index in eligible:
                novelty = (
                    1.0
                    if not selected
                    else float(
                        min(
                            np.linalg.norm(descriptors[index] - descriptors[chosen])
                            for chosen in selected
                        )
                    )
                )
                scored.append(
                    (float(base_scores[index] + diversity_weight * novelty), index, novelty)
                )
            total, best, novelty = max(scored, key=lambda item: (item[0], -item[1]))
            prediction = predictions[best]
            selected.append(best)
            eligible.remove(best)
            selected_rows.append(
                SelectedFrame(
                    index=best,
                    uncertainty_score=total,
                    energy_std_eV_per_atom=prediction.energy_std_eV / max(len(frames[best]), 1),
                    maximum_force_std_eV_A=(
                        prediction.maximum_force_uncertainty_eV_A
                    ),
                    novelty_score=novelty,
                )
            )
        return tuple(selected_rows)


def potential_config_snapshot(config: PolymerPotentialConfig) -> dict[str, Any]:
    return _snapshot_value(asdict(config))


def _snapshot_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _snapshot_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_snapshot_value(item) for item in value]
    return value


def _frame_descriptor(atoms: Any) -> np.ndarray:
    atomic_numbers = np.asarray(atoms.get_atomic_numbers(), dtype=int)
    composition = np.bincount(atomic_numbers, minlength=119)[1:].astype(float)
    composition /= max(float(composition.sum()), 1.0)
    positions = np.asarray(atoms.get_positions(), dtype=float)
    if len(positions) > 1:
        if hasattr(atoms, "get_all_distances"):
            distances = np.asarray(
                atoms.get_all_distances(mic=bool(np.any(atoms.pbc))),
                dtype=float,
            )
        else:
            delta = positions[:, None, :] - positions[None, :, :]
            distances = np.linalg.norm(delta, axis=-1)
        values = distances[np.triu_indices(len(positions), 1)]
        quantiles = np.quantile(values, (0.1, 0.25, 0.5, 0.75, 0.9))
    else:
        quantiles = np.zeros(5, dtype=float)
    volume_per_atom = float(atoms.get_volume()) / max(len(atoms), 1) if np.any(atoms.pbc) else 0.0
    return np.concatenate((composition, quantiles, [volume_per_atom]))


def _standardize(matrix: np.ndarray) -> np.ndarray:
    mean = np.mean(matrix, axis=0)
    scale = np.std(matrix, axis=0)
    scale[scale < 1.0e-12] = 1.0
    return (matrix - mean) / scale


__all__ = [
    "CommitteePrediction",
    "LoadedPolymerPotential",
    "PotentialCommittee",
    "SelectedFrame",
    "build_polymer_jouleweave_config",
    "load_polymer_potential",
    "potential_config_snapshot",
    "train_polymer_potential",
]
