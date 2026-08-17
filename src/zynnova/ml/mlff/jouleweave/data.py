from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from ....data import (
    FieldLevel,
    FieldRole,
    FieldSpec,
    LoaderConfig,
    MaterialSample,
    MissingPolicy,
    TaskSpec,
    create_dataset,
    make_datamodule,
    material_collate,
)
from ....data.source import DatasetSource
from ...workspace import MLWorkspace
from .config import JouleWeaveDataConfig, ReferenceFitMode


def _optional_field(
    name: str,
    source: str,
    *,
    role: FieldRole,
    level: FieldLevel,
) -> FieldSpec:
    return FieldSpec(
        name=name,
        source=source,
        role=role,
        level=level,
        required=False,
        missing=MissingPolicy.MASK,
    )


def jouleweave_task(config: JouleWeaveDataConfig) -> TaskSpec:
    stress = (
        _optional_field(
            "stress",
            config.stress_source,
            role=FieldRole.LABEL,
            level=FieldLevel.GRAPH,
        )
        if config.stress_source
        else None
    )
    extra_targets: list[FieldSpec] = []
    if config.magmoms_source:
        extra_targets.append(
            _optional_field(
                "magmoms",
                config.magmoms_source,
                role=FieldRole.LABEL,
                level=FieldLevel.ATOM,
            )
        )
    if config.charges_source:
        extra_targets.append(
            _optional_field(
                "charges",
                config.charges_source,
                role=FieldRole.LABEL,
                level=FieldLevel.ATOM,
            )
        )
    if config.oxidation_states_source:
        extra_targets.append(
            _optional_field(
                "oxidation_states",
                config.oxidation_states_source,
                role=FieldRole.LABEL,
                level=FieldLevel.ATOM,
            )
        )
    if config.dipole_source:
        extra_targets.append(
            _optional_field(
                "dipole",
                config.dipole_source,
                role=FieldRole.LABEL,
                level=FieldLevel.GRAPH,
            )
        )
    conditions: list[FieldSpec] = []
    for name, source in (
        ("total_charge", config.total_charge_source),
        ("spin", config.spin_source),
        ("fidelity", config.fidelity_source),
    ):
        if source:
            conditions.append(
                _optional_field(
                    name,
                    source,
                    role=FieldRole.CONDITION,
                    level=FieldLevel.GRAPH,
                )
            )
    return TaskSpec.neural_potential(
        name="jouleweave-potential",
        energy=config.energy_source,
        forces=config.forces_source,
        stress=stress,
        conditions=conditions,
        extra_targets=extra_targets,
        material_types=config.material_types,
    )


def prepare_jouleweave_datamodule(
    config: JouleWeaveDataConfig,
    *,
    workspace: MLWorkspace,
    source: DatasetSource | Sequence[MaterialSample] | Iterable[MaterialSample] | None = None,
):
    if source is None:
        root = (
            config.dataset_root
            if config.dataset_root is not None
            else workspace.dataset_dir(config.dataset)
        )
        source = create_dataset(
            config.dataset,
            root=root,
            **dict(config.dataset_kwargs),
        )
    elif not isinstance(source, (DatasetSource, Sequence)):
        source = list(source)
    loader = LoaderConfig(
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        seed=config.seed,
        pin_memory=config.pin_memory,
        collate_fn=jouleweave_collate,
    )
    datamodule = make_datamodule(
        source,
        jouleweave_task(config),
        loader=loader,
        ratios=(config.train_ratio, config.valid_ratio, config.test_ratio),
    )
    datamodule.setup()
    return datamodule


def jouleweave_collate(samples: Sequence[Any]) -> dict[str, Any]:
    """Collate potential samples while preserving sparse atom-wise supervision.

    The generic material collator keeps only target keys present in every sample
    and stacks equal-sized atom arrays. JouleWeave instead concatenates optional
    charge, magnetic-moment, and oxidation-state labels and emits one atom mask
    per target. This permits variable-size crystals and partially labelled
    multi-fidelity datasets without silently discarding supervision.
    """

    import torch

    batch = material_collate(samples)
    atom_target_names = ("magmoms", "charges", "oxidation_states")
    for name in atom_target_names:
        present_values = [
            sample.targets.get(name)
            for sample in samples
            if sample.targets.get(name) is not None
        ]
        if not present_values:
            continue
        trailing_shape: tuple[int, ...] | None = None
        reference_tensor = (
            present_values[0]
            if torch.is_tensor(present_values[0])
            else torch.as_tensor(present_values[0])
        )
        for value in present_values:
            tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
            if tensor.ndim == 0:
                raise ValueError(f"{name} must contain one value per atom")
            current = tuple(tensor.shape[1:])
            if current == (1,):
                current = ()
            if trailing_shape is None:
                trailing_shape = current
            elif current != trailing_shape:
                raise ValueError(
                    f"inconsistent trailing shape for {name}: "
                    f"{current} != {trailing_shape}"
                )
        trailing_shape = trailing_shape or ()

        values: list[Any] = []
        masks: list[Any] = []
        for sample in samples:
            atom_count = int(sample.structure["z"].shape[0])
            value = sample.targets.get(name)
            if value is None:
                values.append(
                    torch.zeros(
                        (atom_count, *trailing_shape),
                        dtype=reference_tensor.dtype,
                    )
                )
                masks.append(torch.zeros((atom_count,), dtype=torch.bool))
                continue
            tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
            if tensor.shape == (atom_count, 1) and not trailing_shape:
                tensor = tensor[:, 0]
            expected_shape = (atom_count, *trailing_shape)
            if tuple(tensor.shape) != expected_shape:
                raise ValueError(
                    f"{name} must have shape {expected_shape}; "
                    f"got {tuple(tensor.shape)} in sample {sample.id!r}"
                )
            if tensor.is_floating_point():
                finite = torch.isfinite(tensor)
                atom_mask = finite if tensor.ndim == 1 else finite.flatten(1).all(dim=1)
                tensor = torch.nan_to_num(tensor)
            elif name == "oxidation_states":
                atom_mask = tensor.reshape(atom_count, -1).ne(-100).all(dim=1)
                tensor = torch.where(atom_mask, tensor.reshape(atom_count), 0)
            else:
                atom_mask = torch.ones((atom_count,), dtype=torch.bool)
            values.append(tensor)
            masks.append(atom_mask)
        batch["targets"][name] = torch.cat(values, dim=0)
        batch["masks"][name] = torch.cat(masks, dim=0)
    return batch


def fit_energy_statistics(
    loader: Any,
    *,
    max_atomic_number: int,
    mode: ReferenceFitMode = "auto",
    ridge: float = 1.0e-8,
) -> tuple[dict[int, float], float, float]:
    """Fit element references and residual per-atom shift/scale on training data."""

    import torch

    compositions: list[Any] = []
    energies: list[Any] = []
    atom_counts: list[Any] = []
    for item in loader:
        structure = item["structure"]
        z = structure["z"].detach().cpu().long()
        batch = structure["batch"].detach().cpu().long()
        target = item["targets"]["energy"].detach().cpu().double().reshape(-1)
        graph_count = target.numel()
        composition = torch.zeros(
            (graph_count, max_atomic_number + 1),
            dtype=torch.float64,
        )
        flat = batch * (max_atomic_number + 1) + z
        composition.view(-1).index_add_(
            0,
            flat,
            torch.ones_like(flat, dtype=torch.float64),
        )
        compositions.append(composition)
        energies.append(target)
        atom_counts.append(structure["natoms"].detach().cpu().double().reshape(-1))
    if not compositions:
        raise ValueError("cannot fit energy statistics from an empty loader")

    composition = torch.cat(compositions, dim=0)
    energy = torch.cat(energies, dim=0)
    natoms = torch.cat(atom_counts, dim=0).clamp_min(1.0)
    present = torch.nonzero(composition.sum(dim=0) > 0, as_tuple=False).reshape(-1)
    present = present[present > 0]
    selected = composition[:, present]

    fit_mode = mode
    if mode == "auto":
        rank = int(torch.linalg.matrix_rank(selected).item()) if selected.numel() else 0
        fit_mode = (
            "least_squares"
            if selected.shape[0] >= selected.shape[1] and rank == selected.shape[1]
            else "mean"
        )
    references: dict[int, float] = {}
    reference_energy = torch.zeros_like(energy)
    if fit_mode == "least_squares" and selected.numel():
        if ridge > 0:
            gram = selected.transpose(0, 1) @ selected
            rhs = selected.transpose(0, 1) @ energy
            coefficients = torch.linalg.solve(
                gram + ridge * torch.eye(gram.shape[0], dtype=gram.dtype),
                rhs,
            )
        else:
            coefficients = torch.linalg.lstsq(selected, energy[:, None]).solution[:, 0]
        references = {
            int(z): float(value)
            for z, value in zip(
                present.tolist(),
                coefficients.tolist(),
                strict=True,
            )
        }
        reference_energy = selected @ coefficients
    elif fit_mode not in {"mean", "none"}:
        raise ValueError(f"unsupported reference fit mode: {fit_mode}")

    if fit_mode == "none":
        return {}, 0.0, 1.0
    residual_per_atom = (energy - reference_energy) / natoms
    shift = float(residual_per_atom.mean())
    scale = float(residual_per_atom.std(unbiased=False))
    return references, shift, max(scale, 1.0e-6)


__all__ = [
    "fit_energy_statistics",
    "jouleweave_collate",
    "jouleweave_task",
    "prepare_jouleweave_datamodule",
]
