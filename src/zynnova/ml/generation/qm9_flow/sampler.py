from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from ....structure import StructureData
from ...common import load_checkpoint, resolve_device
from .config import QM9FlowModelConfig
from .data import center_coordinates
from .model import QM9EquivariantFlow


def sample_qm9_coordinates(
    model: QM9EquivariantFlow,
    atomic_numbers: Iterable[int] | list[list[int]],
    *,
    steps: int = 100,
    noise_scale_A: float = 1.0,
    device: str = "auto",
    seed: int | None = None,
):
    import torch

    resolved = resolve_device(device)
    raw = list(atomic_numbers)
    if not raw:
        raise ValueError("atomic_numbers cannot be empty")
    if isinstance(raw[0], (int, np.integer)):
        compositions = [[int(value) for value in raw]]
    else:
        compositions = [[int(value) for value in item] for item in raw]
    batch_size = len(compositions)
    max_atoms = model.config.max_atoms
    z = torch.zeros((batch_size, max_atoms), device=resolved, dtype=torch.long)
    mask = torch.zeros((batch_size, max_atoms), device=resolved, dtype=torch.bool)
    for batch_index, composition in enumerate(compositions):
        if len(composition) > max_atoms:
            raise ValueError(f"composition has {len(composition)} atoms, max_atoms={max_atoms}")
        z[batch_index, : len(composition)] = torch.as_tensor(
            composition,
            device=resolved,
            dtype=torch.long,
        )
        mask[batch_index, : len(composition)] = True
    generator = None
    if seed is not None:
        generator = torch.Generator(device=resolved).manual_seed(seed)
    positions = torch.randn(
        (batch_size, max_atoms, 3),
        device=resolved,
        dtype=next(model.parameters()).dtype,
        generator=generator,
    ) * noise_scale_A
    positions = center_coordinates(positions, mask)
    model = model.to(resolved).eval()
    dt = 1.0 / steps
    with torch.no_grad():
        for step in range(steps):
            time = torch.full(
                (batch_size,),
                step / steps,
                device=resolved,
                dtype=positions.dtype,
            )
            positions = positions + dt * model(z, positions, time, mask)
            positions = center_coordinates(positions, mask)
    structures = []
    for batch_index, composition in enumerate(compositions):
        count = len(composition)
        structures.append(
            StructureData(
                atomic_numbers=np.asarray(composition, dtype=np.int64),
                positions=positions[batch_index, :count].cpu().numpy(),
                pbc=np.zeros(3, dtype=bool),
                cell=np.zeros((3, 3), dtype=np.float64),
                info={"generator": "zynnova.ml.generation.qm9_flow"},
            )
        )
    return structures[0] if len(structures) == 1 else structures


def save_generated_structures(
    structures,
    directory: str | Path,
    *,
    prefix: str = "qm9-flow",
) -> list[Path]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if isinstance(structures, StructureData):
        structures = [structures]
    paths: list[Path] = []
    for index, structure in enumerate(structures):
        xyz_path = directory / f"{prefix}-{index:05d}.xyz"
        try:
            from ase.io import write

            write(xyz_path, structure.to_ase())
            paths.append(xyz_path)
        except ImportError:
            npz_path = xyz_path.with_suffix(".npz")
            np.savez_compressed(
                npz_path,
                atomic_numbers=structure.atomic_numbers,
                positions=structure.positions,
            )
            paths.append(npz_path)
    return paths


def load_qm9_flow(checkpoint: str | Path, *, device: str = "cpu") -> QM9EquivariantFlow:
    payload = load_checkpoint(checkpoint, map_location=device)
    model = QM9EquivariantFlow(QM9FlowModelConfig(**payload["model_config"]))
    model.load_state_dict(payload["model_state"])
    model.to(resolve_device(device)).eval()
    return model


__all__ = ["load_qm9_flow", "sample_qm9_coordinates", "save_generated_structures"]
