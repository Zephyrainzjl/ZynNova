from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ..encoding import CompiledSample
from ..schema import TaskKind


def material_collate(batch: Sequence[CompiledSample]) -> dict[str, Any]:
    if not batch:
        raise ValueError("cannot collate an empty batch")
    if all(sample.kind is TaskKind.POTENTIAL for sample in batch):
        return _collate_potential(batch)
    return {
        "id": [sample.id for sample in batch],
        "kind": batch[0].kind.value,
        "structure": recursive_collate([sample.structure for sample in batch]),
        "inputs": recursive_collate([sample.inputs for sample in batch]),
        "targets": recursive_collate([sample.targets for sample in batch]),
        "conditions": recursive_collate([sample.conditions for sample in batch]),
        "masks": recursive_collate([sample.masks for sample in batch]),
        "metadata": [sample.metadata for sample in batch],
    }


def _collate_potential(batch: Sequence[CompiledSample]) -> dict[str, Any]:
    torch = _require_torch()
    structures = [sample.structure for sample in batch]
    if not all(isinstance(item, Mapping) for item in structures):
        raise TypeError("potential samples require dictionary structures")
    natoms = torch.as_tensor(
        [int(item["z"].shape[0]) for item in structures],
        dtype=torch.long,
    )
    batch_index = torch.repeat_interleave(torch.arange(len(batch)), natoms)
    pointer = torch.zeros(len(batch) + 1, dtype=torch.long)
    pointer[1:] = torch.cumsum(natoms, dim=0)
    structure = {
        "z": torch.cat([_as_tensor(item["z"], dtype=torch.long) for item in structures]),
        "pos": torch.cat(
            [_as_tensor(item["pos"], dtype=torch.get_default_dtype()) for item in structures]
        ),
        "batch": batch_index,
        "ptr": pointer,
        "natoms": natoms,
        "cell": torch.stack(
            [_as_tensor(item["cell"], dtype=torch.get_default_dtype()) for item in structures]
        ),
        "pbc": torch.stack([_as_tensor(item["pbc"], dtype=torch.bool) for item in structures]),
    }
    optional_atom_fields = ("charge", "mass")
    for name in optional_atom_fields:
        if all(name in item for item in structures):
            structure[name] = torch.cat(
                [_as_tensor(item[name], dtype=torch.get_default_dtype()) for item in structures]
            )
    targets: dict[str, Any] = {}
    names = set.intersection(*(set(sample.targets) for sample in batch)) if batch else set()
    for name in sorted(names):
        values = [sample.targets[name] for sample in batch]
        if name == "forces":
            targets[name] = torch.cat([_as_tensor(value) for value in values])
        else:
            targets[name] = recursive_collate(values)
    return {
        "id": [sample.id for sample in batch],
        "kind": TaskKind.POTENTIAL.value,
        "structure": structure,
        "inputs": recursive_collate([sample.inputs for sample in batch]),
        "targets": targets,
        "conditions": recursive_collate([sample.conditions for sample in batch]),
        "masks": recursive_collate([sample.masks for sample in batch]),
        "metadata": [sample.metadata for sample in batch],
    }


def recursive_collate(values: Sequence[Any]) -> Any:
    if not values:
        return values
    first = values[0]
    if first is None:
        return list(values)
    try:
        from torch_geometric.data import Batch, Data, HeteroData

        if isinstance(first, (Data, HeteroData)):
            return Batch.from_data_list(list(values))
    except ImportError:
        pass
    torch = _require_torch()
    if isinstance(first, torch.Tensor):
        if all(tuple(value.shape) == tuple(first.shape) for value in values):
            return torch.stack(list(values))
        return list(values)
    if isinstance(first, np.ndarray):
        tensors = [torch.as_tensor(value) for value in values]
        if all(tuple(value.shape) == tuple(tensors[0].shape) for value in tensors):
            return torch.stack(tensors)
        return tensors
    if isinstance(first, Mapping):
        common = set.intersection(*(set(value) for value in values))
        return {key: recursive_collate([value[key] for value in values]) for key in sorted(common)}
    if isinstance(first, (bool, int, float, np.number)):
        return torch.as_tensor(values)
    if isinstance(first, str):
        return list(values)
    if isinstance(first, tuple):
        return tuple(
            recursive_collate([value[index] for value in values])
            for index in range(len(first))
        )
    return list(values)


def _as_tensor(value: Any, *, dtype: Any | None = None):
    torch = _require_torch()
    if isinstance(value, torch.Tensor):
        return value.to(dtype=dtype) if dtype is not None else value
    return torch.as_tensor(value, dtype=dtype)


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("PyTorch is required; install zynnova[data]") from exc
    return torch
