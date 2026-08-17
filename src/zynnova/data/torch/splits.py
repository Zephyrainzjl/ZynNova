from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from ..record import MaterialSample


def random_split_indices(
    size: int,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    *,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    _validate_ratios(ratios)
    generator = np.random.default_rng(seed)
    indices = generator.permutation(size)
    train_end = round(size * ratios[0])
    valid_end = train_end + round(size * ratios[1])
    return {
        "train": indices[:train_end],
        "valid": indices[train_end:valid_end],
        "test": indices[valid_end:],
    }


def group_split_indices(
    samples: Sequence[MaterialSample],
    group_by: str | Callable[[MaterialSample], Any],
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    *,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    _validate_ratios(ratios)
    key_function = (
        group_by if callable(group_by) else lambda sample: sample.get(group_by)
    )
    groups: dict[Any, list[int]] = defaultdict(list)
    for index, sample in enumerate(samples):
        groups[key_function(sample)].append(index)
    rng = np.random.default_rng(seed)
    keys = list(groups)
    rng.shuffle(keys)
    target = np.asarray(ratios) * len(samples)
    result: dict[str, list[int]] = {"train": [], "valid": [], "test": []}
    for key in keys:
        split = min(result, key=lambda name: len(result[name]) / max(target[_split_index(name)], 1))
        result[split].extend(groups[key])
    return {name: np.asarray(indices, dtype=np.int64) for name, indices in result.items()}


def scaffold_split_indices(
    samples: Sequence[MaterialSample],
    *,
    smiles_path: str = "metadata.smiles",
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 0,
) -> dict[str, np.ndarray]:
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
    except ImportError as exc:
        raise ImportError("RDKit is required for scaffold splitting") from exc

    def scaffold(sample: MaterialSample) -> str:
        smiles = str(sample.require(smiles_path))
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            return f"invalid:{sample.id}"
        return MurckoScaffoldSmiles(mol=molecule, includeChirality=True)

    return group_split_indices(samples, scaffold, ratios, seed=seed)


def _validate_ratios(ratios: tuple[float, float, float]) -> None:
    if len(ratios) != 3 or any(value < 0 for value in ratios):
        raise ValueError("split ratios must contain three non-negative values")
    if not np.isclose(sum(ratios), 1.0):
        raise ValueError("split ratios must sum to one")


def _split_index(name: str) -> int:
    return {"train": 0, "valid": 1, "test": 2}[name]
