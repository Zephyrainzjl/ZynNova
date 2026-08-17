from __future__ import annotations

from typing import Any

from ....data import LoaderConfig, TaskSpec, create_dataset, make_datamodule
from ...workspace import MLWorkspace
from .config import ZNNPDataConfig


def prepare_rmd17_datamodule(
    config: ZNNPDataConfig,
    *,
    workspace: MLWorkspace,
):
    if config.dataset not in {"rmd17", "md17", "revised_md17"}:
        raise ValueError("the bundled ZNNP example currently targets revised MD17")
    source = create_dataset(
        "rmd17",
        root=workspace.dataset_dir("rmd17"),
        molecule=config.molecule,
        limit=config.limit,
        selection=config.selection,
        seed=config.seed,
        local_file=config.local_file,
        prefer_direct_download=config.prefer_direct_download,
        convert_to_ev=True,
    )
    task = TaskSpec.neural_potential(
        name="znnp-rmd17",
        energy="labels.energy",
        forces="labels.forces",
        material_types=("molecular",),
    )
    loader = LoaderConfig(
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        seed=config.seed,
        pin_memory=True,
    )
    datamodule = make_datamodule(
        source,
        task,
        loader=loader,
        ratios=(config.train_ratio, config.valid_ratio, config.test_ratio),
    )
    datamodule.setup()
    return datamodule


def fit_energy_normalization(loader: Any) -> tuple[float, float]:
    values = []
    for batch in loader:
        energy = batch["targets"]["energy"].reshape(-1).double()
        natoms = batch["structure"]["natoms"].reshape(-1).double()
        values.append(energy / natoms)
    if not values:
        raise ValueError("cannot fit energy normalization from an empty loader")
    import torch

    per_atom = torch.cat(values)
    shift = float(per_atom.mean())
    scale = float(per_atom.std(unbiased=False))
    return shift, max(scale, 1.0e-6)


__all__ = ["fit_energy_normalization", "prepare_rmd17_datamodule"]
