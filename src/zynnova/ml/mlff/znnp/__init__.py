"""ZNNP: an energy-conserving neural-neighbor potential for revised MD17."""

from ...registry import MODELS
from .calculator import load_znnp, load_znnp_calculator, znnp_calculator
from .config import ZNNPConfig, ZNNPDataConfig, ZNNPModelConfig, ZNNPTrainConfig
from .data import fit_energy_normalization, prepare_rmd17_datamodule
from .lammps import (
    LAMMPSRunConfig,
    ZNNPLAMMPSBridge,
    run_znnp_lammps,
    write_znnp_lammps_data,
)
from .model import ZNNP, build_radius_graph
from .trainer import train_znnp


@MODELS.register(
    "mlff",
    "znnp",
    description="Energy-conserving message-passing force field trained on revised MD17",
)
def create_znnp(config: ZNNPModelConfig | None = None) -> ZNNP:
    return ZNNP(config)


__all__ = [
    "LAMMPSRunConfig",
    "ZNNP",
    "ZNNPConfig",
    "ZNNPDataConfig",
    "ZNNPLAMMPSBridge",
    "ZNNPModelConfig",
    "ZNNPTrainConfig",
    "build_radius_graph",
    "create_znnp",
    "fit_energy_normalization",
    "load_znnp",
    "load_znnp_calculator",
    "prepare_rmd17_datamodule",
    "run_znnp_lammps",
    "train_znnp",
    "write_znnp_lammps_data",
    "znnp_calculator",
]
