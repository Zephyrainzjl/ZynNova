"""ZIVAR: stable charge-informed interatomic modelling.

Heavy imports are intentionally lazy.  Importing :mod:`zynnova.ml` therefore
does not import Torch, ASE, e3nn, or the upstream equivariant runtime.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .config import (
    ARCHITECTURE_REVISION,
    NUMERICS_REVISION,
    ZIVAR_VERSION,
    BackboneConfig,
    ElectronicConfig,
    ElectrostaticsConfig,
    OxidationConfig,
    SCFConfig,
    SpinConfig,
    ZIVARConfig,
)

_LAZY_EXPORTS = {
    "BACKBONE_CONTRACT_VERSION": ("backbones", "BACKBONE_CONTRACT_VERSION"),
    "BackboneAdapter": ("backbones", "BackboneAdapter"),
    "BackboneCapabilities": ("backbones", "BackboneCapabilities"),
    "backbone_registration": ("backbones", "backbone_registration"),
    "load_backbone_plugins": ("backbones", "load_backbone_plugins"),
    "register_backbone": ("backbones", "register_backbone"),
    "registered_backbones": ("backbones", "registered_backbones"),
    "validate_backbone_output": ("backbones", "validate_backbone_output"),
    "ZIVAR": ("model", "ZIVAR"),
    "build_zivar": ("model", "build_zivar"),
    "import_legacy_backbone": ("checkpoint", "import_legacy_backbone"),
    "LegacyBackboneImportReport": (
        "checkpoint",
        "LegacyBackboneImportReport",
    ),
    "load_zivar": ("checkpoint", "load_zivar"),
    "save_zivar": ("checkpoint", "save_zivar"),
    "ZIVARCalculator": ("calculator", "ZIVARCalculator"),
    "zivar_calculator": ("calculator", "zivar_calculator"),
    "predict_structure": ("calculator", "predict_structure"),
    "atoms_to_typed_batch": ("data", "atoms_to_typed_batch"),
    "collate_zivar_batches": ("data", "collate_zivar_batches"),
    "relax_structure": ("materials", "relax_structure"),
    "run_molecular_dynamics": ("materials", "run_molecular_dynamics"),
    "equation_of_state": ("materials", "equation_of_state"),
    "calculate_elastic_tensor": ("materials", "calculate_elastic_tensor"),
    "calculate_phonons": ("materials", "calculate_phonons"),
    "run_neb": ("materials", "run_neb"),
    "analyze_li_diffusion": ("battery", "analyze_li_diffusion"),
    "run_li_diffusion": ("battery", "run_li_diffusion"),
    "export_zivar_lammps_bundle": ("lammps", "export_zivar_lammps_bundle"),
    "export_local_backbone_mliap": ("lammps", "export_local_backbone_mliap"),
    "ZIVARLAMMPSCallback": ("lammps", "ZIVARLAMMPSCallback"),
    "ZIVARLoss": ("losses", "ZIVARLoss"),
    "robust_squared": ("losses", "robust_squared"),
    "StableElectronicModel": ("electronic", "StableElectronicModel"),
    "PolarDensityModel": ("polar", "PolarDensityModel"),
    "SpinLatticeHamiltonian": ("magnetism", "SpinLatticeHamiltonian"),
    "llg_midpoint_step": ("spin_dynamics", "llg_midpoint_step"),
    "run_spin_lattice_dynamics": ("spin_dynamics", "run_spin_lattice_dynamics"),
    "solve_qeq": ("qeq", "solve_qeq"),
    "solve_quadratic_scf": ("scf", "solve_quadratic_scf"),
    "SCFConvergenceError": ("errors", "SCFConvergenceError"),
    "ElectroSpinFunctional": ("functional", "ElectroSpinFunctional"),
    "ElectroSpinParameters": ("functional", "ElectroSpinParameters"),
    "ElectronicState": ("types", "ElectronicState"),
    "EnergyBreakdown": ("types", "EnergyBreakdown"),
    "ZIVARBatch": ("types", "ZIVARBatch"),
    "ZIVARPrediction": ("types", "ZIVARPrediction"),
    "ewald_energy": ("ewald_reference", "ewald_energy"),
    "pme_energy": ("pme", "pme_energy"),
    "OxidationStateHead": ("oxidation", "OxidationStateHead"),
    "ZIVARTrainer": ("trainer", "ZIVARTrainer"),
    "assert_model_optimizer_finite": ("trainer", "assert_model_optimizer_finite"),
    "assess_maturity": ("maturity", "assess_maturity"),
    "assert_release_ready": ("maturity", "assert_release_ready"),
    "benchmark_inference": ("benchmark", "benchmark_inference"),
    "benchmark_against_local_backbone": (
        "benchmark",
        "benchmark_against_local_backbone",
    ),
    "benchmark_registered_backbones": (
        "benchmark",
        "benchmark_registered_backbones",
    ),
    "AccuracyMetrics": ("chgnet_parity", "AccuracyMetrics"),
    "CHGNetParityResult": ("chgnet_parity", "CHGNetParityResult"),
    "compare_with_chgnet": ("chgnet_parity", "compare_with_chgnet"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(f"{__name__}.{module_name}"), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *_LAZY_EXPORTS))


__all__ = [
    "ARCHITECTURE_REVISION",
    "BackboneConfig",
    "ElectrostaticsConfig",
    "ElectronicConfig",
    "NUMERICS_REVISION",
    "OxidationConfig",
    "SCFConfig",
    "SpinConfig",
    "ZIVARConfig",
    "ZIVAR_VERSION",
    *_LAZY_EXPORTS,
]
