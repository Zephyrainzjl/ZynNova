from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from .adapters import to_ase_atoms, to_structure_data
from .config import CellMode, OptimizerMethod, RelaxationConfig
from .exceptions import MissingBackendError, PotentialError
from .results import RelaxationResult


def _optimizer_class(method: OptimizerMethod):
    """Resolve only the optimizer requested by the user.

    Optional SciPy optimizers are imported lazily so their availability does
    not affect FIRE/BFGS/LBFGS relaxation.
    """

    method = OptimizerMethod(method)

    try:
        import ase.optimize as ase_opt
    except ImportError as exc:
        raise MissingBackendError(
            "ASE optimizers are required; install ase>=3.23"
        ) from exc

    # FIRE2 是新版 ASE 优化器；旧版 ASE 自动退回 FIRE。
    if method is OptimizerMethod.FIRE2:
        return getattr(ase_opt, "FIRE2", ase_opt.FIRE)

    # SciPy 优化器不再由 ase.optimize 顶层导出。
    if method in {
        OptimizerMethod.SCIPY_BFGS,
        OptimizerMethod.SCIPY_CG,
    }:
        try:
            from ase.optimize.sciopt import (
                SciPyFminBFGS,
                SciPyFminCG,
            )
        except ImportError as exc:
            raise MissingBackendError(
                "SciPy-based ASE optimizers require scipy and "
                "ase.optimize.sciopt"
            ) from exc

        return {
            OptimizerMethod.SCIPY_BFGS: SciPyFminBFGS,
            OptimizerMethod.SCIPY_CG: SciPyFminCG,
        }[method]

    # 仅较新的 ASE 版本具有 CellAwareBFGS。
    if method is OptimizerMethod.CELL_AWARE_BFGS:
        optimizer_cls = getattr(
            ase_opt,
            "CellAwareBFGS",
            None,
        )
        if optimizer_cls is None:
            raise MissingBackendError(
                "CellAwareBFGS requires a recent ASE release"
            )
        return optimizer_cls

    optimizer_names = {
        OptimizerMethod.FIRE: "FIRE",
        OptimizerMethod.BFGS: "BFGS",
        OptimizerMethod.LBFGS: "LBFGS",
        OptimizerMethod.LBFGS_LINESEARCH: "LBFGSLineSearch",
        OptimizerMethod.BFGS_LINESEARCH: "BFGSLineSearch",
        OptimizerMethod.MDMIN: "MDMin",
        OptimizerMethod.GPMIN: "GPMin",
    }

    optimizer_name = optimizer_names[method]
    optimizer_cls = getattr(ase_opt, optimizer_name, None)

    if optimizer_cls is None:
        raise MissingBackendError(
            f"Installed ASE does not provide optimizer "
            f"{optimizer_name!r}"
        )

    return optimizer_cls


def _build_filter(atoms: Any, config: RelaxationConfig):
    if config.cell_mode is CellMode.FIXED:
        return atoms
    properties = set(getattr(atoms.calc, "implemented_properties", ()))
    if "stress" not in properties:
        raise PotentialError("Cell relaxation requires a calculator that provides stress")
    if config.cell_mode is CellMode.CELL_AWARE:
        return atoms
    try:
        from ase import units
        from ase.filters import ExpCellFilter, FrechetCellFilter, StrainFilter, UnitCellFilter
    except ImportError as exc:
        raise MissingBackendError("ASE cell filters are required") from exc
    kwargs = dict(config.filter_kwargs)
    kwargs.setdefault("mask", config.cell_mask)
    kwargs.setdefault("hydrostatic_strain", config.hydrostatic_strain)
    kwargs.setdefault("constant_volume", config.constant_volume)
    kwargs.setdefault("scalar_pressure", config.scalar_pressure_GPa * units.GPa)
    if config.cell_mode is CellMode.FRECHET:
        return FrechetCellFilter(atoms, **kwargs)
    if config.cell_mode is CellMode.UNIT_CELL:
        return UnitCellFilter(atoms, **kwargs)
    if config.cell_mode is CellMode.EXPONENTIAL:
        return ExpCellFilter(atoms, **kwargs)
    if config.cell_mode is CellMode.STRAIN:
        allowed = {"mask"}
        selected = {key: value for key, value in kwargs.items() if key in allowed}
        return StrainFilter(atoms, **selected)
    raise ValueError(f"Unsupported cell mode: {config.cell_mode}")


def relax(
    structure: Any,
    calculator: Any,
    config: RelaxationConfig | None = None,
    *,
    output_directory: str | Path = "zynnova-relax",
) -> RelaxationResult:
    config = config or RelaxationConfig()
    config.validate()
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    atoms = to_ase_atoms(structure)
    initial = to_structure_data(atoms)
    atoms.calc = calculator
    initial_energy = float(atoms.get_potential_energy())
    target = _build_filter(atoms, config)
    optimizer_cls = _optimizer_class(config.optimizer)
    trajectory_path = (
        directory / config.trajectory_filename if config.trajectory_filename else None
    )
    logfile_path = directory / config.logfile if config.logfile else None
    restart_path = directory / config.restart_filename if config.restart_filename else None
    kwargs = dict(config.optimizer_kwargs)
    kwargs.setdefault("logfile", None if logfile_path is None else str(logfile_path))
    kwargs.setdefault("trajectory", None if trajectory_path is None else str(trajectory_path))
    if restart_path is not None:
        kwargs.setdefault("restart", str(restart_path))
    if config.optimizer not in {OptimizerMethod.SCIPY_BFGS, OptimizerMethod.SCIPY_CG}:
        kwargs.setdefault("maxstep", config.maxstep_A)
    if config.optimizer is OptimizerMethod.CELL_AWARE_BFGS:
        kwargs.setdefault("append_trajectory", config.append_trajectory)
    started = time.perf_counter()
    optimizer = optimizer_cls(target, **kwargs)
    if config.optimizer is OptimizerMethod.CELL_AWARE_BFGS:
        converged = bool(
            optimizer.run(
                fmax=config.fmax_eV_per_A,
                smax=config.smax_eV_per_A3,
                steps=config.max_steps,
            )
        )
    else:
        converged = bool(optimizer.run(fmax=config.fmax_eV_per_A, steps=config.max_steps))
    wall = time.perf_counter() - started
    forces = np.asarray(atoms.get_forces(), dtype=float)
    fmax = float(np.linalg.norm(forces, axis=1).max(initial=0.0))
    smax = None
    if np.any(atoms.pbc) and "stress" in getattr(calculator, "implemented_properties", ()):
        smax = float(np.max(np.abs(atoms.get_stress(voigt=False))))
    return RelaxationResult(
        initial_structure=initial,
        final_structure=to_structure_data(atoms),
        converged=converged,
        steps=int(optimizer.nsteps),
        initial_energy_eV=initial_energy,
        final_energy_eV=float(atoms.get_potential_energy()),
        final_fmax_eV_per_A=fmax,
        final_smax_eV_per_A3=smax,
        wall_time_s=wall,
        trajectory_path=trajectory_path,
        logfile_path=logfile_path,
        metadata={
            "optimizer": config.optimizer.value,
            "cell_mode": config.cell_mode.value,
        },
    )


def staged_relax(
    structure: Any,
    calculator: Any,
    stages: list[RelaxationConfig],
    *,
    output_directory: str | Path = "zynnova-relax",
) -> list[RelaxationResult]:
    if not stages:
        raise ValueError("stages cannot be empty")
    current = structure
    results: list[RelaxationResult] = []
    root = Path(output_directory)
    for index, stage in enumerate(stages):
        result = relax(
            current,
            calculator,
            stage,
            output_directory=root / f"stage-{index:02d}",
        )
        results.append(result)
        current = result.final_structure
    return results
