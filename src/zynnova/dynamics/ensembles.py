from __future__ import annotations

from typing import Any

import numpy as np

from .config import Ensemble, MDConfig
from .exceptions import ConfigurationError, MissingBackendError, PotentialError


def _require_ase():
    try:
        from ase import units
    except ImportError as exc:
        raise MissingBackendError(
            "ASE is required for MD integrators; install zynnova[dynamics]"
        ) from exc
    return units


def _rng(seed: int | None):
    return np.random.default_rng(seed)


def initialize_velocities(atoms: Any, config, target_temperature_K: float | None) -> None:
    from .config import VelocityMode

    config.validate(target_temperature_K)
    if config.mode is VelocityMode.KEEP:
        return
    if config.mode is VelocityMode.ZERO:
        atoms.set_velocities(np.zeros((len(atoms), 3), dtype=float))
        return
    try:
        from ase.md.velocitydistribution import (
            MaxwellBoltzmannDistribution,
            Stationary,
            ZeroRotation,
        )
    except ImportError as exc:
        raise MissingBackendError("ASE velocity initialization is unavailable") from exc
    temperature = (
        target_temperature_K
        if config.temperature_K is None
        else float(config.temperature_K)
    )
    rng = _rng(config.seed)
    MaxwellBoltzmannDistribution(
        atoms,
        temperature_K=temperature,
        force_temp=config.force_temperature,
        rng=rng,
    )
    if config.remove_translation:
        Stationary(
            atoms,
            preserve_temperature=config.preserve_temperature_after_cleanup,
        )
    if config.remove_rotation:
        ZeroRotation(
            atoms,
            preserve_temperature=config.preserve_temperature_after_cleanup,
        )


def _require_stress(atoms: Any) -> None:
    calculator = getattr(atoms, "calc", None)
    properties = set(getattr(calculator, "implemented_properties", ()))
    if "stress" not in properties:
        raise PotentialError("NPT dynamics requires a calculator that provides stress")


def build_dynamics(atoms: Any, config: MDConfig):
    """Build an ASE dynamics object for a validated ZynNova configuration."""
    config.validate()
    units = _require_ase()
    dt = config.timestep_fs * units.fs
    temperature = config.temperature_K
    pressure_au = None if config.pressure_GPa is None else config.pressure_GPa * units.GPa
    common = dict(config.extra_integrator_kwargs)
    rng = _rng(config.random_seed)

    if config.ensemble is Ensemble.NVE:
        from ase.md.verlet import VelocityVerlet

        return VelocityVerlet(atoms, timestep=dt, **common)
    if config.ensemble is Ensemble.NVT_LANGEVIN:
        from ase.md.langevin import Langevin

        # ASE's historical ``fixcm=True`` implementation is deprecated and is
        # singular for a one-atom system.  ZynNova removes COM momentum during
        # velocity initialization (and optionally at a user-selected interval),
        # so the integrator should sample with fixcm disabled by default.
        common.setdefault("fixcm", False)
        return Langevin(
            atoms,
            timestep=dt,
            temperature_K=temperature,
            friction=config.friction_per_fs / units.fs,
            rng=rng,
            **common,
        )
    if config.ensemble is Ensemble.NVT_BERENDSEN:
        from ase.md.nvtberendsen import NVTBerendsen

        return NVTBerendsen(
            atoms,
            timestep=dt,
            temperature_K=temperature,
            taut=config.thermostat_time_fs * units.fs,
            **common,
        )
    if config.ensemble is Ensemble.NVT_BUSSI:
        from ase.md.bussi import Bussi

        return Bussi(
            atoms,
            timestep=dt,
            temperature_K=temperature,
            taut=config.thermostat_time_fs * units.fs,
            rng=rng,
            **common,
        )
    if config.ensemble is Ensemble.NVT_ANDERSEN:
        from ase.md.andersen import Andersen

        return Andersen(
            atoms,
            timestep=dt,
            temperature_K=temperature,
            andersen_prob=config.andersen_probability,
            rng=rng,
            **common,
        )
    if config.ensemble is Ensemble.NVT_NOSE_HOOVER:
        try:
            from ase.md.nose_hoover_chain import NoseHooverChainNVT
        except ImportError as exc:
            raise MissingBackendError(
                "NoseHooverChainNVT requires a recent ASE release"
            ) from exc
        return NoseHooverChainNVT(
            atoms,
            timestep=dt,
            temperature_K=temperature,
            tdamp=config.thermostat_time_fs * units.fs,
            tchain=config.thermostat_chain_length,
            tloop=config.thermostat_substeps,
            **common,
        )

    _require_stress(atoms)
    if config.ensemble is Ensemble.NPT_BERENDSEN:
        from ase.md.nptberendsen import NPTBerendsen

        return NPTBerendsen(
            atoms,
            timestep=dt,
            temperature_K=temperature,
            pressure_au=pressure_au,
            taut=config.thermostat_time_fs * units.fs,
            taup=config.barostat_time_fs * units.fs,
            compressibility_au=config.compressibility_GPa_inv / units.GPa,
            **common,
        )
    if config.ensemble is Ensemble.NPT_BERENDSEN_MASKED:
        from ase.md.nptberendsen import Inhomogeneous_NPTBerendsen

        return Inhomogeneous_NPTBerendsen(
            atoms,
            timestep=dt,
            temperature_K=temperature,
            pressure_au=pressure_au,
            taut=config.thermostat_time_fs * units.fs,
            taup=config.barostat_time_fs * units.fs,
            compressibility_au=config.compressibility_GPa_inv / units.GPa,
            mask=config.pressure_mask,
            **common,
        )
    if config.ensemble in {
        Ensemble.NPT_MTK_ISOTROPIC,
        Ensemble.NPT_MTK_FULL,
        Ensemble.NPT_MTK_MASKED,
    }:
        try:
            from ase.md.nose_hoover_chain import IsotropicMTKNPT, MaskedMTKNPT, MTKNPT
        except ImportError as exc:
            raise MissingBackendError("MTK NPT requires a recent ASE release") from exc
        kwargs = dict(
            atoms=atoms,
            timestep=dt,
            temperature_K=temperature,
            pressure_au=pressure_au,
            tdamp=config.thermostat_time_fs * units.fs,
            pdamp=config.barostat_time_fs * units.fs,
            tchain=config.thermostat_chain_length,
            pchain=config.barostat_chain_length,
            tloop=config.thermostat_substeps,
            ploop=config.barostat_substeps,
            **common,
        )
        if config.ensemble is Ensemble.NPT_MTK_ISOTROPIC:
            return IsotropicMTKNPT(**kwargs)
        if config.ensemble is Ensemble.NPT_MTK_FULL:
            return MTKNPT(**kwargs)
        return MaskedMTKNPT(**kwargs, mask=config.cell_mask)
    if config.ensemble is Ensemble.NPT_LANGEVIN_BAOAB:
        try:
            from ase.md.langevinbaoab import LangevinBAOAB
        except ImportError as exc:
            raise MissingBackendError("LangevinBAOAB requires a recent ASE release") from exc
        externalstress = -float(pressure_au)
        return LangevinBAOAB(
            atoms,
            timestep=dt,
            temperature_K=temperature,
            externalstress=externalstress,
            hydrostatic=True,
            T_tau=config.thermostat_time_fs * units.fs,
            P_tau=config.barostat_time_fs * units.fs,
            rng=rng,
            **common,
        )
    if config.ensemble is Ensemble.NPT_MELCHIONNA:
        try:
            from ase.md.melchionna import MelchionnaNPT
        except ImportError:
            from ase.md.npt import NPT as MelchionnaNPT
        if "pfactor" not in common:
            raise ConfigurationError(
                "npt_melchionna requires extra_integrator_kwargs['pfactor']; "
                "the value depends on the expected bulk modulus"
            )
        # ASE interprets a scalar ``externalstress`` as a positive
        # compression pressure and performs the stress-sign conversion itself.
        # Passing ``-pressure_au`` would therefore request tension instead of the
        # user-specified positive pressure.
        common.setdefault("mask", config.pressure_mask)
        return MelchionnaNPT(
            atoms,
            timestep=dt,
            temperature_K=temperature,
            externalstress=pressure_au,
            ttime=config.thermostat_time_fs * units.fs,
            **common,
        )
    raise ValueError(f"Unsupported ensemble: {config.ensemble}")
