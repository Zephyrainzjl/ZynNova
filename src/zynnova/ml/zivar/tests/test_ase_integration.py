from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
ase = pytest.importorskip("ase")

from ase import units
from ase.md.verlet import VelocityVerlet

from zynnova.ml.zivar.calculator import ZIVARCalculator
from zynnova.ml.zivar.config import ZIVARConfig
from zynnova.ml.zivar.model import build_zivar


def _calculator() -> ZIVARCalculator:
    config = ZIVARConfig.convolution(
        dft_level="ase-integration-test",
        backbone__atomic_numbers=(1, 8),
        backbone__channels=8,
        backbone__num_interactions=1,
        backbone__num_bessel=3,
        backbone__radial_mlp=(8,),
        backbone__pair_repulsion=False,
        electronic__hidden=(8,),
        electronic__radial_basis=3,
        electronic__oxidation__enabled=False,
        electrostatics__boundary="isolated",
        spin__mode="disabled",
        spin__require_spin_input=False,
    )
    return ZIVARCalculator(build_zivar(config).double(), dtype="float64")


def test_ase_and_short_md_use_the_same_conservative_core() -> None:
    atoms = ase.Atoms(
        "H2O",
        positions=[[0.0, 0.0, 0.0], [0.95, 0.0, 0.0], [-0.24, 0.92, 0.0]],
    )
    atoms.info["total_charge"] = 0.0
    atoms.calc = _calculator()

    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    assert np.isfinite(energy)
    assert np.isfinite(forces).all()
    assert abs(float(np.sum(atoms.calc.results["charges"]))) <= 1.0e-10
    assert atoms.calc.results["electronic_converged"] is True

    atoms.set_velocities(np.zeros((len(atoms), 3)))
    VelocityVerlet(atoms, timestep=0.1 * units.fs).run(2)
    assert np.isfinite(atoms.get_potential_energy())
    assert np.isfinite(atoms.get_forces()).all()
