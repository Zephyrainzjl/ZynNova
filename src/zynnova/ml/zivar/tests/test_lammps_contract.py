from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from zynnova.ml.zivar.lammps import LAMMPSConfig, lammps_cell, lammps_spin_vectors


def test_triclinic_cell_and_runtime_contract() -> None:
    cell = lammps_cell((1.0, 2.0, 3.0), (5.0, 7.0, 9.0), 0.2, 0.3, -0.1)
    assert np.allclose(cell, ((4.0, 0.0, 0.0), (0.2, 5.0, 0.0), (-0.1, 0.3, 6.0)))
    config = LAMMPSConfig(
        evaluation_mode="replicated",
        device="cuda:local",
        spin_input_mode="by_type",
        spin_vectors_by_type=((0.0, 0.0, 1.0),),
    )
    assert config.evaluation_mode == "replicated"
    assert config.require_release_evidence


def test_invalid_lammps_spin_table_is_rejected() -> None:
    with pytest.raises(ValueError, match="finite xyz triples"):
        LAMMPSConfig(
            spin_input_mode="by_type",
            spin_vectors_by_type=((0.0, float("nan"), 1.0),),
        )


def test_lammps_spin_state_conversion_is_exact_and_strict() -> None:
    state = np.asarray(((1.0, 0.0, 0.0, 2.0), (0.0, 0.0, -1.0, 1.5)))
    assert np.array_equal(
        lammps_spin_vectors(state, 2), ((2.0, 0.0, 0.0), (0.0, 0.0, -1.5))
    )
    with pytest.raises(RuntimeError, match="normalized"):
        lammps_spin_vectors(np.asarray(((2.0, 0.0, 0.0, 1.0),)), 1)


def test_fix_external_rejects_fake_native_spin_integration() -> None:
    with pytest.raises(ValueError, match="cannot supply LAMMPS magnetic force"):
        LAMMPSConfig(spin_evolution="nve/spin")
