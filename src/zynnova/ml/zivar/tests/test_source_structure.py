from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _calls(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Attribute):
            names.append(node.func.attr)
        elif isinstance(node.func, ast.Name):
            names.append(node.func.id)
    return tuple(names)


def test_variational_solver_replaces_the_retired_unrolled_solver() -> None:
    assert (ROOT / "scf.py").is_file()
    scf = (ROOT / "scf.py").read_text(encoding="utf-8")
    assert "solve_quadratic_scf" in scf
    assert "SCFConvergenceError" in scf
    for name in ("electronic.py", "polar.py", "qeq.py", "magnetism.py"):
        source = (ROOT / name).read_text(encoding="utf-8").lower()
        assert "conductancescfsolver" not in source
        assert "scf_steps_train" not in source
        assert "initial_charge_multipoles" not in source
    assert "grad" not in _calls(ROOT / "electronic.py")
    assert "grad" not in _calls(ROOT / "polar.py")
    assert "grad" not in _calls(ROOT / "qeq.py")


def test_training_api_cannot_seed_from_labels() -> None:
    source = (ROOT / "trainer.py").read_text(encoding="utf-8")
    assert "seed_electronic_state_from_labels" not in source
    assert "targets[\"charges\"].detach" not in source
    assert "electronic supervision leaked" in source


def test_all_backbones_and_deployments_use_stable_contract() -> None:
    registry = (ROOT / "backbones" / "__init__.py").read_text(encoding="utf-8")
    assert "_register_convolution()" in registry
    model = (ROOT / "model.py").read_text(encoding="utf-8")
    assert "VariationalElectroSpinModel" in model
    assert 'config.electronic.method == "variational"' in model
    # The old model remains reachable only through an explicit legacy method.
    assert "StableElectronicModel" in model
    assert "SpinLatticeHamiltonian" in model
    assert "ConductanceSCFSolver" not in model
    lammps = (ROOT / "lammps.py").read_text(encoding="utf-8")
    assert "fix_external_global_electrospin" in lammps
    assert "require_electronic_validity" in lammps
    assert "global_lammps_deployable = False" in model
