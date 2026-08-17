from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from zynnova.ml.zivar.config import ElectronicConfig, OxidationConfig
from zynnova.ml.zivar.polar import constrain_charge_monopoles


def _config(boundary: str) -> ElectronicConfig:
    return ElectronicConfig(
        method="polar", energy_coupling="full", hidden=(8,),
        polarization_updates=1, boundary_mode=boundary,
        oxidation=OxidationConfig(enabled=False),
    )


def test_fixed_atoms_fragments_and_total_are_simultaneously_exact() -> None:
    values = torch.tensor([0.2, -0.4, 0.3, 0.1], dtype=torch.float64)
    logits = torch.zeros(4, dtype=torch.float64)
    batch = torch.zeros(4, dtype=torch.long)
    membership = torch.tensor(
        [[0.0], [1.0], [1.0], [0.0]], dtype=torch.float64
    )
    conditions = {
        "total_charge": torch.tensor([1.0], dtype=torch.float64),
        "fragment_membership": membership,
        "fragment_charge": torch.tensor([0.25], dtype=torch.float64),
        "fixed_charge_mask": torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64),
        "fixed_charges": torch.tensor([0.4, 0.0, 0.0, 0.0], dtype=torch.float64),
    }
    result, _, _ = constrain_charge_monopoles(
        values, logits, batch, conditions, _config("fixed_charge")
    )
    assert torch.allclose(result.sum(), torch.tensor(1.0, dtype=torch.float64))
    assert torch.allclose(result[1:3].sum(), torch.tensor(0.25, dtype=torch.float64))
    assert torch.allclose(result[0], torch.tensor(0.4, dtype=torch.float64))


def test_mixed_boundary_constrains_only_closed_region() -> None:
    values = torch.tensor([0.2, -0.4, 0.3], dtype=torch.float64)
    result, _, target = constrain_charge_monopoles(
        values,
        torch.zeros(3, dtype=torch.float64),
        torch.zeros(3, dtype=torch.long),
        {
            "reservoir_mask": torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
            "closed_region_charge": torch.tensor([0.75], dtype=torch.float64),
        },
        _config("mixed"),
    )
    assert target is None
    assert torch.allclose(result[:2].sum(), torch.tensor(0.75, dtype=torch.float64))
    assert torch.allclose(result[2], values[2])


def test_overlapping_fixed_and_fragment_constraints_are_rejected() -> None:
    with pytest.raises(ValueError, match="cannot overlap"):
        constrain_charge_monopoles(
            torch.zeros(2, dtype=torch.float64),
            torch.zeros(2, dtype=torch.float64),
            torch.zeros(2, dtype=torch.long),
            {
                "total_charge": torch.tensor([0.0], dtype=torch.float64),
                "fragment_membership": torch.tensor([[1.0], [0.0]], dtype=torch.float64),
                "fragment_charge": torch.tensor([0.0], dtype=torch.float64),
                "fixed_charge_mask": torch.tensor([1.0, 0.0], dtype=torch.float64),
                "fixed_charges": torch.zeros(2, dtype=torch.float64),
            },
            _config("fixed_charge"),
        )
