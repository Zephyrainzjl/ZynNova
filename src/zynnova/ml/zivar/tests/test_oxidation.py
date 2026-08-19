from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from zynnova.ml.zivar.config import OxidationConfig
from zynnova.ml.zivar.oxidation import OxidationStateHead, exact_charge_balanced_states


def test_exact_oxidation_assignment_obeys_integer_charge() -> None:
    values = torch.arange(-2, 5)
    logits = torch.tensor(
        [
            [-9.0, -9.0, -9.0, 5.0, -9.0, -9.0, -9.0],
            [5.0, -9.0, -9.0, -9.0, -9.0, -9.0, -9.0],
            [-9.0, -9.0, 5.0, -9.0, -9.0, -9.0, -9.0],
        ]
    )
    allowed = torch.ones_like(logits, dtype=torch.bool)
    batch = torch.tensor([0, 0, 0])
    states = exact_charge_balanced_states(
        logits, allowed, batch, torch.tensor([0.0]), values
    )
    assert int(states.sum()) == 0
    assert states.tolist() == [1, -1, 0]


def test_impossible_oxidation_balance_is_rejected() -> None:
    values = torch.tensor((0, 2))
    logits = torch.zeros((2, 2))
    allowed = torch.ones_like(logits, dtype=torch.bool)
    with pytest.raises(ValueError, match="no charge-balanced"):
        exact_charge_balanced_states(
            logits, allowed, torch.zeros(2, dtype=torch.long), torch.tensor([1.0]), values
        )


def test_supervised_head_returns_calibrated_probabilities_and_confidence() -> None:
    config = OxidationConfig(
        enabled=True,
        label_source="MaterialsProject_formal_oxidation_state",
        hidden=(8,),
        temperature=1.5,
    )
    head = OxidationStateHead(4, (3, 8), config).double()
    prediction = head(
        torch.randn(2, 4, dtype=torch.float64),
        torch.tensor([3, 8]),
        torch.zeros(2, dtype=torch.long),
        torch.tensor([0.0], dtype=torch.float64),
        exact=False,
    )
    assert torch.allclose(
        prediction.probabilities.sum(-1), torch.ones(2, dtype=torch.float64)
    )
    assert bool(torch.all((prediction.confidence >= 0) & (prediction.confidence <= 1)))
    assert bool(torch.all(prediction.entropy >= 0))
