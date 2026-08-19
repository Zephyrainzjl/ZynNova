from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
ase = pytest.importorskip("ase")

from zynnova.ml.zivar.data import atoms_to_conditions, collate_zivar_batches
from zynnova.ml.zivar.types import Targets, ZIVARBatch


def test_explicit_zero_spin_state_is_not_discarded() -> None:
    atoms = ase.Atoms("Fe2", positions=[[0, 0, 0], [2, 0, 0]])
    atoms.arrays["spin_vectors"] = np.zeros((2, 3), dtype=float)
    conditions = atoms_to_conditions(atoms, device=torch.device("cpu"), dtype=torch.float64)
    assert "spin_vectors" in conditions
    assert conditions["spin_vectors"].shape == (2, 3)
    assert torch.count_nonzero(conditions["spin_vectors"]) == 0


def test_typed_collator_offsets_edges_and_preserves_graph_boundaries() -> None:
    def item(distance: float) -> ZIVARBatch:
        return ZIVARBatch(
            positions=torch.tensor([[0.0, 0.0, 0.0], [distance, 0.0, 0.0]]),
            atomic_numbers=torch.tensor([3, 8]),
            batch=torch.zeros(2, dtype=torch.long),
            edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
            shifts=torch.zeros(2, 3),
            cell=torch.eye(3).mul(10.0).unsqueeze(0),
            pbc=torch.zeros(1, 3, dtype=torch.bool),
            node_attrs=torch.eye(2),
            ptr=torch.tensor([0, 2]),
            head=torch.zeros(1, dtype=torch.long),
        )

    collated = collate_zivar_batches([item(1.5), item(2.0)])
    assert collated.graph_count == 2
    assert collated.ptr.tolist() == [0, 2, 4]
    assert collated.batch.tolist() == [0, 0, 1, 1]
    assert collated.edge_index[:, 2:].tolist() == [[2, 3], [3, 2]]
    assert collated.head.tolist() == [0, 0]


def test_typed_targets_distinguish_vector_and_legacy_scalar_moments() -> None:
    vectors = torch.ones((2, 3), dtype=torch.float64)
    scalars = torch.tensor([1.2, 0.7], dtype=torch.float64)
    assert Targets(induced_moments=vectors).as_dict()["magmom_vectors"] is vectors
    assert Targets(scalar_moments=scalars).as_dict()["magmoms"] is scalars
    with pytest.raises(ValueError, match="cannot be enabled"):
        Targets(induced_moments=vectors, scalar_moments=scalars)
