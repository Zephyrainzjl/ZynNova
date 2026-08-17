from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


def _zynforge_api() -> tuple[type[Any], type[Any]]:
    """Load the legacy ZynForge fixture dependency only when requested.

    The current source snapshot does not contain ``zynnova.ml.zynforge`` but retains
    historical tests for that removed package. Keeping this import lazy prevents those
    stale tests from blocking collection of unrelated current modules.
    """

    try:
        from zynnova.ml.zynforge.field import (
            JouleWeaveModelConfig,
            ZynForgeSymmetryPotential,
        )
    except ModuleNotFoundError as exc:
        pytest.skip(f"legacy zynforge source is absent from this snapshot: {exc}")
    return JouleWeaveModelConfig, ZynForgeSymmetryPotential


def compact_test_config(**overrides: object) -> Any:
    JouleWeaveModelConfig, _ = _zynforge_api()
    values: dict[str, object] = {
        "hidden_dim": 8,
        "num_layers": 2,
        "num_radial": 4,
        "max_ell": 2,
        "correlation_order": 2,
        "tensor_product_rank": 2,
        "directional_edge_rank": 2,
        "num_attention_heads": 2,
        "num_experts": 2,
        "expert_top_k": 2,
        "max_neighbors": None,
        "use_pair_chemical_bias": False,
        "use_hybrid_irrep_norm": True,
        "use_learned_residual_scales": False,
        "use_electronic_depth_context": False,
        "use_zbl": False,
        "use_dispersion": False,
        "use_qeq": False,
    }
    values.update(overrides)
    return JouleWeaveModelConfig.specialist(**values)


def single_structure(*, periodic: bool = False) -> dict[str, torch.Tensor]:
    dtype = torch.float64
    return {
        "z": torch.tensor([1, 8, 6], dtype=torch.long),
        "pos": torch.tensor(
            [[0.20, 0.30, 0.40], [1.30, 0.40, 0.50], [0.50, 1.60, 0.70]],
            dtype=dtype,
        ),
        "batch": torch.zeros(3, dtype=torch.long),
        "cell": 5.0 * torch.eye(3, dtype=dtype).unsqueeze(0),
        "pbc": torch.full((1, 3), periodic, dtype=torch.bool),
    }


@pytest.fixture
def tiny_model() -> Any:
    _, ZynForgeSymmetryPotential = _zynforge_api()
    torch.manual_seed(271828)
    return ZynForgeSymmetryPotential(compact_test_config()).double().eval()


@pytest.fixture
def nonperiodic_inputs() -> dict[str, torch.Tensor]:
    return single_structure(periodic=False)
