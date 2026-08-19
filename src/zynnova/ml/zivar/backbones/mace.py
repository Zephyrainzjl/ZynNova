"""Default reviewed upstream backbone registration."""

from __future__ import annotations

from typing import Any

from ..backbone import (
    ReferenceBackboneAdapter,
    _forward_kwargs,
    build_reference_backbone,
    installed_mace_version,
)
from .base import BackboneAdapter, BackboneCapabilities, register_backbone


class MaceBackboneAdapter(BackboneAdapter):
    """Contract adapter preserving the legacy ``backbone.model.*`` state keys."""

    def __init__(
        self,
        core: Any,
        *,
        kind: str = "mace",
        architecture: str = "scale-shift-symmetric-contraction",
    ) -> None:
        legacy = ReferenceBackboneAdapter(core)
        super().__init__(
            core,
            kind=kind,
            architecture=architecture,
            implementation=f"mace-torch=={installed_mace_version()}",
            invariant_dim=legacy.invariant_dim,
            atomic_numbers=legacy.atomic_numbers,
            cutoff_A=legacy.cutoff_A,
            capabilities=BackboneCapabilities(local_mliap=True),
        )
        self.irreps_out = legacy.irreps_out
        self.lmax = legacy.lmax
        self.features_per_layer = legacy.features_per_layer
        self.num_interactions = legacy.num_interactions

    def _extract_invariants(self, node_features: Any) -> Any:
        return ReferenceBackboneAdapter._extract_invariants(self, node_features)

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        output = self.model(data, **_forward_kwargs(self.model))
        if not isinstance(output, dict) or output.get("node_feats") is None:
            raise RuntimeError("upstream model did not return node_feats")
        return {
            "energy": output["energy"],
            "node_energy": output.get("node_energy"),
            "invariant_features": self._extract_invariants(output["node_feats"]),
            "raw_output": output,
        }


def build_mace_backbone(config: Any, *, device: Any = "cpu") -> MaceBackboneAdapter:
    core = build_reference_backbone(config, device=device)
    return MaceBackboneAdapter(core, kind="mace")


def register() -> None:
    register_backbone(
        "mace",
        build_mace_backbone,
        description="Default reviewed higher-order equivariant local potential",
        provenance="ACEsuit/mace 0.3.16 official runtime",
    )


__all__ = ["MaceBackboneAdapter", "build_mace_backbone", "register"]
