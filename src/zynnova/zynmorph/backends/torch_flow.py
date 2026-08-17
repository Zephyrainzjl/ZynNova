"""Checkpoint adapter for the built-in conditional 3-D rectified-flow model."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ...core.backend import Availability, module_availability
from ...core.exceptions import BackendUnavailableError
from ..generation import GenerationResult
from ..schema import DEFAULT_PHASE_NAMES, MicrostructureCondition
from ..volume import MicrostructureVolume


class TorchFlowBackend:
    name = "torch-rectified-flow"

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        *,
        device: str = "auto",
        condition_keys: tuple[str, ...] | None = None,
    ) -> None:
        self.checkpoint = None if checkpoint is None else Path(checkpoint)
        self.device = device
        self.condition_keys = condition_keys

    def availability(self) -> Availability:
        torch_status = module_availability("torch")
        if not torch_status.available:
            return torch_status
        if self.checkpoint is None:
            return Availability(False, "checkpoint path was not configured")
        if not self.checkpoint.is_file():
            return Availability(False, f"checkpoint does not exist: {self.checkpoint}")
        return Availability(True, details={"checkpoint": str(self.checkpoint.resolve())})

    def generate(
        self,
        condition: MicrostructureCondition,
        *,
        refinement_steps: int = 0,
        temperature: float = 0.15,
    ) -> GenerationResult:
        del refinement_steps, temperature
        self.availability().require(self.name)
        import torch

        from ..torch_models import ConditionalRectifiedFlow3D, sample_flow

        assert self.checkpoint is not None
        payload = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        config = dict(payload.get("model_config", {}))
        phase_ids = tuple(int(item) for item in payload.get("phase_ids", condition.phases))
        keys = tuple(payload.get("condition_keys", self.condition_keys or ()))
        model = ConditionalRectifiedFlow3D(
            phases=len(phase_ids),
            condition_dim=len(keys),
            **config,
        )
        state = payload.get("model", payload.get("state_dict", payload))
        model.load_state_dict(state, strict=True)
        device = _resolve_device(self.device)
        model.to(device).eval()
        vector = _condition_vector(condition, keys)
        tensor = torch.tensor([vector], device=device, dtype=next(model.parameters()).dtype)
        labels = sample_flow(
            model,
            tensor,
            condition.shape,
            steps=int(payload.get("sampling_steps", 32)),
            seed=condition.seed,
            exact_counts=condition.exact_phase_counts(),
            phase_ids=phase_ids,
        )[0].cpu().numpy()
        volume = MicrostructureVolume(
            labels=labels,
            voxel_size_m=condition.voxel_size_m,
            phase_names=DEFAULT_PHASE_NAMES,
            metadata={"generator": self.name, "checkpoint": str(self.checkpoint.resolve())},
        )
        achieved = {
            phase: int(np.count_nonzero(labels == phase)) for phase in condition.phases
        }
        return GenerationResult(
            volume=volume,
            backend=self.name,
            exact_counts=condition.exact_phase_counts(),
            achieved_counts=achieved,
            refinement_loss=None,
            metadata={"condition_keys": keys},
        )


def _resolve_device(requested: str) -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _condition_vector(
    condition: MicrostructureCondition,
    keys: tuple[str, ...],
) -> list[float]:
    values: Mapping[str, Any] = {
        **{f"phase_fraction.{key}": value for key, value in condition.phase_fractions.items()},
        **condition.manufacturing,
        **condition.descriptor_targets,
    }
    missing = [key for key in keys if key not in values]
    if missing:
        raise BackendUnavailableError(f"checkpoint condition keys are missing: {missing}")
    return [float(values[key]) for key in keys]


__all__ = ["TorchFlowBackend"]
