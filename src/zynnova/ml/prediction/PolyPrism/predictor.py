from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ....data import MaterialSample
from ...common import load_checkpoint, move_to_device, require_torch, resolve_device
from ...polymer_utils import (
    PSMILESTokenizer,
    extract_psmiles,
    polymer_physics_descriptors,
    polymer_record,
)
from ..PolyPrediction.normalizer import MaskedFeatureNormalizer, MaskedTargetNormalizer
from .config import poly_prism_model_config_from_dict
from .data import PolyPrismDataset, poly_prism_collate
from .model import PolyPrismNetwork

torch = require_torch()


@dataclass(slots=True)
class PolyPrismPrediction:
    id: str
    psmiles: str
    mean: dict[str, float]
    aleatoric_standard_deviation: dict[str, float]
    epistemic_standard_deviation: dict[str, float]
    total_standard_deviation: dict[str, float]
    interval_90: dict[str, tuple[float, float]]
    physics_descriptors: dict[str, float]
    ood_diagnostic: float


@dataclass(slots=True)
class LoadedPolyPrism:
    model: PolyPrismNetwork
    tokenizer: PSMILESTokenizer
    target_normalizer: MaskedTargetNormalizer
    condition_normalizer: MaskedFeatureNormalizer
    device: Any


def load_poly_prism(
    checkpoint: str | Path,
    *,
    device: str = "auto",
) -> LoadedPolyPrism:
    resolved_device = resolve_device(device)
    payload = load_checkpoint(checkpoint, map_location=resolved_device)
    model = PolyPrismNetwork(
        poly_prism_model_config_from_dict(payload["model_config"])
    )
    model.load_state_dict(payload["model_state"])
    model.to(resolved_device).eval()
    return LoadedPolyPrism(
        model=model,
        tokenizer=PSMILESTokenizer.from_state_dict(payload["tokenizer"]),
        target_normalizer=MaskedTargetNormalizer.from_state_dict(
            payload["target_normalizer"]
        ),
        condition_normalizer=MaskedFeatureNormalizer.from_state_dict(
            payload["condition_normalizer"]
        ),
        device=resolved_device,
    )


def _sample(
    value: MaterialSample | Any | str,
    *,
    index: int,
    conditions: Mapping[str, float] | None,
    fidelity: str | None,
) -> MaterialSample:
    if isinstance(value, MaterialSample):
        metadata = dict(value.metadata)
        if fidelity is not None:
            metadata["fidelity"] = fidelity
        return value.copy(
            conditions={**value.conditions, **dict(conditions or {})},
            metadata=metadata,
        )
    record = polymer_record(value)
    metadata = {"psmiles": extract_psmiles(value)}
    if fidelity is not None:
        metadata["fidelity"] = fidelity
    return MaterialSample(
        id=getattr(record, "id", f"candidate-{index}"),
        material_type="polymer",
        structure=record,
        conditions=dict(conditions or {}),
        metadata=metadata,
    )


def predict_poly_prism(
    predictor: LoadedPolyPrism,
    polymers: Sequence[MaterialSample | Any | str],
    *,
    conditions: Mapping[str, float] | Sequence[Mapping[str, float]] | None = None,
    fidelity: str | Sequence[str] | None = None,
    batch_size: int = 256,
) -> list[PolyPrismPrediction]:
    if not polymers:
        return []
    if conditions is None or isinstance(conditions, Mapping):
        condition_rows = [dict(conditions or {}) for _ in polymers]
    else:
        condition_rows = [dict(row) for row in conditions]
        if len(condition_rows) != len(polymers):
            raise ValueError("condition row count must match polymer count")
    if fidelity is None or isinstance(fidelity, str):
        fidelity_rows = [fidelity for _ in polymers]
    else:
        fidelity_rows = list(fidelity)
        if len(fidelity_rows) != len(polymers):
            raise ValueError("fidelity row count must match polymer count")
    samples = [
        _sample(
            value,
            index=index,
            conditions=condition_rows[index],
            fidelity=fidelity_rows[index],
        )
        for index, value in enumerate(polymers)
    ]
    dataset = PolyPrismDataset(
        samples,
        tokenizer=predictor.tokenizer,
        target_normalizer=predictor.target_normalizer,
        condition_normalizer=predictor.condition_normalizer,
        max_length=predictor.model.config.max_length,
        fidelity_names=predictor.model.config.fidelity_names,
    )
    dtype = next(predictor.model.parameters()).dtype
    results: list[PolyPrismPrediction] = []
    predictor.model.eval()
    for start in range(0, len(dataset), batch_size):
        stop = min(start + batch_size, len(dataset))
        batch = poly_prism_collate([dataset[index] for index in range(start, stop)])
        batch = move_to_device(batch, predictor.device)
        for name in (
            "node_features",
            "edge_features",
            "node_weights",
            "conditions",
            "raw_conditions",
            "targets",
            "physics_descriptors",
        ):
            batch[name] = batch[name].to(dtype=dtype)
        with torch.no_grad():
            output = predictor.model(batch)
        mean_z = output["mean"].detach().cpu().numpy()
        aleatoric_z = output["aleatoric_variance"].sqrt().detach().cpu().numpy()
        epistemic_z = output["epistemic_variance"].sqrt().detach().cpu().numpy()
        total_z = (
            output["aleatoric_variance"] + output["epistemic_variance"]
        ).sqrt().detach().cpu().numpy()
        ood = output["ood_score"].detach().cpu().numpy()
        for local_index, sample in enumerate(samples[start:stop]):
            mean = predictor.target_normalizer.decode_row(mean_z[local_index])
            aleatoric = predictor.target_normalizer.physical_std(
                mean_z[local_index], aleatoric_z[local_index]
            )
            epistemic = predictor.target_normalizer.physical_std(
                mean_z[local_index], epistemic_z[local_index]
            )
            total = predictor.target_normalizer.physical_std(
                mean_z[local_index], total_z[local_index]
            )
            intervals = {}
            for spec in predictor.target_normalizer.specs:
                radius = 1.6448536269514722 * total[spec.name]
                low, high = mean[spec.name] - radius, mean[spec.name] + radius
                if spec.lower_bound is not None:
                    low = max(low, spec.lower_bound)
                if spec.upper_bound is not None:
                    high = min(high, spec.upper_bound)
                intervals[spec.name] = (float(low), float(high))
            results.append(
                PolyPrismPrediction(
                    id=sample.id,
                    psmiles=extract_psmiles(sample),
                    mean=mean,
                    aleatoric_standard_deviation=aleatoric,
                    epistemic_standard_deviation=epistemic,
                    total_standard_deviation=total,
                    interval_90=intervals,
                    physics_descriptors=polymer_physics_descriptors(sample),
                    ood_diagnostic=float(ood[local_index]),
                )
            )
    return results


def predict_one_poly_prism(
    predictor: LoadedPolyPrism,
    polymer: MaterialSample | Any | str,
    *,
    conditions: Mapping[str, float] | None = None,
    fidelity: str | None = None,
) -> PolyPrismPrediction:
    return predict_poly_prism(
        predictor,
        [polymer],
        conditions=conditions,
        fidelity=fidelity,
    )[0]


__all__ = [
    "LoadedPolyPrism",
    "PolyPrismPrediction",
    "load_poly_prism",
    "predict_one_poly_prism",
    "predict_poly_prism",
]
