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
from .calibration import ConformalCalibrator
from .config import poly_prediction_model_config_from_dict
from .data import PolymerPropertyDataset, polymer_property_collate
from .model import PolyPredictionNetwork
from .normalizer import MaskedFeatureNormalizer, MaskedTargetNormalizer

torch = require_torch()


@dataclass(slots=True)
class PolymerPrediction:
    id: str
    psmiles: str
    mean: dict[str, float]
    standard_deviation: dict[str, float]
    interval_90: dict[str, tuple[float, float]]
    physics_descriptors: dict[str, float]


@dataclass(slots=True)
class LoadedPolyPredictor:
    model: PolyPredictionNetwork
    tokenizer: PSMILESTokenizer
    target_normalizer: MaskedTargetNormalizer
    condition_normalizer: MaskedFeatureNormalizer
    device: Any
    calibrator: ConformalCalibrator | None = None


def load_poly_predictor(
    checkpoint: str | Path,
    *,
    device: str = "auto",
    calibrator: ConformalCalibrator | None = None,
) -> LoadedPolyPredictor:
    resolved_device = resolve_device(device)
    payload = load_checkpoint(checkpoint, map_location=resolved_device)
    model_config = poly_prediction_model_config_from_dict(payload["model_config"])
    model = PolyPredictionNetwork(model_config)
    model.load_state_dict(payload["model_state"])
    model.to(resolved_device)
    model.eval()
    return LoadedPolyPredictor(
        model=model,
        tokenizer=PSMILESTokenizer.from_state_dict(payload["tokenizer"]),
        target_normalizer=MaskedTargetNormalizer.from_state_dict(payload["target_normalizer"]),
        condition_normalizer=MaskedFeatureNormalizer.from_state_dict(
            payload["condition_normalizer"]
        ),
        device=resolved_device,
        calibrator=calibrator,
    )


def _material_sample(
    value: MaterialSample | Any | str,
    *,
    index: int,
    conditions: Mapping[str, float] | None,
) -> MaterialSample:
    if isinstance(value, MaterialSample):
        return value.copy(
            conditions={**value.conditions, **dict(conditions or {})},
        )
    record = polymer_record(value)
    return MaterialSample(
        id=getattr(record, "id", f"candidate-{index}"),
        material_type="polymer",
        structure=record,
        conditions=dict(conditions or {}),
        metadata={"psmiles": extract_psmiles(value)},
    )


def _enable_mc_dropout(module) -> None:
    module.eval()
    for child in module.modules():
        if isinstance(child, torch.nn.Dropout):
            child.train()


def predict_polymers(
    predictor: LoadedPolyPredictor,
    polymers: Sequence[MaterialSample | Any | str],
    *,
    conditions: Mapping[str, float] | Sequence[Mapping[str, float]] | None = None,
    mc_samples: int = 24,
    batch_size: int = 256,
) -> list[PolymerPrediction]:
    if not polymers:
        return []
    if mc_samples < 1:
        raise ValueError("mc_samples must be positive")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if conditions is None or isinstance(conditions, Mapping):
        condition_rows = [dict(conditions or {}) for _ in polymers]
    else:
        condition_rows = [dict(row) for row in conditions]
        if len(condition_rows) != len(polymers):
            raise ValueError("condition row count must match polymer count")
    samples = [
        _material_sample(value, index=index, conditions=condition_rows[index])
        for index, value in enumerate(polymers)
    ]
    dataset = PolymerPropertyDataset(
        samples,
        tokenizer=predictor.tokenizer,
        target_normalizer=predictor.target_normalizer,
        condition_normalizer=predictor.condition_normalizer,
        max_length=predictor.model.config.max_length,
    )
    dtype = next(predictor.model.parameters()).dtype
    _enable_mc_dropout(predictor.model)
    results: list[PolymerPrediction] = []
    for start in range(0, len(dataset), batch_size):
        stop = min(start + batch_size, len(dataset))
        batch = polymer_property_collate([dataset[index] for index in range(start, stop)])
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
        means = []
        variances = []
        with torch.no_grad():
            for _ in range(mc_samples):
                output = predictor.model(batch)
                means.append(output["mean"])
                variances.append(output["log_variance"].exp())
        mean_stack = torch.stack(means)
        normalized_mean = mean_stack.mean(dim=0)
        normalized_variance = (
            torch.stack(variances).mean(dim=0)
            + mean_stack.square().mean(dim=0)
            - normalized_mean.square()
        ).clamp_min(1.0e-12)
        normalized_mean_array = normalized_mean.detach().cpu().numpy()
        normalized_std_array = normalized_variance.sqrt().detach().cpu().numpy()
        for local_index, sample in enumerate(samples[start:stop]):
            mean = predictor.target_normalizer.decode_row(normalized_mean_array[local_index])
            standard_deviation = predictor.target_normalizer.physical_std(
                normalized_mean_array[local_index],
                normalized_std_array[local_index],
            )
            intervals = {}
            for spec in predictor.target_normalizer.specs:
                calibrated = (
                    predictor.calibrator.interval(spec.name, mean[spec.name])
                    if predictor.calibrator
                    else None
                )
                if calibrated is None:
                    radius = 1.6448536269514722 * standard_deviation[spec.name]
                    interval = (
                        mean[spec.name] - radius,
                        mean[spec.name] + radius,
                    )
                else:
                    interval = calibrated
                low, high = interval
                if spec.lower_bound is not None:
                    low = max(low, spec.lower_bound)
                if spec.upper_bound is not None:
                    high = min(high, spec.upper_bound)
                intervals[spec.name] = (float(low), float(high))
            results.append(
                PolymerPrediction(
                    id=sample.id,
                    psmiles=extract_psmiles(sample),
                    mean=mean,
                    standard_deviation=standard_deviation,
                    interval_90=intervals,
                    physics_descriptors=polymer_physics_descriptors(sample),
                )
            )
    predictor.model.eval()
    return results


def predict_polymer(
    predictor: LoadedPolyPredictor,
    polymer: MaterialSample | Any | str,
    *,
    conditions: Mapping[str, float] | None = None,
    mc_samples: int = 24,
) -> PolymerPrediction:
    return predict_polymers(
        predictor,
        [polymer],
        conditions=conditions,
        mc_samples=mc_samples,
    )[0]


__all__ = [
    "LoadedPolyPredictor",
    "PolymerPrediction",
    "load_poly_predictor",
    "predict_polymer",
    "predict_polymers",
]
