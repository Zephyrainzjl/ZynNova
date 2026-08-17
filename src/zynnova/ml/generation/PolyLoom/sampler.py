from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ...common import load_checkpoint, require_torch, resolve_device
from ...polymer_utils import PSMILESTokenizer, polymer_physics_descriptors
from ...prediction.PolyPrediction.normalizer import (
    MaskedFeatureNormalizer,
    MaskedTargetNormalizer,
)
from ...prediction.PolyPrediction.screening import PropertyConstraint
from ...prediction.PolyPrism.predictor import LoadedPolyPrism, predict_poly_prism
from ..PolyGen.representation import (
    decode_polymer_generation_sequence,
    encode_polymer_generation_sequence,
)
from ..PolyGen.theory import assess_paper_mechanisms
from ..PolyGen.validation import validate_generated_polymer
from .config import PolyLoomSamplingConfig, poly_loom_model_config_from_dict
from .model import PolyLoomNetwork

torch = require_torch()


@dataclass(slots=True)
class LoadedPolyLoom:
    model: PolyLoomNetwork
    tokenizer: PSMILESTokenizer
    property_normalizer: MaskedTargetNormalizer
    process_normalizer: MaskedFeatureNormalizer
    device: Any


@dataclass(slots=True)
class PolyLoomGeneratedPolymer:
    psmiles: str
    score: float
    predicted_properties: dict[str, float]
    property_uncertainty: dict[str, float]
    physics_descriptors: dict[str, float]
    theory_checks: dict[str, float | bool]
    port_count: int
    record: Any


@dataclass(slots=True)
class PolyLoomGenerationResult:
    candidates: list[PolyLoomGeneratedPolymer]
    requested_properties: dict[str, float]
    process_conditions: dict[str, float]
    attempted: int
    chemically_valid: int
    rejection_counts: dict[str, int]


def load_poly_loom(
    checkpoint: str | Path,
    *,
    device: str = "auto",
) -> LoadedPolyLoom:
    resolved_device = resolve_device(device)
    payload = load_checkpoint(checkpoint, map_location=resolved_device)
    model = PolyLoomNetwork(
        poly_loom_model_config_from_dict(payload["model_config"])
    )
    model.load_state_dict(payload["model_state"])
    model.to(resolved_device).eval()
    return LoadedPolyLoom(
        model=model,
        tokenizer=PSMILESTokenizer.from_state_dict(payload["tokenizer"]),
        property_normalizer=MaskedTargetNormalizer.from_state_dict(
            payload["property_normalizer"]
        ),
        process_normalizer=MaskedFeatureNormalizer.from_state_dict(
            payload["process_normalizer"]
        ),
        device=resolved_device,
    )


def _condition_tensors(
    generator: LoadedPolyLoom,
    requested: Mapping[str, float],
    process: Mapping[str, float],
    count: int,
    *,
    z_clip: float | None,
):
    unknown_properties = set(requested) - set(generator.property_normalizer.names)
    unknown_process = set(process) - set(generator.process_normalizer.names)
    if unknown_properties:
        raise KeyError(f"unknown generated properties: {sorted(unknown_properties)}")
    if unknown_process:
        raise KeyError(f"unknown process conditions: {sorted(unknown_process)}")
    properties, property_mask = generator.property_normalizer.encode_row(requested)
    process_values, process_mask = generator.process_normalizer.encode_row(process)
    if z_clip is not None:
        properties = np.clip(properties, -z_clip, z_clip)
        process_values = np.clip(process_values, -z_clip, z_clip)
    dtype = next(generator.model.parameters()).dtype

    def repeated(values, *, boolean: bool = False):
        tensor = torch.as_tensor(values, device=generator.device)
        tensor = tensor.to(dtype=torch.bool if boolean else dtype)
        return tensor[None].repeat(count, 1)

    return (
        repeated(properties),
        repeated(property_mask, boolean=True),
        repeated(process_values),
        repeated(process_mask, boolean=True),
    )


def _sample_top_p(logits, *, temperature: float, top_p: float):
    logits = logits / max(temperature, 1.0e-4)
    ordered, indices = logits.sort(dim=-1, descending=True)
    probabilities = ordered.softmax(dim=-1)
    removed = probabilities.cumsum(dim=-1) - probabilities > top_p
    probabilities = ordered.masked_fill(removed, float("-inf")).softmax(dim=-1)
    sampled_ordered = torch.multinomial(
        probabilities.reshape(-1, probabilities.shape[-1]), 1
    ).reshape(*probabilities.shape[:-1])
    return indices.gather(-1, sampled_ordered[..., None]).squeeze(-1)


@torch.no_grad()
def _sample_tokens(
    generator: LoadedPolyLoom,
    conditions,
    *,
    config: PolyLoomSamplingConfig,
):
    properties, property_mask, process, process_mask = conditions
    count = properties.shape[0]
    length_logits = generator.model.predict_length(
        properties, property_mask, process, process_mask
    )
    maximum = min(
        config.maximum_length or generator.model.config.max_length,
        generator.model.config.max_length,
    )
    length_logits[:, : config.minimum_length] = float("-inf")
    if maximum + 1 < length_logits.shape[1]:
        length_logits[:, maximum + 1 :] = float("-inf")
    lengths = torch.multinomial(length_logits.softmax(dim=-1), 1).squeeze(-1)
    tokens = torch.full(
        (count, generator.model.config.max_length),
        generator.tokenizer.pad_id,
        device=generator.device,
        dtype=torch.long,
    )
    positions = torch.arange(tokens.shape[1], device=generator.device)
    attention = positions[None] < lengths[:, None]
    tokens[attention] = generator.tokenizer.mask_id
    tokens[:, 0] = generator.tokenizer.bos_id
    rows = torch.arange(count, device=generator.device)
    tokens[rows, lengths - 1] = generator.tokenizer.eos_id
    self_condition = None
    for step in range(config.refinement_steps):
        time = torch.full(
            (count,),
            1.0 - step / max(config.refinement_steps - 1, 1),
            device=generator.device,
            dtype=properties.dtype,
        ).clamp_min(1.0e-3)
        conditional = generator.model(
            tokens,
            attention,
            time,
            properties,
            property_mask,
            process,
            process_mask,
            self_condition=self_condition,
        )["logits"]
        unconditional = generator.model(
            tokens,
            attention,
            time,
            torch.zeros_like(properties),
            torch.zeros_like(property_mask),
            torch.zeros_like(process),
            torch.zeros_like(process_mask),
            self_condition=self_condition,
        )["logits"]
        logits = unconditional + config.guidance_scale * (conditional - unconditional)
        self_condition = logits.softmax(dim=-1)
        masked = tokens.eq(generator.tokenizer.mask_id) & attention
        remaining_steps = config.refinement_steps - step
        for row in range(count):
            candidates = torch.nonzero(masked[row], as_tuple=False).flatten()
            if not candidates.numel():
                continue
            reveal = max(1, int(np.ceil(candidates.numel() / remaining_steps)))
            confidence = logits[row, candidates].softmax(dim=-1).amax(dim=-1)
            chosen = candidates[confidence.topk(min(reveal, candidates.numel())).indices]
            tokens[row, chosen] = _sample_top_p(
                logits[row, chosen],
                temperature=config.temperature,
                top_p=config.top_p,
            )
    return tokens


def _target_score(
    generator: LoadedPolyLoom,
    predicted: Mapping[str, float],
    requested: Mapping[str, float],
) -> float:
    terms = []
    for name, target in requested.items():
        if name not in predicted:
            continue
        index = generator.property_normalizer.names.index(name)
        scale = max(
            abs(float(target)),
            float(generator.property_normalizer.std[index]),
            1.0e-6,
        )
        terms.append(abs(float(predicted[name]) - float(target)) / scale)
    return -float(np.mean(terms)) if terms else 0.0


def _diverse_top_k(
    candidates: Sequence[PolyLoomGeneratedPolymer],
    count: int,
    *,
    diversity_weight: float,
) -> list[PolyLoomGeneratedPolymer]:
    from ...polymer_utils import tokenize_psmiles

    def similarity(left: str, right: str) -> float:
        left_tokens, right_tokens = set(tokenize_psmiles(left)), set(tokenize_psmiles(right))
        union = left_tokens | right_tokens
        return len(left_tokens & right_tokens) / len(union) if union else 1.0

    pool = list(candidates)
    selected = []
    while pool and len(selected) < count:
        best = max(
            pool,
            key=lambda item: item.score
            + diversity_weight
            * (1.0 if not selected else 1.0 - max(
                similarity(item.psmiles, chosen.psmiles) for chosen in selected
            )),
        )
        selected.append(best)
        pool.remove(best)
    return selected


def generate_poly_loom(
    generator: LoadedPolyLoom,
    requested_properties: Mapping[str, float],
    *,
    process_conditions: Mapping[str, float] | None = None,
    config: PolyLoomSamplingConfig | None = None,
    predictor: LoadedPolyPrism | None = None,
    constraints: Sequence[PropertyConstraint] = (),
) -> PolyLoomGenerationResult:
    config = config or PolyLoomSamplingConfig()
    config.__post_init__()
    if constraints and predictor is None:
        raise ValueError("property constraints require an independent PolyPrism model")
    process_conditions = dict(process_conditions or {})
    count_per_round = config.num_candidates * config.oversample_factor
    attempted = 0
    seen: set[str] = set()
    valid_rows = []
    rejection_counts: dict[str, int] = {}
    for round_index in range(config.max_sampling_rounds):
        torch.manual_seed(config.seed + round_index)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed + round_index)
        conditions = _condition_tensors(
            generator,
            requested_properties,
            process_conditions,
            count_per_round,
            z_clip=config.condition_z_clip,
        )
        token_ids = _sample_tokens(generator, conditions, config=config)
        attempted += count_per_round
        for row in token_ids:
            sequence = generator.tokenizer.decode(row.detach().cpu().tolist())
            try:
                psmiles = decode_polymer_generation_sequence(
                    sequence,
                    generator.model.config.representation,
                    repair_missing_ports=generator.model.config.repair_missing_ports,
                )
            except (RuntimeError, ValueError):
                key = "generation sequence could not be decoded"
                rejection_counts[key] = rejection_counts.get(key, 0) + 1
                continue
            if psmiles in seen:
                continue
            seen.add(psmiles)
            report = validate_generated_polymer(
                psmiles, require_two_ports=config.require_two_ports
            )
            if not report.valid:
                key = report.reason or "invalid generated polymer"
                rejection_counts[key] = rejection_counts.get(key, 0) + 1
                continue
            descriptors = polymer_physics_descriptors(report.record)
            if (
                config.minimum_configurational_entropy_R is not None
                and descriptors["configurational_entropy_R"]
                < config.minimum_configurational_entropy_R
            ):
                key = "below configurational-entropy threshold"
                rejection_counts[key] = rejection_counts.get(key, 0) + 1
                continue
            valid_rows.append((psmiles, report, descriptors))
        if len(valid_rows) >= config.num_candidates * 2:
            break

    if predictor and valid_rows:
        predictions = predict_poly_prism(
            predictor,
            [row[0] for row in valid_rows],
            conditions=process_conditions,
            fidelity="unknown",
        )
        predicted_rows = [item.mean for item in predictions]
        uncertainty_rows = [item.total_standard_deviation for item in predictions]
    elif valid_rows:
        # Use the generator's auxiliary head only when no independent predictor is supplied.
        sequences = [
            generator.tokenizer.encode(
                encode_polymer_generation_sequence(
                    row[0], generator.model.config.representation
                ),
                max_length=generator.model.config.max_length,
            )
            for row in valid_rows
        ]
        ids = torch.as_tensor(
            np.stack([item[0] for item in sequences]), device=generator.device
        )
        masks = torch.as_tensor(
            np.stack([item[1] for item in sequences]), device=generator.device
        )
        with torch.no_grad():
            predicted_z = generator.model.predict_properties(ids, masks).cpu().numpy()
        predicted_rows = [
            generator.property_normalizer.decode_row(row) for row in predicted_z
        ]
        uncertainty_rows = [{} for _ in valid_rows]
    else:
        predicted_rows, uncertainty_rows = [], []

    candidates = []
    for row, predicted, uncertainty in zip(
        valid_rows, predicted_rows, uncertainty_rows, strict=True
    ):
        psmiles, report, descriptors = row
        failed = False
        constraint_score = 0.0
        for constraint in constraints:
            value = predicted.get(constraint.name)
            if value is None:
                failed = True
                break
            if constraint.lower is not None and value < constraint.lower:
                failed = True
            if constraint.upper is not None and value > constraint.upper:
                failed = True
            constraint_score += constraint.weight
        if failed:
            rejection_counts["independent property constraint failed"] = (
                rejection_counts.get("independent property constraint failed", 0) + 1
            )
            continue
        score = _target_score(generator, predicted, requested_properties)
        score += 0.05 * min(descriptors["configurational_entropy_R"], 2.5)
        score += constraint_score
        if uncertainty:
            score -= 0.01 * sum(
                uncertainty.get(name, 0.0) / max(abs(predicted.get(name, 0.0)), 1.0e-8)
                for name in requested_properties
            )
        candidates.append(
            PolyLoomGeneratedPolymer(
                psmiles=psmiles,
                score=score,
                predicted_properties=dict(predicted),
                property_uncertainty=dict(uncertainty),
                physics_descriptors=descriptors,
                theory_checks=assess_paper_mechanisms(psmiles),
                port_count=report.port_count,
                record=report.record,
            )
        )
    selected = _diverse_top_k(
        sorted(candidates, key=lambda item: item.score, reverse=True),
        config.num_candidates,
        diversity_weight=config.diversity_weight,
    )
    return PolyLoomGenerationResult(
        candidates=selected,
        requested_properties=dict(requested_properties),
        process_conditions=process_conditions,
        attempted=attempted,
        chemically_valid=len(valid_rows),
        rejection_counts=rejection_counts,
    )


def save_poly_loom_generation(
    result: PolyLoomGenerationResult,
    path: str | Path,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "requested_properties": result.requested_properties,
        "process_conditions": result.process_conditions,
        "attempted": result.attempted,
        "chemically_valid": result.chemically_valid,
        "rejection_counts": result.rejection_counts,
        "candidates": [
            {
                "psmiles": item.psmiles,
                "score": item.score,
                "predicted_properties": item.predicted_properties,
                "property_uncertainty": item.property_uncertainty,
                "physics_descriptors": item.physics_descriptors,
                "theory_checks": item.theory_checks,
                "port_count": item.port_count,
            }
            for item in result.candidates
        ],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


__all__ = [
    "LoadedPolyLoom",
    "PolyLoomGeneratedPolymer",
    "PolyLoomGenerationResult",
    "generate_poly_loom",
    "load_poly_loom",
    "save_poly_loom_generation",
]
