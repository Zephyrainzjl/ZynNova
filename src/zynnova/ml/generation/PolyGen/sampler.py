from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ...common import load_checkpoint, require_torch, resolve_device
from ...polymer_utils import (
    PSMILESTokenizer,
    polymer_physics_descriptors,
    tokenize_psmiles,
)
from ...prediction.PolyPrediction.normalizer import (
    MaskedFeatureNormalizer,
    MaskedTargetNormalizer,
)
from ...prediction.PolyPrediction.predictor import (
    LoadedPolyPredictor,
    predict_polymers,
)
from ...prediction.PolyPrediction.screening import (
    PropertyConstraint,
    screen_predictions,
)
from .config import (
    PolyGenSamplingConfig,
    poly_gen_model_config_from_dict,
)
from .model import PolymerMaskedFlow
from .representation import (
    POLYMER_SELFIES_PORT_TOKEN,
    decode_polymer_generation_sequence,
    encode_polymer_generation_sequence,
)
from .theory import assess_paper_mechanisms
from .validation import validate_generated_polymer

torch = require_torch()


@dataclass(slots=True)
class LoadedPolyGenerator:
    model: PolymerMaskedFlow
    tokenizer: PSMILESTokenizer
    property_normalizer: MaskedTargetNormalizer
    process_normalizer: MaskedFeatureNormalizer
    device: Any


@dataclass(slots=True)
class GeneratedPolymer:
    psmiles: str
    score: float
    predicted_properties: dict[str, float]
    property_uncertainty: dict[str, float]
    physics_descriptors: dict[str, float]
    theory_checks: dict[str, float | bool]
    port_count: int
    record: Any


@dataclass(slots=True)
class PolyGenerationResult:
    candidates: list[GeneratedPolymer]
    requested_properties: dict[str, float]
    process_conditions: dict[str, float]
    attempted: int
    chemically_valid: int
    rejection_counts: dict[str, int]


def load_poly_generator(
    checkpoint: str | Path,
    *,
    device: str = "auto",
) -> LoadedPolyGenerator:
    resolved_device = resolve_device(device)
    payload = load_checkpoint(checkpoint, map_location=resolved_device)
    config = poly_gen_model_config_from_dict(payload["model_config"])
    model = PolymerMaskedFlow(config)
    model.load_state_dict(payload["model_state"])
    model.to(resolved_device)
    model.eval()
    return LoadedPolyGenerator(
        model=model,
        tokenizer=PSMILESTokenizer.from_state_dict(payload["tokenizer"]),
        property_normalizer=MaskedTargetNormalizer.from_state_dict(payload["property_normalizer"]),
        process_normalizer=MaskedFeatureNormalizer.from_state_dict(payload["process_normalizer"]),
        device=resolved_device,
    )


def _sample_top_p(logits, *, temperature: float, top_p: float):
    scaled = logits / max(temperature, 1.0e-4)
    sorted_logits, sorted_indices = torch.sort(scaled, descending=True, dim=-1)
    probabilities = torch.softmax(sorted_logits, dim=-1)
    cumulative = probabilities.cumsum(dim=-1)
    remove = cumulative - probabilities > top_p
    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    filtered = torch.softmax(sorted_logits, dim=-1)
    leading_shape = filtered.shape[:-1]
    sampled_sorted = torch.multinomial(
        filtered.reshape(-1, filtered.shape[-1]),
        1,
    ).reshape(leading_shape)
    sampled = sorted_indices.gather(-1, sampled_sorted[..., None]).squeeze(-1)
    confidence = filtered.gather(-1, sampled_sorted[..., None]).squeeze(-1)
    return sampled, confidence


def _condition_tensors(
    generator: LoadedPolyGenerator,
    requested: Mapping[str, float],
    process: Mapping[str, float],
    count: int,
    *,
    z_clip: float | None,
):
    unknown_properties = set(requested) - set(generator.property_normalizer.names)
    unknown_process = set(process) - set(generator.process_normalizer.names)
    if unknown_properties:
        raise KeyError(f"unknown generated-property conditions: {sorted(unknown_properties)}")
    if unknown_process:
        raise KeyError(f"unknown process conditions: {sorted(unknown_process)}")
    properties, property_mask = generator.property_normalizer.encode_row(requested)
    process_values, process_mask = generator.process_normalizer.encode_row(process)
    if z_clip is not None:
        properties = np.clip(properties, -z_clip, z_clip)
        process_values = np.clip(process_values, -z_clip, z_clip)
    dtype = next(generator.model.parameters()).dtype
    device = generator.device

    def repeated(array, *, boolean: bool = False):
        tensor = torch.as_tensor(array, device=device)
        tensor = tensor.to(dtype=torch.bool if boolean else dtype)
        return tensor[None].repeat(count, 1)

    return (
        repeated(properties),
        repeated(property_mask, boolean=True),
        repeated(process_values),
        repeated(process_mask, boolean=True),
    )


def _sample_lengths(
    generator: LoadedPolyGenerator,
    conditions,
    config: PolyGenSamplingConfig,
):
    properties, property_mask, process, process_mask = conditions
    logits = generator.model.predict_length(
        properties,
        property_mask,
        process,
        process_mask,
    )
    maximum = min(
        config.maximum_length or generator.model.config.max_length,
        generator.model.config.max_length,
    )
    logits[:, : config.minimum_length] = float("-inf")
    if maximum + 1 < logits.shape[1]:
        logits[:, maximum + 1 :] = float("-inf")
    probabilities = torch.softmax(logits, dim=-1)
    return torch.multinomial(probabilities, 1).squeeze(-1)


def _masked_flow_sample(
    generator: LoadedPolyGenerator,
    conditions,
    *,
    config: PolyGenSamplingConfig,
):
    properties, property_mask, process, process_mask = conditions
    count = properties.shape[0]
    lengths = _sample_lengths(generator, conditions, config)
    max_length = int(lengths.max().item())
    token_ids = torch.full(
        (count, max_length),
        generator.tokenizer.pad_id,
        device=generator.device,
        dtype=torch.long,
    )
    attention_mask = torch.zeros_like(token_ids, dtype=torch.bool)
    for row, length_tensor in enumerate(lengths):
        length = int(length_tensor.item())
        token_ids[row, :length] = generator.tokenizer.mask_id
        token_ids[row, 0] = generator.tokenizer.bos_id
        token_ids[row, length - 1] = generator.tokenizer.eos_id
        attention_mask[row, :length] = True

    force_polymer_selfies_ports = (
        config.require_two_ports and generator.model.config.representation == "polymer_selfies"
    )
    port_token_id = None
    port_positions = None
    if force_polymer_selfies_ports:
        if config.minimum_length < 4:
            raise ValueError("Polymer-SELFIES needs room for BOS, two ports and EOS")
        port_token_id = generator.tokenizer.token_to_id.get(POLYMER_SELFIES_PORT_TOKEN)
        if port_token_id is None:
            raise ValueError(
                "Polymer-SELFIES tokenizer has no port token; "
                "the training data must contain exactly two-port PSMILES"
            )

    special_ids = [
        generator.tokenizer.pad_id,
        generator.tokenizer.bos_id,
        generator.tokenizer.eos_id,
        generator.tokenizer.mask_id,
        generator.tokenizer.unk_id,
    ]

    def guided_logits(current_token_ids, time):
        conditional = generator.model(
            current_token_ids,
            attention_mask,
            time,
            properties,
            property_mask,
            process,
            process_mask,
        )
        unconditional = generator.model(
            current_token_ids,
            attention_mask,
            time,
            torch.zeros_like(properties),
            torch.zeros_like(property_mask),
            torch.zeros_like(process),
            torch.zeros_like(process_mask),
        )
        return unconditional + config.guidance_scale * (conditional - unconditional)

    with torch.no_grad():
        for step in range(config.refinement_steps):
            progress = step / max(config.refinement_steps - 1, 1)
            time = torch.full(
                (count,),
                max(1.0 - progress, 1.0e-3),
                device=generator.device,
                dtype=properties.dtype,
            )
            logits = guided_logits(token_ids, time)
            if force_polymer_selfies_ports and port_positions is None:
                port_positions = torch.zeros_like(token_ids, dtype=torch.bool)
                assert port_token_id is not None
                for row, length_tensor in enumerate(lengths):
                    length = int(length_tensor.item())
                    eligible_positions = torch.arange(
                        1,
                        length - 1,
                        device=generator.device,
                    )
                    scores = logits[row, eligible_positions, port_token_id]
                    selected = eligible_positions[torch.topk(scores, k=2).indices]
                    port_positions[row, selected] = True
                    token_ids[row, selected] = port_token_id
                # Re-evaluate with both learned port positions visible to the
                # denoiser, then forbid additional port tokens everywhere else.
                logits = guided_logits(token_ids, time)
            logits[..., special_ids] = float("-inf")
            if force_polymer_selfies_ports:
                assert port_token_id is not None
                logits[..., port_token_id] = float("-inf")
            sampled, confidence = _sample_top_p(
                logits,
                temperature=config.temperature,
                top_p=config.top_p,
            )
            current_mask = token_ids.eq(generator.tokenizer.mask_id)
            for row in range(count):
                positions = torch.nonzero(current_mask[row], as_tuple=False).flatten()
                if not positions.numel():
                    continue
                if step + 1 == config.refinement_steps:
                    commit_count = positions.numel()
                else:
                    remaining_fraction = math.cos(
                        0.5 * math.pi * (step + 1) / config.refinement_steps
                    )
                    desired_remaining = int(
                        math.ceil((int(lengths[row].item()) - 2) * remaining_fraction)
                    )
                    commit_count = max(positions.numel() - desired_remaining, 1)
                local_confidence = confidence[row, positions]
                selected = positions[
                    torch.topk(
                        local_confidence,
                        k=min(commit_count, positions.numel()),
                    ).indices
                ]
                token_ids[row, selected] = sampled[row, selected]
    return token_ids, attention_mask


def _model_property_predictions(
    generator: LoadedPolyGenerator,
    psmiles_values: Sequence[str],
) -> list[dict[str, float]]:
    rows = [
        generator.tokenizer.encode(
            encode_polymer_generation_sequence(
                psmiles,
                generator.model.config.representation,
            ),
            max_length=generator.model.config.max_length,
        )
        for psmiles in psmiles_values
    ]
    token_ids = torch.as_tensor(
        np.stack([row[0] for row in rows]),
        device=generator.device,
        dtype=torch.long,
    )
    attention = torch.as_tensor(
        np.stack([row[1] for row in rows]),
        device=generator.device,
        dtype=torch.bool,
    )
    with torch.no_grad():
        normalized = generator.model.predict_properties(token_ids, attention)
    normalized = normalized.detach().cpu().numpy()
    return [generator.property_normalizer.decode_row(row) for row in normalized]


def _target_score(
    generator: LoadedPolyGenerator,
    predicted: Mapping[str, float],
    requested: Mapping[str, float],
) -> float:
    if not requested:
        return 0.0
    predicted_z, predicted_mask = generator.property_normalizer.encode_row(predicted)
    target_z, target_mask = generator.property_normalizer.encode_row(requested)
    observed = predicted_mask & target_mask
    if not observed.any():
        return -1.0e6
    return -float(np.abs(predicted_z[observed] - target_z[observed]).mean())


def _token_similarity(first: str, second: str) -> float:
    left = set(tokenize_psmiles(first))
    right = set(tokenize_psmiles(second))
    return len(left & right) / max(len(left | right), 1)


def _diverse_top_k(
    candidates: Sequence[GeneratedPolymer],
    count: int,
    *,
    diversity_weight: float,
) -> list[GeneratedPolymer]:
    pool = list(candidates)
    selected: list[GeneratedPolymer] = []
    while pool and len(selected) < count:
        best = max(
            pool,
            key=lambda candidate: (
                candidate.score
                + diversity_weight
                * (
                    1.0
                    if not selected
                    else 1.0
                    - max(_token_similarity(candidate.psmiles, item.psmiles) for item in selected)
                )
            ),
        )
        selected.append(best)
        pool.remove(best)
    return selected


def _rejection_category(reason: str | None) -> str:
    if not reason:
        return "invalid decoded structure"
    if reason.startswith("chemical parser rejected"):
        return "chemical parser rejected decoded structure"
    if "polymerization ports" in reason:
        return "decoded structure does not have exactly two terminal ports"
    return reason


def generate_polymers(
    generator: LoadedPolyGenerator,
    requested_properties: Mapping[str, float],
    *,
    process_conditions: Mapping[str, float] | None = None,
    config: PolyGenSamplingConfig | None = None,
    predictor: LoadedPolyPredictor | None = None,
    constraints: Sequence[PropertyConstraint] = (),
) -> PolyGenerationResult:
    config = config or PolyGenSamplingConfig()
    config.__post_init__()
    if constraints and predictor is None:
        raise ValueError("property constraints require an independent PolyPrediction model")
    process_conditions = dict(process_conditions or {})
    count_per_round = config.num_candidates * config.oversample_factor
    attempted = 0
    seen_sequences: set[str] = set()
    seen_psmiles: set[str] = set()
    valid_rows: list[tuple[str, Any, dict[str, float]]] = []
    rejection_counts: dict[str, int] = {}
    desired_valid_pool = max(config.num_candidates * 2, config.num_candidates)
    representation = generator.model.config.representation
    for round_index in range(config.max_sampling_rounds):
        torch.manual_seed(config.seed + round_index)
        conditions = _condition_tensors(
            generator,
            requested_properties,
            process_conditions,
            count_per_round,
            z_clip=config.condition_z_clip,
        )
        token_ids, _ = _masked_flow_sample(generator, conditions, config=config)
        attempted += count_per_round
        decoded_sequences = [
            generator.tokenizer.decode(row.detach().cpu().tolist()) for row in token_ids
        ]
        for sequence in decoded_sequences:
            if sequence in seen_sequences:
                continue
            seen_sequences.add(sequence)
            try:
                psmiles = decode_polymer_generation_sequence(
                    sequence,
                    representation,
                    repair_missing_ports=generator.model.config.repair_missing_ports,
                )
            except (RuntimeError, ValueError):
                reason = (
                    "Polymer-SELFIES decoding or port restoration failed"
                    if representation == "polymer_selfies"
                    else "generation sequence could not be decoded"
                )
                rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
                continue
            if psmiles in seen_psmiles:
                continue
            seen_psmiles.add(psmiles)
            report = validate_generated_polymer(
                psmiles,
                require_two_ports=config.require_two_ports,
            )
            if not report.valid:
                reason = _rejection_category(report.reason)
                rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
                continue
            descriptors = polymer_physics_descriptors(report.record)
            if (
                config.minimum_configurational_entropy_R is not None
                and descriptors["configurational_entropy_R"]
                < config.minimum_configurational_entropy_R
            ):
                reason = "below configurational-entropy threshold"
                rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
                continue
            valid_rows.append((psmiles, report, descriptors))
        if len(valid_rows) >= desired_valid_pool:
            break

    if predictor and valid_rows:
        predictor_outputs = predict_polymers(
            predictor,
            [row[0] for row in valid_rows],
            conditions=process_conditions,
        )
        predicted_rows = [item.mean for item in predictor_outputs]
        uncertainty_rows = [item.standard_deviation for item in predictor_outputs]
        screened = screen_predictions(predictor_outputs, constraints) if constraints else []
        constraint_scores = {item.prediction.psmiles: item.score for item in screened}
    else:
        predicted_rows = (
            _model_property_predictions(generator, [row[0] for row in valid_rows])
            if valid_rows
            else []
        )
        uncertainty_rows = [{} for _ in valid_rows]
        constraint_scores = {}

    candidates = []
    for row, predicted, uncertainty in zip(
        valid_rows,
        predicted_rows,
        uncertainty_rows,
        strict=True,
    ):
        psmiles, report, descriptors = row
        score = _target_score(generator, predicted, requested_properties)
        score += 0.05 * min(descriptors["configurational_entropy_R"], 2.5)
        score += constraint_scores.get(psmiles, 0.0)
        if uncertainty:
            relevant = set(requested_properties)
            relevant.update(constraint.name for constraint in constraints)
            score -= 0.01 * sum(
                value / max(abs(predicted.get(name, 0.0)), 1.0e-8)
                for name, value in uncertainty.items()
                if name in relevant
            )
        candidates.append(
            GeneratedPolymer(
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
    return PolyGenerationResult(
        candidates=selected,
        requested_properties=dict(requested_properties),
        process_conditions=process_conditions,
        attempted=attempted,
        chemically_valid=len(valid_rows),
        rejection_counts=rejection_counts,
    )


def save_generation(
    result: PolyGenerationResult,
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
                "psmiles": candidate.psmiles,
                "score": candidate.score,
                "predicted_properties": candidate.predicted_properties,
                "property_uncertainty": candidate.property_uncertainty,
                "physics_descriptors": candidate.physics_descriptors,
                "theory_checks": candidate.theory_checks,
                "port_count": candidate.port_count,
            }
            for candidate in result.candidates
        ],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


__all__ = [
    "GeneratedPolymer",
    "LoadedPolyGenerator",
    "PolyGenerationResult",
    "generate_polymers",
    "load_poly_generator",
    "save_generation",
]
