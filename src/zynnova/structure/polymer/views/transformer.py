from __future__ import annotations

import numpy as np

from ..core.polymer import PolymerRecord
from ..io.json_codec import record_to_dict
from ..schema import RepresentationSchema
from .common import TransformerInputView


TOKEN_TYPES = {
    "special": 0,
    "unit": 1,
    "architecture": 2,
    "condition": 3,
    "property": 4,
}


def to_transformer_view(
    record: PolymerRecord,
    *,
    max_length: int | None = None,
    schema: RepresentationSchema | None = None,
    include_reconstruction: bool = True,
) -> TransformerInputView:
    record.validate()
    if max_length is None and schema is not None:
        max_length = schema.max_tokens
    tokens: list[str] = ["[POLYMER]"]
    token_types: list[int] = [TOKEN_TYPES["special"]]

    for unit_id in sorted(record.units):
        unit = record.units[unit_id]
        tokens.extend(["[UNIT]", unit_id, f"ROLE={unit.role.value}"])
        token_types.extend([TOKEN_TYPES["special"], TOKEN_TYPES["unit"], TOKEN_TYPES["unit"]])
        atom_tokens = [f"Z{atom.atomic_number}" for atom in unit.graph.atoms]
        tokens.extend(atom_tokens)
        token_types.extend([TOKEN_TYPES["unit"]] * len(atom_tokens))
        for port in unit.graph.ports:
            tokens.append(f"PORT={port.port_type}:{port.id}")
            token_types.append(TOKEN_TYPES["unit"])

    tokens.append("[ARCH]")
    token_types.append(TOKEN_TYPES["special"])
    tokens.append(f"TYPE={record.architecture.architecture_type.value}")
    token_types.append(TOKEN_TYPES["architecture"])
    if record.architecture.sequence:
        for unit_id in record.architecture.sequence:
            tokens.append(f"SEQ={unit_id}")
            token_types.append(TOKEN_TYPES["architecture"])
    else:
        for edge in record.architecture.edges:
            tokens.append(
                f"EDGE={edge.source}:{edge.source_port or '*'}>{edge.target}:{edge.target_port or '*'}"
            )
            token_types.append(TOKEN_TYPES["architecture"])

    if schema is not None:
        composition_order = [
            unit_id
            for unit_id, _ in sorted(schema.unit_vocabulary.items(), key=lambda item: item[1])
        ]
    else:
        composition_order = sorted(record.units)
    continuous_values: list[float] = []
    continuous_mask: list[bool] = []
    for unit_id in composition_order:
        if unit_id in {"[PAD]", "[UNK]"}:
            continuous_values.append(0.0)
            continuous_mask.append(False)
        else:
            continuous_values.append(record.ensemble.composition.get(unit_id, 0.0))
            continuous_mask.append(unit_id in record.ensemble.composition)
    for distribution in (
        record.ensemble.degree_of_polymerization,
        record.ensemble.molecular_weight,
    ):
        value = distribution.representative_value() if distribution is not None else None
        continuous_values.append(float(value or 0.0))
        continuous_mask.append(value is not None)
    continuous_values.append(float(record.ensemble.crosslink_density or 0.0))
    continuous_mask.append(record.ensemble.crosslink_density is not None)

    if max_length is not None:
        if max_length < 2:
            raise ValueError("max_length must be at least 2")
        tokens = tokens[: max_length - 1]
        token_types = token_types[: max_length - 1]
    tokens.append("[END]")
    token_types.append(TOKEN_TYPES["special"])

    targets = {
        name: np.asarray(value.value)
        for name, value in record.properties.items()
        if isinstance(value.value, (int, float, list))
    }
    view = TransformerInputView(
        tokens=tokens,
        token_type_ids=np.asarray(token_types, dtype=np.int64),
        attention_mask=np.ones(len(tokens), dtype=bool),
        continuous_features=np.asarray(continuous_values, dtype=np.float32),
        continuous_feature_mask=np.asarray(continuous_mask, dtype=bool),
        targets=targets,
        metadata={
            "record_id": record.id,
            "schema_id": schema.schema_id if schema is not None else None,
            "composition_order": composition_order,
        },
        record_payload=record_to_dict(record) if include_reconstruction else None,
    )
    view.validate()
    return view
