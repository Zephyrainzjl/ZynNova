from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..record import MaterialSample


def json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def structure_to_payload(structure: Any | None) -> dict[str, Any] | None:
    if structure is None:
        return None
    try:
        from ...structure import StructureData

        if isinstance(structure, StructureData):
            return {
                "kind": "structure",
                "atomic_numbers": structure.atomic_numbers.tolist(),
                "positions": structure.positions.tolist(),
                "cell": structure.cell.tolist(),
                "pbc": structure.pbc.tolist(),
                "charges": None if structure.charges is None else structure.charges.tolist(),
                "masses": None if structure.masses is None else structure.masses.tolist(),
                "tags": None if structure.tags is None else structure.tags.tolist(),
                "bonds": None if structure.bonds is None else structure.bonds.tolist(),
                "bond_orders": (
                    None if structure.bond_orders is None else structure.bond_orders.tolist()
                ),
                "arrays": json_value(structure.arrays),
                "info": json_value(structure.info),
                "source": structure.source,
            }
    except ImportError:
        pass
    try:
        from ...structure.polymer import PolymerRecord
        from ...structure.polymer.io.json_codec import record_to_dict

        if isinstance(structure, PolymerRecord):
            return {"kind": "polymer", "record": record_to_dict(structure)}
    except ImportError:
        pass
    if hasattr(structure, "get_atomic_numbers") and hasattr(structure, "get_positions"):
        from ...structure import StructureData

        return structure_to_payload(StructureData.from_ase(structure))
    if isinstance(structure, dict):
        return {"kind": "mapping", "value": json_value(structure)}
    raise TypeError(f"unsupported structure type for serialization: {type(structure).__name__}")


def structure_from_payload(payload: dict[str, Any] | None) -> Any | None:
    if payload is None:
        return None
    kind = payload.get("kind")
    if kind == "structure":
        from ...structure import StructureData

        return StructureData(
            atomic_numbers=payload["atomic_numbers"],
            positions=payload["positions"],
            cell=payload.get("cell", np.zeros((3, 3))),
            pbc=payload.get("pbc", [False, False, False]),
            charges=payload.get("charges"),
            masses=payload.get("masses"),
            tags=payload.get("tags"),
            bonds=payload.get("bonds"),
            bond_orders=payload.get("bond_orders"),
            arrays={key: np.asarray(value) for key, value in payload.get("arrays", {}).items()},
            info=payload.get("info", {}),
            source=payload.get("source"),
        )
    if kind == "polymer":
        from ...structure.polymer.io.json_codec import record_from_dict

        return record_from_dict(payload["record"])
    if kind == "mapping":
        return payload["value"]
    raise ValueError(f"unknown serialized structure kind: {kind!r}")


def sample_to_payload(sample: MaterialSample, *, include_structure: bool = True) -> dict[str, Any]:
    return {
        "id": sample.id,
        "material_type": sample.material_type.value,
        "structure": structure_to_payload(sample.structure) if include_structure else None,
        "features": json_value(sample.features),
        "labels": json_value(sample.labels),
        "conditions": json_value(sample.conditions),
        "metadata": json_value(sample.metadata),
        "provenance": json_value(sample.provenance),
        "split": sample.split,
    }


def sample_from_payload(payload: dict[str, Any]) -> MaterialSample:
    return MaterialSample(
        id=payload["id"],
        material_type=payload["material_type"],
        structure=structure_from_payload(payload.get("structure")),
        features=payload.get("features", {}),
        labels=payload.get("labels", {}),
        conditions=payload.get("conditions", {}),
        metadata=payload.get("metadata", {}),
        provenance=payload.get("provenance", {}),
        split=payload.get("split"),
    )


def dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=json_value)


def loads(text: str) -> Any:
    return json.loads(text)


def safe_sample_name(sample_id: str) -> str:
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."
    cleaned = "".join(character if character in allowed else "_" for character in sample_id)
    return cleaned[:180] or "sample"
