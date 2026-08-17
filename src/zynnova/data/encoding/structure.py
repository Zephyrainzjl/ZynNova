from __future__ import annotations

from typing import Any

import numpy as np

from ..exceptions import SchemaError
from ..record import MaterialSample, MaterialType
from ..schema import StructureEncodingSpec


def encode_structure(
    sample: MaterialSample,
    spec: StructureEncodingSpec,
    *,
    representation_schema: Any | None = None,
) -> Any:
    if sample.structure is None:
        return None
    representation = spec.representation.lower()
    if representation in {"none", "raw"}:
        return sample.structure
    if representation in {"potential", "atomistic"}:
        return _potential_structure(sample)
    if sample.material_type is MaterialType.POLYMER:
        return _encode_polymer(sample, spec, representation_schema=representation_schema)
    if representation in {"graph", "pyg"}:
        graph = _atom_graph(sample, as_pyg=representation == "pyg", options=spec.options)
        return graph
    if representation in {"dense", "dense_graph", "generative"}:
        return _dense_atom_graph(sample, options=spec.options)
    if representation in {"transformer", "tokens", "sequence"}:
        return _atom_tokens(sample, options=spec.options)
    raise SchemaError(f"unsupported structure representation: {spec.representation!r}")


def _coerce_structure(sample: MaterialSample):
    if sample.material_type is MaterialType.POLYMER:
        from ...structure.polymer import record2stru

        try:
            return record2stru(sample.structure)
        except (TypeError, ValueError, AttributeError):
            pass
    from ...structure.common.io import load_structure

    kind = "crystal" if sample.material_type is MaterialType.CRYSTAL else "molecular"
    return load_structure(sample.structure, kind=kind)


def _atom_graph(
    sample: MaterialSample,
    *,
    as_pyg: bool,
    options: dict[str, Any],
) -> Any:
    kwargs = dict(options)
    kwargs.pop("max_nodes", None)
    kwargs.pop("pad_value", None)
    kwargs["as_pyg"] = as_pyg
    if sample.material_type is MaterialType.CRYSTAL:
        from ...structure.crystal import stru2graph
    else:
        from ...structure.molecular import stru2graph
    return stru2graph(sample.structure, **kwargs)


def _potential_structure(sample: MaterialSample) -> dict[str, np.ndarray]:
    structure = _coerce_structure(sample)
    payload = {
        "z": np.asarray(structure.atomic_numbers, dtype=np.int64),
        "pos": np.asarray(structure.positions, dtype=np.float64),
        "cell": np.asarray(structure.cell, dtype=np.float64),
        "pbc": np.asarray(structure.pbc, dtype=bool),
        "natoms": np.asarray([structure.num_atoms], dtype=np.int64),
    }
    if structure.charges is not None:
        payload["charge"] = np.asarray(structure.charges, dtype=np.float64)
    if structure.masses is not None:
        payload["mass"] = np.asarray(structure.masses, dtype=np.float64)
    return payload


def _dense_atom_graph(sample: MaterialSample, *, options: dict[str, Any]) -> dict[str, Any]:
    graph = _atom_graph(sample, as_pyg=False, options=options)
    max_nodes = int(options.get("max_nodes") or graph.num_nodes)
    if graph.num_nodes > max_nodes:
        policy = options.get("overflow", "error")
        if policy == "error":
            raise SchemaError(
                f"sample {sample.id!r} has {graph.num_nodes} nodes, max_nodes={max_nodes}"
            )
        if policy != "truncate":
            raise ValueError(f"unknown overflow policy: {policy!r}")
    count = min(graph.num_nodes, max_nodes)
    node_type = np.zeros(max_nodes, dtype=np.int64)
    node_type[:count] = graph.atomic_numbers[:count]
    node_mask = np.zeros(max_nodes, dtype=bool)
    node_mask[:count] = True
    positions = np.zeros((max_nodes, 3), dtype=np.float64)
    positions[:count] = graph.positions[:count]
    edge_type = np.zeros((max_nodes, max_nodes), dtype=np.int64)
    edge_mask = node_mask[:, None] & node_mask[None, :]
    source, target = graph.edge_index
    keep = (source < count) & (target < count)
    edge_type[source[keep], target[keep]] = 1
    return {
        "node_type": node_type,
        "node_mask": node_mask,
        "node_features": _pad_rows(graph.node_features, max_nodes, count),
        "edge_type": edge_type,
        "edge_mask": edge_mask,
        "positions": positions,
        "coordinate_mask": node_mask.copy(),
        "cell": graph.cell.copy(),
        "pbc": graph.pbc.copy(),
        "level": "atom",
    }


def _pad_rows(values: np.ndarray, max_rows: int, count: int) -> np.ndarray:
    shape = (max_rows, *values.shape[1:])
    output = np.zeros(shape, dtype=values.dtype)
    output[:count] = values[:count]
    return output


def _atom_tokens(sample: MaterialSample, *, options: dict[str, Any]) -> dict[str, Any]:
    structure = _coerce_structure(sample)
    max_length = int(options.get("max_length") or (structure.num_atoms + 2))
    tokens = ["[BOS]", *(f"Z={int(z)}" for z in structure.atomic_numbers), "[EOS]"]
    tokens = tokens[:max_length]
    attention_mask = np.zeros(max_length, dtype=bool)
    attention_mask[: len(tokens)] = True
    tokens.extend(["[PAD]"] * (max_length - len(tokens)))
    return {"tokens": tokens, "attention_mask": attention_mask}


def _encode_polymer(
    sample: MaterialSample,
    spec: StructureEncodingSpec,
    *,
    representation_schema: Any | None,
) -> Any:
    from ...structure.polymer import ViewKind, make_view

    view_name = spec.view.lower()
    mapping = {
        "chemical": ViewKind.CHEMICAL,
        "single_chain": ViewKind.SINGLE_CHAIN,
        "multiscale": ViewKind.MULTISCALE,
        "transformer": ViewKind.TRANSFORMER,
        "generative": ViewKind.GENERATIVE,
        "atom_generative": ViewKind.ATOM_GENERATIVE,
        "generative_atom": ViewKind.ATOM_GENERATIVE,
    }
    if view_name == "graph" and spec.representation in {"pyg", "graph"}:
        view_name = "single_chain"
    try:
        kind = mapping[view_name]
    except KeyError as exc:
        raise SchemaError(f"unsupported polymer view: {spec.view!r}") from exc
    options = dict(spec.options)
    if kind in {ViewKind.TRANSFORMER, ViewKind.GENERATIVE, ViewKind.ATOM_GENERATIVE}:
        if representation_schema is None:
            raise SchemaError(
                f"polymer view {kind.value!r} requires a RepresentationSchema"
            )
        options.setdefault("schema", representation_schema)
    options.setdefault("include_reconstruction", spec.include_reconstruction)
    view = make_view(sample.structure, kind, **options)
    if spec.representation == "pyg":
        from ...structure.polymer import view_to_pyg

        return view_to_pyg(view, include_reconstruction=spec.include_reconstruction)
    return view
