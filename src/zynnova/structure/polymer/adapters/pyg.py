from __future__ import annotations

from typing import Any

import numpy as np

from ..io.json_codec import record_from_dict
from ..views.chemical import ChemicalStructureView
from ..views.generative import GenerativeTensorView
from ..views.multiscale import MultiScaleView
from ..views.single_chain import SingleChainView
from ..views.transformer import TransformerInputView


def _require_torch_pyg() -> tuple[Any, Any, Any]:
    try:
        import torch
        from torch_geometric.data import Data, HeteroData
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "PyTorch and torch-geometric are required; install zynnova[graph]"
        ) from exc
    return torch, Data, HeteroData


def _tensor(torch: Any, array: np.ndarray | None, *, dtype: Any | None = None) -> Any:
    if array is None:
        return None
    result = torch.as_tensor(array)
    return result.to(dtype=dtype) if dtype is not None else result


def chemical_view_to_pyg(
    view: ChemicalStructureView,
    *,
    include_reconstruction: bool = True,
) -> dict[str, Any]:
    torch, Data, _ = _require_torch_pyg()
    result: dict[str, Any] = {}
    for unit_id, graph in view.unit_graphs.items():
        data = Data(
            x=_tensor(torch, graph.node_features, dtype=torch.float32),
            edge_index=_tensor(torch, graph.edge_index, dtype=torch.long),
            edge_attr=_tensor(torch, graph.edge_features, dtype=torch.float32),
            node_type=_tensor(torch, graph.node_type_ids, dtype=torch.long),
            edge_type=_tensor(torch, graph.edge_type_ids, dtype=torch.long),
        )
        if graph.positions is not None:
            data.pos = _tensor(torch, graph.positions, dtype=torch.float32)
        data.unit_id = unit_id
        data.zynnova_view_kind = "chemical_unit"
        data.zynnova_metadata = dict(graph.metadata)
        if include_reconstruction and view.record_payload is not None:
            data.zynnova_record = view.record_payload
        result[unit_id] = data
    return result


def single_chain_view_to_pyg(
    view: SingleChainView,
    *,
    include_reconstruction: bool = True,
) -> Any:
    torch, Data, _ = _require_torch_pyg()
    data = Data(
        x=_tensor(torch, view.node_features, dtype=torch.float32),
        edge_index=_tensor(torch, view.edge_index, dtype=torch.long),
        edge_attr=_tensor(torch, view.edge_features, dtype=torch.float32),
    )
    if view.positions is not None:
        data.pos = _tensor(torch, view.positions, dtype=torch.float32)
    if view.node_type_ids is not None:
        data.node_type = _tensor(torch, view.node_type_ids, dtype=torch.long)
    if view.edge_type_ids is not None:
        data.edge_type = _tensor(torch, view.edge_type_ids, dtype=torch.long)
    if view.graph_features is not None:
        data.graph_features = _tensor(torch, view.graph_features, dtype=torch.float32)
    if view.chain_order is not None:
        data.chain_order = _tensor(torch, view.chain_order, dtype=torch.long)
    if view.backbone_mask is not None:
        data.backbone_mask = _tensor(torch, view.backbone_mask, dtype=torch.bool)
    for name, target in view.targets.items():
        setattr(data, f"y_{name}", _tensor(torch, np.asarray(target)))
    data.zynnova_view_kind = "single_chain"
    data.zynnova_metadata = dict(view.metadata)
    data.zynnova_node_ids = list(view.node_ids)
    data.zynnova_unit_vocabulary = dict(view.unit_order_vocabulary or {})
    if include_reconstruction and view.record_payload is not None:
        data.zynnova_record = view.record_payload
    return data


def multiscale_view_to_pyg(
    view: MultiScaleView,
    *,
    include_reconstruction: bool = True,
) -> Any:
    torch, _, HeteroData = _require_torch_pyg()
    data = HeteroData()
    for node_type, features in view.node_features.items():
        data[node_type].x = _tensor(torch, features, dtype=torch.float32)
        data[node_type].zynnova_ids = list(view.node_ids.get(node_type, []))
    for relation in view.relations:
        key = (relation.source_type, relation.relation, relation.target_type)
        data[key].edge_index = _tensor(torch, relation.edge_index, dtype=torch.long)
        if relation.edge_features is not None:
            data[key].edge_attr = _tensor(
                torch, relation.edge_features, dtype=torch.float32
            )
    data["polymer"].x = _tensor(
        torch, view.graph_features.reshape(1, -1), dtype=torch.float32
    )
    if view.spatial.get("coordinates") is not None:
        coordinates = np.asarray(view.spatial["coordinates"])
        resolution = view.spatial.get("resolution")
        node_type = "unit" if resolution == "repeat_unit" else "spatial_node"
        if node_type not in data.node_types:
            data[node_type].x = torch.zeros((coordinates.shape[0], 1), dtype=torch.float32)
        data[node_type].pos = _tensor(torch, coordinates, dtype=torch.float32)
        if view.spatial.get("spatial_edge_index") is not None:
            key = (node_type, "near", node_type)
            data[key].edge_index = _tensor(
                torch, view.spatial["spatial_edge_index"], dtype=torch.long
            )
    for name, target in view.targets.items():
        setattr(data["polymer"], f"y_{name}", _tensor(torch, np.asarray(target)))
    data.zynnova_view_kind = "multiscale"
    data.zynnova_metadata = dict(view.metadata)
    if include_reconstruction and view.record_payload is not None:
        data.zynnova_record = view.record_payload
    return data


def generative_view_to_pyg(
    view: GenerativeTensorView,
    *,
    include_reconstruction: bool = True,
) -> Any:
    torch, Data, _ = _require_torch_pyg()
    data = Data(
        x=_tensor(torch, view.node_features, dtype=torch.float32),
        node_type=_tensor(torch, view.node_type, dtype=torch.long),
        node_mask=_tensor(torch, view.node_mask, dtype=torch.bool),
        edge_type=_tensor(torch, view.edge_type, dtype=torch.long),
        edge_mask=_tensor(torch, view.edge_mask, dtype=torch.bool),
        composition_logits=_tensor(torch, view.composition_logits, dtype=torch.float32),
        composition_mask=_tensor(torch, view.composition_mask, dtype=torch.bool),
        transition_logits=_tensor(torch, view.transition_logits, dtype=torch.float32),
        transition_mask=_tensor(torch, view.transition_mask, dtype=torch.bool),
        continuous_features=_tensor(torch, view.continuous_features, dtype=torch.float32),
        continuous_feature_mask=_tensor(
            torch, view.continuous_feature_mask, dtype=torch.bool
        ),
    )
    if view.coordinates is not None:
        data.pos = _tensor(torch, view.coordinates, dtype=torch.float32)
        data.coordinate_mask = _tensor(torch, view.coordinate_mask, dtype=torch.bool)
    data.zynnova_view_kind = "generative"
    data.zynnova_level = view.level
    data.zynnova_vocabularies = view.vocabularies
    data.zynnova_feature_names = list(view.feature_names)
    data.zynnova_metadata = dict(view.metadata)
    if include_reconstruction and view.record_payload is not None:
        data.zynnova_record = view.record_payload
    return data


def generative_view_from_pyg(data: Any) -> GenerativeTensorView:
    def array(name: str) -> np.ndarray:
        value = getattr(data, name)
        return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)

    view = GenerativeTensorView(
        level=str(getattr(data, "zynnova_level", "unit")),
        node_type=array("node_type"),
        node_mask=array("node_mask"),
        node_features=array("x"),
        edge_type=array("edge_type"),
        edge_mask=array("edge_mask"),
        composition_logits=array("composition_logits"),
        composition_mask=array("composition_mask"),
        transition_logits=array("transition_logits"),
        transition_mask=array("transition_mask"),
        continuous_features=array("continuous_features"),
        continuous_feature_mask=array("continuous_feature_mask"),
        coordinates=array("pos") if hasattr(data, "pos") else None,
        coordinate_mask=array("coordinate_mask") if hasattr(data, "coordinate_mask") else None,
        vocabularies=dict(getattr(data, "zynnova_vocabularies", {})),
        feature_names=list(getattr(data, "zynnova_feature_names", [])),
        metadata=dict(getattr(data, "zynnova_metadata", {})),
        record_payload=getattr(data, "zynnova_record", None),
    )
    view.validate()
    return view



def single_chain_view_from_pyg(data: Any) -> SingleChainView:
    def array(name: str) -> np.ndarray | None:
        if not hasattr(data, name):
            return None
        value = getattr(data, name)
        return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)

    view = SingleChainView(
        node_features=array("x"),
        edge_index=array("edge_index"),
        edge_features=array("edge_attr"),
        positions=array("pos"),
        node_ids=list(getattr(data, "zynnova_node_ids", [])),
        node_type_ids=array("node_type"),
        edge_type_ids=array("edge_type"),
        graph_features=array("graph_features"),
        metadata=dict(getattr(data, "zynnova_metadata", {})),
        record_payload=getattr(data, "zynnova_record", None),
        chain_order=array("chain_order"),
        backbone_mask=array("backbone_mask"),
        unit_order_vocabulary=dict(getattr(data, "zynnova_unit_vocabulary", {})),
    )
    view.validate()
    return view

def pyg_to_record(data: Any):
    if isinstance(data, dict):
        if not data:
            raise ValueError("empty PyG mapping")
        data = next(iter(data.values()))
    payload = getattr(data, "zynnova_record", None)
    if payload is None:
        raise ValueError(
            "PyG object has no lossless record payload; decode a generated graph "
            "with generative_view_from_pyg() and generative2record()"
        )
    return record_from_dict(payload)


def view_to_pyg(view: Any, *, include_reconstruction: bool = True) -> Any:
    if isinstance(view, ChemicalStructureView):
        return chemical_view_to_pyg(view, include_reconstruction=include_reconstruction)
    if isinstance(view, SingleChainView):
        return single_chain_view_to_pyg(view, include_reconstruction=include_reconstruction)
    if isinstance(view, MultiScaleView):
        return multiscale_view_to_pyg(view, include_reconstruction=include_reconstruction)
    if isinstance(view, GenerativeTensorView):
        return generative_view_to_pyg(view, include_reconstruction=include_reconstruction)
    if isinstance(view, TransformerInputView):
        raise TypeError("TransformerInputView is not a graph and cannot be converted to PyG")
    raise TypeError(f"unsupported view type: {type(view).__name__}")
