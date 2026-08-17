"""Unified, reversible polymer representations.

Public conversion pairs mirror the existing crystal and molecular namespaces::

    stru2simple <-> simple2stru
    stru2graph  <-> graph2stru
    stru2record <-> record2stru

The same :class:`PolymerRecord` can be projected to chemistry-only, single-chain,
multiscale, Transformer, or graph-generative training views.
"""

from .adapters import (
    DEFAULT_ADAPTER_REGISTRY,
    AdapterRegistry,
    DatasetAdapter,
    FunctionalDatasetAdapter,
    architecture_to_networkx,
    chemical_view_to_pyg,
    collate_generative_views,
    collate_transformer_views,
    generative_view_from_pyg,
    generative_view_to_pyg,
    molecular_graph_from_smiles,
    multiscale_view_to_pyg,
    pyg_to_record,
    single_chain_view_from_pyg,
    single_chain_view_to_pyg,
    view_to_pyg,
)
from .codec import PolymerCodec
from .core import *
from .core import __all__ as _core_all
from .decoder import (
    chemical2record,
    generative2record,
    multiscale2record,
    single_chain2record,
    transformer2record,
    view2record,
    view2stru,
)
from .factory import ViewKind, make_view, record2view
from .graph2stru import graph2stru
from .io import load_json, load_zpoly, record_from_dict, record_to_dict, save_json, save_zpoly
from .record_conversion import record2stru, simple2stru, stru2record, stru2simple
from .schema import RepresentationSchema
from .stru2graph import stru2graph, stru2pyg, to_graph
from .views import (
    ChemicalStructureView,
    GenerativeTensorView,
    generative_view_from_logits,
    GraphTensorView,
    MultiScaleView,
    RelationTable,
    SingleChainView,
    TransformerInputView,
    to_chemical_structure_view,
    to_generative_view,
    to_multiscale_view,
    to_single_chain_view,
    to_transformer_view,
)

__all__ = [
    *_core_all,
    "PolymerCodec",
    "RepresentationSchema",
    "ViewKind",
    "make_view",
    "record2view",
    "stru2record",
    "record2stru",
    "stru2simple",
    "simple2stru",
    "stru2graph",
    "stru2pyg",
    "graph2stru",
    "to_graph",
    "view2record",
    "view2stru",
    "chemical2record",
    "single_chain2record",
    "multiscale2record",
    "transformer2record",
    "generative2record",
    "ChemicalStructureView",
    "SingleChainView",
    "MultiScaleView",
    "TransformerInputView",
    "GenerativeTensorView",
    "generative_view_from_logits",
    "GraphTensorView",
    "RelationTable",
    "to_chemical_structure_view",
    "to_single_chain_view",
    "to_multiscale_view",
    "to_transformer_view",
    "to_generative_view",
    "chemical_view_to_pyg",
    "single_chain_view_to_pyg",
    "single_chain_view_from_pyg",
    "multiscale_view_to_pyg",
    "generative_view_to_pyg",
    "generative_view_from_pyg",
    "view_to_pyg",
    "pyg_to_record",
    "collate_generative_views",
    "collate_transformer_views",
    "molecular_graph_from_smiles",
    "architecture_to_networkx",
    "DatasetAdapter",
    "FunctionalDatasetAdapter",
    "AdapterRegistry",
    "DEFAULT_ADAPTER_REGISTRY",
    "record_to_dict",
    "record_from_dict",
    "save_json",
    "load_json",
    "save_zpoly",
    "load_zpoly",
]
