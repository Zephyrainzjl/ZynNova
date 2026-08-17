from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class GraphTensorView:
    """Backend-neutral sparse graph training view.

    ``record_payload`` is optional. Keeping it makes the transformation exactly
    reversible without reparsing. It can be disabled for compact large-scale
    training datasets; generated graphs are then decoded from node/edge types.
    """

    node_features: np.ndarray
    edge_index: np.ndarray
    edge_features: np.ndarray | None = None
    positions: np.ndarray | None = None
    node_ids: list[str] = field(default_factory=list)
    node_type_ids: np.ndarray | None = None
    edge_type_ids: np.ndarray | None = None
    graph_features: np.ndarray | None = None
    targets: dict[str, np.ndarray] = field(default_factory=dict)
    masks: dict[str, np.ndarray] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    record_payload: dict[str, Any] | None = None

    def validate(self) -> None:
        self.node_features = np.asarray(self.node_features, dtype=np.float32)
        self.edge_index = np.asarray(self.edge_index, dtype=np.int64)
        if self.node_features.ndim != 2:
            raise ValueError("node_features must have shape [N, F]")
        if self.edge_index.ndim != 2 or self.edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, E]")
        n = self.node_features.shape[0]
        if self.edge_index.size:
            if self.edge_index.min() < 0 or self.edge_index.max() >= n:
                raise ValueError("edge_index contains an invalid node index")
        if self.edge_features is not None:
            self.edge_features = np.asarray(self.edge_features, dtype=np.float32)
            if self.edge_features.shape[0] != self.edge_index.shape[1]:
                raise ValueError("edge_features rows must match edge count")
        if self.positions is not None:
            self.positions = np.asarray(self.positions, dtype=np.float64)
            if self.positions.shape != (n, 3):
                raise ValueError("positions must have shape [N, 3]")
        if self.node_type_ids is not None:
            self.node_type_ids = np.asarray(self.node_type_ids, dtype=np.int64)
            if self.node_type_ids.shape != (n,):
                raise ValueError("node_type_ids must have shape [N]")
        if self.edge_type_ids is not None:
            self.edge_type_ids = np.asarray(self.edge_type_ids, dtype=np.int64)
            if self.edge_type_ids.shape != (self.edge_index.shape[1],):
                raise ValueError("edge_type_ids must have shape [E]")


@dataclass
class TransformerInputView:
    tokens: list[str]
    token_type_ids: np.ndarray
    attention_mask: np.ndarray
    continuous_features: np.ndarray
    continuous_feature_mask: np.ndarray
    targets: dict[str, np.ndarray] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    record_payload: dict[str, Any] | None = None

    def validate(self) -> None:
        n = len(self.tokens)
        self.token_type_ids = np.asarray(self.token_type_ids, dtype=np.int64)
        self.attention_mask = np.asarray(self.attention_mask, dtype=bool)
        if self.token_type_ids.shape != (n,) or self.attention_mask.shape != (n,):
            raise ValueError("token arrays must match token count")
        self.continuous_features = np.asarray(self.continuous_features, dtype=np.float32)
        self.continuous_feature_mask = np.asarray(
            self.continuous_feature_mask, dtype=bool
        )
        if self.continuous_features.shape != self.continuous_feature_mask.shape:
            raise ValueError("continuous feature mask shape mismatch")
