from __future__ import annotations

from ...common import require_torch
from .config import PolyPredictionModelConfig

torch = require_torch()
nn = torch.nn


def scatter_sum(values, index, dim_size: int):
    result = values.new_zeros((dim_size, *values.shape[1:]))
    if values.numel():
        result.index_add_(0, index, values)
    return result


class PolymerGraphLayer(nn.Module):
    def __init__(self, hidden_dim: int, edge_feature_dim: int, dropout: float) -> None:
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_feature_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )
        self.update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, hidden, edge_index, edge_features):
        if edge_index.numel():
            receiver, sender = edge_index
            message = self.message(
                torch.cat((hidden[receiver], hidden[sender], edge_features), dim=-1)
            )
            aggregate = scatter_sum(message, receiver, hidden.shape[0])
        else:
            aggregate = torch.zeros_like(hidden)
        joined = torch.cat((hidden, aggregate), dim=-1)
        gate = self.gate(joined)
        update = self.update(joined)
        return self.norm(hidden + gate * update)


class PolyPredictionNetwork(nn.Module):
    """Multiview heteroscedastic network for condition-dependent polymer properties."""

    def __init__(self, config: PolyPredictionModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or PolyPredictionModelConfig()
        self.config.__post_init__()
        if self.config.vocab_size < 5:
            raise ValueError(
                "vocab_size is unset; prepare data first or restore a checkpoint tokenizer"
            )
        hidden = self.config.hidden_dim
        self.token_embedding = nn.Embedding(self.config.vocab_size, hidden, padding_idx=0)
        self.position_embedding = nn.Embedding(self.config.max_length, hidden)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=self.config.attention_heads,
            dim_feedforward=hidden * self.config.feedforward_multiplier,
            dropout=self.config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.sequence_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.config.sequence_layers,
            norm=nn.LayerNorm(hidden),
        )
        self.node_projection = nn.Sequential(
            nn.Linear(self.config.node_feature_dim, hidden),
            nn.SiLU(),
            nn.LayerNorm(hidden),
        )
        self.graph_layers = nn.ModuleList(
            PolymerGraphLayer(hidden, self.config.edge_feature_dim, self.config.dropout)
            for _ in range(self.config.graph_layers)
        )
        condition_input = max(2 * len(self.config.condition_names), 1)
        self.condition_encoder = nn.Sequential(
            nn.Linear(condition_input, hidden),
            nn.SiLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
        )
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden * 3, hidden * 3),
            nn.Sigmoid(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden * 3, hidden * 2),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(hidden * 2, hidden),
            nn.LayerNorm(hidden),
        )
        self.property_context = nn.Parameter(torch.empty(len(self.config.property_specs), hidden))
        nn.init.normal_(self.property_context, std=0.02)
        self.output = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(hidden, 2),
        )

    @property
    def property_names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.config.property_specs)

    def encode(self, batch):
        token_ids = batch["token_ids"].long()
        attention_mask = batch["attention_mask"].bool()
        positions = torch.arange(token_ids.shape[1], device=token_ids.device)
        sequence = self.token_embedding(token_ids) + self.position_embedding(positions)[None]
        sequence = self.sequence_encoder(
            sequence,
            src_key_padding_mask=~attention_mask,
        )
        mask = attention_mask.to(sequence.dtype)[..., None]
        sequence_pool = (sequence * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

        graph_hidden = self.node_projection(batch["node_features"])
        for layer in self.graph_layers:
            graph_hidden = layer(
                graph_hidden,
                batch["edge_index"].long(),
                batch["edge_features"],
            )
        graph_count = token_ids.shape[0]
        graph_index = batch["graph_index"].long()
        node_weights = batch["node_weights"].to(graph_hidden.dtype)[..., None]
        graph_pool = scatter_sum(
            graph_hidden * node_weights,
            graph_index,
            graph_count,
        )
        graph_weight = scatter_sum(node_weights, graph_index, graph_count).clamp_min(1.0e-8)
        graph_pool = graph_pool / graph_weight

        if self.config.condition_names:
            condition_values = batch["conditions"]
            condition_mask = batch["condition_mask"].to(condition_values.dtype)
            condition_input = torch.cat(
                (condition_values * condition_mask, condition_mask),
                dim=-1,
            )
        else:
            condition_input = sequence_pool.new_zeros((sequence_pool.shape[0], 1))
        condition_pool = self.condition_encoder(condition_input)
        joined = torch.cat((sequence_pool, graph_pool, condition_pool), dim=-1)
        gate = self.fusion_gate(joined)
        fused = self.fusion(joined * gate)
        return fused

    def forward(self, batch):
        embedding = self.encode(batch)
        property_hidden = embedding[:, None, :] + self.property_context[None, :, :]
        output = self.output(property_hidden)
        mean = output[..., 0]
        log_variance = output[..., 1].clamp(
            self.config.min_log_variance,
            self.config.max_log_variance,
        )
        return {
            "mean": mean,
            "log_variance": log_variance,
            "embedding": embedding,
        }


__all__ = ["PolyPredictionNetwork", "PolymerGraphLayer", "scatter_sum"]
