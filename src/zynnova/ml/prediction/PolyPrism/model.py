from __future__ import annotations

from ...common import require_torch
from ..PolyPrediction.model import PolymerGraphLayer, scatter_sum
from .config import PolyPrismModelConfig

torch = require_torch()
nn = torch.nn


class SparseExpertBlock(nn.Module):
    """Token-wise top-k mixture of experts with an exposed balancing loss."""

    def __init__(self, hidden: int, experts: int, top_k: int, dropout: float) -> None:
        super().__init__()
        self.top_k = top_k
        self.router = nn.Linear(hidden, experts, bias=False)
        self.experts = nn.ModuleList(
            nn.Sequential(
                nn.Linear(hidden, hidden * 4),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden * 4, hidden),
            )
            for _ in range(experts)
        )
        self.norm = nn.LayerNorm(hidden)

    def forward(self, hidden):
        probabilities = self.router(hidden).softmax(dim=-1)
        weights, indices = probabilities.topk(self.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
        mixed = torch.zeros_like(hidden)
        for expert_index, expert in enumerate(self.experts):
            selection = indices.eq(expert_index)
            coefficient = (weights * selection).sum(dim=-1, keepdim=True)
            mixed = mixed + coefficient * expert(hidden)
        importance = probabilities.mean(dim=tuple(range(probabilities.ndim - 1)))
        load = torch.stack(
            [indices.eq(index).float().mean() for index in range(len(self.experts))]
        )
        balance = len(self.experts) * (importance * load).sum()
        return self.norm(hidden + mixed), balance


class PolyPrismNetwork(nn.Module):
    """Multiview, multi-fidelity, property-query polymer predictor.

    The property tokens query sequence, periodic-graph and process-condition
    tokens. Sparse experts then specialize by property family while sharing the
    common polymer representation.
    """

    def __init__(self, config: PolyPrismModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or PolyPrismModelConfig()
        self.config.__post_init__()
        if self.config.vocab_size < 5:
            raise ValueError("vocab_size is unset; prepare data before constructing PolyPrism")
        hidden = self.config.hidden_dim
        self.token_embedding = nn.Embedding(self.config.vocab_size, hidden, padding_idx=0)
        self.position_embedding = nn.Embedding(self.config.max_length, hidden)
        layer = nn.TransformerEncoderLayer(
            hidden,
            self.config.attention_heads,
            hidden * self.config.feedforward_multiplier,
            self.config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.sequence_encoder = nn.TransformerEncoder(
            layer, self.config.sequence_layers, norm=nn.LayerNorm(hidden)
        )
        self.node_projection = nn.Linear(self.config.node_feature_dim, hidden)
        self.graph_layers = nn.ModuleList(
            PolymerGraphLayer(hidden, self.config.edge_feature_dim, self.config.dropout)
            for _ in range(self.config.graph_layers)
        )
        condition_width = max(2 * len(self.config.condition_names), 1)
        self.condition_encoder = nn.Sequential(
            nn.Linear(condition_width, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
        )
        self.fidelity_embedding = nn.Embedding(len(self.config.fidelity_names), hidden)
        self.property_queries = nn.Parameter(
            torch.empty(len(self.config.property_specs), hidden)
        )
        nn.init.normal_(self.property_queries, std=0.02)
        self.cross_attention = nn.ModuleList(
            nn.MultiheadAttention(
                hidden,
                self.config.attention_heads,
                dropout=self.config.dropout,
                batch_first=True,
            )
            for _ in range(self.config.fusion_layers)
        )
        self.cross_norms = nn.ModuleList(
            nn.LayerNorm(hidden) for _ in range(self.config.fusion_layers)
        )
        self.expert_blocks = nn.ModuleList(
            SparseExpertBlock(
                hidden,
                self.config.num_experts,
                self.config.top_k_experts,
                self.config.dropout,
            )
            for _ in range(self.config.fusion_layers)
        )
        width = 4 if self.config.uncertainty == "evidential" else 2
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(hidden, width),
        )

    @property
    def property_names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.config.property_specs)

    def _views(self, batch):
        token_ids = batch["token_ids"].long()
        mask = batch["attention_mask"].bool()
        positions = torch.arange(token_ids.shape[1], device=token_ids.device)
        sequence = self.token_embedding(token_ids) + self.position_embedding(positions)[None]
        sequence = self.sequence_encoder(sequence, src_key_padding_mask=~mask)
        sequence_mask = mask

        graph = self.node_projection(batch["node_features"])
        for layer in self.graph_layers:
            graph = layer(graph, batch["edge_index"].long(), batch["edge_features"])
        graph_count = token_ids.shape[0]
        graph_index = batch["graph_index"].long()
        weights = batch["node_weights"].to(graph.dtype)[:, None]
        graph_pool = scatter_sum(graph * weights, graph_index, graph_count)
        graph_pool /= scatter_sum(weights, graph_index, graph_count).clamp_min(1.0e-8)

        if self.config.condition_names:
            values = batch["conditions"]
            observed = batch["condition_mask"].to(values.dtype)
            condition_input = torch.cat((values * observed, observed), dim=-1)
        else:
            condition_input = graph_pool.new_zeros((graph_count, 1))
        condition = self.condition_encoder(condition_input)
        fidelity = batch.get("fidelity_index")
        if fidelity is None:
            fidelity = torch.zeros(graph_count, dtype=torch.long, device=token_ids.device)
        fidelity = self.fidelity_embedding(fidelity.long())
        summary = torch.stack((graph_pool, condition, fidelity), dim=1)
        views = torch.cat((sequence, summary), dim=1)
        summary_mask = torch.ones((graph_count, 3), dtype=torch.bool, device=mask.device)
        return views, torch.cat((sequence_mask, summary_mask), dim=1)

    def forward(self, batch):
        views, view_mask = self._views(batch)
        queries = self.property_queries[None].expand(views.shape[0], -1, -1)
        balances = []
        for attention, norm, experts in zip(
            self.cross_attention, self.cross_norms, self.expert_blocks, strict=True
        ):
            update, _ = attention(
                queries, views, views, key_padding_mask=~view_mask, need_weights=False
            )
            queries = norm(queries + update)
            queries, balance = experts(queries)
            balances.append(balance)
        raw = self.head(queries)
        mean = raw[..., 0]
        if self.config.uncertainty == "evidential":
            evidence = torch.nn.functional.softplus(raw[..., 1:]) + 1.0e-6
            nu = evidence[..., 0]
            alpha = evidence[..., 1] + 1.0
            beta = evidence[..., 2]
            aleatoric = beta / (alpha - 1.0).clamp_min(1.0e-6)
            epistemic = aleatoric / nu.clamp_min(1.0e-6)
            output = {
                "mean": mean,
                "nu": nu,
                "alpha": alpha,
                "beta": beta,
                "aleatoric_variance": aleatoric,
                "epistemic_variance": epistemic,
                "log_variance": (aleatoric + epistemic).clamp_min(1.0e-8).log(),
            }
        else:
            log_variance = raw[..., 1].clamp(
                self.config.min_log_variance, self.config.max_log_variance
            )
            output = {
                "mean": mean,
                "log_variance": log_variance,
                "aleatoric_variance": log_variance.exp(),
                "epistemic_variance": torch.zeros_like(log_variance),
            }
        output["embedding"] = queries.mean(dim=1)
        output["ood_score"] = output["embedding"].square().mean(dim=-1).sqrt()
        output["expert_balance_loss"] = torch.stack(balances).mean()
        return output


__all__ = ["PolyPrismNetwork", "SparseExpertBlock"]
