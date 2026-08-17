from __future__ import annotations

import math

from ...common import require_torch
from .config import PolyLoomModelConfig

torch = require_torch()
nn = torch.nn


def log_snr_time_embedding(time, hidden_dim: int):
    time = time.clamp(1.0e-5, 1.0 - 1.0e-5)
    log_snr = 2.0 * (torch.cos(time * math.pi / 2).clamp_min(1.0e-5).log()
                     - torch.sin(time * math.pi / 2).clamp_min(1.0e-5).log())
    half = hidden_dim // 2
    frequency = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, device=time.device, dtype=time.dtype)
        / max(half - 1, 1)
    )
    angles = log_snr[:, None] * frequency[None]
    result = torch.cat((angles.sin(), angles.cos()), dim=-1)
    return torch.nn.functional.pad(result, (0, hidden_dim - result.shape[-1]))


class PolyLoomNetwork(nn.Module):
    """Self-conditioned conditional discrete-flow model for polymer repeat units."""

    def __init__(self, config: PolyLoomModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or PolyLoomModelConfig()
        self.config.__post_init__()
        if self.config.vocab_size < 5:
            raise ValueError("vocab_size is unset; prepare data before constructing PolyLoom")
        hidden = self.config.hidden_dim
        self.token_embedding = nn.Embedding(self.config.vocab_size, hidden, padding_idx=0)
        self.position_embedding = nn.Embedding(self.config.max_length, hidden)
        self.time_encoder = nn.Sequential(
            nn.Linear(hidden, hidden * 2), nn.SiLU(), nn.Linear(hidden * 2, hidden)
        )
        width = 2 * (
            len(self.config.property_specs) + len(self.config.process_condition_names)
        )
        self.condition_encoder = nn.Sequential(
            nn.Linear(max(width, 1), hidden * 2),
            nn.SiLU(),
            nn.Linear(hidden * 2, hidden),
            nn.LayerNorm(hidden),
        )
        self.self_condition_projection = nn.Linear(self.config.vocab_size, hidden, bias=False)
        layer = nn.TransformerEncoderLayer(
            hidden,
            self.config.attention_heads,
            hidden * self.config.feedforward_multiplier,
            self.config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.backbone = nn.TransformerEncoder(
            layer, self.config.num_layers, norm=nn.LayerNorm(hidden)
        )
        self.expert_router = nn.Linear(hidden, self.config.num_experts)
        self.experts = nn.ModuleList(
            nn.Sequential(
                nn.Linear(hidden, hidden * 2), nn.SiLU(), nn.Linear(hidden * 2, hidden)
            )
            for _ in range(self.config.num_experts)
        )
        self.token_head = nn.Linear(hidden, self.config.vocab_size, bias=False)
        self.token_head.weight = self.token_embedding.weight
        self.property_head = nn.Linear(hidden, len(self.config.property_specs))
        self.length_head = nn.Linear(hidden, self.config.max_length + 1)
        self.endpoint_head = nn.Linear(hidden, 1)

    def encode_conditions(self, properties, property_mask, process, process_mask):
        parts = []
        for values, mask in ((properties, property_mask), (process, process_mask)):
            if values.shape[1]:
                observed = mask.to(values.dtype)
                parts.extend((values * observed, observed))
        if parts:
            joined = torch.cat(parts, dim=-1)
        else:
            joined = properties.new_zeros((properties.shape[0], 1))
        return self.condition_encoder(joined)

    def forward(
        self,
        token_ids,
        attention_mask,
        time,
        properties,
        property_mask,
        process_conditions,
        process_condition_mask,
        *,
        self_condition=None,
    ):
        position = torch.arange(token_ids.shape[1], device=token_ids.device)
        condition = self.encode_conditions(
            properties, property_mask, process_conditions, process_condition_mask
        )
        hidden = (
            self.token_embedding(token_ids)
            + self.position_embedding(position)[None]
            + self.time_encoder(log_snr_time_embedding(time, self.config.hidden_dim))[:, None]
            + condition[:, None]
        )
        if self.config.self_conditioning and self_condition is not None:
            hidden = hidden + self.self_condition_projection(self_condition)
        hidden = self.backbone(hidden, src_key_padding_mask=~attention_mask.bool())
        routing = self.expert_router(condition).softmax(dim=-1)
        expert_update = sum(
            routing[:, index, None, None] * expert(hidden)
            for index, expert in enumerate(self.experts)
        )
        hidden = hidden + expert_update
        importance = routing.mean(dim=0)
        balance = self.config.num_experts * importance.square().sum()
        return {
            "logits": self.token_head(hidden),
            "endpoint_logits": self.endpoint_head(hidden).squeeze(-1),
            "expert_balance_loss": balance,
            "hidden": hidden,
        }

    def predict_properties(self, token_ids, attention_mask):
        batch = token_ids.shape[0]
        dtype, device = self.token_embedding.weight.dtype, token_ids.device
        properties = torch.zeros(
            (batch, len(self.config.property_specs)), device=device, dtype=dtype
        )
        process = torch.zeros(
            (batch, len(self.config.process_condition_names)), device=device, dtype=dtype
        )
        result = self(
            token_ids, attention_mask, torch.zeros(batch, device=device, dtype=dtype),
            properties, torch.zeros_like(properties, dtype=torch.bool),
            process, torch.zeros_like(process, dtype=torch.bool),
        )
        mask = attention_mask.to(result["hidden"].dtype)[..., None]
        pooled = (result["hidden"] * mask).sum(1) / mask.sum(1).clamp_min(1.0)
        return self.property_head(pooled)

    def predict_length(self, properties, property_mask, process, process_mask):
        return self.length_head(
            self.encode_conditions(properties, property_mask, process, process_mask)
        )


__all__ = ["PolyLoomNetwork", "log_snr_time_embedding"]
