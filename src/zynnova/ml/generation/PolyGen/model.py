from __future__ import annotations

import math

from ...common import require_torch
from .config import PolyGenModelConfig

torch = require_torch()
nn = torch.nn


def sinusoidal_time_embedding(time, hidden_dim: int):
    half = hidden_dim // 2
    frequency = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, device=time.device, dtype=time.dtype)
        / max(half - 1, 1)
    )
    angles = time[:, None] * frequency[None, :] * 1000.0
    embedding = torch.cat((angles.sin(), angles.cos()), dim=-1)
    if embedding.shape[-1] < hidden_dim:
        embedding = torch.nn.functional.pad(embedding, (0, 1))
    return embedding


class PolymerMaskedFlow(nn.Module):
    """Conditional masked discrete-flow Transformer over PSMILES tokens."""

    def __init__(self, config: PolyGenModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or PolyGenModelConfig()
        self.config.__post_init__()
        if self.config.vocab_size < 5:
            raise ValueError(
                "vocab_size is unset; prepare data first or restore a checkpoint tokenizer"
            )
        hidden = self.config.hidden_dim
        self.token_embedding = nn.Embedding(self.config.vocab_size, hidden, padding_idx=0)
        self.position_embedding = nn.Embedding(self.config.max_length, hidden)
        self.time_encoder = nn.Sequential(
            nn.Linear(hidden, hidden * 2),
            nn.SiLU(),
            nn.Linear(hidden * 2, hidden),
        )
        condition_width = 2 * (
            len(self.config.property_specs) + len(self.config.process_condition_names)
        )
        self.condition_encoder = nn.Sequential(
            nn.Linear(max(condition_width, 1), hidden * 2),
            nn.SiLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(hidden * 2, hidden),
            nn.LayerNorm(hidden),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=self.config.attention_heads,
            dim_feedforward=hidden * self.config.feedforward_multiplier,
            dropout=self.config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=self.config.num_layers,
            norm=nn.LayerNorm(hidden),
        )
        self.token_head = nn.Linear(hidden, self.config.vocab_size, bias=False)
        self.token_head.weight = self.token_embedding.weight
        self.property_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(hidden, len(self.config.property_specs)),
        )
        self.length_prior = nn.Parameter(torch.zeros(hidden))
        self.length_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.config.max_length + 1),
        )

    @property
    def property_names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.config.property_specs)

    def encode_conditions(
        self,
        properties,
        property_mask,
        process_conditions,
        process_condition_mask,
    ):
        batch_size = properties.shape[0]
        parts = []
        if properties.shape[1]:
            property_mask_float = property_mask.to(properties.dtype)
            parts.extend((properties * property_mask_float, property_mask_float))
        if process_conditions.shape[1]:
            process_mask_float = process_condition_mask.to(process_conditions.dtype)
            parts.extend(
                (
                    process_conditions * process_mask_float,
                    process_mask_float,
                )
            )
        if not parts:
            condition_input = properties.new_zeros((batch_size, 1))
        else:
            condition_input = torch.cat(parts, dim=-1)
        return self.condition_encoder(condition_input)

    def encode_tokens(
        self,
        token_ids,
        attention_mask,
        time,
        condition_embedding,
    ):
        positions = torch.arange(token_ids.shape[1], device=token_ids.device)
        hidden = self.token_embedding(token_ids)
        hidden = hidden + self.position_embedding(positions)[None]
        hidden = (
            hidden
            + self.time_encoder(sinusoidal_time_embedding(time, self.config.hidden_dim))[:, None, :]
        )
        hidden = hidden + condition_embedding[:, None, :]
        return self.transformer(hidden, src_key_padding_mask=~attention_mask.bool())

    def forward(
        self,
        token_ids,
        attention_mask,
        time,
        properties,
        property_mask,
        process_conditions,
        process_condition_mask,
    ):
        condition = self.encode_conditions(
            properties,
            property_mask,
            process_conditions,
            process_condition_mask,
        )
        hidden = self.encode_tokens(token_ids, attention_mask, time, condition)
        return self.token_head(hidden)

    def predict_properties(self, token_ids, attention_mask):
        batch_size = token_ids.shape[0]
        dtype = self.token_embedding.weight.dtype
        device = token_ids.device
        properties = torch.zeros(
            (batch_size, len(self.config.property_specs)),
            device=device,
            dtype=dtype,
        )
        property_mask = torch.zeros_like(properties, dtype=torch.bool)
        process = torch.zeros(
            (batch_size, len(self.config.process_condition_names)),
            device=device,
            dtype=dtype,
        )
        process_mask = torch.zeros_like(process, dtype=torch.bool)
        condition = self.encode_conditions(
            properties,
            property_mask,
            process,
            process_mask,
        )
        hidden = self.encode_tokens(
            token_ids,
            attention_mask,
            torch.zeros(batch_size, device=device, dtype=dtype),
            condition,
        )
        mask = attention_mask.to(hidden.dtype)[..., None]
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.property_head(pooled)

    def predict_length(
        self,
        properties,
        property_mask,
        process_conditions,
        process_condition_mask,
    ):
        condition = self.encode_conditions(
            properties,
            property_mask,
            process_conditions,
            process_condition_mask,
        )
        return self.length_head(condition + self.length_prior[None])


__all__ = ["PolymerMaskedFlow", "sinusoidal_time_embedding"]
