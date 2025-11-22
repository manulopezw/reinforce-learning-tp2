"""
Implementación de referencia del Decision Transformer (Opción 1 de la guía).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class DecisionTransformer(nn.Module):
    def __init__(
        self,
        num_items: int = 752,
        num_groups: int = 8,
        hidden_dim: int = 128,
        n_layers: int = 3,
        n_heads: int = 4,
        context_length: int = 20,
        max_timestep: int = 200,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_items = num_items
        self.hidden_dim = hidden_dim
        self.context_length = context_length

        # Embeddings
        self.item_embedding = nn.Embedding(num_items, hidden_dim)
        self.group_embedding = nn.Embedding(num_groups, hidden_dim)
        self.rtg_embedding = nn.Linear(1, hidden_dim)
        self.timestep_embedding = nn.Embedding(max_timestep, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.predict_item = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_items),
        )

        self.ln = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        timesteps: torch.Tensor,
        user_groups: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            states: (batch, seq_len) item IDs vistos
            actions: (batch, seq_len) item IDs (para autoregresión)
            returns_to_go: (batch, seq_len, 1)
            timesteps: (batch, seq_len)
            user_groups: (batch,)
            attention_mask: máscara causal opcional
        Returns:
            item_logits: (batch, seq_len, num_items)
        """
        batch_size, seq_len = states.shape

        state_emb = self.item_embedding(states)
        action_emb = self.item_embedding(actions)
        rtg_emb = self.rtg_embedding(returns_to_go)
        time_emb = self.timestep_embedding(timesteps)

        group_emb = self.group_embedding(user_groups).unsqueeze(1)
        group_emb = group_emb.expand(-1, seq_len, -1)

        h = state_emb + action_emb + rtg_emb + time_emb + group_emb
        h = self.ln(h)

        causal_mask = (
            attention_mask.to(h.device)
            if attention_mask is not None
            else self._generate_causal_mask(seq_len).to(h.device)
        )

        # padding_mask: True donde hay padding para enmascarar en el encoder
        key_padding = None
        if padding_mask is not None:
            key_padding = padding_mask == 0

        h = self.transformer(h, mask=causal_mask, src_key_padding_mask=key_padding)
        item_logits = self.predict_item(h)
        return item_logits

    def _generate_causal_mask(self, seq_len: int) -> torch.Tensor:
        # Máscara superior triangular para evitar mirar al futuro
        mask = torch.triu(torch.ones(seq_len, seq_len) * float("-inf"), diagonal=1)
        return mask


__all__ = ["DecisionTransformer"]
