import torch
import torch.nn as nn


class DecisionTransformer(nn.Module):
    def __init__(
        self,
        num_items=752,
        num_groups=8,
        hidden_dim=128,
        n_layers=3,
        n_heads=4,
        context_length=20,
        max_timestep=200,
        dropout=0.1,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.context_length = context_length
        self.num_items = num_items

        # === EMBEDDINGS ===
        self.item_embedding = nn.Embedding(num_items, hidden_dim)
        self.group_embedding = nn.Embedding(num_groups, hidden_dim)
        self.rtg_embedding = nn.Linear(1, hidden_dim)
        self.timestep_embedding = nn.Embedding(max_timestep, hidden_dim)

        # === TRANSFORMER ===
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # === PREDICTION HEAD ===
        self.predict_item = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_items),
        )

        self.ln = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        states,  # (batch, seq_len)
        actions,
        returns_to_go,
        timesteps,
        user_groups,
        attention_mask=None,
    ):
        batch_size, seq_len = states.shape

        state_emb = self.item_embedding(states)
        action_emb = self.item_embedding(actions)
        rtg_emb = self.rtg_embedding(returns_to_go)
        time_emb = self.timestep_embedding(timesteps)

        group_emb = self.group_embedding(user_groups).unsqueeze(1)
        group_emb = group_emb.expand(-1, seq_len, -1)

        h = state_emb + action_emb + rtg_emb + time_emb + group_emb
        h = self.ln(h)

        if attention_mask is None:
            attention_mask = self._generate_causal_mask(seq_len).to(h.device)

        h = self.transformer(h, mask=attention_mask)
        item_logits = self.predict_item(h)
        return item_logits

    def _generate_causal_mask(self, seq_len):
        mask = torch.triu(torch.ones(seq_len, seq_len) * float("-inf"), diagonal=1)
        return mask


__all__ = ["DecisionTransformer"]
