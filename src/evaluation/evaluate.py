"""
Evaluación del Decision Transformer en usuarios de test (Parte 3.4).
"""
from __future__ import annotations

import numpy as np
import torch

from src.evaluation.metrics import hit_rate_at_k, ndcg_at_k, mrr


@torch.no_grad()
def evaluate_model(model, test_data, device, target_return=None, k_list=(5, 10, 20)):
    """
    Evalúa el modelo en cold-start users siguiendo la guía.
    """
    model.eval()

    metrics = {f"HR@{k}": [] for k in k_list}
    metrics.update({f"NDCG@{k}": [] for k in k_list})
    metrics["MRR"] = []

    context_len = getattr(model, "context_length", 20)

    for user in test_data:
        group = user["group"]
        items = user["items"]
        ratings = user["ratings"]

        for t in range(context_len, len(items)):
            history_items = items[t - context_len : t]
            history_ratings = ratings[t - context_len : t]

            rtg_val = sum(history_ratings) if target_return is None else target_return

            states = torch.tensor(history_items, dtype=torch.long).unsqueeze(0).to(device)
            actions = torch.tensor(history_items, dtype=torch.long).unsqueeze(0).to(device)
            rtg_input = torch.full(
                (1, context_len, 1), rtg_val, dtype=torch.float32, device=device
            )
            timesteps = torch.arange(context_len, dtype=torch.long, device=device).unsqueeze(0)
            groups = torch.tensor([group], dtype=torch.long, device=device)

            logits = model(states, actions, rtg_input, timesteps, groups)
            predictions = logits[0, -1, :]
            target_item = items[t]
            target_tensor = torch.tensor([target_item], device=device)

            for k in k_list:
                metrics[f"HR@{k}"].append(
                    hit_rate_at_k(predictions.unsqueeze(0), target_tensor, k)
                )
                metrics[f"NDCG@{k}"].append(
                    ndcg_at_k(predictions.unsqueeze(0), target_tensor, k)
                )
            metrics["MRR"].append(mrr(predictions.unsqueeze(0), target_tensor))

    return {key: float(np.mean(values)) for key, values in metrics.items()}


__all__ = ["evaluate_model"]
