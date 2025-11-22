"""
Métricas de ranking (Parte 3.3): HR@K, NDCG@K, MRR.
"""
from __future__ import annotations

import numpy as np
import torch


def hit_rate_at_k(predictions: torch.Tensor, targets: torch.Tensor, k: int = 10) -> float:
    top_k = torch.topk(predictions, k, dim=1).indices
    hits = (top_k == targets.unsqueeze(1)).any(dim=1).float()
    return hits.mean().item()


def ndcg_at_k(predictions: torch.Tensor, targets: torch.Tensor, k: int = 10) -> float:
    top_k_indices = torch.topk(predictions, k, dim=1).indices
    relevance = (top_k_indices == targets.unsqueeze(1)).float()
    ranks = torch.arange(1, k + 1, device=predictions.device).float()
    dcg = (relevance / torch.log2(ranks + 1)).sum(dim=1)
    idcg = 1.0 / np.log2(2)
    ndcg = dcg / idcg
    return ndcg.mean().item()


def mrr(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    sorted_indices = torch.argsort(predictions, dim=1, descending=True)
    ranks = (sorted_indices == targets.unsqueeze(1)).nonzero()[:, 1] + 1
    rr = 1.0 / ranks.float()
    return rr.mean().item()


__all__ = ["hit_rate_at_k", "ndcg_at_k", "mrr"]
