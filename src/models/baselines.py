"""
Baselines simples según referencia (Parte 3.1).
"""
from __future__ import annotations

import numpy as np
from config_dataset import NUM_ITEMS


class PopularityRecommender:
    """
    Recomienda items más populares (no personalizados).
    """

    def __init__(self, num_items: int = NUM_ITEMS):
        self.num_items = num_items
        self.item_counts = None
        self.popular_items = None

    def fit(self, trajectories):
        all_items = np.concatenate([traj["items"] for traj in trajectories])
        self.item_counts = np.bincount(all_items, minlength=self.num_items)
        self.popular_items = np.argsort(self.item_counts)[::-1]

    def recommend(self, user_history, k: int = 10):
        recommendations = []
        seen = set(user_history)
        for item in self.popular_items:
            if item not in seen:
                recommendations.append(int(item))
            if len(recommendations) == k:
                break
        return recommendations


__all__ = ["PopularityRecommender"]
