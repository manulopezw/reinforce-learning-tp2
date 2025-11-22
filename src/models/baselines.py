"""
Baselines simples (Parte 3.1): Popularity.
"""
from __future__ import annotations

import numpy as np
from config_dataset import NUM_ITEMS


class PopularityRecommender:
    """
    Recomienda los items más populares en el dataset (sin personalización).
    """

    def __init__(self, num_items: int = NUM_ITEMS):
        self.num_items = num_items
        self.item_counts: np.ndarray | None = None
        self.popular_items: np.ndarray | None = None

    def fit(self, trajectories):
        """
        Cuenta frecuencia de cada item en las trayectorias de train.
        """
        all_items = np.concatenate([traj["items"] for traj in trajectories])
        self.item_counts = np.bincount(all_items, minlength=self.num_items)
        self.popular_items = np.argsort(self.item_counts)[::-1]

    def recommend(self, user_history, k: int = 10):
        """
        Retorna top-k items más populares que el usuario no ha visto.
        """
        if self.popular_items is None:
            raise RuntimeError("Debes llamar a fit() antes de recommend().")
        seen = set(user_history)
        recs = []
        for item in self.popular_items:
            if item not in seen:
                recs.append(int(item))
            if len(recs) == k:
                break
        return recs


__all__ = ["PopularityRecommender"]
