"""
Preprocesamiento para convertir el DataFrame raw al formato Decision Transformer.
"""
from __future__ import annotations

from typing import List, Dict, Any

import numpy as np


def create_dt_dataset(df_train) -> List[Dict[str, Any]]:
    """
    Convierte el DataFrame raw (user_id, user_group, items, ratings)
    a una lista de trayectorias con:
    - items
    - ratings
    - returns_to_go
    - timesteps
    - user_group
    """
    trajectories: List[Dict[str, Any]] = []

    for _, row in df_train.iterrows():
        items = row["items"]
        ratings = row["ratings"]
        group = int(row["user_group"])

        returns = np.zeros(len(ratings), dtype=np.float32)
        returns[-1] = ratings[-1]
        for t in range(len(ratings) - 2, -1, -1):
            returns[t] = ratings[t] + returns[t + 1]

        trajectory = {
            "items": items.astype(np.int64),
            "ratings": ratings.astype(np.float32),
            "returns_to_go": returns,
            "timesteps": np.arange(len(items), dtype=np.int64),
            "user_group": group,
        }
        trajectories.append(trajectory)

    return trajectories


def validate_preprocessing(trajectories: List[Dict[str, Any]]) -> bool:
    """
    Validaciones básicas del dataset procesado.
    Lanza AssertionError si alguna comprobación falla.
    """
    required_keys = {"items", "ratings", "returns_to_go", "timesteps", "user_group"}

    for traj in trajectories:
        assert required_keys.issubset(traj.keys()), "Faltan keys en trayectoria"
        items = traj["items"]
        ratings = traj["ratings"]
        returns = traj["returns_to_go"]
        timesteps = traj["timesteps"]

        assert len(items) == len(ratings) == len(returns) == len(
            timesteps
        ), "Las secuencias deben tener igual longitud"
        assert np.isclose(returns[0], ratings.sum()), "returns_to_go[0] debe ser suma total"
        assert np.isclose(returns[-1], ratings[-1]), "returns_to_go[-1] debe igualar rating final"

    return True


__all__ = ["create_dt_dataset", "validate_preprocessing"]
