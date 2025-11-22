"""
Dataset de PyTorch para el Decision Transformer (Parte 2.2).
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class RecommendationDataset(Dataset):
    """
    Devuelve ventanas de longitud fija para entrenamiento autoregresivo.
    """

    def __init__(self, trajectories, context_length: int = 20):
        """
        Args:
            trajectories: lista producida por create_dt_dataset()
            context_length: tamaño de ventana (timesteps)
        """
        self.trajectories = trajectories
        self.context_length = context_length

    def __len__(self) -> int:  # pragma: no cover
        return len(self.trajectories)

    def __getitem__(self, idx: int):
        traj = self.trajectories[idx]

        items = traj["items"]
        ratings = traj["ratings"]
        rtg = traj["returns_to_go"]
        timesteps = traj["timesteps"]
        group = traj["user_group"]

        seq_len = min(len(items), self.context_length)
        if len(items) > self.context_length:
            start_idx = np.random.randint(0, len(items) - self.context_length + 1)
        else:
            start_idx = 0
        end_idx = start_idx + seq_len

        # Ventana de la trayectoria
        states = items[start_idx:end_idx]
        actions = items[start_idx:end_idx]

        # Targets: siguiente item (último padding con -1)
        targets = np.full(self.context_length, -1, dtype=np.int64)
        shifted = items[start_idx + 1 : end_idx]
        targets[: seq_len - 1] = shifted

        # Returns-to-go
        rtg_seq = np.zeros((self.context_length, 1), dtype=np.float32)
        rtg_seq[:seq_len] = rtg[start_idx:end_idx].reshape(-1, 1)

        # Timesteps
        time_seq = np.zeros(self.context_length, dtype=np.int64)
        time_seq[:seq_len] = timesteps[start_idx:end_idx]

        # Padding states/actions
        states_padded = np.zeros(self.context_length, dtype=np.int64)
        actions_padded = np.zeros(self.context_length, dtype=np.int64)
        states_padded[:seq_len] = states
        actions_padded[:seq_len] = actions

        return {
            "states": torch.tensor(states_padded, dtype=torch.long),
            "actions": torch.tensor(actions_padded, dtype=torch.long),
            "rtg": torch.tensor(rtg_seq, dtype=torch.float32),
            "timesteps": torch.tensor(time_seq, dtype=torch.long),
            "groups": torch.tensor(group, dtype=torch.long),
            "targets": torch.tensor(targets, dtype=torch.long),
        }


__all__ = ["RecommendationDataset"]
