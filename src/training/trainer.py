"""
Loop de entrenamiento según referencia (Parte 2.3).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def train_decision_transformer(
    model,
    train_loader: DataLoader,
    optimizer,
    device,
    num_epochs=50,
    return_history=False,
):
    """
    Entrena el Decision Transformer.
    Loss: Cross-entropy entre item predicho y item verdadero.
    """
    model.train()

    history = []

    for epoch in range(num_epochs):
        total_loss = 0

        for batch in train_loader:
            states = batch["states"].to(device)
            actions = batch["actions"].to(device)
            rtg = batch["rtg"].to(device)
            timesteps = batch["timesteps"].to(device)
            groups = batch["groups"].to(device)
            targets = batch["targets"].to(device)

            logits = model(states, actions, rtg, timesteps, groups)

            loss = F.cross_entropy(
                logits.reshape(-1, model.num_items),
                targets.reshape(-1),
                ignore_index=-1,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        history.append(avg_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

    if return_history:
        return model, history
    return model


__all__ = ["train_decision_transformer"]
