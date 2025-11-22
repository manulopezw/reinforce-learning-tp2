"""
Loop de entrenamiento para Decision Transformer (Parte 2.3).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def train_decision_transformer(
    model: torch.nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int = 50,
    val_loader: DataLoader | None = None,
):
    """
    Entrena el modelo con cross-entropy sobre los próximos items.
    """
    model.to(device)
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

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

        avg_train = total_loss / len(train_loader)
        history["train_loss"].append(avg_train)

        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
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
                    val_loss += loss.item()
            avg_val = val_loss / len(val_loader)
            history["val_loss"].append(avg_val)

        print(f"Epoch {epoch + 1}/{num_epochs} - train_loss={avg_train:.4f}")
        if val_loader is not None:
            print(f"  val_loss={avg_val:.4f}")

    return model, history


__all__ = ["train_decision_transformer"]
