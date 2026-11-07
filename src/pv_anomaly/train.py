"""Training loop and optimization helpers for PV anomaly models."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy with label smoothing."""

    def __init__(self, smoothing: float = 0.1) -> None:
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute label-smoothed cross entropy loss."""
        confidence = 1.0 - self.smoothing
        log_probs = torch.log_softmax(pred, dim=1)
        nll_loss = -log_probs.gather(dim=1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -log_probs.mean(dim=1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Train the model for one epoch."""
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    train_bar = tqdm(train_loader, desc="Training")
    for inputs, labels in train_bar:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data).item()
        total_samples += inputs.size(0)

        train_bar.set_postfix({
            "Loss": f"{loss.item():.4f}",
            "Acc": f"{running_corrects / total_samples:.4f}",
        })

    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects / total_samples
    return epoch_loss, epoch_acc


def validate_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Validate the model for one epoch."""
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    with torch.no_grad():
        val_bar = tqdm(val_loader, desc="Validation")
        for inputs, labels in val_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).item()
            total_samples += inputs.size(0)

            val_bar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "Acc": f"{running_corrects / total_samples:.4f}",
            })

    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects / total_samples
    return epoch_loss, epoch_acc


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    scheduler: ReduceLROnPlateau,
    device: torch.device,
    *,
    epochs: int,
    patience_early_stop: int,
    model_save_path: Path,
    hyperparams: dict[str, Any],
) -> tuple[dict[str, list[float]], float, float, datetime, datetime]:
    """Run the full training loop with early stopping."""
    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
    }

    best_val_loss = float("inf")
    best_val_acc = 0.0
    patience_counter = 0
    best_model_state = None

    start_time = datetime.now()
    print(f"Training started at: {start_time}")
    print(f"Max epochs: {epochs}")
    print(f"Early stopping patience: {patience_early_stop}")
    print("-" * 60)

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 40)

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"Learning Rate: {current_lr:.2e}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print(f"✅ New best model! Val Acc: {val_acc:.4f}")

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": best_model_state,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_acc": best_val_acc,
                    "best_val_loss": best_val_loss,
                    "hyperparams": hyperparams,
                },
                model_save_path / "best_model.pth",
            )
        else:
            patience_counter += 1
            print(f"⏳ No improvement. Patience: {patience_counter}/{patience_early_stop}")

        if patience_counter >= patience_early_stop:
            print(f"\n🛑 Early stopping triggered after {epoch + 1} epochs")
            print(f"Best validation accuracy: {best_val_acc:.4f}")
            break

    end_time = datetime.now()
    print(f"\n{'=' * 60}")
    print("Training completed!")
    print(f"Training duration: {end_time - start_time}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Total epochs trained: {len(history['train_loss'])}")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("✅ Best model weights loaded")

    return history, best_val_loss, best_val_acc, start_time, end_time
