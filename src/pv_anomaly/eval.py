"""Evaluation helpers for PV anomaly models."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, list[int], list[int]]:
    """Evaluate model on a test set and collect predictions."""
    model.eval()
    test_loss = 0.0
    test_corrects = 0
    total_samples = 0
    all_preds: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        test_bar = tqdm(test_loader, desc="Testing")
        for inputs, labels in test_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            test_loss += loss.item() * inputs.size(0)
            test_corrects += torch.sum(preds == labels.data).item()
            total_samples += inputs.size(0)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

            test_bar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "Acc": f"{test_corrects / total_samples:.4f}",
            })

    test_loss = test_loss / total_samples
    test_acc = test_corrects / total_samples

    return test_loss, test_acc, all_preds, all_labels
