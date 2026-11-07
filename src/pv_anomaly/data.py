"""Data loading and transform helpers for PV anomaly classification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, random_split
from torchvision.datasets import ImageFolder


@dataclass
class DataLoaders:
    """Container for datasets and dataloaders."""

    full_dataset: ImageFolder
    train_dataset: Subset
    val_dataset: Subset
    test_dataset: Subset
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    class_names: list[str]
    class_to_idx: dict[str, int]


def create_transforms(input_shape: tuple[int, int, int]) -> tuple[transforms.Compose, transforms.Compose]:
    """Create training and validation/test transforms."""
    train_transform = transforms.Compose([
        transforms.Resize((input_shape[0], input_shape[1])),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((input_shape[0], input_shape[1])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_transform, val_test_transform


def create_data_loaders(
    dataset_path: Path,
    input_shape: tuple[int, int, int],
    batch_size: int,
    validation_split: float,
    test_split: float,
    *,
    split_seed: int = 42,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoaders:
    """Load dataset, split it, and create data loaders."""
    train_transform, val_test_transform = create_transforms(input_shape)

    full_dataset = ImageFolder(root=dataset_path, transform=train_transform)

    total_size = len(full_dataset)
    test_size = int(test_split * total_size)
    val_size = int(validation_split * total_size)
    train_size = total_size - test_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(split_seed),
    )

    val_dataset.dataset = ImageFolder(root=dataset_path, transform=val_test_transform)
    test_dataset.dataset = ImageFolder(root=dataset_path, transform=val_test_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return DataLoaders(
        full_dataset=full_dataset,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        class_names=full_dataset.classes,
        class_to_idx=full_dataset.class_to_idx,
    )
