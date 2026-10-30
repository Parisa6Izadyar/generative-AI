"""Official MNIST partitions and deterministic sample selection."""

import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset


def select_indices(size, count, seed):
    if not 1 <= count <= size:
        raise ValueError(f"Sample count must be between 1 and {size}.")
    return np.random.default_rng(seed).permutation(size)[:count]


def normalize_pixels(pixels):
    """Map uint8 MNIST pixels to the Tanh output range."""
    if pixels.dtype != torch.uint8 or pixels.ndim != 3 or pixels.shape[1:] != (28, 28):
        raise ValueError("Expected uint8 MNIST pixels with shape (N, 28, 28).")
    return pixels.unsqueeze(1).float().div(127.5).sub(1)


def load_mnist(root, train_samples, eval_samples, seed, download=False):
    from torchvision.datasets import MNIST

    root = Path(root)
    try:
        train = MNIST(root, train=True, download=download)
        test = MNIST(root, train=False, download=download)
    except RuntimeError as exc:
        raise ValueError(
            "MNIST is unavailable. Run train with --download once while online, "
            "or use the smoke command for an offline execution check."
        ) from exc
    train_indices = select_indices(len(train), train_samples, seed)
    reference_indices = select_indices(len(test), 2 * eval_samples, seed + 1)
    train_data = TensorDataset(
        normalize_pixels(train.data[train_indices]), train.targets[train_indices]
    )
    test_data = TensorDataset(normalize_pixels(test.data), test.targets)
    reference = test_data.tensors[0][reference_indices[:eval_samples]]
    comparison = test_data.tensors[0][reference_indices[eval_samples:]]
    raw = root / "MNIST" / "raw"
    checksums = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(raw.glob("*-ubyte"))
    }
    manifest = {
        "dataset": "MNIST",
        "official_train_size": len(train),
        "official_test_size": len(test),
        "train_samples": len(train_data),
        "reference_samples": eval_samples,
        "real_comparison_samples": eval_samples,
        "train_indices_sha256": hashlib.sha256(
            np.asarray(train_indices, dtype="<i8").tobytes()
        ).hexdigest(),
        "reference_indices_sha256": hashlib.sha256(
            np.asarray(reference_indices, dtype="<i8").tobytes()
        ).hexdigest(),
        "raw_file_sha256": checksums,
        "normalization": "uint8 / 127.5 - 1",
        "reference_partition": "official test; disjoint reference and real-comparison subsets",
    }
    return train_data, test_data, reference, comparison, manifest


def synthetic_data(seed=42):
    """Small random tensors for exercising code paths without a download."""
    generator = torch.Generator().manual_seed(seed)
    train_images = torch.rand(64, 1, 28, 28, generator=generator) * 2 - 1
    test_images = torch.rand(64, 1, 28, 28, generator=generator) * 2 - 1
    train = TensorDataset(train_images, torch.arange(64) % 10)
    test = TensorDataset(test_images, torch.arange(64) % 10)
    return train, test, test_images[:32], test_images[32:], {
        "dataset": "synthetic random tensors; execution check only",
        "train_samples": 64, "reference_samples": 32, "real_comparison_samples": 32,
    }
