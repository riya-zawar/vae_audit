"""Controlled corruption injection for audit evaluation.

corruption.py — Controlled dataset corruption for audit experiments.

Three corruption types, each tracked with a ground-truth flag so we can
compute AUROC for the audit metrics (flags are NEVER passed to the VAE):

  1. Gaussian input noise  — pixel-level additive noise N(0, σ²)
  2. Label flips           — random label reassignment at rate ε
  3. OOD injection         — replace MNIST samples with Fashion-MNIST samples

Each function returns:
    images  : torch.Tensor (N, 1, 28, 28)  — corrupted dataset images
    labels  : torch.Tensor (N,)            — (possibly flipped) labels
    flags   : torch.Tensor (N,)            — 1 if corrupted, 0 if clean
    c_types : list[str] of length N        — 'clean'|'noise'|'label'|'ood'
"""

import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset


def load_mnist(root='./data_cache', train=True):
    """Download and return MNIST as tensors (N, 1, 28, 28) in [0,1]."""
    ds = datasets.MNIST(
        root=root, train=train, download=True,
        transform=transforms.ToTensor()
    )
    loader = DataLoader(ds, batch_size=len(ds), shuffle=False)
    images, labels = next(iter(loader))
    return images, labels


def load_fashion_mnist(root='./data_cache', train=True):
    """Download and return Fashion-MNIST as tensors (N, 1, 28, 28) in [0,1]."""
    ds = datasets.FashionMNIST(
        root=root, train=train, download=True,
        transform=transforms.ToTensor()
    )
    loader = DataLoader(ds, batch_size=len(ds), shuffle=False)
    images, labels = next(iter(loader))
    return images, labels


def inject_gaussian_noise(images, labels, sigma=0.3, rate=0.15, rng=None):
    """
    Add Gaussian noise N(0, σ²) to a random subset of images.

    Args:
        images : (N, 1, 28, 28)
        labels : (N,)
        sigma  : std of additive noise
        rate   : fraction of dataset to corrupt
        rng    : np.random.Generator for reproducibility

    Returns:
        images_out, labels, flags, c_types
    """
    if rng is None:
        rng = np.random.default_rng(42)

    N = images.size(0)
    n_corrupt = int(N * rate)
    corrupt_idx = rng.choice(N, size=n_corrupt, replace=False)

    images_out = images.clone()
    noise = torch.tensor(
        rng.normal(0, sigma, size=(n_corrupt, 1, 28, 28)),
        dtype=torch.float32
    )
    images_out[corrupt_idx] = torch.clamp(
        images_out[corrupt_idx] + noise, 0.0, 1.0
    )

    flags   = torch.zeros(N, dtype=torch.long)
    c_types = ['clean'] * N
    for i in corrupt_idx:
        flags[i]   = 1
        c_types[i] = 'noise'

    return images_out, labels, flags, c_types


def inject_label_flips(images, labels, epsilon=0.15, n_classes=10, rng=None):
    """
    Randomly flip labels to a different class at rate ε.

    Args:
        images   : (N, 1, 28, 28)
        labels   : (N,)
        epsilon  : fraction of labels to flip
        n_classes: number of classes (10 for MNIST)

    Returns:
        images, labels_out, flags, c_types
    """
    if rng is None:
        rng = np.random.default_rng(42)

    N = labels.size(0)
    n_corrupt = int(N * epsilon)
    corrupt_idx = rng.choice(N, size=n_corrupt, replace=False)

    labels_out = labels.clone()
    for i in corrupt_idx:
        orig = labels_out[i].item()
        candidates = [c for c in range(n_classes) if c != orig]
        labels_out[i] = rng.choice(candidates)

    flags   = torch.zeros(N, dtype=torch.long)
    c_types = ['clean'] * N
    for i in corrupt_idx:
        flags[i]   = 1
        c_types[i] = 'label'

    return images, labels_out, flags, c_types


def inject_ood(mnist_images, mnist_labels, fashion_images,
               rate=0.10, rng=None):
    """
    Replace a fraction of MNIST samples with Fashion-MNIST samples (OOD).
    OOD samples retain a dummy label of -1 so they are excluded from
    class-centroid computation in Mahalanobis scoring.

    Args:
        mnist_images   : (N, 1, 28, 28)
        mnist_labels   : (N,)
        fashion_images : (M, 1, 28, 28)  — OOD source
        rate           : fraction to replace

    Returns:
        images_out, labels_out, flags, c_types
    """
    if rng is None:
        rng = np.random.default_rng(42)

    N = mnist_images.size(0)
    n_corrupt = int(N * rate)
    corrupt_idx = rng.choice(N, size=n_corrupt, replace=False)
    fashion_idx = rng.choice(fashion_images.size(0), size=n_corrupt, replace=False)

    images_out = mnist_images.clone()
    labels_out = mnist_labels.clone()

    images_out[corrupt_idx] = fashion_images[fashion_idx]
    labels_out[corrupt_idx] = -1   # sentinel — excluded from class stats

    flags   = torch.zeros(N, dtype=torch.long)
    c_types = ['clean'] * N
    for i in corrupt_idx:
        flags[i]   = 1
        c_types[i] = 'ood'

    return images_out, labels_out, flags, c_types


def build_corrupted_dataset(
    sigma=0.3,
    noise_rate=0.15,
    epsilon=0.15,
    ood_rate=0.10,
    train=True,
    seed=42,
    root='./data_cache'
):
    """
    Build a single dataset with all three corruption types injected
    on disjoint subsets of MNIST training data.

    Corruption budget is split evenly so corruptions do not overlap:
        noise_rate  → 15% of dataset
        epsilon     → 15% of dataset  (applied to non-noise samples)
        ood_rate    → 10% of dataset  (applied to remaining clean samples)

    Returns:
        dataset    : TensorDataset(images, labels, flags)
        c_types    : list[str] of length N — corruption type per sample
        flag_array : np.array of 0/1, shape (N,)

    The `flags` tensor is the ground truth for AUROC computation.
    It is NEVER used during VAE training.
    """
    rng = np.random.default_rng(seed)

    print("Loading MNIST...")
    images, labels = load_mnist(root=root, train=train)
    print("Loading Fashion-MNIST (OOD source)...")
    fashion_imgs, _ = load_fashion_mnist(root=root, train=train)

    N = images.size(0)
    flags   = torch.zeros(N, dtype=torch.long)
    c_types = ['clean'] * N

    # --- Build non-overlapping index pools ---
    all_idx      = np.arange(N)
    n_noise      = int(N * noise_rate)
    n_label      = int(N * epsilon)
    n_ood        = int(N * ood_rate)

    shuffled     = rng.permutation(all_idx)
    noise_idx    = shuffled[:n_noise]
    label_idx    = shuffled[n_noise: n_noise + n_label]
    ood_idx      = shuffled[n_noise + n_label: n_noise + n_label + n_ood]

    # --- Gaussian noise ---
    noise = torch.tensor(
        rng.normal(0, sigma, size=(n_noise, 1, 28, 28)), dtype=torch.float32
    )
    images[noise_idx] = torch.clamp(images[noise_idx] + noise, 0.0, 1.0)
    for i in noise_idx:
        flags[i]   = 1
        c_types[i] = 'noise'

    # --- Label flips ---
    for i in label_idx:
        orig       = labels[i].item()
        candidates = [c for c in range(10) if c != orig]
        labels[i]  = int(rng.choice(candidates))
        flags[i]   = 1
        c_types[i] = 'label'

    # --- OOD injection ---
    fashion_sel = rng.choice(fashion_imgs.size(0), size=n_ood, replace=False)
    images[ood_idx]  = fashion_imgs[fashion_sel]
    labels[ood_idx]  = -1
    for i in ood_idx:
        flags[i]   = 1
        c_types[i] = 'ood'

    # Summary
    n_clean = (flags == 0).sum().item()
    print(f"\nDataset composition (N={N}):")
    print(f"  Clean  : {n_clean:>6d} ({100*n_clean/N:.1f}%)")
    print(f"  Noise  : {n_noise:>6d} ({100*n_noise/N:.1f}%)")
    print(f"  Label  : {n_label:>6d} ({100*n_label/N:.1f}%)")
    print(f"  OOD    : {n_ood:>6d}   ({100*n_ood/N:.1f}%)\n")

    dataset    = TensorDataset(images, labels, flags)
    flag_array = flags.numpy()
    return dataset, c_types, flag_array