"""
metrics.py — Three per-sample audit scores derived from a trained VAE.

These are the core methodological contribution of the paper.
No ground-truth corruption labels are used here — scores are computed
purely from the VAE's encoder and decoder outputs.

Metric 1 — Per-sample KL divergence
    KL_i = -0.5 · Σ_d (1 + log σ²_id - μ²_id - σ²_id)
    Interpretation: how strongly the sample's posterior deviates from N(0,I).
    High KL → sample is hard to encode; likely anomalous.

Metric 2 — Mahalanobis distance
    d_M(z_i, c_k) = sqrt((z_i - μ_k)^T Σ_k^{-1} (z_i - μ_k))
    where μ_k, Σ_k are the mean and covariance of class k in latent space.
    Computed against the predicted class (from true label when available).
    Interpretation: how far the sample is from its class cluster.
    High d_M → latent position inconsistent with class.

Metric 3 — Reconstruction probability (An & Cho, 2015)
    log p(x|z) ≈ (1/K) Σ_{k=1}^K log p(x | z_k),  z_k ~ q(z|x)
    Approximated as mean BCE over K Monte Carlo samples.
    Interpretation: how well the generative model explains the sample.
    Low recon_prob → sample is poorly explained; likely anomalous.

Note: metrics 1 and 3 are inverted (higher = more anomalous) for consistent
AUROC computation. For Mahalanobis, higher distance = more anomalous.
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F


@torch.no_grad()
def compute_all_audit_scores(
    model,
    dataset,
    labels_arr,
    batch_size = 256,
    K          = 20,
    device     = None
):
    """
    Compute all three audit scores for every sample in the dataset.

    Args:
        model      : trained VAE (eval mode set internally)
        dataset    : TensorDataset(images, labels, flags)
        labels_arr : np.array of true labels (N,) — used only for Mahalanobis
                     class centroid computation; -1 for OOD samples
        batch_size : inference batch size
        K          : Monte Carlo samples for reconstruction probability
        device     : compute device

    Returns:
        scores : dict with keys 'kl', 'mahal', 'recon_prob'
                 each value is np.array of shape (N,)
                 Higher value = more suspicious for all three metrics.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model.eval()
    model.to(device)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_mu   = []
    all_logv = []
    all_kl   = []
    all_recon_prob = []
    all_images = []

    print("Computing encoder statistics and reconstruction probabilities...")
    for batch in loader:
        x, _, _ = batch
        x = x.to(device)
        x_flat = x.view(x.size(0), -1)

        # Encode
        mu, logv = model.encoder(x_flat)

        # KL per sample: -0.5 * sum_d (1 + logv - mu^2 - exp(logv))
        kl_per_sample = -0.5 * torch.sum(
            1 + logv - mu.pow(2) - logv.exp(), dim=1
        )   # (B,) — higher = more anomalous

        # Reconstruction probability via K Monte Carlo samples
        recon_probs = []
        for _ in range(K):
            std = torch.exp(0.5 * logv)
            eps = torch.randn_like(std)
            z_k   = mu + std * eps
            x_hat = model.decoder(z_k)
            # Log-likelihood under Bernoulli: -BCE (higher = better fit)
            log_p = -F.binary_cross_entropy(x_hat, x_flat, reduction='none').sum(dim=1)
            recon_probs.append(log_p.unsqueeze(1))

        # Mean log-likelihood over K samples (higher = better explained)
        recon_prob = torch.cat(recon_probs, dim=1).mean(dim=1)  # (B,)
        # Negate so higher = more anomalous (consistent with KL direction)
        recon_prob_score = -recon_prob

        all_mu.append(mu.cpu())
        all_logv.append(logv.cpu())
        all_kl.append(kl_per_sample.cpu())
        all_recon_prob.append(recon_prob_score.cpu())
        all_images.append(x.cpu())

    all_mu    = torch.cat(all_mu,    dim=0).numpy()   # (N, 20)
    all_kl    = torch.cat(all_kl,    dim=0).numpy()   # (N,)
    all_recon = torch.cat(all_recon_prob, dim=0).numpy()  # (N,)

    # Mahalanobis distance
    print("Computing Mahalanobis distances...")
    mahal_scores = _mahalanobis_scores(all_mu, labels_arr)

    scores = {
        'kl':         all_kl,
        'mahal':      mahal_scores,
        'recon_prob': all_recon,
    }

    print("Audit scores computed.")
    return scores, all_mu


def _mahalanobis_scores(z_all, labels_arr):
    """
    Compute per-sample Mahalanobis distance to the nearest class centroid.

    Class statistics are computed only from samples with valid labels (≥ 0).
    OOD samples (label = -1) get their distance computed to the nearest class.

    Args:
        z_all      : (N, latent_dim) latent means
        labels_arr : (N,) integer class labels; -1 for OOD

    Returns:
        distances : (N,) float array — higher = more anomalous
    """
    unique_classes = np.unique(labels_arr[labels_arr >= 0])
    N, D           = z_all.shape

    # Compute per-class mean and covariance (only from valid-label samples)
    class_means = {}
    class_covs  = {}
    for c in unique_classes:
        idx = np.where(labels_arr == c)[0]
        z_c = z_all[idx]
        class_means[c] = np.mean(z_c, axis=0)                          # (D,)
        cov = np.cov(z_c.T) + 1e-6 * np.eye(D)                        # (D, D)
        class_covs[c]  = np.linalg.inv(cov)                            # Σ^{-1}

    distances = np.zeros(N)
    for i in range(N):
        z_i = z_all[i]
        label = labels_arr[i]

        if label >= 0:
            # Use true class centroid
            diff = z_i - class_means[label]
            cov_inv = class_covs[label]
            distances[i] = np.sqrt(diff @ cov_inv @ diff)
        else:
            # OOD: distance to nearest class centroid
            min_d = float('inf')
            for c in unique_classes:
                diff  = z_i - class_means[c]
                cov_inv = class_covs[c]
                d     = np.sqrt(diff @ cov_inv @ diff)
                if d < min_d:
                    min_d = d
            distances[i] = min_d

    return distances