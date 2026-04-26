"""
vae.py — Variational Autoencoder architecture.

Architecture (MNIST input: 1 × 28 × 28 = 784 dims):

  Input (784)
      │
  FC 784 → 512, ReLU, Dropout(0.2)
      │
  FC 512 → 256, ReLU, Dropout(0.2)
      │
  ┌───┴────┐
  μ (256→20)   log σ² (256→20)
      │
  z = μ + σ·ε    ← reparameterization trick
      │
  FC 20 → 256, ReLU
      │
  FC 256 → 512, ReLU
      │
  FC 512 → 784, Sigmoid
      │
  Reconstruction x̂ (784)

Tensor shapes at each layer are annotated inline.
Loss: ELBO = E[log p(x|z)] - KL(q(z|x) || p(z))
           = -BCE(x̂, x)   - 0.5 * Σ(1 + log σ² - μ² - σ²)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


LATENT_DIM = 20
INPUT_DIM  = 784        # 28 × 28 flattened
H1_DIM     = 512
H2_DIM     = 256


class Encoder(nn.Module):
    """
    q(z|x): maps input x → (μ, log σ²) in latent space R^LATENT_DIM.

    Dropout on hidden layers acts as the overfitting regulariser.
    Both μ and log σ² are unconstrained linear outputs; σ² = exp(log σ²)
    is always positive by construction.
    """
    def __init__(self, input_dim=INPUT_DIM, h1=H1_DIM, h2=H2_DIM,
                 latent_dim=LATENT_DIM, dropout=0.2):
        super().__init__()

        self.fc1     = nn.Linear(input_dim, h1)     # (B, 784) → (B, 512)
        self.fc2     = nn.Linear(h1, h2)             # (B, 512) → (B, 256)
        self.fc_mu   = nn.Linear(h2, latent_dim)     # (B, 256) → (B, 20)
        self.fc_logv = nn.Linear(h2, latent_dim)     # (B, 256) → (B, 20)
        self.drop    = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, 784)
        h = F.relu(self.fc1(x))   # (B, 512)
        h = self.drop(h)
        h = F.relu(self.fc2(h))   # (B, 256)
        h = self.drop(h)
        mu    = self.fc_mu(h)     # (B, 20)
        logv  = self.fc_logv(h)   # (B, 20)  — log σ²
        return mu, logv


class Decoder(nn.Module):
    """
    p(x|z): maps latent code z → reconstructed input x̂ ∈ [0,1]^INPUT_DIM.

    Sigmoid output treats each pixel as an independent Bernoulli.
    No dropout in decoder — we want deterministic reconstruction at inference.
    """
    def __init__(self, latent_dim=LATENT_DIM, h2=H2_DIM, h1=H1_DIM,
                 output_dim=INPUT_DIM):
        super().__init__()

        self.fc1 = nn.Linear(latent_dim, h2)    # (B, 20)  → (B, 256)
        self.fc2 = nn.Linear(h2, h1)            # (B, 256) → (B, 512)
        self.fc3 = nn.Linear(h1, output_dim)    # (B, 512) → (B, 784)

    def forward(self, z):
        # z: (B, 20)
        h = F.relu(self.fc1(z))      # (B, 256)
        h = F.relu(self.fc2(h))      # (B, 512)
        x_hat = torch.sigmoid(self.fc3(h))  # (B, 784) ∈ [0,1]
        return x_hat


class VAE(nn.Module):
    """
    Full VAE: encoder + reparameterization + decoder.

    Reparameterization trick:
        z = μ + σ · ε,   ε ~ N(0, I)
    Keeps gradient flow through the stochastic node (Kingma & Welling, 2013).
    """
    def __init__(self, latent_dim=LATENT_DIM, dropout=0.2):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder    = Encoder(latent_dim=latent_dim, dropout=dropout)
        self.decoder    = Decoder(latent_dim=latent_dim)

    def reparameterize(self, mu, logv):
        """
        Sample z ~ q(z|x) = N(μ, σ²I) using the reparameterization trick.
        During eval, returns μ directly (deterministic encoding).
        """
        if self.training:
            std = torch.exp(0.5 * logv)   # σ = exp(log σ² / 2)
            eps = torch.randn_like(std)    # ε ~ N(0, I), same shape as σ
            return mu + std * eps          # z = μ + σ·ε
        return mu

    def forward(self, x):
        """
        x: (B, 1, 28, 28)  →  flattened  →  encoder  →  z  →  decoder  →  x̂

        Returns:
            x_hat : (B, 784)  reconstructed input
            mu    : (B, 20)   posterior mean
            logv  : (B, 20)   posterior log-variance
            z     : (B, 20)   sampled latent code
        """
        x_flat = x.view(x.size(0), -1)          # (B, 784)
        mu, logv = self.encoder(x_flat)          # (B, 20), (B, 20)
        z = self.reparameterize(mu, logv)        # (B, 20)
        x_hat = self.decoder(z)                  # (B, 784)
        return x_hat, mu, logv, z


def elbo_loss(x_hat, x, mu, logv, beta=1.0):
    """
    ELBO loss = Reconstruction loss + β · KL divergence.

    Reconstruction: Binary Cross-Entropy summed over pixels, averaged over batch.
        BCE = -Σ [x·log(x̂) + (1-x)·log(1-x̂)]

    KL divergence (closed form for Gaussian prior N(0,I)):
        KL = -0.5 · Σ (1 + log σ² - μ² - σ²)

    β=1 → standard VAE (Kingma & Welling, 2013)
    β>1 → disentangled β-VAE (Higgins et al., 2017)

    Args:
        x_hat : (B, 784) reconstructed
        x     : (B, 1, 28, 28) original
        mu    : (B, 20)
        logv  : (B, 20)
        beta  : KL weight (default 1.0)

    Returns:
        total_loss, recon_loss, kl_loss  — all scalar tensors
    """
    x_flat = x.view(x.size(0), -1)   # (B, 784)

    recon_loss = F.binary_cross_entropy(x_hat, x_flat, reduction='sum') / x.size(0)

    # KL per dimension, summed over latent dims, averaged over batch
    kl_loss = -0.5 * torch.mean(
        torch.sum(1 + logv - mu.pow(2) - logv.exp(), dim=1)
    )

    total = recon_loss + beta * kl_loss
    return total, recon_loss, kl_loss