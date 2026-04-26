# VAE Latent Space Geometry as a Data Quality Metric

Auditing noisy, mislabeled, and out-of-distribution samples using
Variational Autoencoder latent geometry — without access to ground-truth
corruption labels.

## Project structure

```
vae_audit/
├── data/
│   └── corruption.py        # Controlled corruption injection
├── models/
│   └── vae.py               # VAE encoder, decoder, full model
├── audit/
│   └── metrics.py           # Per-sample KL, Mahalanobis, recon probability
├── experiments/
│   ├── train.py             # Training loop with ELBO loss
│   └── evaluate.py          # AUROC evaluation per metric × corruption type
├── utils/
│   └── visualise.py         # UMAP + latent scatter plots
├── run_pipeline.py          # Single entry point — runs everything
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python run_pipeline.py
```

Results (plots + CSV) saved to `results/`.

## Corruption types

| Type | Parameter | Default |
|------|-----------|---------|
| Gaussian input noise | σ (std of additive noise) | 0.3 |
| Label flips | ε (fraction of labels flipped) | 0.15 |
| OOD injection | r (fraction replaced with Fashion-MNIST) | 0.10 |

## Audit metrics

| Metric | Source | What it captures |
|--------|--------|-----------------|
| Per-sample KL divergence | Encoder posterior vs. N(0,I) | How far sample is from prior |
| Mahalanobis distance | Position vs. class centroid in z | Inter-class displacement |
| Reconstruction probability | Sampled decoder likelihood | Fidelity under generative model |

## Reproducing paper results

All random seeds are fixed (seed=42). Results are deterministic across runs.