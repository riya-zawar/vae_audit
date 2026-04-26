"""
run_pipeline.py — Single entry point. Run this file to execute the full pipeline.

Usage:
    python run_pipeline.py

Steps:
    1. Build corrupted MNIST + Fashion-MNIST dataset
    2. Train VAE (images only — no corruption flags used)
    3. Compute 3 audit scores per sample
    4. Evaluate AUROC per metric × corruption type
    5. Save all plots and CSVs to results/
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np

from data.corruption      import build_corrupted_dataset
from experiments.train    import train_vae
from audit.metrics        import compute_all_audit_scores
from experiments.evaluate import compute_auroc_matrix, combined_score_auroc, save_results, print_results
from utils.visualise      import (plot_latent_umap, plot_loss_curves,
                                   plot_roc_curves, plot_score_distributions,
                                   plot_reconstructions)

# ── Config ────────────────────────────────────────────────────────────────────
SEED        = 42
LATENT_DIM  = 20
BETA        = 1.0       # 1.0 = standard VAE
EPOCHS      = 50
BATCH_SIZE  = 128
LR          = 1e-3
DROPOUT     = 0.2

NOISE_SIGMA = 0.3       # Gaussian noise std
NOISE_RATE  = 0.15      # 15% of dataset
LABEL_EPS   = 0.15      # 15% label flip rate
OOD_RATE    = 0.10      # 10% OOD injection

RESULTS_DIR = 'results'
CKPT_PATH   = 'results/vae_best.pt'
# ─────────────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")

    # ── Step 1: Build dataset ────────────────────────────────────────────────
    print("=" * 55)
    print("STEP 1 — Building corrupted dataset")
    print("=" * 55)
    dataset, c_types, flag_array = build_corrupted_dataset(
        sigma       = NOISE_SIGMA,
        noise_rate  = NOISE_RATE,
        epsilon     = LABEL_EPS,
        ood_rate    = OOD_RATE,
        train       = True,
        seed        = SEED,
        root        = './data_cache'
    )
    # Extract labels for Mahalanobis (label array, not passed to VAE training)
    labels_arr = dataset.tensors[1].numpy()

    # ── Step 2: Train VAE ────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("STEP 2 — Training VAE")
    print("=" * 55)
    model, history = train_vae(
        dataset     = dataset,
        latent_dim  = LATENT_DIM,
        beta        = BETA,
        epochs      = EPOCHS,
        batch_size  = BATCH_SIZE,
        lr          = LR,
        dropout     = DROPOUT,
        save_path   = CKPT_PATH,
        device      = device,
        seed        = SEED,
        verbose     = True
    )

    # ── Step 3: Compute audit scores ─────────────────────────────────────────
    print("\n" + "=" * 55)
    print("STEP 3 — Computing audit scores")
    print("=" * 55)
    scores, z_all = compute_all_audit_scores(
        model       = model,
        dataset     = dataset,
        labels_arr  = labels_arr,
        batch_size  = 512,
        K           = 20,
        device      = device
    )

    # ── Step 4: Evaluate AUROC ───────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("STEP 4 — Evaluating AUROC")
    print("=" * 55)
    df, roc_data = compute_auroc_matrix(scores, c_types, flag_array)
    combined_aucs = combined_score_auroc(scores, c_types)
    print_results(df, combined_aucs)
    save_results(df, combined_aucs, roc_data, out_dir=RESULTS_DIR)

    # ── Step 5: Visualise ────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("STEP 5 — Generating plots")
    print("=" * 55)
    digit_labels = labels_arr.copy()
    digit_labels[digit_labels < 0] = 10   # map OOD (-1) to dummy class 10

    plot_loss_curves(history, out_dir=RESULTS_DIR)
    plot_latent_umap(z_all, c_types, digit_labels, out_dir=RESULTS_DIR)
    plot_roc_curves(roc_data, out_dir=RESULTS_DIR)
    plot_score_distributions(scores, c_types, out_dir=RESULTS_DIR)
    plot_reconstructions(model, dataset, c_types,
                         n_per_type=8, device=device, out_dir=RESULTS_DIR)

    print("\n" + "=" * 55)
    print("DONE. All outputs saved to results/")
    print("=" * 55)
    print("Files produced:")
    for f in sorted(os.listdir(RESULTS_DIR)):
        print(f"  results/{f}")


if __name__ == '__main__':
    main()
