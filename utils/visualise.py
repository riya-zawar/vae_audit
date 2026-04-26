"""
visualise.py — All plots for the paper's Results section.

Produces:
  1. latent_umap.png       — UMAP projection coloured by corruption type
  2. latent_umap_class.png — UMAP projection coloured by digit class
  3. loss_curves.png       — Training/validation ELBO over epochs
  4. roc_curves.png        — ROC curves for each metric × corruption type
  5. score_distributions.png — Boxplots of audit scores by corruption type
  6. reconstructions.png   — Sample images vs. their reconstructions
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    from sklearn.decomposition import PCA
    UMAP_AVAILABLE = False
    print("umap-learn not found — falling back to PCA for 2D projection.")


CORRUPTION_COLORS = {
    'clean': '#4C9BE8',
    'noise': '#E8794C',
    'label': '#E84C6A',
    'ood':   '#9B4CE8',
}
CORRUPTION_LABELS = {
    'clean': 'Clean',
    'noise': 'Gaussian noise',
    'label': 'Label flip',
    'ood':   'OOD (Fashion-MNIST)',
}
METRIC_LABELS = {
    'kl':         'Per-sample KL divergence',
    'mahal':      'Mahalanobis distance',
    'recon_prob': 'Reconstruction probability score',
}


def plot_latent_umap(z_all, c_types, digit_labels, out_dir='results'):
    """
    2D UMAP (or PCA) projection of the latent space.
    Produces two plots: coloured by corruption type, and by digit class.
    """
    os.makedirs(out_dir, exist_ok=True)
    c_types_arr = np.array(c_types)

    print(f"Fitting {'UMAP' if UMAP_AVAILABLE else 'PCA'} on {z_all.shape[0]} latent vectors...")
    if UMAP_AVAILABLE:
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15,
                            min_dist=0.1)
    else:
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2, random_state=42)

    z_2d = reducer.fit_transform(z_all)

    # ── Plot 1: by corruption type ──
    fig, ax = plt.subplots(figsize=(8, 6))
    for ctype in ['clean', 'noise', 'label', 'ood']:
        mask = c_types_arr == ctype
        if mask.sum() == 0:
            continue
        alpha = 0.3 if ctype == 'clean' else 0.7
        size  = 3   if ctype == 'clean' else 10
        ax.scatter(
            z_2d[mask, 0], z_2d[mask, 1],
            c=CORRUPTION_COLORS[ctype],
            label=CORRUPTION_LABELS[ctype],
            alpha=alpha, s=size, linewidths=0
        )
    ax.set_xlabel('Dimension 1', fontsize=11)
    ax.set_ylabel('Dimension 2', fontsize=11)
    ax.set_title('Latent space: corruption type' + (' (UMAP)' if UMAP_AVAILABLE else ' (PCA)'),
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, markerscale=3)
    ax.grid(True, alpha=0.2, linewidth=0.5)
    plt.tight_layout()
    out = os.path.join(out_dir, 'latent_umap.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")

    # ── Plot 2: by digit class ──
    digit_labels_arr = np.array(digit_labels)
    class_colors = plt.cm.tab10(np.linspace(0, 1, 10))
    fig, ax = plt.subplots(figsize=(8, 6))
    for c in range(10):
        mask = digit_labels_arr == c
        if mask.sum() == 0:
            continue
        ax.scatter(z_2d[mask, 0], z_2d[mask, 1],
                   c=[class_colors[c]], label=str(c),
                   alpha=0.4, s=3, linewidths=0)
    ax.set_xlabel('Dimension 1', fontsize=11)
    ax.set_ylabel('Dimension 2', fontsize=11)
    ax.set_title('Latent space: digit class' + (' (UMAP)' if UMAP_AVAILABLE else ' (PCA)'),
                 fontsize=12, fontweight='bold')
    ax.legend(title='Digit', fontsize=8, markerscale=4,
              loc='upper right', ncol=2)
    ax.grid(True, alpha=0.2, linewidth=0.5)
    plt.tight_layout()
    out = os.path.join(out_dir, 'latent_umap_class.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


def plot_loss_curves(history, out_dir='results'):
    """Training and validation ELBO loss curves over epochs."""
    os.makedirs(out_dir, exist_ok=True)
    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history['train_loss'], label='Train ELBO', color='#4C9BE8')
    axes[0].plot(epochs, history['val_loss'],   label='Val ELBO',   color='#E8794C')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].set_title('ELBO loss (train vs. val)', fontweight='bold')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history['train_recon'], label='Reconstruction', color='#4C9BE8')
    axes[1].plot(epochs, history['train_kl'],    label='KL divergence',  color='#9B4CE8')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Loss component')
    axes[1].set_title('ELBO components (train)', fontweight='bold')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(out_dir, 'loss_curves.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


def plot_roc_curves(roc_data, out_dir='results'):
    """3×3 grid of ROC curves: metric (rows) × corruption type (cols)."""
    os.makedirs(out_dir, exist_ok=True)
    metrics     = ['kl', 'mahal', 'recon_prob']
    c_types_ord = ['noise', 'label', 'ood']
    c_labels    = {'noise': 'Gaussian noise', 'label': 'Label flip', 'ood': 'OOD'}

    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    colors = {'noise': '#E8794C', 'label': '#E84C6A', 'ood': '#9B4CE8'}

    for ri, metric in enumerate(metrics):
        for ci, c_type in enumerate(c_types_ord):
            ax = axes[ri][ci]
            fpr, tpr, auc = roc_data[(metric, c_type)]
            ax.plot(fpr, tpr, color=colors[c_type], lw=1.8,
                    label=f'AUC = {auc:.3f}')
            ax.plot([0, 1], [0, 1], 'k--', lw=0.8, alpha=0.5)
            ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
            ax.grid(True, alpha=0.2)
            ax.legend(fontsize=9, loc='lower right')
            if ri == 0: ax.set_title(c_labels[c_type], fontsize=10, fontweight='bold')
            if ci == 0: ax.set_ylabel(METRIC_LABELS[metric], fontsize=9)
            if ri == 2: ax.set_xlabel('FPR', fontsize=9)

    plt.suptitle('ROC curves: audit metric × corruption type', fontsize=13,
                 fontweight='bold', y=1.01)
    plt.tight_layout()
    out = os.path.join(out_dir, 'roc_curves.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


def plot_score_distributions(scores, c_types, out_dir='results'):
    """Boxplot distributions of each audit score broken down by corruption type."""
    os.makedirs(out_dir, exist_ok=True)
    c_types_arr = np.array(c_types)
    order       = ['clean', 'noise', 'label', 'ood']
    colors      = [CORRUPTION_COLORS[c] for c in order]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, metric in zip(axes, ['kl', 'mahal', 'recon_prob']):
        data_by_type = []
        labels_by_type = []
        for ctype in order:
            mask = c_types_arr == ctype
            if mask.sum() > 0:
                data_by_type.append(scores[metric][mask])
                labels_by_type.append(CORRUPTION_LABELS[ctype])

        bp = ax.boxplot(data_by_type, labels=labels_by_type,
                        patch_artist=True, showfliers=False,
                        medianprops={'color': 'black', 'linewidth': 1.5})
        for patch, color in zip(bp['boxes'], [colors[order.index(c)]
                                               for c in order
                                               if (c_types_arr == c).sum() > 0]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_title(METRIC_LABELS[metric], fontsize=10, fontweight='bold')
        ax.set_ylabel('Score (higher = more suspicious)', fontsize=9)
        ax.tick_params(axis='x', labelrotation=15, labelsize=8)
        ax.grid(True, axis='y', alpha=0.3)

    plt.suptitle('Audit score distributions by corruption type', fontsize=12,
                 fontweight='bold')
    plt.tight_layout()
    out = os.path.join(out_dir, 'score_distributions.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


def plot_reconstructions(model, dataset, c_types, n_per_type=8,
                         device='cpu', out_dir='results'):
    """
    Show original vs. reconstructed images for each corruption type.
    Samples the n_per_type highest-scoring (most suspicious) examples per type.
    """
    import torch
    os.makedirs(out_dir, exist_ok=True)
    c_types_arr = np.array(c_types)
    order       = ['clean', 'noise', 'label', 'ood']

    model.eval()
    fig, axes = plt.subplots(len(order) * 2, n_per_type,
                             figsize=(n_per_type * 1.2, len(order) * 2.4))

    with torch.no_grad():
        for row_pair, ctype in enumerate(order):
            idx_list = np.where(c_types_arr == ctype)[0]
            if len(idx_list) == 0:
                continue
            chosen = idx_list[:n_per_type]

            for col, idx in enumerate(chosen):
                x, _, _ = dataset[idx]
                x_t   = x.unsqueeze(0).to(device)
                x_flat = x_t.view(1, -1)
                mu, logv = model.encoder(x_flat)
                x_hat = model.decoder(mu)   # deterministic encoding

                orig = x.squeeze().numpy()
                recon = x_hat.squeeze().cpu().numpy().reshape(28, 28)

                r_orig  = row_pair * 2
                r_recon = row_pair * 2 + 1

                axes[r_orig][col].imshow(orig,  cmap='gray', vmin=0, vmax=1)
                axes[r_recon][col].imshow(recon, cmap='gray', vmin=0, vmax=1)
                for r in [r_orig, r_recon]:
                    axes[r][col].axis('off')

            label_str = CORRUPTION_LABELS[ctype]
            axes[row_pair * 2][0].set_ylabel(f'{label_str}\n(orig)',
                                              fontsize=7, rotation=90, labelpad=2)
            axes[row_pair * 2 + 1][0].set_ylabel('recon', fontsize=7,
                                                   rotation=90, labelpad=2)

    plt.suptitle('Original vs. reconstructed images by corruption type',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    out = os.path.join(out_dir, 'reconstructions.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")