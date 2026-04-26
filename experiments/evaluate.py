"""
evaluate.py — AUROC evaluation of audit metrics against corruption types.

For each metric × corruption type combination, binary AUROC is computed:
    - Positive class: corrupted samples of that type
    - Negative class: all clean samples

This gives a 3 × 3 result matrix:
                   KL div    Mahalanobis   Recon prob
    Gaussian noise  AUC       AUC           AUC
    Label flip      AUC       AUC           AUC
    OOD             AUC       AUC           AUC

AUC = 0.5 → metric cannot detect this corruption type
AUC = 1.0 → metric perfectly separates corrupted from clean
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


CORRUPTION_TYPES = ['noise', 'label', 'ood']
METRIC_NAMES     = ['kl', 'mahal', 'recon_prob']
METRIC_LABELS    = {
    'kl':         'Per-sample KL',
    'mahal':      'Mahalanobis dist.',
    'recon_prob': 'Recon. probability',
}
CORRUPTION_LABELS = {
    'noise': 'Gaussian noise',
    'label': 'Label flip',
    'ood':   'OOD injection',
}


def compute_auroc_matrix(scores, c_types, flags):
    """
    Compute AUROC for every metric × corruption type pair.

    Args:
        scores   : dict {'kl': (N,), 'mahal': (N,), 'recon_prob': (N,)}
        c_types  : list[str] length N — 'clean'|'noise'|'label'|'ood'
        flags    : np.array (N,) — 1 if any corruption, 0 if clean

    Returns:
        results  : pd.DataFrame shape (3 corruptions × 3 metrics) with AUROC
        roc_data : dict of (fpr, tpr) curves for plotting
    """
    c_types_arr = np.array(c_types)
    results     = {}
    roc_data    = {}

    for metric in METRIC_NAMES:
        score_arr   = scores[metric]
        results[metric] = {}

        for c_type in CORRUPTION_TYPES:
            # Binary label: 1 if this specific corruption type, 0 if clean
            # Ignore other corruption types for this binary comparison
            mask = (c_types_arr == c_type) | (c_types_arr == 'clean')
            y_true  = (c_types_arr[mask] == c_type).astype(int)
            y_score = score_arr[mask]

            if y_true.sum() == 0:
                auc = float('nan')
                fpr, tpr = np.array([0, 1]), np.array([0, 1])
            else:
                auc = roc_auc_score(y_true, y_score)
                fpr, tpr, _ = roc_curve(y_true, y_score)

            results[metric][c_type] = round(auc, 4)
            roc_data[(metric, c_type)] = (fpr, tpr, auc)

    df = pd.DataFrame(results).rename(
        index    = CORRUPTION_LABELS,
        columns  = METRIC_LABELS
    )
    return df, roc_data


def combined_score_auroc(scores, c_types):
    """
    Compute AUROC for a combined score = mean-normalised sum of all three metrics.
    Tests whether combining metrics improves detection (cf. Osada et al., 2023).

    Returns:
        combined_auc_per_type : dict {corruption_type: auc}
    """
    c_types_arr = np.array(c_types)

    # Normalise each metric to [0,1] range
    def norm(arr):
        mn, mx = arr.min(), arr.max()
        return (arr - mn) / (mx - mn + 1e-9)

    combined = (
        norm(scores['kl']) +
        norm(scores['mahal']) +
        norm(scores['recon_prob'])
    ) / 3.0

    combined_aucs = {}
    for c_type in CORRUPTION_TYPES:
        mask   = (c_types_arr == c_type) | (c_types_arr == 'clean')
        y_true = (c_types_arr[mask] == c_type).astype(int)
        y_sc   = combined[mask]
        if y_true.sum() == 0:
            combined_aucs[c_type] = float('nan')
        else:
            combined_aucs[c_type] = round(roc_auc_score(y_true, y_sc), 4)

    return combined_aucs


def save_results(df, combined_aucs, roc_data, out_dir='results'):
    """Save AUROC table and ROC curve data to CSV."""
    os.makedirs(out_dir, exist_ok=True)

    # Main table
    csv_path = os.path.join(out_dir, 'auroc_table.csv')
    df.to_csv(csv_path)
    print(f"AUROC table saved: {csv_path}")

    # Combined score row
    combined_row = {
        CORRUPTION_LABELS[c]: combined_aucs[c]
        for c in CORRUPTION_TYPES
    }
    pd.DataFrame([combined_row], index=['Combined score']).to_csv(
        os.path.join(out_dir, 'auroc_combined.csv')
    )

    # ROC curve data
    for (metric, c_type), (fpr, tpr, auc) in roc_data.items():
        fname = os.path.join(out_dir, f'roc_{metric}_{c_type}.csv')
        pd.DataFrame({'fpr': fpr, 'tpr': tpr}).to_csv(fname, index=False)

    return csv_path


def print_results(df, combined_aucs):
    """Pretty-print the result table to console."""
    print("\n" + "=" * 60)
    print("AUROC TABLE (higher = better audit detection)")
    print("=" * 60)
    print(df.to_string())
    print("\nCombined score AUROC (normalised sum of all 3 metrics):")
    for c, auc in combined_aucs.items():
        print(f"  {CORRUPTION_LABELS[c]:20s}: {auc:.4f}")
    print("=" * 60)