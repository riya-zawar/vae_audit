"""Training loop with ELBO loss.

train.py — VAE training loop.

Pseudocode (maps directly to paper Methodology §3.3):

    INITIALISE VAE(latent_dim=20), Adam(lr=1e-3)
    FOR epoch = 1 to N_EPOCHS:
        FOR each batch (x, _, _) in train_loader:     ← labels/flags ignored
            x_hat, mu, logv, z = VAE.forward(x)
            loss, recon, kl    = ELBO(x_hat, x, mu, logv, beta)
            loss.backward()
            optimizer.step()
        val_loss = EVALUATE(val_loader)
        CHECKPOINT if val_loss improves
    RETURN best model

Note: corruption flags are in the dataset but deliberately excluded from
training — the VAE sees only (image, label) and learns the clean manifold
structure. Labels are also unused during VAE training.
"""

import os
import torch
from torch.utils.data import DataLoader, random_split

from models.vae import VAE, elbo_loss


def train_vae(
    dataset,
    latent_dim      = 20,
    beta            = 1.0,
    epochs          = 50,
    batch_size      = 128,
    lr              = 1e-3,
    val_split       = 0.1,
    dropout         = 0.2,
    save_path       = 'results/vae_best.pt',
    device          = None,
    seed            = 42,
    verbose         = True
):
    """
    Train a VAE on the corrupted dataset.

    The model has no access to corruption flags — it learns from images only.
    Best checkpoint is saved based on validation ELBO.

    Args:
        dataset    : TensorDataset(images, labels, flags)
        latent_dim : dimensionality of z (default 20)
        beta       : KL weight in ELBO (1.0 = standard VAE)
        epochs     : training epochs
        batch_size : mini-batch size
        lr         : Adam learning rate
        val_split  : fraction of data held out for validation
        dropout    : encoder dropout rate
        save_path  : where to save best model weights
        device     : 'cpu' | 'cuda' | None (auto-detect)
        seed       : random seed
        verbose    : print epoch stats

    Returns:
        model      : trained VAE (best checkpoint loaded)
        history    : dict with lists of train/val losses per epoch
    """
    torch.manual_seed(seed)

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on: {device}")

    # --- Train / val split ---
    N_val   = int(len(dataset) * val_split)
    N_train = len(dataset) - N_val
    train_ds, val_ds = random_split(
        dataset, [N_train, N_val],
        generator=torch.Generator().manual_seed(seed)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=(device == 'cuda'))
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=0)

    # --- Model & optimiser ---
    model     = VAE(latent_dim=latent_dim, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, verbose=verbose
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    history    = {'train_loss': [], 'val_loss': [],
                  'train_recon': [], 'train_kl': []}
    best_val   = float('inf')

    for epoch in range(1, epochs + 1):
        # ── Training ──
        model.train()
        t_loss = t_recon = t_kl = 0.0

        for batch in train_loader:
            x, _, _ = batch          # labels and flags deliberately ignored
            x = x.to(device)

            optimizer.zero_grad()
            x_hat, mu, logv, z = model(x)
            loss, recon, kl    = elbo_loss(x_hat, x, mu, logv, beta=beta)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            t_loss  += loss.item()
            t_recon += recon.item()
            t_kl    += kl.item()

        n_batches   = len(train_loader)
        t_loss     /= n_batches
        t_recon    /= n_batches
        t_kl       /= n_batches

        # ── Validation ──
        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x, _, _ = batch
                x = x.to(device)
                x_hat, mu, logv, z = model(x)
                loss, _, _         = elbo_loss(x_hat, x, mu, logv, beta=beta)
                v_loss            += loss.item()
        v_loss /= len(val_loader)

        scheduler.step(v_loss)

        history['train_loss'].append(t_loss)
        history['val_loss'].append(v_loss)
        history['train_recon'].append(t_recon)
        history['train_kl'].append(t_kl)

        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(f"Epoch {epoch:3d}/{epochs} | "
                  f"Train ELBO: {t_loss:.2f} "
                  f"(Recon: {t_recon:.2f}, KL: {t_kl:.2f}) | "
                  f"Val ELBO: {v_loss:.2f}")

        # ── Checkpoint ──
        if v_loss < best_val:
            best_val = v_loss
            torch.save({
                'epoch':      epoch,
                'model_state': model.state_dict(),
                'val_loss':   best_val,
                'config': {
                    'latent_dim': latent_dim,
                    'beta':       beta,
                    'dropout':    dropout,
                }
            }, save_path)

    # Load best weights
    ckpt = torch.load(save_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    print(f"\nBest val ELBO: {best_val:.2f} at epoch {ckpt['epoch']}")
    print(f"Checkpoint saved: {save_path}")

    return model, history