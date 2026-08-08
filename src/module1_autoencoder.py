"""
module1_autoencoder.py
Module 1 — fifth anomaly method: an autoencoder (PyTorch).

Train an autoencoder to reconstruct CLEAN reference data. Normal rows reconstruct
well (low error); anomalous rows reconstruct poorly (high error). The reconstruction
error is the anomaly score. This complements the other detectors: it is a non-linear,
reconstruction-based method (Sakurada & Yairi, 2014; the deep-AD family in Pang et al., 2021).

Author: Neha Pandit 
"""

import numpy as np

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

SEED = 42


def _normalise(x):
    x = np.asarray(x, dtype=float)
    lo, hi = x.min(), x.max()
    return np.zeros_like(x) if hi - lo == 0 else (x - lo) / (hi - lo)


if _HAS_TORCH:

    class Autoencoder(nn.Module):
        """Small symmetric autoencoder: input -> hidden -> bottleneck -> hidden -> input."""

        def __init__(self, input_dim, hidden=16, bottleneck=4):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, bottleneck), nn.ReLU(),
            )
            self.decoder = nn.Sequential(
                nn.Linear(bottleneck, hidden), nn.ReLU(),
                nn.Linear(hidden, input_dim),
            )

        def forward(self, x):
            return self.decoder(self.encoder(x))

    def train_autoencoder(X_ref, epochs=20, batch_size=512, lr=1e-3, hidden=None, bottleneck=None):
   
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        X = torch.tensor(np.asarray(X_ref), dtype=torch.float32)
        input_dim = X.shape[1]
        if hidden is None:
            hidden = max(8, input_dim * 2)
        if bottleneck is None:
            bottleneck = max(1, input_dim // 2)   # strictly smaller than input_dim
        model = Autoencoder(input_dim, hidden=hidden, bottleneck=bottleneck)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        n = len(X)
        model.train()
        for _ in range(epochs):
            perm = torch.randperm(n)
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                batch = X[idx]
                opt.zero_grad()
                loss = loss_fn(model(batch), batch)
                loss.backward()
                opt.step()
        return model

    def score_autoencoder(model, X):
        """Anomaly score in [0,1] from per-row reconstruction error (higher = anomalous)."""
        model.eval()
        with torch.no_grad():
            Xt = torch.tensor(np.asarray(X), dtype=torch.float32)
            recon = model(Xt)
            err = ((recon - Xt) ** 2).mean(dim=1).numpy()
        return _normalise(err)

else:  # pragma: no cover - torch missing

    def train_autoencoder(*a, **k):
        raise ImportError("PyTorch not installed. Run: pip install torch")

    def score_autoencoder(*a, **k):
        raise ImportError("PyTorch not installed. Run: pip install torch")
