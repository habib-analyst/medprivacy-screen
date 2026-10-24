"""Dataset loading for MedMNIST .npz files (PneumoniaMNIST by default)."""
from __future__ import annotations

import numpy as np


def load_medmnist_npz(path: str):
    """Return (train_X, train_y, test_X, test_y) as float32 images in [0,1]
    shaped (N, 1, 28, 28) and int64 labels."""
    d = np.load(path)
    tr_x = d["train_images"].astype(np.float32) / 255.0
    va_x = d["val_images"].astype(np.float32) / 255.0
    te_x = d["test_images"].astype(np.float32) / 255.0
    # Pool train+val: both are 'candidate training patients'; the official
    # test split stays untouched as the utility holdout.
    x_pool = np.concatenate([tr_x, va_x], axis=0)
    y_pool = np.concatenate(
        [d["train_labels"].ravel(), d["val_labels"].ravel()]
    ).astype(np.int64)
    y_test = d["test_labels"].ravel().astype(np.int64)
    x_pool = x_pool[:, None, :, :]
    te_x = te_x[:, None, :, :]
    return x_pool, y_pool, te_x, y_test
