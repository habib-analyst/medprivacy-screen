"""Small CNN classifier + training/scoring helpers (CPU, torch)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(2)


class SmallCNN(nn.Module):
    """Compact CNN for 1x28x28 inputs, ~50k params. Deliberately ordinary:
    the screener must work on the kind of model a small clinic team trains."""

    def __init__(self, n_classes: int = 2):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, n_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 2)  # 14x14
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), 2)  # 7x7
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x).flatten(1)
        return self.fc(x)


def train_model(X: np.ndarray, y: np.ndarray, seed: int, epochs: int = 12,
                batch: int = 256, lr: float = 1e-3) -> SmallCNN:
    """Train a SmallCNN on (X, y) with a fixed seed. Returns the model in eval mode."""
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = SmallCNN(n_classes=int(y.max()) + 1)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xt = torch.from_numpy(X)
    yt = torch.from_numpy(y)
    n = len(X)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss = F.cross_entropy(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
    model.eval()
    return model


@torch.no_grad()
def true_class_stats(model: SmallCNN, X: np.ndarray, y: np.ndarray,
                     batch: int = 512):
    """Per-example statistics under `model`.

    Returns dict with:
      prob  : P(true class)
      logit : log(p/(1-p)) of the true class (LiRA-style logit)
      loss  : cross-entropy loss of the true class
    """
    model.eval()
    probs = []
    Xt = torch.from_numpy(X)
    for i in range(0, len(X), batch):
        logits = model(Xt[i:i + batch])
        probs.append(F.softmax(logits, dim=1).numpy())
    P = np.concatenate(probs, axis=0)
    p = P[np.arange(len(y)), y]
    p_clip = np.clip(p, 1e-6, 1 - 1e-6)
    logit = np.log(p_clip / (1 - p_clip))
    loss = -np.log(p_clip)
    return {"prob": p.astype(np.float64), "logit": logit.astype(np.float64),
            "loss": loss.astype(np.float64)}


@torch.no_grad()
def accuracy(model: SmallCNN, X: np.ndarray, y: np.ndarray,
             batch: int = 512) -> float:
    model.eval()
    Xt = torch.from_numpy(X)
    preds = []
    for i in range(0, len(X), batch):
        preds.append(model(Xt[i:i + batch]).argmax(1).numpy())
    return float((np.concatenate(preds) == y).mean())
