"""Vulnerability signals.

Two independent signal families are computed:

A) Target-model signals (no extra training): per-example loss / confidence of
   the deployed model on its own training examples. Memorized examples have
   abnormally low loss.

B) Shadow-model signals (independent models, never the target): from K shadow
   models each trained on a random half of the pool we get, per example, the
   distribution of true-class logits when the example was IN vs OUT of
   training.
     - shadow_gap     = mean(logit | IN) - mean(logit | OUT)  (memorization gap)
     - atypicality    = mean loss when OUT (hard/unusual examples leak more)
     - LiRA parameters: (mu_in, sd_in, mu_out, sd_out) for the attack oracle.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def shadow_membership(n_pool: int, k_shadows: int, seed: int) -> np.ndarray:
    """Boolean (K, N) balanced membership matrix: every example is IN
    exactly K//2 shadows and OUT of the rest (randomized per example), so
    every example gets equally reliable IN/OUT statistics. Each shadow's
    training set is then ~half the pool."""
    rng = np.random.default_rng(seed)
    M = np.zeros((k_shadows, n_pool), dtype=bool)
    n_in = k_shadows // 2
    for j in range(n_pool):
        rows = rng.choice(k_shadows, size=n_in, replace=False)
        M[rows, j] = True
    return M


def shadow_signals(logits: np.ndarray, membership: np.ndarray) -> dict:
    """logits: (K, N) true-class logits from each shadow model on every pool
    example. membership: (K, N) bool. Returns per-example signal arrays."""
    K, N = logits.shape
    mu_in = np.zeros(N); sd_in = np.zeros(N)
    mu_out = np.zeros(N); sd_out = np.zeros(N)
    mean_out_loss = np.zeros(N)
    for j in range(N):
        li = logits[membership[:, j], j]
        lo = logits[~membership[:, j], j]
        mu_in[j], mu_out[j] = li.mean(), lo.mean()
        sd_in[j] = li.std(ddof=1) if len(li) > 1 else 1.0
        sd_out[j] = lo.std(ddof=1) if len(lo) > 1 else 1.0
        # loss when unseen: -log sigmoid(logit)
        mean_out_loss[j] = np.logaddexp(0.0, -lo).mean()
    sd_in = np.maximum(sd_in, 1e-3)
    sd_out = np.maximum(sd_out, 1e-3)
    return {
        "shadow_gap": mu_in - mu_out,
        "atypicality": mean_out_loss,
        "mu_in": mu_in, "sd_in": sd_in, "mu_out": mu_out, "sd_out": sd_out,
    }


def rank_pct(scores: np.ndarray) -> np.ndarray:
    """Rank percentile in [0, 1]; higher score -> higher percentile."""
    return stats.rankdata(scores) / len(scores)


def combined_rank(*score_arrays: np.ndarray) -> np.ndarray:
    """Average of rank percentiles across signals (higher = more vulnerable)."""
    return np.mean([rank_pct(s) for s in score_arrays], axis=0)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return float(stats.spearmanr(a, b).statistic)
