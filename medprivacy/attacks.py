"""Membership-inference attacks used as the validation oracle.

Two attacks, both standard:
  * Loss-threshold attack (Ye et al.): score = -loss under the target model.
  * Offline LiRA (Carlini et al. 2022): per-example Gaussian likelihood ratio
    of the target model's true-class logit under shadow IN vs OUT statistics.

An attack 'identifies' a member when its score exceeds the threshold that
gives a fixed false-positive rate (default 10%) on known non-members.
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score


def loss_attack_scores(loss: np.ndarray) -> np.ndarray:
    return -loss


def lira_attack_scores(target_logit: np.ndarray, mu_in: np.ndarray,
                       sd_in: np.ndarray, mu_out: np.ndarray,
                       sd_out: np.ndarray) -> np.ndarray:
    """log Pr(logit | IN) - log Pr(logit | OUT), per example."""
    return (stats.norm.logpdf(target_logit, mu_in, sd_in)
            - stats.norm.logpdf(target_logit, mu_out, sd_out))


def threshold_at_fpr(nonmember_scores: np.ndarray, fpr: float) -> float:
    return float(np.quantile(nonmember_scores, 1.0 - fpr))


def evaluate_attack(scores: np.ndarray, is_member: np.ndarray,
                    fpr: float = 0.10) -> dict:
    """Global attack metrics + per-member success indicator."""
    auc = float(roc_auc_score(is_member.astype(int), scores))
    thr = threshold_at_fpr(scores[~is_member], fpr)
    flagged = scores > thr
    success = flagged & is_member
    tpr = float(success.sum() / is_member.sum())
    # TPR at 1% FPR as a secondary operating point
    thr01 = threshold_at_fpr(scores[~is_member], 0.01)
    tpr01 = float(((scores > thr01) & is_member).sum() / is_member.sum())
    return {"auc": auc, "tpr_at_fpr": tpr, "tpr_at_fpr01": tpr01,
            "threshold": thr, "success": success,
            "n_identified": int(success.sum())}
