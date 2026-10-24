"""Concentration analysis: how much of the *realized* attack success sits in
the top-k% of a vulnerability ranking?

If vulnerability were uniform, the top k% of any ranking would hold ~k% of
successful identifications (the random-ranking baseline). Concentration well
above that line is what makes screening useful: a defender can act on a small
flagged set and remove a large share of the actual leakage.
"""
from __future__ import annotations

import numpy as np

DEFAULT_KS = (1, 5, 10, 20)


def concentration(member_scores: np.ndarray, member_success: np.ndarray,
                  ks=DEFAULT_KS, n_perm: int = 200, seed: int = 0) -> dict:
    """member_scores: ranking signal for members only (higher = flagged first).
    member_success: bool, whether the attack identified this member.
    Returns {k: share_of_success_in_top_k} plus a permutation baseline."""
    n = len(member_scores)
    total = int(member_success.sum())
    order = np.argsort(-member_scores)
    sorted_success = member_success[order]
    out = {}
    for k in ks:
        m = max(1, int(round(n * k / 100.0)))
        out[f"top{k}"] = float(sorted_success[:m].sum() / total) if total else 0.0
    rng = np.random.default_rng(seed)
    base = {f"top{k}": [] for k in ks}
    for _ in range(n_perm):
        perm = rng.permutation(n)
        ps = member_success[perm]
        for k in ks:
            m = max(1, int(round(n * k / 100.0)))
            base[f"top{k}"].append(float(ps[:m].sum() / total) if total else 0.0)
    baseline = {k: float(np.mean(v)) for k, v in base.items()}
    baseline_sd = {k: float(np.std(v)) for k, v in base.items()}
    return {"share": out, "random_baseline_mean": baseline,
            "random_baseline_sd": baseline_sd, "n_members": n,
            "n_success": total}
