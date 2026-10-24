import numpy as np

from medprivacy.signals import (combined_rank, rank_pct, shadow_membership,
                                shadow_signals, spearman)


def test_shadow_membership_balanced_counts():
    M = shadow_membership(200, 12, seed=1)
    assert M.shape == (12, 200)
    in_counts = M.sum(axis=0)
    # balanced design: every example is IN exactly half the shadows
    assert (in_counts == 6).all()
    # each shadow trains on roughly half the pool
    sizes = M.sum(axis=1)
    assert ((sizes > 70) & (sizes < 130)).all()


def test_shadow_signals_gap_positive_for_memorized():
    rng = np.random.default_rng(0)
    K, N = 10, 50
    M = shadow_membership(N, K, seed=2)
    logits = rng.normal(0, 1, (K, N))
    logits[M] += 2.0  # examples look 'memorized' when IN
    sh = shadow_signals(logits, M)
    assert sh["shadow_gap"].mean() > 1.5
    assert (sh["sd_in"] > 0).all() and (sh["sd_out"] > 0).all()


def test_rank_pct_and_combined():
    s = np.array([0.0, 1.0, 2.0, 3.0])
    rp = rank_pct(s)
    assert rp[0] < rp[-1] and rp.max() == 1.0
    c = combined_rank(s, s[::-1].copy())
    assert c.shape == (4,)


def test_spearman_perfect():
    a = np.arange(20.0)
    assert spearman(a, a) == 1.0
    assert spearman(a, a[::-1].copy()) == -1.0
