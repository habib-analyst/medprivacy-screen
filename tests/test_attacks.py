import numpy as np

from medprivacy.attacks import (evaluate_attack, lira_attack_scores,
                                loss_attack_scores, threshold_at_fpr)


def test_loss_attack_scores_negates():
    loss = np.array([0.1, 2.0, 0.5])
    assert (loss_attack_scores(loss) == -loss).all()


def test_threshold_at_fpr():
    rng = np.random.default_rng(0)
    s = rng.normal(0, 1, 10000)
    thr = threshold_at_fpr(s, 0.10)
    assert abs((s > thr).mean() - 0.10) < 0.01


def test_evaluate_attack_perfect_separation():
    scores = np.concatenate([np.ones(50) * 2, np.zeros(50)])
    is_member = np.array([True] * 50 + [False] * 50)
    ev = evaluate_attack(scores, is_member, fpr=0.10)
    assert ev["auc"] == 1.0
    assert ev["n_identified"] == 50
    assert ev["success"].sum() == 50


def test_lira_scores_favor_in_distribution():
    # target logit close to IN mean, far from OUT mean -> positive score
    s = lira_attack_scores(np.array([2.0]), np.array([2.0]), np.array([0.5]),
                           np.array([-2.0]), np.array([0.5]))
    assert s[0] > 0
    s2 = lira_attack_scores(np.array([-2.0]), np.array([2.0]), np.array([0.5]),
                            np.array([-2.0]), np.array([0.5]))
    assert s2[0] < 0
