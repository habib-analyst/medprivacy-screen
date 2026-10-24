import numpy as np

from medprivacy.screen import concentration


def test_concentration_perfect_ranking():
    n = 1000
    scores = np.arange(n, dtype=float)  # highest score = last example
    success = np.zeros(n, dtype=bool)
    success[-50:] = True  # all successes are exactly the top 5%
    out = concentration(scores, success, seed=0)
    assert out["share"]["top5"] == 1.0
    assert out["share"]["top10"] == 1.0
    assert out["n_success"] == 50


def test_concentration_random_ranking_near_baseline():
    rng = np.random.default_rng(3)
    n = 2000
    scores = rng.normal(size=n)
    success = rng.random(n) < 0.2
    out = concentration(scores, success, seed=1)
    # random ranking: top10 share should be near 0.10, like the baseline
    assert abs(out["share"]["top10"] - 0.10) < 0.05
    assert abs(out["random_baseline_mean"]["top10"] - 0.10) < 0.02


def test_concentration_zero_success_safe():
    out = concentration(np.arange(10.0), np.zeros(10, dtype=bool))
    assert out["n_success"] == 0
    assert out["share"]["top5"] == 0.0
