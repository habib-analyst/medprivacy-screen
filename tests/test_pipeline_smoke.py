import numpy as np

from medprivacy.experiment import run_experiment


def _toy_data(seed=0):
    """Two Gaussian blobs as fake 'images': learnable, tiny, fast."""
    rng = np.random.default_rng(seed)
    n_pool, n_test = 240, 80
    y_pool = rng.integers(0, 2, n_pool)
    y_test = rng.integers(0, 2, n_test)
    X_pool = rng.normal(0, 1, (n_pool, 1, 28, 28)).astype(np.float32)
    X_test = rng.normal(0, 1, (n_test, 1, 28, 28)).astype(np.float32)
    X_pool[y_pool == 1, 0, :14] += 1.5
    X_test[y_test == 1, 0, :14] += 1.5
    return X_pool, y_pool.astype(np.int64), X_test, y_test.astype(np.int64)


def test_experiment_smoke():
    X_pool, y_pool, X_test, y_test = _toy_data()
    cfg = dict(pool_size=200, k_shadows=4, epochs=2, target_seeds=(0,),
               fpr=0.10, mitigation_ks=(10,))
    res = run_experiment(X_pool, y_pool, X_test, y_test, cfg)
    assert "aggregate" in res
    assert 0.0 <= res["aggregate"]["attack_auc_lira"] <= 1.0
    assert "seed0_drop10" in res["mitigation"]
    shares = res["aggregate"]["concentration_lira_combined"]
    assert set(shares) == {"top1", "top5", "top10", "top20"}
