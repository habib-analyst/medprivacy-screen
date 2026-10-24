"""End-to-end screening experiment.

Protocol (per target seed):
  1. Shadow stage: K shadow models on random halves of a fixed pool.
     -> per-example shadow signals (gap, atypicality, LiRA parameters).
  2. Target stage: a target model trains on a random half of the pool
     (its 'patients'); the other half are known non-members.
     -> two live attacks (loss-threshold, LiRA) are run for real.
     -> concentration of each attack's realized success in the top-k% of
        each screener ranking, vs a permutation (random-ranking) baseline.
  3. Mitigation: drop the top 5% / 10% most vulnerable members (combined
     ranking), retrain from scratch, re-run both attacks and test accuracy.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import attacks, screen, signals
from .model import accuracy, train_model, true_class_stats

FULL = dict(pool_size=3072, k_shadows=20, epochs=40, batch=128,
            target_seeds=(0, 1), fpr=0.10, mitigation_ks=(5, 10))
QUICK = dict(pool_size=768, k_shadows=6, epochs=4, batch=128,
             target_seeds=(0,), fpr=0.10, mitigation_ks=(10,))


def _concentration_block(member_signals: dict, success: np.ndarray,
                         seed: int) -> dict:
    return {name: screen.concentration(sig, success, seed=seed)
            for name, sig in member_signals.items()}


def run_experiment(X_pool, y_pool, X_test, y_test, cfg: dict,
                   out_dir: str | None = None) -> dict:
    t0 = time.time()
    rng = np.random.default_rng(12345)
    pool_idx = rng.permutation(len(X_pool))[: cfg["pool_size"]]
    Xp, yp = X_pool[pool_idx], y_pool[pool_idx]
    N = len(Xp)

    # ---- Stage 1: shadows -------------------------------------------------
    membership = signals.shadow_membership(N, cfg["k_shadows"], seed=999)
    shadow_logits = np.zeros((cfg["k_shadows"], N))
    for k in range(cfg["k_shadows"]):
        m = train_model(Xp[membership[k]], yp[membership[k]],
                        seed=10_000 + k, epochs=cfg["epochs"], batch=cfg.get("batch", 256))
        shadow_logits[k] = true_class_stats(m, Xp, yp)["logit"]
    sh = signals.shadow_signals(shadow_logits, membership)

    results = {"config": {k: (list(v) if isinstance(v, tuple) else v)
                          for k, v in cfg.items()},
               "pool_size": N, "per_seed": {}, "mitigation": {}}

    for s in cfg["target_seeds"]:
        srng = np.random.default_rng(50_000 + s)
        member_mask = np.zeros(N, dtype=bool)
        member_mask[srng.permutation(N)[: N // 2]] = True

        target = train_model(Xp[member_mask], yp[member_mask],
                             seed=20_000 + s, epochs=cfg["epochs"], batch=cfg.get("batch", 256))
        tstats = true_class_stats(target, Xp, yp)
        test_acc = accuracy(target, X_test, y_test)

        loss_scores = attacks.loss_attack_scores(tstats["loss"])
        lira_scores = attacks.lira_attack_scores(
            tstats["logit"], sh["mu_in"], sh["sd_in"], sh["mu_out"], sh["sd_out"])
        ev_loss = attacks.evaluate_attack(loss_scores, member_mask, cfg["fpr"])
        ev_lira = attacks.evaluate_attack(lira_scores, member_mask, cfg["fpr"])

        # Screener rankings over members only (flagging a training set).
        midx = np.where(member_mask)[0]
        sig_members = {
            "target_loss": -tstats["loss"][midx],
            "shadow_gap": sh["shadow_gap"][midx],
            "atypicality": sh["atypicality"][midx],
        }
        sig_members["combined"] = signals.combined_rank(
            sig_members["target_loss"], sig_members["shadow_gap"])

        seed_res = {
            "n_members": int(member_mask.sum()),
            "target_test_accuracy": test_acc,
            "attack_loss": {k: v for k, v in ev_loss.items()
                            if k not in ("success",)},
            "attack_lira": {k: v for k, v in ev_lira.items()
                            if k not in ("success",)},
            "signal_agreement_spearman": {
                "target_loss_vs_shadow_gap": signals.spearman(
                    sig_members["target_loss"], sig_members["shadow_gap"]),
                "target_loss_vs_atypicality": signals.spearman(
                    sig_members["target_loss"], sig_members["atypicality"]),
                "shadow_gap_vs_atypicality": signals.spearman(
                    sig_members["shadow_gap"], sig_members["atypicality"]),
            },
            "signal_auc_for_lira_success": {
                name: float(roc_auc_score(ev_lira["success"][midx], sig))
                for name, sig in sig_members.items()},
            "signal_auc_for_loss_success": {
                name: float(roc_auc_score(ev_loss["success"][midx], sig))
                for name, sig in sig_members.items()},
            "concentration_loss_attack": _concentration_block(
                sig_members, ev_loss["success"][midx], seed=s),
            "concentration_lira_attack": _concentration_block(
                sig_members, ev_lira["success"][midx], seed=s),
        }
        results["per_seed"][str(s)] = seed_res

        if out_dir:
            df = pd.DataFrame({
                "pool_index": midx,
                "label": yp[midx],
                "sig_target_loss": sig_members["target_loss"],
                "sig_shadow_gap": sig_members["shadow_gap"],
                "sig_atypicality": sig_members["atypicality"],
                "sig_combined": sig_members["combined"],
                "lira_success": ev_lira["success"][midx],
                "loss_success": ev_loss["success"][midx],
            })
            df.to_csv(f"{out_dir}/scores_seed{s}.csv", index=False)
            n_flag = max(1, int(round(len(df) * 0.10)))
            df.sort_values("sig_combined", ascending=False).head(n_flag).to_csv(
                f"{out_dir}/flagged_top10_seed{s}.csv", index=False)

        # ---- Stage 3: mitigation -----------------------------------------
        combined_pool = np.full(N, np.nan)
        combined_pool[midx] = sig_members["combined"]
        order = midx[np.argsort(-sig_members["combined"])]
        for k_pct in cfg["mitigation_ks"]:
            n_drop = int(round(len(midx) * k_pct / 100.0))
            dropped = order[:n_drop]
            kept_mask = member_mask.copy()
            kept_mask[dropped] = False
            mit = train_model(Xp[kept_mask], yp[kept_mask],
                              seed=30_000 + s, epochs=cfg["epochs"], batch=cfg.get("batch", 256))
            mstats = true_class_stats(mit, Xp, yp)
            m_loss = attacks.evaluate_attack(
                attacks.loss_attack_scores(mstats["loss"]), kept_mask,
                cfg["fpr"])
            m_lira_scores = attacks.lira_attack_scores(
                mstats["logit"], sh["mu_in"], sh["sd_in"],
                sh["mu_out"], sh["sd_out"])
            m_lira = attacks.evaluate_attack(m_lira_scores, kept_mask,
                                             cfg["fpr"])
            kept_idx = np.where(kept_mask)[0]
            base_on_kept = {
                "loss_identified": int(ev_loss["success"][kept_idx].sum()),
                "lira_identified": int(ev_lira["success"][kept_idx].sum()),
            }
            results["mitigation"][f"seed{s}_drop{k_pct}"] = {
                "n_dropped": int(n_drop),
                "baseline": {
                    "identified_total_loss": ev_loss["n_identified"],
                    "identified_total_lira": ev_lira["n_identified"],
                    "identified_on_kept_loss": base_on_kept["loss_identified"],
                    "identified_on_kept_lira": base_on_kept["lira_identified"],
                    "test_accuracy": test_acc,
                },
                "after": {
                    "identified_total_loss": m_loss["n_identified"],
                    "identified_total_lira": m_lira["n_identified"],
                    "attack_auc_loss": m_loss["auc"],
                    "attack_auc_lira": m_lira["auc"],
                    "tpr_at_fpr_loss": m_loss["tpr_at_fpr"],
                    "tpr_at_fpr_lira": m_lira["tpr_at_fpr"],
                    "test_accuracy": accuracy(mit, X_test, y_test),
                },
            }

    results["runtime_sec"] = round(time.time() - t0, 1)
    results["aggregate"] = _aggregate(results)
    return results


def _aggregate(results: dict) -> dict:
    """Mean headline numbers across target seeds."""
    seeds = list(results["per_seed"].values())
    agg = {
        "attack_auc_loss": float(np.mean([s["attack_loss"]["auc"]
                                          for s in seeds])),
        "attack_auc_lira": float(np.mean([s["attack_lira"]["auc"]
                                          for s in seeds])),
        "target_test_accuracy": float(np.mean([s["target_test_accuracy"]
                                               for s in seeds])),
        "spearman_target_loss_vs_shadow_gap": float(np.mean(
            [s["signal_agreement_spearman"]["target_loss_vs_shadow_gap"]
             for s in seeds])),
        "concentration_lira_combined": {},
        "concentration_loss_combined": {},
        "concentration_lira_shadow_gap": {},
    }
    for k in ("top1", "top5", "top10", "top20"):
        agg["concentration_lira_combined"][k] = float(np.mean(
            [s["concentration_lira_attack"]["combined"]["share"][k]
             for s in seeds]))
        agg["concentration_loss_combined"][k] = float(np.mean(
            [s["concentration_loss_attack"]["combined"]["share"][k]
             for s in seeds]))
        agg["concentration_lira_shadow_gap"][k] = float(np.mean(
            [s["concentration_lira_attack"]["shadow_gap"]["share"][k]
             for s in seeds]))
    rb = seeds[0]["concentration_lira_attack"]["combined"]["random_baseline_mean"]
    agg["random_baseline_lira"] = rb
    return agg
