"""Command-line interface.

  python -m medprivacy.cli screen --data data/pneumoniamnist.npz \
      --out outputs --results results.json [--quick]
  python -m medprivacy.cli report --results results.json

`screen` runs the full screening protocol on a MedMNIST .npz dataset:
shadow stage, target stage, live-attack validation with concentration
curves, and the remove-top-k mitigation measurement. It writes
results.json plus per-seed score tables and the flagged-patient lists
(outputs/flagged_top10_seed*.csv).
"""
from __future__ import annotations

import argparse
import json
import os

from .data import load_medmnist_npz
from .experiment import FULL, QUICK, run_experiment


def _screen(args):
    os.makedirs(args.out, exist_ok=True)
    X_pool, y_pool, X_test, y_test = load_medmnist_npz(args.data)
    cfg = QUICK if args.quick else FULL
    results = run_experiment(X_pool, y_pool, X_test, y_test, cfg,
                             out_dir=args.out)
    results["dataset"] = "PneumoniaMNIST (MedMNIST v2, train+val pooled)"
    with open(args.results, "w") as f:
        json.dump(results, f, indent=2)
    _print_summary(results)


def _print_summary(results: dict):
    agg = results["aggregate"]
    print("=== medprivacy-screen summary ===")
    print(f"global attack AUC (what existing tools report): "
          f"loss={agg['attack_auc_loss']:.3f}  lira={agg['attack_auc_lira']:.3f}")
    print(f"target test accuracy: {agg['target_test_accuracy']:.3f}")
    print(f"signal agreement (Spearman, target-loss vs shadow-gap): "
          f"{agg['spearman_target_loss_vs_shadow_gap']:.3f}")
    print("concentration of LiRA attack success (combined ranking):")
    for k in ("top1", "top5", "top10", "top20"):
        print(f"  {k}: {agg['concentration_lira_combined'][k]:.3f} "
              f"(random baseline {agg['random_baseline_lira'][k]:.3f})")
    for key, m in results["mitigation"].items():
        b, a = m["baseline"], m["after"]
        print(f"mitigation {key}: identified (LiRA) "
              f"{b['identified_total_lira']} -> {a['identified_total_lira']}, "
              f"test acc {b['test_accuracy']:.3f} -> {a['test_accuracy']:.3f}")


def _report(args):
    with open(args.results) as f:
        _print_summary(json.load(f))


def main():
    ap = argparse.ArgumentParser(prog="medprivacy-screen")
    sub = ap.add_subparsers(required=True)
    p = sub.add_parser("screen")
    p.add_argument("--data", default="data/pneumoniamnist.npz")
    p.add_argument("--out", default="outputs")
    p.add_argument("--results", default="results.json")
    p.add_argument("--quick", action="store_true")
    p.set_defaults(fn=_screen)
    p = sub.add_parser("report")
    p.add_argument("--results", default="results.json")
    p.set_defaults(fn=_report)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
