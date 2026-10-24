#!/usr/bin/env python3
"""Run the medprivacy-screen experiment and write results.json + artifacts.

Usage: python run_experiments.py [--quick] [--data data/pneumoniamnist.npz]
                                  [--out outputs]
"""
from __future__ import annotations

import argparse
import json
import os

from medprivacy.data import load_medmnist_npz
from medprivacy.experiment import FULL, QUICK, run_experiment


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--data", default="data/pneumoniamnist.npz")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--results", default="results.json")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    X_pool, y_pool, X_test, y_test = load_medmnist_npz(args.data)
    cfg = QUICK if args.quick else FULL
    results = run_experiment(X_pool, y_pool, X_test, y_test, cfg,
                             out_dir=args.out)
    results["dataset"] = "PneumoniaMNIST (MedMNIST v2, train+val pooled)"
    with open(args.results, "w") as f:
        json.dump(results, f, indent=2)
    agg = results["aggregate"]
    print(json.dumps(agg, indent=2))
    print(f"runtime_sec={results['runtime_sec']}")


if __name__ == "__main__":
    main()
