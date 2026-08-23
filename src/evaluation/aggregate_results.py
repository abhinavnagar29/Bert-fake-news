"""Scan a results/ directory tree for metrics.json files and build a single
comparison table (CSV + printed markdown) across all trained models.

Usage:
    python -m src.evaluation.aggregate_results --results_dir results --out results/comparison_table.csv
"""
import argparse
import glob
import json
import os

import pandas as pd


def load_all_metrics(results_dir: str) -> pd.DataFrame:
    rows = []
    for path in glob.glob(os.path.join(results_dir, "**", "metrics.json"), recursive=True):
        with open(path) as f:
            m = json.load(f)
        rows.append(
            {
                "run_dir": os.path.relpath(os.path.dirname(path), results_dir),
                "model": m.get("model", "unknown"),
                "seed": m.get("seed"),
                "train_fraction": m.get("train_fraction", 1.0),
                "accuracy": m.get("accuracy"),
                "precision": m.get("precision_weighted"),
                "recall": m.get("recall_weighted"),
                "f1": m.get("f1_weighted"),
            }
        )
    return pd.DataFrame(rows)


def summarize_by_model(df: pd.DataFrame) -> pd.DataFrame:
    """Mean +/- std F1/accuracy per model (collapses multi-seed runs)."""
    grouped = df.groupby("model").agg(
        n_runs=("f1", "count"),
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        f1_mean=("f1", "mean"),
        f1_std=("f1", "std"),
    ).reset_index()
    return grouped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", type=str, default="results")
    ap.add_argument("--out", type=str, default="results/comparison_table.csv")
    args = ap.parse_args()

    df = load_all_metrics(args.results_dir)
    if df.empty:
        print(f"No metrics.json files found under {args.results_dir}/. Run some training scripts first.")
        return

    df.to_csv(args.out, index=False)
    print(f"Wrote raw comparison table to {args.out}")

    summary = summarize_by_model(df)
    summary_path = os.path.splitext(args.out)[0] + "_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Wrote per-model summary to {summary_path}\n")
    print(summary.to_markdown(index=False))


if __name__ == "__main__":
    main()
