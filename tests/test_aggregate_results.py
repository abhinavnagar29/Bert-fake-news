import json
import os

from src.evaluation.aggregate_results import load_all_metrics, summarize_by_model


def _write_metrics(path, **kwargs):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(kwargs, f)


def test_load_all_metrics_finds_nested_runs(tmp_path):
    root = str(tmp_path)
    _write_metrics(os.path.join(root, "bert_base_seed1", "metrics.json"), model="bert-base-uncased", seed=1, accuracy=0.84, f1_weighted=0.83, precision_weighted=0.82, recall_weighted=0.84)
    _write_metrics(os.path.join(root, "bert_base_seed2", "metrics.json"), model="bert-base-uncased", seed=2, accuracy=0.85, f1_weighted=0.84, precision_weighted=0.83, recall_weighted=0.85)
    _write_metrics(os.path.join(root, "baseline", "metrics.json"), model="tfidf_logreg", seed=42, accuracy=0.72, f1_weighted=0.71, precision_weighted=0.70, recall_weighted=0.72)

    df = load_all_metrics(root)
    assert len(df) == 3
    assert set(df["model"]) == {"bert-base-uncased", "tfidf_logreg"}


def test_summarize_by_model_averages_across_seeds(tmp_path):
    root = str(tmp_path)
    _write_metrics(os.path.join(root, "r1", "metrics.json"), model="bert-base-uncased", seed=1, accuracy=0.80, f1_weighted=0.80, precision_weighted=0.80, recall_weighted=0.80)
    _write_metrics(os.path.join(root, "r2", "metrics.json"), model="bert-base-uncased", seed=2, accuracy=0.90, f1_weighted=0.90, precision_weighted=0.90, recall_weighted=0.90)

    df = load_all_metrics(root)
    summary = summarize_by_model(df)
    row = summary[summary["model"] == "bert-base-uncased"].iloc[0]
    assert row["n_runs"] == 2
    assert abs(row["f1_mean"] - 0.85) < 1e-9


def test_load_all_metrics_empty_dir(tmp_path):
    df = load_all_metrics(str(tmp_path))
    assert df.empty
