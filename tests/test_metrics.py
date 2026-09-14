import json
import os

from src.evaluation.metrics import compute_metrics, save_metrics


def test_compute_metrics_perfect_predictions():
    labels = [0, 0, 1, 1]
    preds = [0, 0, 1, 1]
    m = compute_metrics(labels, preds)
    assert m["accuracy"] == 1.0
    assert m["f1_weighted"] == 1.0
    assert m["confusion_matrix"] == [[2, 0], [0, 2]]


def test_compute_metrics_known_confusion_matrix():
    # 2 real correctly predicted, 1 real predicted as fake, 3 fake correct, 1 fake predicted as real
    labels = [0, 0, 0, 1, 1, 1, 1]
    preds = [0, 0, 1, 1, 1, 1, 0]
    m = compute_metrics(labels, preds)
    cm = m["confusion_matrix"]
    # rows = actual [real, fake], cols = predicted [real, fake]
    assert cm[0][0] == 2  # real correctly predicted real
    assert cm[0][1] == 1  # real wrongly predicted fake
    assert cm[1][0] == 1  # fake wrongly predicted real
    assert cm[1][1] == 3  # fake correctly predicted fake
    assert abs(m["accuracy"] - 5 / 7) < 1e-6


def test_save_metrics_writes_valid_json(tmp_path):
    m = {"accuracy": 0.9}
    out_path = os.path.join(str(tmp_path), "metrics.json")
    save_metrics(m, out_path, extra={"model": "test-model"})
    with open(out_path) as f:
        loaded = json.load(f)
    assert loaded["accuracy"] == 0.9
    assert loaded["model"] == "test-model"
