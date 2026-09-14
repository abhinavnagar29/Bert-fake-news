"""Metric computation and result-artifact saving (JSON + plots)."""
import json
import os

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_metrics(labels, preds) -> dict:
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision_weighted": precision_score(labels, preds, average="weighted", zero_division=0),
        "recall_weighted": recall_score(labels, preds, average="weighted", zero_division=0),
        "f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0),
        "confusion_matrix": confusion_matrix(labels, preds).tolist(),
        "classification_report": classification_report(
            labels, preds, target_names=["Real (0)", "Fake (1)"], zero_division=0
        ),
    }


def save_metrics(metrics: dict, out_path: str, extra: dict = None):
    payload = dict(metrics)
    if extra:
        payload.update(extra)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    return out_path


def plot_confusion_matrix(cm, out_path: str, title: str = "Confusion Matrix"):
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Predicted Real", "Predicted Fake"],
        yticklabels=["Actual Real", "Actual Fake"],
        ax=ax,
        linewidths=0.5,
    )
    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_training_curves(train_losses, test_losses, train_accs, test_accs, out_path: str):
    import matplotlib.pyplot as plt

    epochs_x = list(range(1, len(train_losses) + 1))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs_x, train_losses, "b-o", label="Train Loss")
    ax1.plot(epochs_x, test_losses, "r-s", label="Test Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.set_title("Loss Curves")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(epochs_x, train_accs, "b-o", label="Train Acc")
    ax2.plot(epochs_x, test_accs, "r-s", label="Test Acc")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy Curves")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_data_scaling_curve(fractions, f1_scores, out_path: str):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot([f * 100 for f in fractions], f1_scores, "o-", linewidth=2, markersize=8)
    ax.set_xlabel("Training set size used (%)")
    ax.set_ylabel("Test F1-score")
    ax.set_title("Performance vs. Training Set Size")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
