"""TF-IDF + Logistic Regression baseline.

Usage:
    python -m src.training.train_baseline --data_dir data/fakeddit --out_dir results/baseline
"""
import argparse
import os
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.data.preprocess import prepare_dataset
from src.evaluation.metrics import compute_metrics, plot_confusion_matrix, save_metrics
from src.utils.seed import set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="results/baseline")
    ap.add_argument("--n_per_class", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_features", type=int, default=20000)
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    train_df, test_df, stats = prepare_dataset(args.data_dir, n_per_class=args.n_per_class, seed=args.seed)
    print(f"Train: {stats.train_size} | Test: {stats.test_size}")

    vectorizer = TfidfVectorizer(max_features=args.max_features, ngram_range=(1, 2), stop_words="english")
    X_train = vectorizer.fit_transform(train_df["clean_title"])
    X_test = vectorizer.transform(test_df["clean_title"])

    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=args.seed)
    t0 = time.time()
    clf.fit(X_train, train_df["label"])
    train_time = time.time() - t0

    preds = clf.predict(X_test)
    metrics = compute_metrics(test_df["label"].tolist(), preds.tolist())

    print(f"Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1_weighted']:.4f}")
    print(metrics["classification_report"])

    plot_confusion_matrix(
        metrics["confusion_matrix"],
        os.path.join(args.out_dir, "confusion_matrix.png"),
        title="TF-IDF + Logistic Regression -- Confusion Matrix",
    )
    save_metrics(
        metrics,
        os.path.join(args.out_dir, "metrics.json"),
        extra={"model": "tfidf_logreg", "train_time_sec": train_time, "seed": args.seed},
    )
    print(f"Saved results to {args.out_dir}/")


if __name__ == "__main__":
    main()
