"""Fine-tune a HuggingFace BertForSequenceClassification-style model.

Works for bert-base-uncased, distilbert-base-uncased, roberta-base, etc.
Supports --seed (multi-seed reporting) and --train_fraction (data-scaling
ablation) so the SAME script drives Task 3, the multi-seed experiment, and
the data-scaling experiment -- just with different CLI flags.

Usage:
    python -m src.training.train_transformer \
        --data_dir data/fakeddit --model_name bert-base-uncased \
        --out_dir results/bert_base --seed 42

    # data-scaling ablation point at 25% of training data
    python -m src.training.train_transformer \
        --data_dir data/fakeddit --model_name bert-base-uncased \
        --out_dir results/bert_base_25pct --train_fraction 0.25
"""
import argparse
import os

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from src.data.dataset import FakedditDataset
from src.data.preprocess import prepare_dataset
from src.evaluation.error_analysis import build_error_report, category_counts
from src.evaluation.metrics import (
    compute_metrics,
    plot_confusion_matrix,
    plot_training_curves,
    save_metrics,
)
from src.training.engine import evaluate, train_one_epoch
from src.utils.seed import set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--model_name", type=str, default="bert-base-uncased")
    ap.add_argument("--out_dir", type=str, default="results/bert_base")
    ap.add_argument("--n_per_class", type=int, default=2500)
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_fraction", type=float, default=1.0, help="fraction of TRAIN split to use (data-scaling ablation)")
    ap.add_argument("--save_model", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    train_df, test_df, stats = prepare_dataset(
        args.data_dir, n_per_class=args.n_per_class, seed=args.seed, train_fraction=args.train_fraction
    )
    print(f"Train: {stats.train_size} | Test: {stats.test_size} | train_fraction={args.train_fraction}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_loader = DataLoader(
        FakedditDataset(train_df, tokenizer, args.max_len), batch_size=args.batch_size, shuffle=True
    )
    test_loader = DataLoader(FakedditDataset(test_df, tokenizer, args.max_len), batch_size=args.batch_size)

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2).to(device)

    no_decay = ["bias", "LayerNorm.weight"]
    param_groups = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(param_groups, lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(args.warmup_ratio * total_steps), num_training_steps=total_steps
    )
    loss_fn = nn.CrossEntropyLoss()

    train_losses, train_accs, test_losses, test_accs = [], [], [], []
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, scheduler, loss_fn, device)
        te_loss, te_acc, _, _ = evaluate(model, test_loader, loss_fn, device)
        train_losses.append(tr_loss)
        train_accs.append(tr_acc)
        test_losses.append(te_loss)
        test_accs.append(te_acc)
        print(f"  train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} test_loss={te_loss:.4f} test_acc={te_acc:.4f}")

    _, _, preds, labels = evaluate(model, test_loader, loss_fn, device)
    metrics = compute_metrics(labels, preds)
    print(f"Final Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1_weighted']:.4f}")

    plot_training_curves(train_losses, test_losses, train_accs, test_accs, os.path.join(args.out_dir, "training_curves.png"))
    plot_confusion_matrix(metrics["confusion_matrix"], os.path.join(args.out_dir, "confusion_matrix.png"), title=f"{args.model_name} -- Confusion Matrix")
    save_metrics(
        metrics,
        os.path.join(args.out_dir, "metrics.json"),
        extra={
            "model": args.model_name,
            "seed": args.seed,
            "train_fraction": args.train_fraction,
            "train_size": stats.train_size,
            "hyperparams": vars(args),
        },
    )

    errors = build_error_report(test_df, preds, labels, top_k=10)
    errors.to_csv(os.path.join(args.out_dir, "error_examples.csv"), index=False)
    print("Error category counts:\n", category_counts(errors))

    if args.save_model:
        model.save_pretrained(os.path.join(args.out_dir, "checkpoint"))
        tokenizer.save_pretrained(os.path.join(args.out_dir, "checkpoint"))

    print(f"Saved results to {args.out_dir}/")


if __name__ == "__main__":
    main()
