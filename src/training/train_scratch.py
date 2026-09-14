"""Train the from-scratch BertModel (Task 1 architecture) on Fakeddit.

Usage:
    python -m src.training.train_scratch --data_dir data/fakeddit --out_dir results/scratch_bert
"""
import argparse
import os

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import BertTokenizer, get_linear_schedule_with_warmup

from src.data.dataset import FakedditDataset
from src.data.preprocess import prepare_dataset
from src.evaluation.metrics import (
    compute_metrics,
    plot_confusion_matrix,
    plot_training_curves,
    save_metrics,
)
from src.models.bert import BertModel
from src.training.engine import evaluate, train_one_epoch
from src.utils.seed import set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="results/scratch_bert")
    ap.add_argument("--n_per_class", type=int, default=2500)
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--embed_dim", type=int, default=512)
    ap.add_argument("--num_heads", type=int, default=8)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    train_df, test_df, stats = prepare_dataset(args.data_dir, n_per_class=args.n_per_class, seed=args.seed)
    print(f"Train: {stats.train_size} | Test: {stats.test_size}")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    train_loader = DataLoader(
        FakedditDataset(train_df, tokenizer, args.max_len), batch_size=args.batch_size, shuffle=True
    )
    test_loader = DataLoader(
        FakedditDataset(test_df, tokenizer, args.max_len), batch_size=args.batch_size
    )

    model = BertModel(
        vocab_size=tokenizer.vocab_size,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        ff_dim=args.embed_dim * 4,
        num_classes=2,
    ).to(device)

    total_steps = len(train_loader) * args.epochs
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
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

    plot_training_curves(
        train_losses, test_losses, train_accs, test_accs,
        os.path.join(args.out_dir, "training_curves.png"),
    )
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        os.path.join(args.out_dir, "confusion_matrix.png"),
        title="Scratch BERT -- Confusion Matrix",
    )
    save_metrics(
        metrics,
        os.path.join(args.out_dir, "metrics.json"),
        extra={"model": "scratch_bert", "seed": args.seed, "hyperparams": vars(args)},
    )

    torch.save(model.state_dict(), os.path.join(args.out_dir, "model_state_dict.pt"))
    print(f"Saved results and checkpoint to {args.out_dir}/")


if __name__ == "__main__":
    main()
