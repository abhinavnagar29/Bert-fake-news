"""Fine-tune a HuggingFace BertForSequenceClassification-style model.

Works for bert-base-uncased, distilbert-base-uncased, roberta-base, etc.
Supports --seed (multi-seed reporting) and --train_fraction (data-scaling
ablation) so the SAME script drives Task 3, the multi-seed experiment, and
the data-scaling experiment -- just with different CLI flags.

CHANGES vs. the original version of this file:
  1. Uses `prepare_dataset_with_val` (train/val/test, not train/test) so the
     test set is never looked at until the single final evaluation.
  2. Selects the best checkpoint by validation F1 (in-memory, via
     copy.deepcopy of the state dict -- no extra disk I/O per epoch), not
     by whichever epoch happens to run last.
  3. Fits temperature scaling (src.calibration.temperature_scaling) on the
     validation logits from the *best* checkpoint, and reports both raw and
     calibrated ECE/confidence on the held-out test set.
  4. Test set is evaluated exactly once, using the best checkpoint.

Usage:
    python -m src.training.train_transformer \
        --data_dir data/fakeddit --model_name bert-base-uncased \
        --out_dir results/bert_base_final --seed 42 --save_model

    # data-scaling ablation point at 25% of training data
    python -m src.training.train_transformer \
        --data_dir data/fakeddit --model_name bert-base-uncased \
        --out_dir results/bert_base_25pct --train_fraction 0.25
"""
import argparse
import copy
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from src.calibration.temperature_scaling import TemperatureScaler, expected_calibration_error
from src.data.dataset import FakedditDataset
from src.data.preprocess import prepare_dataset_with_val
from src.evaluation.error_analysis import build_error_report, category_counts
from src.evaluation.metrics import (
    compute_metrics,
    plot_confusion_matrix,
    plot_training_curves,
    save_metrics,
)
from src.training.engine import evaluate_with_logits, train_one_epoch
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
    ap.add_argument("--val_size", type=float, default=0.1, help="fraction of full balanced set held out for validation")
    ap.add_argument("--test_size", type=float, default=0.2, help="fraction of full balanced set held out for test")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_fraction", type=float, default=1.0, help="fraction of TRAIN split to use (data-scaling ablation)")
    ap.add_argument("--save_model", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    train_df, val_df, test_df, stats = prepare_dataset_with_val(
        args.data_dir,
        n_per_class=args.n_per_class,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
        train_fraction=args.train_fraction,
    )
    print(
        f"Train: {stats.train_size} | Val: {stats.val_size} | Test: {stats.test_size} "
        f"| train_fraction={args.train_fraction}"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_loader = DataLoader(
        FakedditDataset(train_df, tokenizer, args.max_len), batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(FakedditDataset(val_df, tokenizer, args.max_len), batch_size=args.batch_size)
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

    train_losses, train_accs, val_losses, val_accs = [], [], [], []
    best_val_f1 = -1.0
    best_state_dict = None
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, scheduler, loss_fn, device)
        va_loss, va_acc, va_preds, va_labels, _va_logits = evaluate_with_logits(model, val_loader, loss_fn, device)
        va_metrics = compute_metrics(va_labels, va_preds)

        train_losses.append(tr_loss)
        train_accs.append(tr_acc)
        val_losses.append(va_loss)
        val_accs.append(va_acc)
        print(
            f"  train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} "
            f"val_loss={va_loss:.4f} val_acc={va_acc:.4f} val_f1={va_metrics['f1_weighted']:.4f}"
        )

        # Checkpoint selection uses VALIDATION F1 only -- the test set is
        # never touched inside this loop.
        if va_metrics["f1_weighted"] > best_val_f1:
            best_val_f1 = va_metrics["f1_weighted"]
            best_state_dict = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            print(f"  -> new best checkpoint (val_f1={best_val_f1:.4f})")

    print(f"Loading best checkpoint from epoch {best_epoch} (val_f1={best_val_f1:.4f})")
    model.load_state_dict(best_state_dict)

    # Fit temperature scaling on the VALIDATION logits of the best checkpoint.
    _, _, _, val_labels_final, val_logits_final = evaluate_with_logits(model, val_loader, loss_fn, device)
    scaler = TemperatureScaler()
    fitted_temperature = scaler.fit(val_logits_final, torch.tensor(val_labels_final))
    print(f"Fitted temperature: {fitted_temperature:.4f}")

    # Single final evaluation on the held-out TEST set.
    te_loss, te_acc, preds, labels, test_logits = evaluate_with_logits(model, test_loader, loss_fn, device)
    metrics = compute_metrics(labels, preds)
    print(f"Final Test Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1_weighted']:.4f}")

    raw_probs = F.softmax(test_logits, dim=-1)
    calibrated_probs = scaler.calibrated_probs(test_logits)
    test_labels_tensor = torch.tensor(labels)
    ece_before = expected_calibration_error(raw_probs, test_labels_tensor)
    ece_after = expected_calibration_error(calibrated_probs, test_labels_tensor)
    print(f"Test ECE before calibration: {ece_before.ece:.4f} | after: {ece_after.ece:.4f}")

    plot_training_curves(train_losses, val_losses, train_accs, val_accs, os.path.join(args.out_dir, "training_curves.png"))
    plot_confusion_matrix(metrics["confusion_matrix"], os.path.join(args.out_dir, "confusion_matrix.png"), title=f"{args.model_name} -- Confusion Matrix (test)")

    save_metrics(
        metrics,
        os.path.join(args.out_dir, "metrics.json"),
        extra={
            "model": args.model_name,
            "seed": args.seed,
            "train_fraction": args.train_fraction,
            "train_size": stats.train_size,
            "val_size": stats.val_size,
            "test_size": stats.test_size,
            "best_epoch": best_epoch,
            "best_val_f1": best_val_f1,
            "temperature": fitted_temperature,
            "test_ece_before_calibration": ece_before.ece,
            "test_ece_after_calibration": ece_after.ece,
            "hyperparams": vars(args),
        },
    )

    errors = build_error_report(test_df, preds, labels, top_k=10, seed=args.seed)
    errors.to_csv(os.path.join(args.out_dir, "error_examples.csv"), index=False)
    print("Error category counts:\n", category_counts(errors))

    if args.save_model:
        ckpt_dir = os.path.join(args.out_dir, "checkpoint")
        model.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)
        # Save temperature + label map alongside the HF-standard files so the
        # Streamlit app can reload calibrated confidence without retraining.
        with open(os.path.join(ckpt_dir, "calibration.json"), "w") as f:
            json.dump(
                {
                    "temperature": fitted_temperature,
                    "id2label": {"0": "real", "1": "fake"},
                    "label2id": {"real": 0, "fake": 1},
                    "test_ece_before_calibration": ece_before.ece,
                    "test_ece_after_calibration": ece_after.ece,
                },
                f,
                indent=2,
            )

    print(f"Saved results to {args.out_dir}/")


if __name__ == "__main__":
    main()
