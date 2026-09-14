"""Shared training / evaluation loops.

Works with both our from-scratch BertModel (which returns a dict with a
'logits' key) and HuggingFace models (which return an object with a .logits
attribute), via the `get_logits` helper.

`evaluate_with_logits` is new (additive): it's identical to `evaluate` but
also returns raw logits, which the production training path needs to fit
temperature scaling on the validation set and to run the single final test
evaluation. `train_one_epoch` and `evaluate` are unchanged so
train_baseline.py and train_scratch.py (and their tests) are unaffected.
"""
import torch
import torch.nn as nn


def get_logits(outputs):
    if isinstance(outputs, dict):
        return outputs["logits"]
    return outputs.logits


def train_one_epoch(model, loader, optimizer, scheduler, loss_fn, device, log_every=20):
    model.train()
    total_loss, n_correct, n_total = 0.0, 0, 0

    for step, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        logits = get_logits(outputs)
        loss = loss_fn(logits, labels)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=-1)
        n_correct += (preds == labels).sum().item()
        n_total += labels.size(0)

        if log_every and (step + 1) % log_every == 0:
            print(
                f"  Step {step + 1:>4}/{len(loader)} "
                f"| Loss: {total_loss / (step + 1):.4f} "
                f"| Acc: {n_correct / n_total:.4f}"
            )

    return total_loss / len(loader), n_correct / n_total


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss, n_correct, n_total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels = batch["label"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        logits = get_logits(outputs)
        loss = loss_fn(logits, labels)
        preds = logits.argmax(dim=-1)

        total_loss += loss.item()
        n_correct += (preds == labels).sum().item()
        n_total += labels.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    return total_loss / len(loader), n_correct / n_total, all_preds, all_labels


@torch.no_grad()
def evaluate_with_logits(model, loader, loss_fn, device):
    """Same as `evaluate`, but also returns the raw (uncalibrated) logits
    for every example, stacked into a single (N, n_classes) tensor on CPU.
    Needed for temperature-scaling fit and for computing calibrated
    confidence at inference time.
    """
    model.eval()
    total_loss, n_correct, n_total = 0.0, 0, 0
    all_preds, all_labels, all_logits = [], [], []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels = batch["label"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        logits = get_logits(outputs)
        loss = loss_fn(logits, labels)
        preds = logits.argmax(dim=-1)

        total_loss += loss.item()
        n_correct += (preds == labels).sum().item()
        n_total += labels.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        all_logits.append(logits.detach().cpu())

    stacked_logits = torch.cat(all_logits, dim=0)
    return total_loss / len(loader), n_correct / n_total, all_preds, all_labels, stacked_logits
