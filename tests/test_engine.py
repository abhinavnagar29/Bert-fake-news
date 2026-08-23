import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

from src.models.bert import BertModel
from src.training.engine import evaluate, get_logits, train_one_epoch


def _dict_collate(batch):
    input_ids = torch.stack([b[0] for b in batch])
    attention_mask = torch.stack([b[1] for b in batch])
    token_type_ids = torch.stack([b[2] for b in batch])
    label = torch.stack([b[3] for b in batch])
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        "label": label,
    }


def _make_synthetic_loader(n=32, seq_len=8, vocab=50, batch_size=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(1, vocab, (n, seq_len), generator=g)
    attention_mask = torch.ones(n, seq_len, dtype=torch.long)
    token_type_ids = torch.zeros(n, seq_len, dtype=torch.long)
    # Trivially learnable synthetic task: label = 1 if first token id is even
    labels = (input_ids[:, 0] % 2 == 0).long()
    ds = TensorDataset(input_ids, attention_mask, token_type_ids, labels)
    return DataLoader(ds, batch_size=batch_size, collate_fn=_dict_collate)


def test_get_logits_handles_dict_output():
    class DummyDict:
        pass

    d = {"logits": torch.randn(2, 2)}
    assert torch.equal(get_logits(d), d["logits"])


def test_train_one_epoch_reduces_loss_on_synthetic_task():
    torch.manual_seed(0)
    model = BertModel(vocab_size=50, embed_dim=16, num_heads=2, num_layers=1, ff_dim=32, num_classes=2)
    loader = _make_synthetic_loader()
    optimizer = AdamW(model.parameters(), lr=5e-3)
    loss_fn = nn.CrossEntropyLoss()

    first_loss, _ = train_one_epoch(model, loader, optimizer, None, loss_fn, "cpu", log_every=0)
    # run several more epochs; loss should trend down on this trivial task
    losses = [first_loss]
    for _ in range(8):
        l, _ = train_one_epoch(model, loader, optimizer, None, loss_fn, "cpu", log_every=0)
        losses.append(l)

    assert losses[-1] < losses[0]


def test_evaluate_returns_expected_shapes_and_types():
    model = BertModel(vocab_size=50, embed_dim=16, num_heads=2, num_layers=1, ff_dim=32, num_classes=2)
    loader = _make_synthetic_loader(n=16, batch_size=4)
    loss_fn = nn.CrossEntropyLoss()

    avg_loss, acc, preds, labels = evaluate(model, loader, loss_fn, "cpu")
    assert isinstance(avg_loss, float)
    assert 0.0 <= acc <= 1.0
    assert len(preds) == 16
    assert len(labels) == 16
