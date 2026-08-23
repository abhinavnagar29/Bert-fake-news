import pandas as pd
import torch

from src.data.dataset import FakedditDataset


class MockTokenizer:
    """Minimal stand-in for a HuggingFace tokenizer, whitespace-based, so the
    dataset padding/truncation logic can be tested without network access."""

    def __call__(self, text, max_length, padding, truncation, return_tensors):
        tokens = text.split()[: max_length - 2]  # room for CLS/SEP
        ids = [101] + [hash(t) % 1000 + 2 for t in tokens] + [102]
        attn = [1] * len(ids)
        if padding == "max_length":
            pad_len = max_length - len(ids)
            ids = ids + [0] * pad_len
            attn = attn + [0] * pad_len
        return {
            "input_ids": torch.tensor([ids]),
            "attention_mask": torch.tensor([attn]),
        }


def _sample_df():
    return pd.DataFrame(
        {
            "clean_title": [
                "scientists discover new species of fish",
                "shocking you wont believe what happened next",
            ],
            "label": [0, 1],
        }
    )


def test_dataset_length():
    ds = FakedditDataset(_sample_df(), MockTokenizer(), max_len=16)
    assert len(ds) == 2


def test_dataset_item_shapes_and_padding():
    max_len = 16
    ds = FakedditDataset(_sample_df(), MockTokenizer(), max_len=max_len)
    item = ds[0]
    assert item["input_ids"].shape == (max_len,)
    assert item["attention_mask"].shape == (max_len,)
    assert item["token_type_ids"].shape == (max_len,)
    assert item["label"].item() == 0

    # attention_mask should be 1 exactly where input_ids is non-pad content
    n_real = int(item["attention_mask"].sum().item())
    assert n_real <= max_len
    assert torch.all(item["attention_mask"][n_real:] == 0)


def test_dataset_labels_match_input():
    ds = FakedditDataset(_sample_df(), MockTokenizer(), max_len=16)
    assert ds[0]["label"].item() == 0
    assert ds[1]["label"].item() == 1


def test_token_type_ids_always_zero():
    ds = FakedditDataset(_sample_df(), MockTokenizer(), max_len=16)
    for i in range(len(ds)):
        assert torch.all(ds[i]["token_type_ids"] == 0)
