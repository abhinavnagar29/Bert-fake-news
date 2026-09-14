"""Tests for the production save/reload/inference path.

These use `hf-internal-testing/tiny-random-bert` -- a tiny (a few hundred KB)
randomly-initialized BERT checkpoint that the `transformers` library's own
test suite uses for exactly this purpose -- instead of the real
bert-base-uncased / distilbert-base-uncased, so these tests run in seconds
and don't require a GPU. They exercise the SAME code path
(save_pretrained -> from_pretrained -> inference) that the real production
model goes through; they do not test model quality/accuracy, only that the
mechanics work end-to-end. If the sandbox running these tests has no
network access to reach the Hugging Face Hub, the fixture skips the whole
module rather than failing (see `_require_network`).
"""
import json
import os

import pytest
import torch

TINY_MODEL_ID = "hf-internal-testing/tiny-random-bert"


def _require_network():
    try:
        from transformers import AutoTokenizer

        AutoTokenizer.from_pretrained(TINY_MODEL_ID)
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"No access to Hugging Face Hub in this environment: {exc}")


@pytest.fixture(scope="module")
def tiny_model_and_tokenizer():
    _require_network()
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TINY_MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(TINY_MODEL_ID, num_labels=2)
    model.eval()
    return model, tokenizer


@pytest.fixture
def saved_checkpoint_dir(tmp_path, tiny_model_and_tokenizer):
    model, tokenizer = tiny_model_and_tokenizer
    ckpt_dir = tmp_path / "checkpoint"
    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)
    with open(ckpt_dir / "calibration.json", "w") as f:
        json.dump({"temperature": 1.37, "id2label": {"0": "real", "1": "fake"}, "label2id": {"real": 0, "fake": 1}}, f)
    return str(ckpt_dir)


def _predict(model, tokenizer, text, max_len=64):
    model.eval()
    enc = tokenizer(text, max_length=max_len, padding="max_length", truncation=True, return_tensors="pt")
    with torch.no_grad():
        out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
    probs = torch.softmax(out.logits, dim=-1)
    return probs


# ---- saved-model compatibility --------------------------------------------------

def test_save_pretrained_produces_expected_files(saved_checkpoint_dir):
    files = os.listdir(saved_checkpoint_dir)
    assert "config.json" in files
    assert any(f.startswith("tokenizer") for f in files)
    assert any(f in files for f in ("pytorch_model.bin", "model.safetensors"))
    assert "calibration.json" in files


def test_reloaded_model_matches_original_predictions(tiny_model_and_tokenizer, saved_checkpoint_dir):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    original_model, original_tokenizer = tiny_model_and_tokenizer
    reloaded_model = AutoModelForSequenceClassification.from_pretrained(saved_checkpoint_dir)
    reloaded_tokenizer = AutoTokenizer.from_pretrained(saved_checkpoint_dir)

    text = "Scientists confirm the earth orbits the sun"
    original_probs = _predict(original_model, original_tokenizer, text)
    reloaded_probs = _predict(reloaded_model, reloaded_tokenizer, text)

    assert torch.allclose(original_probs, reloaded_probs, atol=1e-5)


def test_calibration_json_round_trips(saved_checkpoint_dir):
    with open(os.path.join(saved_checkpoint_dir, "calibration.json")) as f:
        cal = json.load(f)
    assert cal["id2label"]["1"] == "fake"
    assert cal["label2id"]["fake"] == 1
    assert cal["temperature"] > 0


# ---- inference correctness --------------------------------------------------

def test_inference_returns_valid_probability_distribution(saved_checkpoint_dir):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model = AutoModelForSequenceClassification.from_pretrained(saved_checkpoint_dir)
    tokenizer = AutoTokenizer.from_pretrained(saved_checkpoint_dir)

    probs = _predict(model, tokenizer, "Breaking: local council approves new library budget")
    assert probs.shape == (1, 2)
    assert torch.isclose(probs.sum(), torch.tensor(1.0), atol=1e-5)
    assert (probs >= 0).all() and (probs <= 1).all()


# ---- edge cases --------------------------------------------------

def test_inference_handles_empty_string(saved_checkpoint_dir):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model = AutoModelForSequenceClassification.from_pretrained(saved_checkpoint_dir)
    tokenizer = AutoTokenizer.from_pretrained(saved_checkpoint_dir)
    probs = _predict(model, tokenizer, "")
    assert probs.shape == (1, 2)


def test_inference_handles_whitespace_only_string(saved_checkpoint_dir):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model = AutoModelForSequenceClassification.from_pretrained(saved_checkpoint_dir)
    tokenizer = AutoTokenizer.from_pretrained(saved_checkpoint_dir)
    probs = _predict(model, tokenizer, "     ")
    assert probs.shape == (1, 2)


def test_inference_truncates_very_long_input_without_crashing(saved_checkpoint_dir):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model = AutoModelForSequenceClassification.from_pretrained(saved_checkpoint_dir)
    tokenizer = AutoTokenizer.from_pretrained(saved_checkpoint_dir)
    very_long_text = "breaking news " * 500  # far beyond max_len=64 tokens
    probs = _predict(model, tokenizer, very_long_text, max_len=64)
    assert probs.shape == (1, 2)


def test_inference_handles_non_ascii_and_emoji(saved_checkpoint_dir):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model = AutoModelForSequenceClassification.from_pretrained(saved_checkpoint_dir)
    tokenizer = AutoTokenizer.from_pretrained(saved_checkpoint_dir)
    probs = _predict(model, tokenizer, "Se dice que \u00e9sto es verdad \ud83d\ude80\ud83d\udd25 100%!!")
    assert probs.shape == (1, 2)


def test_inference_is_deterministic_in_eval_mode(saved_checkpoint_dir):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model = AutoModelForSequenceClassification.from_pretrained(saved_checkpoint_dir)
    tokenizer = AutoTokenizer.from_pretrained(saved_checkpoint_dir)
    text = "Local team wins championship after dramatic final"
    probs_a = _predict(model, tokenizer, text)
    probs_b = _predict(model, tokenizer, text)
    assert torch.allclose(probs_a, probs_b, atol=1e-6)
