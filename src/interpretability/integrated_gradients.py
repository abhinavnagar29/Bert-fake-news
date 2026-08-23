"""Layer Integrated Gradients token attribution for the fine-tuned classifier.

Requires `captum` (pip install captum) and a fine-tuned
BertForSequenceClassification model + its tokenizer. Not covered by the CPU
unit tests in tests/ because it requires downloading bert-base-uncased and a
trained checkpoint; run it manually per NEXT_STEPS.md Step 8.
"""
from __future__ import annotations

import torch

SPECIAL_TOKENS = {"[CLS]", "[SEP]", "[PAD]"}


def make_forward_fn(model):
    """Rank-agnostic forward function for Captum: accepts either token ids
    (rank 2) or embeddings (rank 3)."""

    def forward_fn(inputs, attention_mask=None, token_type_ids=None):
        if inputs.dim() == 2:
            outputs = model(
                input_ids=inputs,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
        else:
            outputs = model(
                inputs_embeds=inputs,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
        return outputs.logits

    return forward_fn


def compute_ig(
    text: str,
    model,
    tokenizer,
    target_class: int,
    device: str = "cpu",
    n_steps: int = 300,
    internal_batch_size: int = 2,
    max_len: int = 64,
) -> dict:
    """Tokenise `text`, run LayerIntegratedGradients, return per-token scores.

    Baseline: all-zero embeddings. Attribution is summed over the embedding
    dimension and normalized by its L1 norm for interpretability.
    """
    from captum.attr import LayerIntegratedGradients

    model.eval()
    enc = tokenizer(text, max_length=max_len, truncation=True, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    token_type_ids = torch.zeros_like(input_ids)

    forward_fn = make_forward_fn(model)
    embedding_layer = model.bert.embeddings.word_embeddings
    lig = LayerIntegratedGradients(forward_fn, embedding_layer)

    baseline_ids = torch.zeros_like(input_ids)

    attributions, delta = lig.attribute(
        inputs=input_ids,
        baselines=baseline_ids,
        additional_forward_args=(attention_mask, token_type_ids),
        target=target_class,
        n_steps=n_steps,
        internal_batch_size=internal_batch_size,
        return_convergence_delta=True,
    )

    scores = attributions.sum(dim=-1).squeeze(0)
    scores = scores / (scores.norm() + 1e-12)
    scores = scores.detach().cpu().numpy().tolist()

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().tolist())

    return {
        "text": text,
        "tokens": tokens,
        "scores": scores,
        "convergence_delta": float(delta.abs().item()),
        "target_class": target_class,
    }


def top_k_tokens(result: dict, k: int = 5, exclude_special: bool = True):
    pairs = list(zip(result["tokens"], result["scores"]))
    if exclude_special:
        pairs = [(t, s) for t, s in pairs if t not in SPECIAL_TOKENS]
    return sorted(pairs, key=lambda x: abs(x[1]), reverse=True)[:k]
