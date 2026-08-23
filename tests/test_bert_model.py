import torch

from src.models.bert import BertModel


def _tiny_model(num_classes=2):
    return BertModel(
        vocab_size=200,
        embed_dim=32,
        num_heads=4,
        num_layers=2,
        ff_dim=64,
        max_pos=64,
        num_classes=num_classes,
        dropout=0.0,
    )


def test_forward_output_shapes():
    batch, seq_len = 3, 10
    model = _tiny_model()
    input_ids = torch.randint(0, 200, (batch, seq_len))
    attention_mask = torch.ones(batch, seq_len, dtype=torch.long)
    token_type_ids = torch.zeros(batch, seq_len, dtype=torch.long)

    out = model(input_ids, attention_mask, token_type_ids)

    assert out["logits"].shape == (batch, 2)
    assert out["pooled_output"].shape == (batch, 32)
    assert out["last_hidden_state"].shape == (batch, seq_len, 32)
    assert len(out["enc_outputs"]) == 2  # num_layers
    for enc in out["enc_outputs"]:
        assert enc.shape == (batch, seq_len, 32)


def test_padding_does_not_change_prediction_for_unpadded_tokens():
    """Adding fully-masked padding tokens should not change the pooled [CLS]
    representation (padding attention weight ~0), verifying the mask is
    actually wired through the whole stack, not just the first layer."""
    torch.manual_seed(0)
    model = _tiny_model()
    model.eval()

    seq_len = 6
    input_ids_short = torch.randint(1, 200, (1, seq_len))
    mask_short = torch.ones(1, seq_len, dtype=torch.long)

    pad_len = 4
    input_ids_padded = torch.cat(
        [input_ids_short, torch.zeros(1, pad_len, dtype=torch.long)], dim=1
    )
    mask_padded = torch.cat(
        [mask_short, torch.zeros(1, pad_len, dtype=torch.long)], dim=1
    )

    with torch.no_grad():
        out_short = model(input_ids_short, mask_short)
        out_padded = model(input_ids_padded, mask_padded)

    assert torch.allclose(
        out_short["pooled_output"], out_padded["pooled_output"], atol=1e-4
    )


def test_output_is_deterministic_in_eval_mode():
    model = _tiny_model()
    model.eval()
    input_ids = torch.randint(0, 200, (2, 8))
    with torch.no_grad():
        out1 = model(input_ids)["logits"]
        out2 = model(input_ids)["logits"]
    assert torch.allclose(out1, out2)


def test_param_count_is_positive_and_finite():
    model = _tiny_model()
    total = sum(p.numel() for p in model.parameters())
    assert total > 0
    for p in model.parameters():
        assert torch.isfinite(p).all()
