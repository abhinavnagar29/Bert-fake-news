import torch

from src.models.encoder import BertPooler, TransformerEncoderLayer


def test_encoder_layer_output_shapes():
    batch, seq_len, embed_dim, heads, ff_dim = 2, 9, 32, 4, 128
    layer = TransformerEncoderLayer(embed_dim, heads, ff_dim)
    x = torch.randn(batch, seq_len, embed_dim)

    out, attn_out, ff_out = layer(x)

    assert out.shape == (batch, seq_len, embed_dim)
    assert attn_out.shape == (batch, seq_len, embed_dim)
    assert ff_out.shape == (batch, seq_len, embed_dim)


def test_encoder_layer_respects_attention_mask_shape():
    batch, seq_len, embed_dim, heads = 2, 6, 16, 2
    layer = TransformerEncoderLayer(embed_dim, heads, ff_dim=32)
    x = torch.randn(batch, seq_len, embed_dim)
    mask = torch.ones(batch, seq_len)
    out, _, _ = layer(x, attention_mask=mask)
    assert out.shape == (batch, seq_len, embed_dim)


def test_pooler_extracts_cls_and_shape():
    batch, seq_len, embed_dim = 3, 5, 24
    pooler = BertPooler(embed_dim)
    hidden = torch.randn(batch, seq_len, embed_dim)
    pooled = pooler(hidden)
    assert pooled.shape == (batch, embed_dim)
    # tanh activation bounds output in (-1, 1)
    assert torch.all(pooled <= 1.0) and torch.all(pooled >= -1.0)
