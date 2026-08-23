import torch

from src.models.attention import MultiHeadSelfAttention


def test_output_shape_matches_input():
    batch, seq_len, embed_dim, heads = 4, 12, 64, 8
    layer = MultiHeadSelfAttention(embed_dim, heads)
    x = torch.randn(batch, seq_len, embed_dim)
    out = layer(x)
    assert out.shape == (batch, seq_len, embed_dim)


def test_rejects_non_divisible_heads():
    try:
        MultiHeadSelfAttention(embed_dim=65, num_heads=8)
        assert False, "expected ValueError for non-divisible embed_dim/num_heads"
    except ValueError:
        pass


def test_padding_mask_zeros_out_attention_to_pad_tokens():
    """A padded position should receive ~0 attention weight from real tokens."""
    torch.manual_seed(0)
    batch, seq_len, embed_dim, heads = 1, 6, 32, 4
    layer = MultiHeadSelfAttention(embed_dim, heads, dropout=0.0)
    layer.eval()

    x = torch.randn(batch, seq_len, embed_dim)
    # first 4 tokens real, last 2 are padding
    attention_mask = torch.tensor([[1, 1, 1, 1, 0, 0]])

    _, attn_weights = layer(x, attention_mask=attention_mask, return_attn_weights=True)
    # attn_weights: (batch, heads, seq_len, seq_len) -- weight given by each
    # query position to each key position. Padded KEY positions (last two
    # columns) should get ~0 weight from every query.
    pad_attention = attn_weights[0, :, :, 4:]
    assert torch.allclose(pad_attention, torch.zeros_like(pad_attention), atol=1e-5)


def test_attention_weights_sum_to_one():
    torch.manual_seed(0)
    batch, seq_len, embed_dim, heads = 2, 5, 16, 2
    layer = MultiHeadSelfAttention(embed_dim, heads, dropout=0.0)
    layer.eval()
    x = torch.randn(batch, seq_len, embed_dim)
    _, attn_weights = layer(x, return_attn_weights=True)
    row_sums = attn_weights.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)
