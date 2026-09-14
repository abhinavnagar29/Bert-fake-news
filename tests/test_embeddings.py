import torch

from src.models.embeddings import BertEmbeddings


def test_embedding_output_shapes():
    vocab_size, embed_dim, max_pos, batch, seq_len = 1000, 64, 128, 3, 10
    emb = BertEmbeddings(vocab_size, embed_dim, max_pos)
    input_ids = torch.randint(0, vocab_size, (batch, seq_len))

    combined, tok_emb, pos_emb, seg_emb = emb(input_ids)

    assert combined.shape == (batch, seq_len, embed_dim)
    assert tok_emb.shape == (batch, seq_len, embed_dim)
    assert pos_emb.shape == (1, seq_len, embed_dim)
    assert seg_emb.shape == (batch, seq_len, embed_dim)


def test_default_token_type_ids_are_zero():
    """When token_type_ids is None, it should default to an all-zero tensor
    (i.e. behave identically to explicitly passing zeros)."""
    vocab_size, embed_dim, max_pos, batch, seq_len = 500, 32, 64, 2, 7
    torch.manual_seed(0)
    emb = BertEmbeddings(vocab_size, embed_dim, max_pos)
    emb.eval()
    input_ids = torch.randint(0, vocab_size, (batch, seq_len))

    out_default, *_ = emb(input_ids)
    out_explicit, *_ = emb(input_ids, token_type_ids=torch.zeros_like(input_ids))

    assert torch.allclose(out_default, out_explicit)


def test_padding_idx_zero_gives_zero_token_embedding_before_norm():
    vocab_size, embed_dim, max_pos = 100, 16, 32
    emb = BertEmbeddings(vocab_size, embed_dim, max_pos)
    pad_vec = emb.token_embeddings(torch.tensor([0]))
    assert torch.allclose(pad_vec, torch.zeros_like(pad_vec))
