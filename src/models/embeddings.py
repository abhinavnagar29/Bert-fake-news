"""BERT-style input embeddings: token + positional + segment."""
import torch
import torch.nn as nn


class BertEmbeddings(nn.Module):
    """Combines token, positional and segment embeddings.

        combined = TokenEmb(input_ids) + PositionEmb(0..seq_len-1) + SegmentEmb(token_type_ids)
        output   = Dropout( LayerNorm( combined ) )

    All three tables are learned (as in the original BERT paper), unlike the
    sinusoidal positions used in the vanilla Transformer.
    """

    def __init__(
        self,
        vocab_size: int = 30522,
        embed_dim: int = 512,
        max_pos: int = 512,
        type_vocab: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.position_embeddings = nn.Embedding(max_pos, embed_dim)
        self.segment_embeddings = nn.Embedding(type_vocab, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("position_ids", torch.arange(max_pos).unsqueeze(0))

    def forward(self, input_ids: torch.Tensor, token_type_ids: torch.Tensor = None):
        _, seq_len = input_ids.size()

        tok_emb = self.token_embeddings(input_ids)
        pos_emb = self.position_embeddings(self.position_ids[:, :seq_len])

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        seg_emb = self.segment_embeddings(token_type_ids)

        combined = tok_emb + pos_emb + seg_emb
        combined = self.norm(combined)
        combined = self.dropout(combined)
        return combined, tok_emb, pos_emb, seg_emb
