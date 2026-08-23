"""Transformer encoder block and BERT pooler."""
import torch
import torch.nn as nn

from .attention import MultiHeadSelfAttention
from .feedforward import FeedForwardLayer


class TransformerEncoderLayer(nn.Module):
    """One BERT encoder block with Post-LayerNorm (original BERT paper).

        Sub-layer 1:  out1 = LayerNorm( x + Dropout( MHSA(x) ) )
        Sub-layer 2:  out2 = LayerNorm( out1 + Dropout( FFN(out1) ) )

    Returns the block output plus the pre-residual MHSA/FFN outputs so their
    shapes can be logged / inspected (used by Task 1's shape report).
    """

    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.feed_forward = FeedForwardLayer(embed_dim, ff_dim, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None):
        attn_out = self.attention(x, attention_mask)
        x = self.norm1(x + self.dropout(attn_out))

        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))

        return x, attn_out, ff_out


class BertPooler(nn.Module):
    """Extracts the [CLS] hidden state and passes it through Dense + Tanh."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        cls_hidden = hidden_states[:, 0, :]
        return self.activation(self.dense(cls_hidden))
