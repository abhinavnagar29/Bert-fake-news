"""Multi-Head Self-Attention implemented from scratch (Vaswani et al., 2017)."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    """Scaled Dot-Product Multi-Head Attention.

    For each head h (h = 0 .. H-1):
        head_h = Attention(X.W_Q_h, X.W_K_h, X.W_V_h)
               = softmax( Q.K^T / sqrt(d_k) ) . V
    Full output = Concat(head_0, ..., head_{H-1}) . W_O

    Args:
        embed_dim: model dimension (d_model).
        num_heads: number of attention heads H. Must divide embed_dim.
        dropout: dropout applied to attention weights.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}"
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.W_q = nn.Linear(embed_dim, embed_dim, bias=True)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=True)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=True)
        self.W_o = nn.Linear(embed_dim, embed_dim, bias=True)
        self.attn_dropout = nn.Dropout(dropout)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, seq_len, embed_dim) -> (batch, num_heads, seq_len, head_dim)."""
        batch, seq_len, _ = x.size()
        return x.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,  # (batch, seq_len, embed_dim)
        attention_mask: torch.Tensor = None,  # (batch, seq_len), 1=keep 0=pad
        return_attn_weights: bool = False,
    ):
        batch, seq_len, _ = x.size()

        Q = self.split_heads(self.W_q(x))  # (batch, heads, seq, head_dim)
        K = self.split_heads(self.W_k(x))
        V = self.split_heads(self.W_v(x))

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (batch, heads, seq, seq)

        if attention_mask is not None:
            extended_mask = attention_mask[:, None, None, :].float()
            scores = scores + (1.0 - extended_mask) * -1e9

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        context = torch.matmul(attn_weights, V)  # (batch, heads, seq, head_dim)
        context = context.transpose(1, 2).contiguous().view(batch, seq_len, self.embed_dim)

        output = self.W_o(context)
        if return_attn_weights:
            return output, attn_weights
        return output
