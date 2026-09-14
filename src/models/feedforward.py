"""Position-wise Feed-Forward Network used inside each encoder block."""
import torch
import torch.nn as nn


class FeedForwardLayer(nn.Module):
    """Two-layer MLP applied identically to every token position.

        FFN(x) = GELU( x.W_1 + b_1 ).W_2 + b_2

    BERT uses GELU (not ReLU). Inner dimension is conventionally 4 * embed_dim.
    """

    def __init__(self, embed_dim: int, ff_dim: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, embed_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
