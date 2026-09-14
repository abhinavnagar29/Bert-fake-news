"""Encoder-only BERT implemented from scratch, with a classification head."""
import torch
import torch.nn as nn

from .embeddings import BertEmbeddings
from .encoder import BertPooler, TransformerEncoderLayer


class BertModel(nn.Module):
    """Encoder-only BERT built from scratch.

    Data flow:
        input_ids
          -> BertEmbeddings   (token + position + segment)
          -> N x TransformerEncoderLayer
          -> BertPooler       ([CLS] vector)
          -> Dropout + Linear (classifier head)
          -> logits

    Default hyperparameters match the assignment spec (embed_dim=512,
    num_heads=8, num_layers=2), but the class is fully configurable so it can
    be used in the model-size / data-scaling ablations.
    """

    def __init__(
        self,
        vocab_size: int = 30522,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 2,
        ff_dim: int = 2048,
        max_pos: int = 512,
        num_classes: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embeddings = BertEmbeddings(vocab_size, embed_dim, max_pos, 2, dropout)
        self.encoder_layers = nn.ModuleList(
            [
                TransformerEncoderLayer(embed_dim, num_heads, ff_dim, dropout)
                for _ in range(num_layers)
            ]
        )
        self.pooler = BertPooler(embed_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        token_type_ids: torch.Tensor = None,
    ) -> dict:
        combined, tok_emb, pos_emb, seg_emb = self.embeddings(input_ids, token_type_ids)

        h = combined
        enc_outputs, attn_outputs, ff_outputs = [], [], []
        for layer in self.encoder_layers:
            h, attn_out, ff_out = layer(h, attention_mask)
            enc_outputs.append(h)
            attn_outputs.append(attn_out)
            ff_outputs.append(ff_out)

        pooled = self.pooler(h)
        logits = self.classifier(pooled)

        return {
            "logits": logits,
            "pooled_output": pooled,
            "last_hidden_state": h,
            "enc_outputs": enc_outputs,
            "attn_outputs": attn_outputs,
            "ff_outputs": ff_outputs,
            "token_embeddings": tok_emb,
            "position_embeddings": pos_emb,
            "segment_embeddings": seg_emb,
            "combined_embeddings": combined,
        }
