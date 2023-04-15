import torch
import torch.nn as nn
from .encoder_layer import EncoderLayer
from .positional_encoding import PositionalEncoding
from .embed import Embeddings


class Encoder(nn.Module):
    def __init__(self, Embedding: Embeddings, d_model,
                 num_heads, num_layers,
                 d_ff, device="cpu", dropout=0.3, efficient_mha=False):
        super().__init__()

        self.embedding = Embedding

        self.PE = PositionalEncoding(
            d_model, device=device)

        self.encoders = nn.ModuleList([EncoderLayer(
            d_model,
            num_heads,
            d_ff,
            dropout,
            efficient_mha
        ) for layer in range(num_layers)])

    def forward(self, x, mask=None):
        # shape(x) = [B x SRC_seq_len]

        embeddings = self.embedding(x)
        encoding = self.PE(embeddings)
        # shape(embeddings) = [B x SRC_seq_len x D]
        # shape(encoding) = [B x SRC_seq_len x D]

        for encoder in self.encoders:
            encoding, encoder_attention_weights = encoder(encoding, mask)
            # shape(encoding) = [B x SRC_seq_len x D]
            # shape(encoder_attention_weights) = [B x num_heads x SRC_seq_len x SRC_seq_len]

        return encoding, encoder_attention_weights
