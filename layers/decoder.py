import torch
import torch.nn as nn
from .decoder_layer import DecoderLayer
from .embed import Embeddings
from .positional_encoding import PositionalEncoding


class Decoder(nn.Module):
    def __init__(self, Embedding: Embeddings, d_model,
                 num_heads, num_layers,
                 d_ff, device="cpu", dropout=0.3, efficient_mha=False):
        super().__init__()

        self.embedding = Embedding

        self.PE = PositionalEncoding(
            d_model, device=device)

        self.dropout = nn.Dropout(dropout)

        self.decoders = nn.ModuleList([DecoderLayer(
            d_model,
            num_heads,
            d_ff,
            dropout,
            efficient_mha
        ) for layer in range(num_layers)])

    def forward(self, x, encoder_output, trg_mask, src_mask):
        # shape(x) = [B x TRG_seq_len]

        embeddings = self.embedding(x)
        encoding = self.PE(embeddings)
        # shape(embeddings) = [B x TRG_seq_len x D]
        # shape(encoding) = [B x TRG_seq_len x D]

        for decoder in self.decoders:
            encoding, masked_mha_attn_weights, enc_dec_mha_attn_weights = decoder(encoding, encoder_output, trg_mask, src_mask)
            # shape(encoding) = [B x TRG_seq_len x D]
            # shape(masked_mha_attn_weights) = [B x num_heads x TRG_seq_len x TRG_seq_len]
            # shape(enc_dec_mha_attn_weights) = [B x num_heads x TRG_seq_len x SRC_seq_len]

        return encoding, masked_mha_attn_weights, enc_dec_mha_attn_weights
