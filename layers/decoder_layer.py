import torch.nn as nn
from .mha import MultiHeadAttention
from .efficient_mha import MultiHeadAttention as EfficientMultiHeadAttention
from .pwffn import PWFFN
from .residual_layer_norm import ResidualLayerNorm


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.3, efficient_mha=False):
        super().__init__()
        self.norm_1 = ResidualLayerNorm(d_model)
        self.norm_2 = ResidualLayerNorm(d_model)
        self.norm_3 = ResidualLayerNorm(d_model)

        if efficient_mha:
            self.masked_mha = EfficientMultiHeadAttention(d_model, num_heads, dropout)
            self.enc_dec_mha = EfficientMultiHeadAttention(d_model, num_heads, dropout)
        else:
            self.masked_mha = MultiHeadAttention(d_model, num_heads, dropout)
            self.enc_dec_mha = MultiHeadAttention(d_model, num_heads, dropout)

        self.ff = PWFFN(d_model, d_ff)

    def forward(self, x, encoder_outputs, trg_mask, src_mask):
        # shape(x) = [B x TRG_seq_len x D]
        # shape(encoder_outputs) = [B x SRC_seq_len x D]

        masked_mha, masked_mha_attn_weights = self.masked_mha(x, x, x, mask=trg_mask)
        # shape(masked_mha) = [B x TRG_seq_len x D]
        # shape(masked_mha_attn_weights) = [B x num_heads x TRG_seq_len x TRG_seq_len]

        norm1 = self.norm_1(masked_mha, x)
        # shape(norm1) = [B x TRG_seq_len x D]

        enc_dec_mha, enc_dec_mha_attn_weights = self.enc_dec_mha(norm1, encoder_outputs, encoder_outputs, mask=src_mask)
        # shape(enc_dec_mha) = [B x TRG_seq_len x D]
        # shape(enc_dec_mha_attn_weights) = [B x num_heads x TRG_seq_len x SRC_seq_len]

        norm2 = self.norm_2(enc_dec_mha, norm1)
        # shape(norm2) = [B x TRG_seq_len x D]

        ff = self.ff(norm2)
        norm3 = self.norm_3(ff, norm2)
        # shape(ff) = [B x TRG_seq_len x D]
        # shape(norm3) = [B x TRG_seq_len x D]

        return norm3, masked_mha_attn_weights, enc_dec_mha_attn_weights
