import torch.nn as nn
from .residual_layer_norm import ResidualLayerNorm
from .mha import MultiHeadAttention
from .efficient_mha import MultiHeadAttention as EfficientMultiHeadAttention
from .pwffn import PWFFN


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.3, efficient_mha=False):
        super().__init__()

        # initalize these
        self.norm_1 = ResidualLayerNorm(d_model, dropout)
        self.norm_2 = ResidualLayerNorm(d_model, dropout)

        if efficient_mha:
            self.mha = EfficientMultiHeadAttention(d_model, num_heads, dropout)
        else:
            self.mha = MultiHeadAttention(d_model, num_heads, dropout)

        self.ff = PWFFN(d_model, d_ff, dropout)

    def forward(self, x, mask):
        # shape(x) = [B x seq_len x D]

        mha, encoder_attention_weights = self.mha(x, x, x, mask)
        # shape(mha) = [B x seq_len x D]
        # shape(encoder_attention_weights) = [B x num_heads x seq_len x seq_len]

        norm1 = self.norm_1(mha, x)
        # shape(norm1) = [B x seq_len x D]

        ff = self.ff(norm1)
        norm2 = self.norm_2(ff, norm1)
        # shape(ff) = [B x seq_len x D]
        # shape(norm2) = [B x seq_len x D]

        return norm2, encoder_attention_weights
