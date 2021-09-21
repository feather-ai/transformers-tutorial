import torch
import torch.nn as nn
from .embed import Embeddings
from .encoder import Encoder
from .decoder import Decoder


class Transformer(nn.Module):
    def __init__(self, src_vocab_len, trg_vocab_len, d_model, d_ff,
                 num_layers, num_heads, src_pad_idx, trg_pad_idx, dropout=0.3, device="cpu", efficient_mha=False):
        super().__init__()

        self.num_heads = num_heads
        self.device = device
        self.efficient_mha = efficient_mha

        encoder_Embedding = Embeddings(
            src_vocab_len, src_pad_idx, d_model)
        decoder_Embedding = Embeddings(
            trg_vocab_len, trg_pad_idx, d_model)

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx

        self.encoder = Encoder(encoder_Embedding, d_model,
                               num_heads, num_layers, d_ff, device, dropout, efficient_mha)
        self.decoder = Decoder(decoder_Embedding, d_model,
                               num_heads, num_layers, d_ff, device, dropout, efficient_mha)

        self.linear_layer = nn.Linear(d_model, trg_vocab_len)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def create_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1)
        if self.efficient_mha:
            src_mask = src_mask.unsqueeze(2)
        return src_mask

    def create_trg_mask(self, trg):
        if self.efficient_mha:
            trg_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
            mask = torch.ones((1, self.num_heads, trg.shape[1], trg.shape[1])).triu(1).to(self.device)
        else:
            trg_mask = (trg != self.trg_pad_idx).unsqueeze(1)
            mask = torch.ones((1, trg.shape[1], trg.shape[1])).triu(1).to(self.device)
        mask = mask == 0
        trg_mask = trg_mask & mask
        return trg_mask

    def forward(self, src, trg):
        # shape(src) = [B x SRC_seq_len]
        # shape(trg) = [B x TRG_seq_len]

        src_mask = self.create_src_mask(src)
        trg_mask = self.create_trg_mask(trg)
        # shape(src_mask) = [B x 1 x SRC_seq_len]
        # shape(trg_mask) = [B x 1 x TRG_seq_len]

        encoder_outputs, encoder_mha_attn_weights = self.encoder(src, src_mask)
        # shape(encoder_outputs) = [B x SRC_seq_len x D]
        # shape(encoder_mha_attn_weights) = [B x num_heads x SRC_seq_len x SRC_seq_len]

        decoder_outputs, _, enc_dec_mha_attn_weights = self.decoder(
            trg, encoder_outputs, trg_mask, src_mask)
        # shape(decoder_outputs) = [B x SRC_seq_len x D]
        # shape(enc_dec_mha_attn_weights) = [B x num_heads x TRG_seq_len x SRC_seq_len]

        logits = self.linear_layer(decoder_outputs)
        # shape(logits) = [B x TRG_seq_len x TRG_vocab_size]

        return logits
