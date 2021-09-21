import torch.nn as nn
import torch


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.3, max_seq_len=200, device="cpu"):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_seq_len, d_model).to(device)
        pos = torch.arange(0, max_seq_len).unsqueeze(1).float()

        two_i = torch.arange(0, d_model, step=2).float()
        div_term = torch.pow(10000, (two_i/torch.Tensor([d_model]))).float()
        pe[:, 0::2] = torch.sin(pos/div_term)
        pe[:, 1::2] = torch.cos(pos/div_term)

        pe = pe.unsqueeze(0)

        # assigns the first argument to a class variable
        # i.e. self.pe
        self.register_buffer("pe", pe)

    def forward(self, x):
        # shape(x) = [B x seq_len x D]
        one_batch_pe: torch.Tensor = self.pe[:, :x.shape[1]].detach()
        repeated_pe = one_batch_pe.repeat([x.shape[0], 1, 1]).detach()
        x = x.add(repeated_pe)
        # shape(x) = [B x seq_len x D]
        return self.dropout(x)
