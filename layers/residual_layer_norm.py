import torch.nn as nn


class ResidualLayerNorm(nn.Module):
    def __init__(self, d_model, dropout=0.3):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, residual):
        # In the video this was:
        #   ln = self.layer_norm(x + residual)
        #   return self.dropout(ln)
        # The above does not lead to convergence. We must dropout x for convergence.
        # Why doesn't this work? Because we send the output of the layernorm to an attention block.
        # So some values would be zeroed out if dropout is enabled. Obviously MHA doesn't know what to attend to then.
        # We can dropout(x) though because 1) we're adding a residual to it, so dropped out values won't be zero
        # and 2), the layernorm has an additive beta parameter which provides a non-zero value to a tensor
        ln = self.layer_norm(self.dropout(x) + residual)
        return ln
