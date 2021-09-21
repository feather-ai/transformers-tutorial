# %%
import torch
import torch.nn as nn
import math as m
import torch.nn.functional as F

# %%
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=4, num_heads=2, dropout=0.3):
        super().__init__()

        # d_q, d_k, d_v
        self.d = d_model//num_heads

        self.d_model = d_model
        self.num_heads = num_heads

        self.dropout = nn.Dropout(dropout)

        ##create a list of layers for K, and a list of layers for V
        
        self.linear_Qs = nn.ModuleList([nn.Linear(d_model, self.d)
                                        for _ in range(num_heads)])
        self.linear_Ks = nn.ModuleList([nn.Linear(d_model, self.d)
                                        for _ in range(num_heads)])
        self.linear_Vs = nn.ModuleList([nn.Linear(d_model, self.d)
                                        for _ in range(num_heads)])

        self.mha_linear = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V):
        # shape(Q) = [B x seq_len x D/num_heads] = [B x T x d_k]
        # shape(K, V) = [B x T x d_k]

        Q_K_matmul = torch.matmul(Q, K.permute(0, 2, 1))
        scores = Q_K_matmul/m.sqrt(self.d)
        # shape(scores) = [B x seq_len x seq_len]

        attention_weights = F.softmax(scores, dim=-1)
        # shape(attention_weights) = [B x seq_len x seq_len]

        output = torch.matmul(attention_weights, V)
        # shape(output) = [B x seq_len x D/num_heads]

        return output, attention_weights

    def forward(self, x):
        # shape(x) = [B x seq_len x D]

        Q = [linear_Q(x) for linear_Q in self.linear_Qs]
        K = [linear_K(x) for linear_K in self.linear_Ks]
        V = [linear_V(x) for linear_V in self.linear_Vs]
        # shape(Q, K, V) = [B x seq_len x D/num_heads] * num_heads

        output_per_head = []
        attn_weights_per_head = []
        # shape(output_per_head) = [B x seq_len x D/num_heads] * num_heads
        # shape(attn_weights_per_head) = [B x seq_len x seq_len] * num_heads
        for Q_, K_, V_ in zip(Q, K, V):
            
            ##run scaled_dot_product_attention
            output, attn_weight = self.scaled_dot_product_attention(Q_, K_, V_)
            # shape(output) = [B x seq_len x D/num_heads]
            # shape(attn_weights_per_head) = [B x seq_len x seq_len]
            output_per_head.append(output)
            attn_weights_per_head.append(attn_weight)

        output = torch.cat(output_per_head, -1)
        attn_weights = torch.stack(attn_weights_per_head).permute(1, 0, 2, 3)
        # shape(output) = [B x seq_len x D]
        # shape(attn_weights) = [B x num_heads x seq_len x seq_len]
        
        projection = self.dropout(self.mha_linear(output))

        return projection, attn_weights

