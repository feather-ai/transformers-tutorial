# %%
import torch
import torch.nn.functional as F
import math
import numpy as np

# %%
def scaled_dot_product_attention(Q, K, V, dk=4):
    ##matmul Q and K
    QK = torch.matmul(Q, K.T)
    
    ##scale QK by the sqrt of dk
    matmul_scaled = QK / math.sqrt(dk)
    
    attention_weights = F.softmax(matmul_scaled, dim=-1)

    ## matmul attention_weights by V
    output = torch.matmul(attention_weights, V)

    return output, attention_weights

# %%
def print_attention(Q, K, V, n_digits = 3):
    temp_out, temp_attn = scaled_dot_product_attention(Q, K, V)
    temp_out, temp_attn = temp_out.numpy(), temp_attn.numpy()
    print ('Attention weights are:')
    print (np.round(temp_attn, n_digits))
    print()
    print ('Output is:')
    print (np.around(temp_out, n_digits))

# %%
temp_k = torch.Tensor([[10,0,0],
                      [0,10,0],
                      [0,0,10],
                      [0,0,10]])  # (4, 3)

temp_v = torch.Tensor([[   1,0, 1],
                      [  10,0, 2],
                      [ 100,5, 0],
                      [1000,6, 0]])  # (4, 3)

# %%
# This `query` aligns with the second `key`,
# so the second `value` is returned.
temp_q = torch.Tensor([[0, 10, 0]])  # (1, 3)
print_attention(temp_q, temp_k, temp_v)

# %%
# This query aligns with a repeated key (third and fourth), 
# so all associated values get averaged.
temp_q = torch.Tensor([[0, 0, 10]])  # (1, 3)
print_attention(temp_q, temp_k, temp_v)

# %%
# This query aligns equally with the first and second key, 
# so their values get averaged.
temp_q = torch.Tensor([[10, 10, 0]])  # (1, 3)
print_attention(temp_q, temp_k, temp_v)

# %%
temp_q = torch.Tensor([[0, 10, 0], [0, 0, 10], [10, 10, 0]])  # (3, 3)
print_attention(temp_q, temp_k, temp_v)