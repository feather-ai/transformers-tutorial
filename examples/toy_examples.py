# %%
import torch
from mha import MultiHeadAttention
import os
import sys
import pathlib

PACKAGE_PARENT = pathlib.Path.cwd().parent
sys.path.append(str(PACKAGE_PARENT))


# %%
toy_encodings = torch.Tensor([[[0.0, 0.1, 0.2, 0.3], [1.0, 1.1, 1.2, 1.3], [2.0, 2.1, 2.2, 2.3]]]) 
# shape(toy_encodings) = [B, T, D] = (1, 3, 4)
print("Toy Encodings:\n", toy_encodings)

toy_MHA_layer = MultiHeadAttention(d_model=4, num_heads=2)
toy_MHA, _ = toy_MHA_layer(toy_encodings)
print("Toy MHA: \n", toy_MHA)
print("Toy MHA Shape: \n", toy_MHA.shape)

# %%
from layers.residual_layer_norm import ResidualLayerNorm
toy_prev_x = torch.randn(1, 3, 4)
toy_norm_layer = ResidualLayerNorm(d_model=4)
toy_norm = toy_norm_layer(toy_encodings, toy_prev_x)
print("Toy Norm: \n", toy_norm)
print("Toy Norm shape: \n", toy_norm.shape)

# %%
from layers.pwffn import PWFFN
toy_PWFFN_layer = PWFFN(d_model=4, d_ff=16)
toy_PWFFN = toy_PWFFN_layer(toy_norm)
print("Toy PWFFN: \n", toy_PWFFN)
print("Toy PWFFN Shape: \n", toy_PWFFN.shape)

# %%
from layers.embed import Embeddings
toy_vocab = torch.LongTensor([[1, 2, 3, 4, 0, 0]])

toy_embedding_layer = Embeddings(5, pad_idx=0, d_model=4)
toy_embeddings = toy_embedding_layer(toy_vocab)

print("Toy Embeddings: \n", toy_embeddings)
print("Toy Embeddings Shape: \n", toy_embeddings.shape)

# %%
from layers.positional_encoding import PositionalEncoding
toy_PE_layer = PositionalEncoding(d_model=4)
toy_PEs = toy_PE_layer(toy_embeddings)

print("Toy PE: \n", toy_PEs)
print("Toy PE Shape: \n", toy_PEs.shape)

print(toy_PE_layer.pe[0, 0])

# %%
# from layers.encoder import Encoder
# toy_encoder = Encoder(toy_embedding_layer, 4, 2, 2, 8)
# toy_encoder_output, toy_encoder_attn = toy_encoder(toy_vocab)

# print("Toy Encodings: \n", toy_encoder_output)
# print("Toy Encoder Attn Weights: \n", toy_encoder_attn)
# print("Toy Encodings Shape: \n", toy_encoder_output.shape)
# print("Toy Encodings Attn Weights Shape: \n", toy_encoder_attn.shape)

# %%
from create_mask import create_mask
toy_mask = create_mask(10)
print("Toy Mask: \n", toy_mask)
print("Toy Mask Shape: \n", toy_mask.shape)

# %%
toy_scores = torch.arange(100).reshape(1, 10, 10)
print("Toy Scores: \n", toy_scores)
print("Toy Scores Shape: \n", toy_scores.shape)

toy_scores = toy_scores.masked_fill(toy_mask == 0, -1)
print("Toy Scores Masked: \n", toy_scores)
