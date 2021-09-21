import torch


def create_mask(size):
    # since this mask is the same for a batch being fed into the model,
    # we will the mask Tensor with the batch size = 1.
    # Broadcasting will allow us to replicate this mask across all the other elements in the batch
    mask = torch.ones((1, size, size)).triu(1)
    mask = mask == 0
    return(mask)