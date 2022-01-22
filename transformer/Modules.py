import torch
import torch.nn as nn
import torch.nn.functional as F

# For the self-attention module
class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, temperature, attn_dropout=0.2):
        super().__init__()

        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3)) # While k is (batch_size, heads, length of key, M_k) dimensional, transpose the last 2 dimensions

        if mask is not None:
            attn = attn.masked_fill(mask, -1e9) # mask with -infty values on future value correlations

        attn = self.dropout(F.softmax(attn, dim=-1)) # softmax over the last dimension of the computed attention, with dropout of attention (randomly 0ing attn values). 
        output = torch.matmul(attn, v) # multiply with the value matrix
        # computation of single-head attention values done!: output
        return output, attn
