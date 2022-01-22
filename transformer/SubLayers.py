import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import transformer.Constants as Constants
from transformer.Modules import ScaledDotProductAttention

# use the transformer.Modules to make Multi-head attention modules
class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention module """

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1, normalize_before=True):
        super().__init__()

        self.normalize_before = normalize_before
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        # d_model is the length of each sample embedding
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False) # weights for all the heads together, used for query, key, value computation, so no bias
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.xavier_uniform_(self.w_qs.weight) # xavier initialization of the weights of query, key, value
        nn.init.xavier_uniform_(self.w_ks.weight)
        nn.init.xavier_uniform_(self.w_vs.weight)

        self.fc = nn.Linear(d_v * n_head, d_model) # linear fully-connected layer to compute cumulative attention values over all heads
        nn.init.xavier_uniform_(self.fc.weight) # Xavier initialization of weight

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5, attn_dropout=dropout) # single-head attention object, temp = \sqrt{M_k}

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6) # normalization over M elements of embedding
        self.dropout = nn.Dropout(dropout) # dropout layer

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1) # batch_size, length of q, k, v

        residual = q
        if self.normalize_before:
            q = self.layer_norm(q)

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)  # For head axis broadcasting.

        output, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        output = output.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        output = self.dropout(self.fc(output)) # combining multi-head to one set of attention scores using fc, applying dropout
        output += residual # residual connections, same as input to the self-attention module

        if not self.normalize_before: # if you hadn't normalized before, normalize now
            output = self.layer_norm(output)
        return output, attn


class PositionwiseFeedForward(nn.Module):
    """ Two-layer position-wise feed-forward neural network. """
    # inputs are the multihead attention outputs

    def __init__(self, d_in, d_hid, dropout=0.1, normalize_before=True):
        super().__init__()

        self.normalize_before = normalize_before

        self.w_1 = nn.Linear(d_in, d_hid) # d_in: d_model, d_out = n_head
        self.w_2 = nn.Linear(d_hid, d_in)

        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        if self.normalize_before:
            x = self.layer_norm(x)

        x = F.gelu(self.w_1(x)) # gelu: Gaussian Error Linear Unit: x∗Φ(x)
        x = self.dropout(x)
        x = self.w_2(x)
        x = self.dropout(x)
        x = x + residual # introduce residual connection for better training

        if not self.normalize_before:
            x = self.layer_norm(x) # if hadn't normalized earlier, normalize here
        return x
