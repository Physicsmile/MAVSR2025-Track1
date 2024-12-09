
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotAttention(nn.Module):
    def __init__(self, p=0.):
        """
        Compute the scaled dot self-attention, where the attention is computed from any two dimentions in the same input.

        Args:
            p: dropout ratio. Zero by default.
        """
        assert 0. <= p <= 1.
        super(ScaledDotAttention, self).__init__()
        self.dropout = nn.Dropout(p=p)

    def forward(self, Q, K, V, d_k, attn_mask=None):
        """
        Forward pass of ScaledDotAttention.

        Args:
            Q: queries.
            K: keys.
            V: values.
            d_k: the dimension of keys, used for normalizing attentions.
            attn_mask: attention mask. None by default, which means not applying any mask.

        Returns: context, attentions.

        """
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        if attn_mask is not None:
            scores.masked_fill_(attn_mask, -1e4)
            # scores.masked_fill_(attn_mask, -1e20)     # not work in fp16 mode
        attns = self.dropout(torch.softmax(scores, dim=-1))
        context = torch.matmul(attns, V)
        return context, attns


class MultiHeadAttention(nn.Module):
    def __init__(self, d_k, d_v, d_model, num_heads, p=0.):
        """

        References: <<Attention is all you need>>. https://arxiv.org/pdf/1706.03762.pdf.

        Args:
            d_k: the dimension of keys.
            d_v: the dimension of values.
            d_model: the dimension of model. Should be equal to num_heads * d_k.
            num_heads: number of attention heads.
            p: dropout ratio. Zero by default.
        """
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.num_heads = num_heads

        self.W_Q = nn.Linear(d_model, d_k * num_heads)
        self.W_K = nn.Linear(d_model, d_k * num_heads)
        self.W_V = nn.Linear(d_model, d_v * num_heads)
        self.W_out = nn.Linear(d_v * num_heads, d_model)

        self.attn_block = ScaledDotAttention(p)      
        self.dropout_p = p

        # Kaiming normalization
        # References: <<Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification>>
        nn.init.normal_(self.W_Q.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.W_K.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.W_V.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))
        nn.init.normal_(self.W_out.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

    def forward(self, Q, K, V, attn_mask):
        """

        Args:
            Q: Queries.
            K: Keys.
            V: Values.
            attn_mask: attention mask(optional).

        Returns: output, attentions.

        """
        N = Q.shape[0]
        d_k, d_v = self.d_k, self.d_v
        num_heads = self.num_heads

        Qs = self.W_Q(Q).view(N, -1, num_heads, d_k).transpose(1, 2)
        Ks = self.W_K(K).view(N, -1, num_heads, d_k).transpose(1, 2)
        Vs = self.W_V(V).view(N, -1, num_heads, d_v).transpose(1, 2)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, num_heads, 1, 1)
            attn_mask = attn_mask.bool()


        context, attn = self.attn_block(Qs, Ks, Vs, d_k, attn_mask)

        context = context.transpose(1, 2).contiguous().reshape(N, -1, d_v * num_heads)

        context = self.W_out(context)
        out = context
        return out, attn
   