import math

import numpy as np
import torch
import torch.nn as nn

from models.transformer.utils import pos_sinusoid_embedding


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    @staticmethod
    def forward(X):
        return X * X.sigmoid()


class GLU(nn.Module):
    def __init__(self, dim):
        super(GLU, self).__init__()
        self.dim = dim

    def forward(self, X):
        out, gate = X.chunk(2, dim=self.dim)
        return out * gate.sigmoid()


class DepthWiseConv1d(nn.Module):
    def __init__(self, idim, odim, k, s, p):
        """

        Args:
            idim: input dimension
            odim: output dimension
            k: kernel size
            s: stride
            p: padding
        """
        super(DepthWiseConv1d, self).__init__()
        assert odim % idim == 0
        self.conv = nn.Conv1d(idim, odim, k, s, p, groups=idim)

    def forward(self, X):
        return self.conv(X)


class PointWiseConv1d(nn.Module):
    def __init__(self, idim, odim, s, p):
        """

        Args:
            idim: input dimension
            odim: output dimension
            s: stride
            p: padding
        """
        super(PointWiseConv1d, self).__init__()
        self.conv = nn.Conv1d(idim, odim, (1,), s, p)

    def forward(self, X):
        return self.conv(X)


class RelativePosMultiHeadAttention(nn.Module):
    def __init__(self, dim, n, p):
        """
        Multi-head Attention with relative position encoding
        Args:
            dim: encoder's feature dimension
            n: number of heads
            p: dropout ratio
        """
        super(RelativePosMultiHeadAttention, self).__init__()
        assert dim % n == 0
        hdim = dim // n                     # dimention per head
        self.dim, self.n, self.hdim = dim, n, hdim
        self.W_Q = nn.Linear(dim, dim)
        self.W_K = nn.Linear(dim, dim)
        self.W_V = nn.Linear(dim, dim)
        self.W_pos = nn.Linear(dim, dim, bias=False)    # linear projection of position embeddings
        self.dp = nn.Dropout(p=p)
        self.u = nn.Parameter(torch.Tensor(n, hdim))
        self.v = nn.Parameter(torch.Tensor(n, hdim))
        self.W_out = nn.Linear(dim, dim)
        torch.nn.init.xavier_normal_(self.u)
        torch.nn.init.xavier_normal_(self.v)

    @staticmethod
    def _relative_shift(X):
        b, n, q_len, k_len = X.shape
        zeros = X.new_zeros(b, n, q_len, 1)
        X_pad = torch.cat([zeros, X], dim=-1).view(b, n, k_len + 1, q_len)
        X = X_pad[:, :, 1:].view_as(X)
        return X

    def forward(self, q, k, v, pos_emb, mask=None):
        b = q.size(0)
        n, hdim = self.n, self.hdim
        q = self.W_Q(q).reshape(b, -1, n, hdim)                     # b, seq_len, num_heads, head_dim
        k = self.W_K(k).reshape(b, -1, n, hdim)
        v = self.W_V(v).reshape(b, -1, n, hdim)
        pos_emb = self.W_pos(pos_emb).reshape(b, -1, n, hdim)       # b, seq_len, num_heads, head_dim
        # content: (b, n, l, hd) * (b, n, l, hd) -> (b, n, hd, l)
        content = torch.matmul((q + self.u).permute(0, 2, 1, 3), k.permute(0, 2, 3, 1))
        pos = torch.matmul((q + self.v).permute(0, 2, 1, 3), pos_emb.permute(0, 2, 3, 1))
        pos = self._relative_shift(pos)

        score = (content + pos) / math.sqrt(self.dim)

        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, n, 1, 1)             # b, q_len, k_len -> b, num_heads, q_len, k_len
            score.masked_fill_(mask, -1e4)
        attn = torch.softmax(score, dim=-1)
        attn = self.dp(attn)
        context = torch.matmul(attn, v.permute(0, 2, 1, 3))
        context = context.permute(0, 2, 1, 3).contiguous().reshape(b, -1, n * hdim)
        context = self.W_out(context)
        return context, attn


class FeedForwardModule(nn.Module):
    def __init__(self, idim, p, e=4, prenorm=False):
        """

        Args:
            idim: input dimension
            p: doprout
            e: expansion factor
        """
        super(FeedForwardModule, self).__init__()
        self.ln = nn.LayerNorm(idim)                            # only layer norm the last dimension
        self.lin1 = nn.Linear(idim, idim * e, bias=True)        # e: expansion factor, 4 by default
        self.swish = Swish()
        self.dp1 = nn.Dropout(p=p)
        self.lin2 = nn.Linear(idim * e, idim, bias=True)
        self.dp2 = nn.Dropout(p=p)
        self.prenorm = prenorm

    def forward(self, X):               # B, T, dim
        if self.prenorm:
            out = X
        else:
            out = self.ln(X)
        out = self.lin1(out)
        out = self.swish(out)
        out = self.dp1(out)
        out = self.lin2(out)
        out = self.dp2(out)
        out = 0.5 * out + X
        if self.prenorm:
            return self.ln(out)
        else:
            return out


class RelativePosMHSA(nn.Module):
    def __init__(self, tgt_len, dim, n, p, prenorm=False):
        super(RelativePosMHSA, self).__init__()
        self.pos_emb = nn.Embedding.from_pretrained(pos_sinusoid_embedding(tgt_len, dim), freeze=True)
        self.ln = nn.LayerNorm(dim)
        self.attn = RelativePosMultiHeadAttention(dim, n, p)
        self.dp = nn.Dropout(p=p)
        self.prenorm = prenorm

    def forward(self, X, mask=None):
        b, l, dim = X.size()
        device = X.device
        if self.prenorm:
            out = X
        else:
            out = self.ln(X)
        pos_emb = self.pos_emb(torch.arange(l, device=device)).repeat(b, 1, 1)       # (batchsize, seq_len, d_model)
        out, attn = self.attn(out, out, out, pos_emb, mask)
        out = self.dp(out)
        # don't forget this residual connection
        out = out + X
        if self.prenorm:
            return self.ln(out), attn
        else:
            return out, attn


class ConvModule(nn.Module):
    def __init__(self, k, dim, p, prenorm=False):
        super(ConvModule, self).__init__()
        self.ln = nn.LayerNorm(dim)
        self.pw_conv1 = PointWiseConv1d(dim, 2 * dim, 1, 0)
        self.glu = GLU(dim=1)
        self.dw_conv = DepthWiseConv1d(dim, dim, k, 1, (k - 1) // 2)
        self.bn = nn.BatchNorm1d(dim)
        self.swish = Swish()
        self.pw_conv2 = PointWiseConv1d(dim, dim, 1, 0)
        self.dp = nn.Dropout(p=p)
        self.prenorm = prenorm

    def forward(self, X):               # B, T, dim
        if self.prenorm:
            out = X
        else:
            out = self.ln(X)
        out = out.permute(0, 2, 1)      # B, dim, T
        out = self.pw_conv1(out)        # B, 2*dim, T
        out = self.glu(out)             # B, dim, T
        out = self.dw_conv(out)         # B, dim, T
        out = self.bn(out)
        out = self.swish(out)
        out = self.pw_conv2(out)
        out = self.dp(out)
        out = out.permute(0, 2, 1)      # (B, T, dim)
        out = out + X
        if self.prenorm:
            return self.ln(out)
        else:
            return out
