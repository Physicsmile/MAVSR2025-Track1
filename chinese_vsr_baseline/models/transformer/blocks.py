
import torch
import torch.nn as nn

from .attentions import MultiHeadAttention
from .posFFN import PoswiseFFN


class UniEncoderBlock(nn.Module):
    def __init__(self, dim, n, dff, prenorm, dropout_posffn, dropout_attn):
        """

        Unidirectional encoder block

        Args:
            dim: input dimension
            n: number of attention heads
            dff: dimention of PosFFN (Positional FeedForward)
            prenorm: whether to use pre-norm
            dropout_posffn: dropout ratio of PosFFN
            dropout_attn: dropout ratio of attention module
        """
        assert dim % n == 0
        hdim = dim // n     # dimension of each attention head
        super(UniEncoderBlock, self).__init__()
        self.prenorm = prenorm
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.multi_head_attn = MultiHeadAttention(hdim, hdim, dim, n, dropout_attn)
        self.poswise_ffn = PoswiseFFN(dim, dff, p=dropout_posffn)

    def forward(self, enc_in, attn_mask):
        if self.prenorm:
            enc_in = self.norm1(enc_in)
        residual = enc_in
        context, attn = self.multi_head_attn(enc_in, enc_in, enc_in, attn_mask)
        if self.prenorm:
            out = residual + context
        else:
            out = self.norm1(residual + context)

        if self.prenorm:
            out = self.norm2(out)
        residual = out
        out = self.poswise_ffn(out)
        if self.prenorm:
            out = residual + out
        else:
            out = self.norm2(residual + out)

        return out, attn


class UniDecoderBlock(nn.Module):
    def __init__(self, dim, n, dff, prenorm, dropout_posffn, dropout_attn):
        """

        Unidirectional decoder block

        Args:
            dim: input dimension
            n: number of attention heads
            dff: dimention of PosFFN (Positional FeedForward)
            prenorm: whether to use pre-norm
            dropout_posffn: dropout ratio of PosFFN
            dropout_attn: dropout ratio of attention module
        """
        super(UniDecoderBlock, self).__init__()
        assert dim % n == 0
        hdim = dim // n
        self.prenorm = prenorm
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.dec_attn = MultiHeadAttention(hdim, hdim, dim, n, dropout_attn)
        self.enc_dec_attn = MultiHeadAttention(hdim, hdim, dim, n, dropout_attn)
        self.poswise_ffn = PoswiseFFN(dim, dff, p=dropout_posffn)

    def forward(self, dec_in, enc_out, dec_mask, dec_enc_mask):
        if self.prenorm:
            dec_in = self.norm1(dec_in)
        residual = dec_in
        context, dec_attns = self.dec_attn(dec_in, dec_in, dec_in, dec_mask)
        if self.prenorm:
            dec_out = residual + context
        else:
            dec_out = self.norm1(residual + context)

        if self.prenorm:
            dec_out = self.norm2(dec_out)
        residual = dec_out
        context, dec_enc_attns = self.enc_dec_attn(dec_out, enc_out, enc_out, dec_enc_mask)
        if self.prenorm:
            dec_out = residual + context
        else:
            dec_out = self.norm2(residual + context)

        if self.prenorm:
            dec_out = self.norm3(dec_out)
        residual = dec_out
        out = self.poswise_ffn(dec_out)
        if self.prenorm:
            dec_out = residual + out
        else:
            dec_out = self.norm2(residual + out)

        return dec_out, dec_attns, dec_enc_attns
