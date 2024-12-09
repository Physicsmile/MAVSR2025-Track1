import torch
import torch.nn as nn

from .modules import FeedForwardModule, RelativePosMHSA, ConvModule


class ConformerEncoderBlock(nn.Module):
    def __init__(self, kernel_size, tgt_len, dim, ff_e, n, feed_p, attn_p, conv_p, prenorm=False):
        """
        The encoder block of Conformer

        References:
            <<Conformer:Convolution-augmented Transformer for Speech Recognition>>

        Args:
            kernel_size: the size of the convolution kernel used in ConvModule
            tgt_len:
            dim:
            ff_e: the expansion factor in the FeedForwardModule
            n: num_heads
            p: dropout ratio. Set all dropout ratios to 0.1
        """
        super(ConformerEncoderBlock, self).__init__()
        self.feedfoward1 = FeedForwardModule(dim, feed_p, ff_e, prenorm=prenorm)
        self.mhsa = RelativePosMHSA(tgt_len, dim, n, attn_p, prenorm=prenorm)
        self.conv = ConvModule(kernel_size, dim, conv_p, prenorm=prenorm)
        self.feedfoward2 = FeedForwardModule(dim, feed_p, prenorm=prenorm)
        self.ln = nn.LayerNorm(dim)
        self.prenorm = prenorm

    def forward(self, X, mask):
        if self.prenorm:
            X = self.ln(X)
        out = self.feedfoward1(X)
        out, attn = self.mhsa(out, mask)
        out = self.conv(out)
        out = self.feedfoward2(out)
        if self.prenorm:
            return out, attn
        else:
            return self.ln(out), attn
