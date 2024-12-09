import os
import sys
import logging

import torch
import torch.nn as nn

from .blocks import ConformerEncoderBlock

formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
logging.basicConfig(
    format=formatter,
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ConformerEncoder(nn.Module):
    def __init__(
            self, enc_dim, ff_e, num_layers, num_heads, kernel_size, tgt_len,
            dropout_posffn, dropout_attn, dropout_conv,
            prenorm=False, block_type="ConformerEncoderBlock"
    ):
        """

        References: <<Conformer: Convolution-augmented Transformer for Speech Recognition>>

        Args:
            enc_dim: input's dim
            ff_e: expansion factor of PosFFN
            dim: encoder's dim
            num_layers: number of encoder layers
            num_heads: number of attention heads
            kernel_size: the size of the kernel used in ConformerConvModule
            tgt_len: the maximum length of input sequences
            p: dropout ratio
        """
        super(ConformerEncoder, self).__init__()

        block = eval(block_type)
        self.layers = nn.ModuleList([
            block(kernel_size, tgt_len, enc_dim, ff_e, num_heads, dropout_posffn, dropout_attn, dropout_conv, prenorm)
            for _ in range(num_layers)
        ])

        self.state = "Model name: {}, " \
                     "block type: {}, " \
                     "input dim: {}, " \
                     "ff expansion: {}, " \
                     "num_layers: {}, " \
                     "num_heads: {}, " \
                     "kernel_size: {}, " \
                     "tgt_len: {}, " \
                     "dropout_posffn: {}, dropout_attn: {}, dropout_conv: {}, " \
                     "prenorm: {}.".format(
            self.__class__.__name__, block_type, enc_dim, ff_e, num_layers, num_heads, kernel_size, tgt_len,
            dropout_posffn, dropout_attn, dropout_conv, prenorm
        )

        logger.info(self)

    def __repr__(self):
        return self.state + '\n' + super().__repr__()

    def forward(self, X, X_lens, mask=None):
        out = X
        attns = []
        for i, layer in enumerate(self.layers):
            out, attn = layer(out, mask)
            attns.append(attn)
        return out, X_lens, attns