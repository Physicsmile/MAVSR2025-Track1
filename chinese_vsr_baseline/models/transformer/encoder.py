import os
import sys
import logging

import numpy as np
import torch
import torch.nn as nn

from .blocks import UniEncoderBlock
from .utils import pos_sinusoid_embedding, PosEmbedding

formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
logging.basicConfig(
    format=formatter,
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class UniEncoder(nn.Module):
    def __init__(
            self, dropout_emb, dropout_posffn, dropout_attn,
            num_layers, enc_dim, num_heads, dff, tgt_len,
            prenorm=False, block_type="UniEncoderBlock"
    ):
        """

        Unidirectional Transformer Encoder.

        Args:
            dropout_emb: dropout ratio of Position Embeddings.
            dropout_posffn: dropout ratio of PosFFN.
            dropout_attn: dropout ratio of attention module.
            num_layers: number of encoder layers
            enc_dim: input dimension of encoder
            num_heads: number of attention heads
            dff: dimensionf of PosFFN
            tgt_len: the maximum length of sequences
            prenorm: whether to use pre-norm
            block_type: which encoder block to use
        """
        super(UniEncoder, self).__init__()
        self.tgt_len = tgt_len                         
        self.pos_emb = nn.Embedding.from_pretrained(pos_sinusoid_embedding(tgt_len, enc_dim), freeze=True)
        self.emb_dropout = nn.Dropout(dropout_emb)     
        block = eval(block_type)
        self.layers = nn.ModuleList(
            [block(enc_dim, num_heads, dff, prenorm, dropout_posffn, dropout_attn) for _ in range(num_layers)]
        )
        self._init_weights()

        self.state = "Model name: {}, " \
                     "num_layers: {}, " \
                     "enc_dim: {}, " \
                     "num_heads: {}, " \
                     "dff: {}, " \
                     "block_type: {}.".format(
            self.__class__.__name__, num_layers, enc_dim, num_heads, dff, block_type
        )
        logger.info(self)

    def __repr__(self):
        return self.state + '\n' + super().__repr__()

    def forward(self, X, X_lens, mask=None):
        """

        Args:
            X: (B, T, dim)
            X_lens:
            mask: (B, T, T)

        Returns:

        """
        batch_size, seq_len, d_model = X.shape
        out = X + self.pos_emb(torch.arange(seq_len, device=X.device)) 
        out = self.emb_dropout(out)

        attns = []
        for layer in self.layers:
            out, attn = layer(out, mask)
            attns.append(attn)
        return out, X_lens, attns

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = int(np.prod(m.kernel_size)) * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

from torch.nn import TransformerEncoderLayer, TransformerEncoder

class TMEncoder(nn.Module):
    def __init__(
            self, dropout_emb, dropout_posffn, dropout_attn,
            num_layers, enc_dim, num_heads, dff, tgt_len,
            prenorm=False, block_type=None
    ):
        """

        The original Transformer Encoder.

        Args:
            dropout_emb: dropout ratio of Position Embeddings.
            dropout_enc: dropout ratio of encoder.
            num_layers: number of encoder layers
            enc_dim: input dimension of encoder
            num_heads: number of attention heads
            dff: dimensionf of PosFFN
            tgt_len: the maximum length of sequences
            prenorm: whether to use pre-norm
            block_type: which encoder block to use
        """
        super(TMEncoder, self).__init__()
        self.num_heads = num_heads
        self.tgt_len = tgt_len
        self.pos_emb = PosEmbedding(tgt_len, enc_dim)
        self.emb_dropout = nn.Dropout(dropout_emb)
        if num_layers > 0:
            self.enc = TransformerEncoder(
                TransformerEncoderLayer(d_model=enc_dim, nhead=num_heads, dim_feedforward=dff, \
                                        dropout=dropout_attn, batch_first=True, norm_first=prenorm),
                num_layers
            )
        else:
            self.enc = torch.nn.Identity()

    def forward(self, X, X_lens, src_mask=None):
        """

        Args:
            X: (B, T, dim)
            X_lens:
            mask: (B, T, T)

        Returns:

        """

        out = X + self.pos_emb(X)
        out = self.emb_dropout(out)

        if isinstance(self.enc, torch.nn.Identity):
            out = out
        else:
            out = self.enc(out, src_key_padding_mask=src_mask)
        return out, X_lens, None