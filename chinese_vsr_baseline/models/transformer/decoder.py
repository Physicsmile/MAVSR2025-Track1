import os
import sys
import logging

import numpy as np
import torch
import torch.nn as nn

from .blocks import UniDecoderBlock
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


class UniDecoder(nn.Module):
    def __init__(
            self, dropout_emb, dropout_posffn, dropout_attn,
            num_layers, dec_dim, num_heads, dff, tgt_len, tgt_vocab_size,
            prenorm=False, block_type="UniDecoderBlock"
    ):
        """
        Unidirectional Transformer Decoder.

        Args:
            dropout_emb: dropout ratio of Position Embeddings.
            dropout_posffn: dropout ratio of PosFFN.
            dropout_attn: dropout ratio of attention module.
            num_layers: number of encoder layers
            dec_dim: input dimension of decoder
            num_heads: number of attention heads
            dff: dimensionf of PosFFN
            tgt_len: the target length to be embedded.
            tgt_vocab_size: the target vocabulary size.
            prenorm: whether to use pre-norm
            block_type: which encoder block to use
        """
        super(UniDecoder, self).__init__()

        self.tgt_emb = nn.Embedding(tgt_vocab_size, dec_dim)
        self.pos_emb = nn.Embedding.from_pretrained(pos_sinusoid_embedding(tgt_len, dec_dim), freeze=True)
        self.dropout_emb = nn.Dropout(p=dropout_emb)                      

        block = eval(block_type)
        self.layers = nn.ModuleList(
            [block(dec_dim, num_heads, dff, prenorm, dropout_posffn, dropout_attn) for _ in range(num_layers)]
        )
        self._init_weights()

    def forward(self, labels, enc_out, dec_mask, dec_enc_mask):
        bs, tgt_len = labels.shape            
        dec_out = self.tgt_emb(labels) + self.pos_emb(torch.arange(tgt_len, device=labels.device))  
        dec_out = self.dropout_emb(dec_out)   
        
        dec_attns, dec_enc_attns = [], []
        for i, layer in enumerate(self.layers):
            dec_out, dec_attn, dec_enc_attn = layer(dec_out, enc_out, dec_mask, dec_enc_mask)
            dec_attns.append(dec_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_out, dec_attns, dec_enc_attns

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = int(np.prod(m.kernel_size)) * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

from torch.nn import TransformerDecoder, TransformerDecoderLayer

class TMDecoder(nn.Module):
    def __init__(
        self, dropout_emb, dropout_posffn, dropout_attn,
        num_layers, dec_dim, num_heads, dff, tgt_len, tgt_vocab_size,
        prenorm=False, block_type=None
    ):
        """
        Unidirectional Transformer Decoder.

        Args:
            dropout_emb: dropout ratio of Position Embeddings.
            dropout_posffn: dropout ratio of PosFFN.
            dropout_attn: dropout ratio of attention module.
            num_layers: number of encoder layers
            dec_dim: input dimension of decoder
            num_heads: number of attention heads
            dff: dimensionf of PosFFN
            tgt_len: the target length to be embedded.
            tgt_vocab_size: the target vocabulary size.
            prenorm: whether to use pre-norm
            block_type: which encoder block to use
        """
        super(TMDecoder, self).__init__()
        self.num_heads = num_heads
        self.tgt_emb = nn.Embedding(tgt_vocab_size, dec_dim)
        self.pos_emb = PosEmbedding(tgt_len, dec_dim)
        self.dropout_emb = nn.Dropout(p=dropout_emb)                            
        if num_layers > 0:
            self.layers = TransformerDecoder(
                TransformerDecoderLayer(d_model=dec_dim, nhead=num_heads, dim_feedforward=dff, \
                                        dropout=dropout_attn, batch_first=True, norm_first=prenorm),
                num_layers
            )
        else:
            self.layers = torch.nn.Identity()
    
    def forward(self, labels, enc_out, dec_mask, dec_enc_mask):
        tgt_emb = self.tgt_emb(labels) + self.pos_emb(labels)
        tgt_emb = self.dropout_emb(tgt_emb)
        
        dec_mask = dec_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        dec_enc_mask = dec_enc_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        dec_mask = dec_mask.reshape(-1, dec_mask.size(2), dec_mask.size(3))
        dec_enc_mask = dec_enc_mask.reshape(-1, dec_enc_mask.size(2), dec_enc_mask.size(3))

        dec_out = self.layers(tgt_emb, enc_out, tgt_mask=dec_mask, memory_mask=dec_enc_mask)
        return dec_out, [], []
