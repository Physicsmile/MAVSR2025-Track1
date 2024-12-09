import torch
import torch.nn as nn
from torch.nn import functional as F
import cv2
import os
import numpy as np
from models.beam_search import DecoderInterface


def get_len_mask(b, max_len, feat_lens, device):
    """
    Get the self-attention mask of the encoder, which input is the output of the visual frontend.

    Args:
        b:
        max_len:
        feat_lens:
        device:

    Returns: attn_mask, (b, max_len, max_len). Every atten_mask[i] is a square matrix, whose entries in the
            upper triangle is 0 and otherwise 1(Entry being 1 means masking this position).
            Notice that the dtype of attn_mask must be int(torch.int8, torch.int16, torch.int32, etc)

        e.g.: batch size = 2, max feat len = 4, feat_lens = [2, 3], then the mask is:
          [[[0, 0, 1, 1],
            [0, 0, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1]],
           [[0, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [1, 1, 1, 1]]]
    """
    attn_mask = torch.ones((b, max_len, max_len), device=device)
    for i in range(b):
        attn_mask[i, :feat_lens[i], :feat_lens[i]] = 0
    return attn_mask.to(torch.bool)

def get_pad_mask(b, max_len, feat_lens, device):
    '''
    Get the pad mask of the encoder, which input is the output of the visual frontend.

    Args:
        b:
        max_len:
        feat_lens:
        device:

    Returns: pad_mask, (b, max_len). Every pad_mask[i] is a vector, the last few positions are 1 meaning to be masked.
    '''
    pad_mask = torch.zeros((b, max_len), device=device)
    for i in range(b):
        pad_mask[i, feat_lens[i]:] = 1
    return pad_mask.to(torch.bool)

def get_subsequent_mask(b, max_len, device):
    """
    Supposing the input is X, this function makes X only visible to the inputs before it

    Notes:
        Why it should be a triu: one can't see the tokens after it. In other words, token_0 can't see token_1, ...,
        token_n. Since mask[i, j] should be 1 if token_i can't see token_j, the mask matrix should be upper triangular
        matrix. As a result, the attention matrix will be lower triangluar matrix.

    Args:
        b: batch-size.
        max_len: the length of the whole seqeunce.
        device: cuda or cpu.

    Returns:
        a matrix like:
            [[0, 1, 1, 1, 1],
             [0, 0, 1, 1, 1],
             [0, 0, 0, 1, 1],
             [0, 0, 0, 0, 1],
             [0, 0, 0, 0, 0]]
        but of data type bool.
    """

    return torch.triu(torch.ones((b, max_len, max_len), device=device), diagonal=1).to(torch.bool)     # or .to(torch.uint8)


def get_enc_dec_mask(b, max_feat_len, feat_lens, max_label_len, device):
    """
    Get the mask used in the attention between encoder's output and decoder's input.
    Args:
        b: batch_size
        max_feat_len:
        feat_lens: a list. The i-th item is the length of the length of i-th input.
        max_label_len:
        device:

    Returns:
        e.g.: batch size = 2, max feat len = 4, max label len = 2, feat_lens = [2, 3], then the mask is:
            [[[0, 0, 1, 1],
              [0, 0, 1, 1]],
             [[0, 0, 0, 1],
              [0, 0, 0, 1]]]
        but of data type bool.
    """
    attn_mask = torch.zeros((b, max_label_len, max_feat_len), device=device)       # (b, seq_q, seq_k)
    for i in range(b):
        attn_mask[i, :, feat_lens[i]:] = 1
    return attn_mask.to(torch.bool)


def get_attn_key_mask(seq_q, seq_k, pad_token):
    """
    Make special tokens(such PAD, BOS, EOS) invisible to network forward and backward propagation.
    This is only used in the decoder because there's a visual frontend and no explicit labels in the encoder.
    Use get_visual_attn_mask instead when applied after ResNet frontend in the decoder.

    Notes:
        - being equal to the arg 'pad_token' means that it has no contributions to gradients and shuoldn't be involved
          in the later computations

    Args:
        pad_token: the special id, such as 'sos', 'eos', 'ignore', etc.
        seq_q: the input query sequence, (b, max_len)
        seq_k: the input key sequence, (b, max_len)

    Returns:

    """
    assert seq_q.dim() == 2 and seq_k.dim() == 2
    q_dim, k_dim = seq_q.shape[1], seq_k.shape[1]
    return seq_k.detach().eq(pad_token).unsqueeze(1).repeat((1, q_dim, 1)).to(torch.bool)


class TMDecoding(DecoderInterface):
    def __init__(
            self, frontend: nn.Module, encoder: nn.Module, decoder: nn.Module,
            frontend_dim: int, enc_in_dim: int, enc_out_dim: int,
            dec_in_dim: int, dec_out_dim: int, vocab: int,

    ):
        """

        Args:
            frontend:
            encoder:
            decoder:
            frontend_dim:
            enc_in_dim:
            enc_out_dim:
            dec_in_dim:
            dec_out_dim:
            n_cls:
        """
        super().__init__()
        self.frontend = frontend
        self.encoder = encoder
        self.decoder = decoder

        self.frontend_dim = frontend_dim
        self.enc_in_dim = enc_in_dim
        self.enc_out_dim = enc_out_dim
        assert enc_out_dim == dec_in_dim
        self.dec_in_dim = dec_in_dim
        self.dec_out_dim = dec_out_dim
        self.vocab = vocab

        if frontend_dim != enc_in_dim:
            self.lin1 = nn.Linear(frontend_dim, enc_in_dim)
        else:
            self.lin1 = nn.Identity()
        self.lin2 = nn.Linear(dec_out_dim, vocab)
        self.enc_linear = nn.Linear(enc_out_dim, vocab)   

        self.word_embedding = nn.Embedding(vocab, dec_in_dim)

    def init_cache(self, X: torch.Tensor, X_lens: torch.Tensor, beam_size: int):
        out, X_lens = self.frontend(X, X_lens)  # (B*bms, T, frontend_dim)
        out = self.lin1(out)  # (B*bms, T, enc_in_dim)
        b, T, dim = out.size()
        device = out.device
        out = out.unsqueeze(1).repeat(1, beam_size, 1, 1).view(-1, T, dim)  # (B, T, dim) -> (B*bms, T, dim)
        X_lens = X_lens.unsqueeze(1).repeat(1, beam_size).view(-1)
        enc_mask = get_len_mask(b * beam_size, T, X_lens, device)
        enc_out, X_lens, enc_attns = self.encoder(out, X_lens, enc_mask)
        return enc_out, X_lens

    def forward_one_step(self, cache, tokens: torch.Tensor):
        enc_out, X_lens = cache
        device = enc_out.device
        b, max_feat_len, *_ = enc_out.size()
        max_label_len = tokens.size(1)
        assert b == tokens.size(0)
        dec_mask = get_subsequent_mask(b, max_label_len, device)
        dec_enc_mask = get_enc_dec_mask(b, max_feat_len, X_lens, max_label_len, device)
        dec_out, *_ = self.decoder(tokens, enc_out, dec_mask, dec_enc_mask)
        logits = self.lin2(dec_out)[:, -1]       
        logp = torch.log_softmax(logits, dim=-1)
        return logp, cache

    def forward(self, X: torch.Tensor, X_lens: torch.Tensor, labels: torch.Tensor):
        X_lens, labels = X_lens.long(), labels.long()
        b = X.size(0)
        max_feat_len = X.size(1)
        max_label_len = labels.size(1)
        device = X.device

        # frontend
        front_out, X_lens = self.frontend(X, X_lens)
        front_out = self.lin1(front_out)                                # (B, T, enc_in_dim)

        # encoder
        pad_mask = get_len_mask(b, max_feat_len, X_lens, device) 
        enc_out, X_lens, enc_attns = self.encoder(front_out, X_lens, pad_mask)

        ctc_scores = self.enc_linear(enc_out)  # for ctc loss

        # decoder
        dec_mask = get_subsequent_mask(b, max_label_len, device)
        dec_enc_mask = get_enc_dec_mask(b, max_feat_len, X_lens, max_label_len, device)
        dec_out, dec_attns, dec_enc_attns = self.decoder(labels, enc_out, dec_mask, dec_enc_mask)
        logits = self.lin2(dec_out)

        return {
            "logits": logits,
            "ctc_scores": ctc_scores,
            "attns": (enc_attns, dec_attns, dec_enc_mask),
            "input_lens": X_lens
        }
