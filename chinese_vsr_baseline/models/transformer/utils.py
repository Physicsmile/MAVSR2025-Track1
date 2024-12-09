
import numpy as np
import torch
import torch.nn as nn


def pos_sinusoid_embedding(seq_len, d_model):
    """
    Generate frozen position embeddings with sinusiod functions.
    See <<Attention is all you need>> for more details.

    Args:
        seq_len: the length of the desired sequence to be embedded.
        d_model: model's dimension, typically equal to 512.

    Returns: position embeddings.

    """
    embeddings = np.zeros((seq_len, d_model))
    for i in range(d_model):
        f = np.sin if i % 2 == 0 else np.cos
        embeddings[:, i] = f(np.arange(0, seq_len) / np.power(1e4, 2 * (i // 2) / d_model))
    return torch.from_numpy(embeddings.astype(np.float32))


def reverse_dim1(labels, lens):
    """
    (Deprecated)Apply sequence reverse in dim 1.

    Args:
        labels:
        lens:

    Returns:

    """
    new_labels = labels.new_zeros(labels.shape)
    for i in range(labels.size(0)):
        new_labels[i, lens[i]:] = labels[i, lens[i]:]
        new_labels[i, :lens[i]] = labels[i, :lens[i]].flip(0)
    return new_labels

class PosEmbedding(nn.Module):
    def __init__(self, max_len, d_model):
        super(PosEmbedding, self).__init__()
        self.d_model = d_model

        self.encoding = torch.zeros(max_len, d_model)

        self.encoding.requires_grad = False

        pos = torch.arange(0, max_len)

        pos = pos.unsqueeze(-1)

        ii = torch.arange(0, d_model, step=2)

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (ii / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (ii / d_model))) \
            if d_model % 2 == 0 else torch.cos(pos / (10000 ** ((ii[:-1]) / d_model)))


    def forward(self, x):
        out = self.encoding[:x.size(1), :].to(x.device)
        return out