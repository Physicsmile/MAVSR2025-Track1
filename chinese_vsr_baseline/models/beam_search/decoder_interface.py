from typing import Union, Tuple

import torch
import torch.nn as nn


class DecoderInterface(nn.Module):
    def __init__(self):
        super().__init__()

    def init_cache(self, feats: torch.Tensor, feat_lens: torch.Tensor, beam_size: int):
        return NotImplementedError()

    def forward_one_step(self, cache: Union[Tuple, torch.Tensor], tokens: torch.Tensor):
        raise NotImplementedError()