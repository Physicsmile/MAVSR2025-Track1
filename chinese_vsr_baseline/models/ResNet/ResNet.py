import os
import sys
import logging
from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np

from .blocks import BasicBlock

formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
logging.basicConfig(
    format=formatter,
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ResNet(nn.Module):
    def __init__(self, block_type, in_planes, layer_dims, layer_widths,
                 conv=nn.Conv2d, norm_layer=nn.BatchNorm2d, pool=nn.AdaptiveAvgPool2d((1, 1))):
        """
        Notes:
            Typically, the argument conv, norm_layer and pool stay unchanged and always designed for images.
            However, they're allowed to be customized for some other inputs.
            For example, conv=nn.Conv1d, norm_layer=nn.BatchNorm1d and pool=nn.AdaptiveAvgPool1d((1, 1)) are suitble
            for 1-d signals, such as audios.
        Args:
            block_type: e.g. BasicBlock
            in_planes: e.g. 64
            layer_dims: [512, 512, 64, 128]
            layer_widths: e.g. [2, 2, 2, 2]
            conv:
            norm_layer:
            pool:
        """
        super(ResNet, self).__init__()
        assert len(layer_dims) == len(layer_widths)

        self.in_planes = in_planes
        self.conv = conv
        self.norm_layer = norm_layer
        self.avgpool = pool
        self.convs = []

        # strides: down-sampling output size. For example, (112, 112) -> (112, 112) -> (56, 56) -> (28, 28) -> (14, 14)
        strides = [1] + [2] * (len(layer_widths) - 1)
        # strides = [2] * (len(layer_widths))
        for i, (out_plane, layer_width, stride) in enumerate(zip(layer_dims, layer_widths, strides)):
            subblock_name = f"{block_type}_{i}"
            subblock = self._make_layers(i, eval(block_type), out_plane, layer_width, stride, 1)
            self.convs.append((subblock_name, subblock))
        self.convs = nn.Sequential(OrderedDict(self.convs))

        self.bn = nn.BatchNorm1d(self.in_planes)  # now self.in_planes == out_plane after self._make_layers

        self._init_weights()

        self.state = "Model name: {}, " \
                     "block_type: {}, " \
                     "in_planes: {}, " \
                     "layer_dims: {}, " \
                     "layer_widths: {}, ".format(
            self.__class__.__name__, block_type, in_planes, layer_dims, layer_widths
        )
        # logger.info(self)

    def __repr__(self):
        return self.state + '\n' + super().__repr__()

    def _make_layers(self, i, block, out_plane, layer_width, stride=2, padding=1):
        # The default padding should be 1 and shouldn't be changed
        in_and_out_dims = [(self.in_planes, out_plane * block.expansion)]
        in_and_out_dims.extend(
            [(out_plane * block.expansion, out_plane * block.expansion)] * (layer_width - 1)
        )

        layers = []
        strides = [stride] + [1] * (len(in_and_out_dims) - 1)
        for j, ((idim, odim), stride) in enumerate(zip(in_and_out_dims, strides)):
            if stride != 1 or idim != odim:
                # Down-sampling for two times
                assert stride == 2 or (idim / odim == 2 and idim % odim == 0)
                down_sample = nn.Sequential(
                    self.conv(idim, odim, 1, 2, 0),  # kernel size: 1, stride: 2, padding: 0
                    self.norm_layer(odim),
                )
            else:
                down_sample = None
            layers.append((f"blocks_{i}_{j}", block(idim, odim, stride, padding, down_sample)))
        self.in_planes = in_and_out_dims[-1][-1]
        return nn.Sequential(OrderedDict(layers))

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                n = int(np.prod(m.kernel_size)) * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, X):
        # (N, C, H, W) -> (N, C)
        out = X
        for conv in self.convs:
            out = conv(out)  # (N, C, H, W) -> (N, C, h, w)
        out = self.avgpool(out)  # (N, C, h, w) -> (N, C, 1, 1)
        out = out.view(out.size(0), -1)
        out = self.bn(out)  # (N, C) -> (N, C)
        return out

    
def ResNet18(block_type="BasicBlock", in_planes=64):
    return ResNet(
        block_type=block_type,
        in_planes=in_planes,
        layer_dims=[in_planes, 128, 256, 512],
        layer_widths=[2, 2, 2, 2]
    )
