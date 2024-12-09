import os
import sys
import logging

import torch
import torch.nn as nn

from models import ResNet

logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def conv3d():
    return nn.Sequential(
        nn.Conv3d(
            1, 64,
            (5, 7, 7), (1, 2, 2), (2, 3, 3)
        ),
        nn.BatchNorm3d(64),
        nn.ReLU(True),
        nn.MaxPool3d((1, 3, 3), (1, 2, 2), (0, 1, 1))
    )


def conv2p1d():
    return nn.Sequential(
        nn.Conv3d(
            1, 45,
            kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False
        ),
        nn.BatchNorm3d(45),
        nn.ReLU(True),
        nn.Conv3d(
            45, 64,
            kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0), bias=False
        ),
        nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
    )


class Conv3dResNet(nn.Module):
    def __init__(self, resnet_type, block_type="BasicBlock", c3d_type="conv3d"):
        assert resnet_type in ("ResNet18", "ResNet34", "ResNet152")
        super().__init__()

        # (B, C, seq_len, 88, 88) -> (B, C, seq_len, 24, 24)
        logger.info(f"Using {c3d_type} as 3D convolution.")
        if c3d_type == "conv3d":
            self.conv3d = conv3d()
        elif c3d_type == "conv2p1d":
            self.conv3d = conv2p1d()
        else:
            raise NotImplementedError(f"{c3d_type} is not implemented.")

        # (B*C, seq_len, 24, 24) -> (B*C, seq_len, 512)
        logger.info(f"Using {resnet_type}.")
        if resnet_type == "ResNet18":
            self.resnet = ResNet.ResNet18(block_type)
        else:
            raise NotImplementedError

    def forward(self, X, X_lens):
        # print("X.shape",X.shape)
        out = self.conv3d(X.transpose(1, 2)).transpose(1, 2)  # (N, L, 64, 88, 88)
        # print("out1.shape",out.shape)
        N, L, odim, H, W = out.size()
        out = out.contiguous().view(-1, odim, H, W)  # (N*T, C, h, w)
        # print("out2.shape",out.shape)
        out = self.resnet(out).view(N, L, 512)  # (N, T, 512)
        # print("return res:",out.shape)
        return out, X_lens