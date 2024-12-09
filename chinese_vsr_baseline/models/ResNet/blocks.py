import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
            self, inplane, outplane, stride, padding=1, down_sample=None,
            bias=False, se=False, expansion=1, activation_type="prelu"
    ):
        # Structure: conv1 -> bn1 -> relu1 -> conv2 -> bn2 (-> downsampling) -> add residual
        super(BasicBlock, self).__init__()
        self.expansion = expansion
        self.down_sample = down_sample

        self.conv1 = nn.Conv2d(inplane, outplane, (3, 3), (stride, stride), padding, bias=bias)
        self.bn1 = nn.BatchNorm2d(outplane)
        # The setting of the second Conv2d: kernel size: (3, 3), padding: (1, 1), stride: 1
        self.conv2 = nn.Conv2d(outplane, outplane, (3, 3), (1, 1), padding, bias=bias)
        self.bn2 = nn.BatchNorm2d(outplane)

        if activation_type == "relu":
            self.relu1 = nn.ReLU(inplace=True)
            self.relu2 = nn.ReLU(inplace=True)
        elif activation_type == "prelu":
            self.relu1 = nn.PReLU(num_parameters=outplane)
            self.relu2 = nn.PReLU(num_parameters=outplane)
        else:
            raise NotImplementedError(f"Activation {activation_type} not implemented!!!")

        if se:
            raise NotImplementedError("Squeeze-and-Excitation not implemented")
        if expansion != 1:
            raise NotImplementedError

    def forward(self, X):
        out = self.conv1(X)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        residual = self.down_sample(X) if self.down_sample is not None else X
        return self.relu2(out + residual)

