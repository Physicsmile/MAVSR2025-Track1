import torch
import torch.nn as nn


class PoswiseFFN(nn.Module):
    def __init__(self, d_model, d_ff, p=0.):
        """
        Positional-wise Feedforward Nerwork. See <<Attention is all you need>> for more details.
        Basically, it's equal to two linear fully connected layers with relu activation.

        Args:
            d_model: model's input dimension.
            d_ff: `bottleneck` dimension.
            p: dropout ratio.
        """
        super(PoswiseFFN, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.conv1 = nn.Conv1d(d_model, d_ff, 1, 1, 0)
        self.conv2 = nn.Conv1d(d_ff, d_model, 1, 1, 0)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=p)

    def forward(self, X):
        out = self.conv1(X.transpose(1, 2))    
        out = self.relu(out)
        out = self.conv2(out).transpose(1, 2)  
        out = self.dropout(out)
        return out
