import torch
import torch.nn as nn


class FeatSubSampling(nn.Module):
    def __init__(self):
        super(FeatSubSampling, self).__init__()
        self.convs = None
        self.lin = None
        self.dp = None

    def forward(self, X):                   # (N, t, 512)
        X = X.unsqueeze(1)                  # (N, c=1, t, 512)
        out = self.convs(X)                 # (N, odim, f(t), f(512))
        out = out.permute(0, 2, 1, 3)       # (N, f(t), odim, f(512))
        b, t = out.shape[:2]
        out = out.reshape(b, t, -1)         # (N, f(t), odim*f(512))
        out = self.lin(out)                 # (N, f(t), dim
        out = self.dp(out)
        return out


class FeatSubSampling1(nn.Module):
    """Doesn't perform any subsampling"""
    def __init__(self, idim, odim, p):
        super(FeatSubSampling1, self).__init__()
        if idim == odim:
            self.lin = nn.Identity()
        else:
            self.lin = nn.Sequential(
                nn.Linear(idim, odim),
                nn.Dropout(p=p)
            )

    def forward(self, X):
        return self.lin(X)


class FeatSubSampling2(FeatSubSampling):
    """Subsamples visualfrontend's features' lengths to 1/2"""
    def __init__(self, idim, odim, p):
        # For example, the input is the visual features shaped as (N, T, 512), then `512` is the idim
        # odim is the dim of the COnformer Encoder
        super(FeatSubSampling2, self).__init__()
        def f(x): return (x - 1) // 2
        self.convs = nn.Sequential(
            nn.Conv2d(1, odim, (3, 3), (2, 2), (0, 0)),
            nn.ReLU(inplace=True)
        )
        # This linear layer is super huge, almost as huge as 4096 * 4096
        self.lin = nn.Linear(odim * f(idim), odim)
        self.dp = nn.Dropout(p=p)


class FeatSubSampling4(FeatSubSampling):
    """Subsamples visualfrontend's features' lengths to 1/4"""
    def __init__(self, idim, odim, p):
        super(FeatSubSampling4, self).__init__()
        def f(x): return (x - 1) // 2
        self.convs = nn.Sequential(
            nn.Conv2d(1, odim, (3, 3), (2, 2), (0, 0)),
            nn.ReLU(inplace=True),
            nn.Conv2d(odim, odim, (3, 3), (2, 2), (0, 0)),
            nn.ReLU(inplace=True),
        )
        self.lin = nn.Linear(odim * f(f(idim)), odim)
        self.dp = nn.Dropout(p=p)
