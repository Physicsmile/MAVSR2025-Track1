import os
import sys
import logging

import torch
import torch.nn as nn

logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# References: https://github.com/wenet-e2e/wenet/blob/main/wenet/transformer/label_smoothing_loss.py
class LabelSmoothingLoss(nn.Module):
    """Label-smoothing loss.

    In a standard CE loss, the label's data distribution is:
    [0,1,2] ->
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]

    In the smoothing version CE Loss,some probabilities
    are taken from the true label prob (1.0) and are divided
    among other labels.

    e.g.
    smoothing=0.1
    [0,1,2] ->
    [
        [0.9, 0.05, 0.05],
        [0.05, 0.9, 0.05],
        [0.05, 0.05, 0.9],
    ]

    Args:
        size (int): the number of class
        padding_idx (int): padding class id which will be ignored for loss
        smoothing (float): smoothing rate (0.0 means the conventional CE)
        normalize_length (bool):
            normalize loss by sequence length if True
            normalize loss by batch size if False
    """
    def __init__(self,
                 size: int,
                 padding_idx: int,
                 smoothing: float,
                 normalize_length: bool = True):
        """Construct an LabelSmoothingLoss object."""
        super(LabelSmoothingLoss, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="none")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.normalize_length = normalize_length

        logger.info(', '.join([
            f"loss: {self.__class__.__name__}",
            f"ignore index: {self.padding_idx}",
            f"smoothing: {smoothing}",
            f"vocab size: {size}",
            f"normalize length: {normalize_length}"
        ]))

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute loss between x and target.

        The model outputs and data labels tensors are flatten to
        (batch*seqlen, class) shape and a mask is applied to the
        padding part which should not be calculated for loss.

        Args:
            x (torch.Tensor): prediction **logits** (batch * seqlen, class)
            target (torch.Tensor):
                target signal masked with self.padding_id (batch * seqlen, )
        Returns:
            loss (torch.Tensor) : The KL loss, scalar float value
        """
        assert x.size(1) == self.size           # prevent that the vocab size is wrong
        batch_size = x.size(0)
        x = x.view(-1, self.size)               # flattening
        target = target.view(-1)

        true_dist = torch.zeros_like(x)         # (B, vocab)
        true_dist.fill_(self.smoothing / (self.size - 1))
        ignore = (target == self.padding_idx)     # (B,)
        total = len(target) - ignore.sum().item()
        target = target.masked_fill(ignore, 0)  # avoid -1 index
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        lgsf = torch.log_softmax(x, dim=1)

        if lgsf.isnan().any():
            raise ValueError("nan in log_softmax")
        kl = self.criterion(torch.log_softmax(x, dim=1), true_dist)
        denom = total if self.normalize_length else batch_size

        nan = kl.isnan().any()
        inf = kl.isinf().any()
        c = kl.isnan().sum() if nan else kl.isinf().sum()
        n = kl.numel()
        if nan or inf:
            kl[kl.isnan()] = 0.            
            kl[kl.isinf()] = 0.
            logger.warning(
                f"Device {torch.cuda.current_device()}: "
                f"Found {'NaN' if nan else 'Inf'} (element: {c}/{n} -> {c / n * 100:.2f}%) in the loss. Skipped"
            )

        return kl.masked_fill(ignore.unsqueeze(1), 0).sum() / denom

