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


class CTC(nn.Module):
    """
    using torch.nn.CTCLoss
    NOTE: in nn.CTCLoss(log_probs, targets, input_lengths, target_lengths)
        Args: 
            log_probs:  log_softmax(probs), [T, B, C]
            targets:    


    in this CTC
    Args:
        scores:     ctc scores of a batch of sequences
        labels:     
        feat_lens:  
        label_lens: 
    """
    def __init__(
            # self, weight, blank, enabled, reduction="mean", zero_infinity=True
            self, blank, enabled, reduction="mean", zero_infinity=True
    ):
        """
        in cfg
            ctc:
                weight: 0.1
                ctc_args:
                    blank: ${constants.special_ids.blank}
                    enabled: true
                    reduction: "mean"
                    zero_infinity: true
        """
        super().__init__()
        self.enabled = enabled
        self.ctc = nn.CTCLoss(blank=blank, reduction=reduction, zero_infinity=zero_infinity)
        logger.info(
            f"loss: {self.__class__.__name__}, "
            f"enabled: {enabled}, "
            f"reduction: {reduction}, "
            f"zero_infinity: {zero_infinity}."
        )

    def forward(self, scores, labels, feat_lens, label_lens):
        if self.enabled:
            scores = scores.float() # CTCscores
            log_probs = scores.transpose(0, 1).log_softmax(2)  # (B, T, vocab) -> (T, B, vocab)
            return self.ctc(log_probs, labels, feat_lens, label_lens)
        else:
            return torch.tensor(0.)