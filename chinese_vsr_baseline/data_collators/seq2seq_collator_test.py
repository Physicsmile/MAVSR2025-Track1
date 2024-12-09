import os
import sys
import logging
import warnings

import torch

logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Seq2SeqCollatorTest(object):
    def __init__(self, special_ids):
        if isinstance(special_ids, str):
            special_ids = eval(special_ids)
        assert isinstance(special_ids, dict) and all([t in special_ids for t in ("sos", "eos", "ignore", "blank")])
        self.sos = special_ids["sos"]
        self.eos = special_ids["eos"]
        self.blank = special_ids["blank"]
        self.ignore = special_ids["ignore"]

        self.batch_size_multiple_by_eight = False
        if self.batch_size_multiple_by_eight:
            logger.warning("self.batch_size_multiple_by_eight = True")

    def collate(self, batch):
        # The features should be torch.Tensor instead of numpy.ndarray
        # pad_dim: 0
        # The keys of an input batch: feat, label, feat_len, label_len

        max_feat_len, max_label_len = float("-inf"), float("-inf")
        batch = batch[0]

        for data in batch:
            max_feat_len = max(data["feat_len"], max_feat_len)

        if self.batch_size_multiple_by_eight:
            if max_feat_len % 8 != 0:
                max_feat_len += 8 - max_feat_len % 8

        feat_out_shape = (len(batch), max_feat_len,) + batch[0]["feat"].shape[1:]


        feats = torch.zeros(feat_out_shape)

        for i, data in enumerate(batch):
            n_elem = len(data.keys())
            feats[i, :data["feat_len"], ...] = data["feat"]

        return {
            "feats": feats.float(),
            "feat_lens": torch.tensor([t["feat_len"] for t in batch], dtype=torch.long),
        }

    def __call__(self, *args, **kwargs):
        return self.collate(*args, **kwargs)

    def __repr__(self):
        return "collator name: {}, " \
               "sos: {}, eos: {}, blank, ignore: {}".format(
            self.__class__.__name__, self.sos, self.eos, self.blank, self.ignore
        )