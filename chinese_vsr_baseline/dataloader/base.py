import os
import sys
import logging
import random
import pickle
from typing import Union

import numpy as np
import torch
from torch.utils.data import Dataset
from turbojpeg import TurboJPEG, TJPF_BGR

from data_augmentation import DataAugumentation
from my_tokenizers import MappingTokenizer

logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)


class Seq2SeqBaseVideo(Dataset):
    is_training: bool
    aug_args: dict
    tokenizer: Union[MappingTokenizer]

    max_len: int
    batch_size: int
    base_frms: int

    samples: list
    decoder: TurboJPEG

    def __init__(
            self, is_training: bool, aug_args: dict,
            tokenizer: Union[MappingTokenizer],
            max_len: int, batch_size: int, base_frms: int,
            sample_init_args: tuple
    ):
        self.is_training = is_training
        self.max_len = max_len
        self.base_frms = base_frms
        if self.is_training:
            self.batch_size = batch_size
        else:
            self.batch_size = 1

        self.samples = []
        self.aug_args = aug_args
        self.decoder = TurboJPEG()
        self.tokenizer = tokenizer
        self.mean_batch_size = -1.
        self._init_samples(*sample_init_args)

        self.shuffle(shuffle_minibatch=self.is_training)
        
    def _init_samples(self, *args):
        raise NotImplementedError()

    def _init_minibatches(self):
        self.minibatches = []
        self.minibatches.append([])
        max_frm_len = -1
        samples_in_batch = 0

        if self.is_training:   
            sorted_samples = sorted(self.samples, key=lambda x: x["frm_len"])
        else:                  
            sorted_samples = self.samples

        for sample in sorted_samples:
            frm_len = sample["frm_len"]
            max_frm_len = max(frm_len, max_frm_len)
            samples_in_batch += 1
            num_frms_in_batch = max_frm_len * samples_in_batch
            mm_overflow = num_frms_in_batch > self.base_frms or samples_in_batch > self.batch_size
            if mm_overflow:   
                self.minibatches.append([])
                samples_in_batch = 1
                max_frm_len = frm_len
            self.minibatches[-1].append(sample)

    def shuffle(self, shuffle_minibatch=True):
        if self.is_training:
            random.shuffle(self.samples)
        self._init_minibatches()

        if shuffle_minibatch:
            assert self.is_training
            random.shuffle(self.minibatches)
        self.mean_batch_size = len(self.samples) / len(self.minibatches)

    def _get_one_sample(self, sample):
        pkl_path = sample["pkl_path"]
        trans = sample["trans"]
        labels = self.tokenizer.tokenize(trans)
        feats = [
            self.decoder.decode(t, pixel_format=TJPF_BGR) for t in pickle.load(open(pkl_path, "rb"))
        ]
        feats = np.stack(feats, axis=0).transpose(0, 3, 1, 2) 
        if self.aug_args.get('mpc_aug') is not None:
            feats = DataAugumentation(
            feats, self.is_training,
            crop_size=(88, 88), p=self.aug_args['p'],
            drop_frm=self.aug_args["drop_frm"], Tmask_p=self.aug_args["temporal_masking"],mpc_aug=True,
        )
        else:
            feats = DataAugumentation(
                feats, self.is_training,
                crop_size=(88, 88), p=self.aug_args['p'],
                drop_frm=self.aug_args["drop_frm"], Tmask_p=self.aug_args["temporal_masking"],
            )
        
        feats = np.expand_dims(feats, axis=1) 
        return {
            "feat": torch.from_numpy(np.ascontiguousarray(feats)).float(),
            "feat_len": len(feats),
            "label": torch.tensor(labels).long(),
            "label_len": len(labels),
        }

    def __getitem__(self, idx):
        uncollated_samples = [self._get_one_sample(t) for t in self.minibatches[idx]]
        return uncollated_samples

    def __len__(self):
        return len(self.minibatches)
