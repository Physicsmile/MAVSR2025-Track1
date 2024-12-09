import os
import sys
import logging
import json
from pathlib import Path
import warnings
from typing import Union

from my_tokenizers import MappingTokenizer

import csv
import pickle
from .base import Seq2SeqBaseVideo
import random
from typing import Union
import numpy as np
import torch
from torch.utils.data import Dataset
from turbojpeg import TurboJPEG, TJPF_BGR

from data_augmentation import DataAugumentation


logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)



class MOV20(Seq2SeqBaseVideo):
    def __init__(
            self, phase: str, dset_dir: str, trans_dir: str,
            max_len: int, batch_size: int, base_frms: int,
            aug_args: dict, tokenizer: MappingTokenizer,
    ):
        assert phase in ("test", "val")
        self.phase = phase
        super().__init__(
            self.phase == "train", aug_args, tokenizer, max_len, batch_size, base_frms,
            sample_init_args=(dset_dir, trans_dir)
        )

    def _init_samples(self, dset_dir: str, trans_dir: str):
        dset_dir = Path(dset_dir)
        trans_dir = Path(trans_dir)

        all_id = []
        with open(trans_dir, 'r', encoding='utf-8') as csvfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader: 
                _, d = row
                all_id.append(d) 

        self.samples = []
        for id in all_id:
            pkl_path = dset_dir.joinpath(f"{id}.pkl")
            if pkl_path.exists():
                self.samples.append({
                    "pkl_path": pkl_path,
                    "sample_id": id,
                    "frm_len": 1,
                })
       
    def _get_one_sample(self, sample):
        pkl_path = sample["pkl_path"]
        feats = [
            self.decoder.decode(t, pixel_format=TJPF_BGR) for t in pickle.load(open(pkl_path, "rb"))
        ]
        
        feats = np.stack(feats, axis=0).transpose(0, 3, 1, 2)  # (T, H, W, C) -> (T, C, H, W)
        feats = DataAugumentation(
            feats, self.is_training,
            crop_size=(80, 80), p=self.aug_args['p'],
            drop_frm=self.aug_args["drop_frm"], Tmask_p=self.aug_args["temporal_masking"],
        )
        feats = np.expand_dims(feats, axis=1)  # T, 1, H, W

        return {
            "feat": torch.from_numpy(np.ascontiguousarray(feats)).float(),
            "feat_len": len(feats),
        }

