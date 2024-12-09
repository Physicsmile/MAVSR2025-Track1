from torch.utils.data import DataLoader

import data_collators
import my_tokenizers


def tv101_init_dataloder(cfg):
    from .tv101 import TV101
    collator_cls = getattr(data_collators, cfg.collator_cls)
    collator_args = cfg.collator_args or {}
    collator = collator_cls(**collator_args)

    tokenizer_cls = getattr(my_tokenizers, cfg.tokenizer_cls)
    tokenizer_args = cfg.tokenizer_args or {}
    tokenizer = tokenizer_cls(**tokenizer_args)

    dset = TV101(
        phase=cfg.phase, dset_dir=cfg.dset_dir,trans_dir=cfg.trans_dir,
        max_len=cfg.max_len, batch_size=cfg.batch_size, base_frms=cfg.base_frms,
        aug_args=cfg.aug_args, tokenizer=tokenizer
    )
    
    return DataLoader(
        dataset=dset,
        batch_size=1,
        num_workers=4 if dset.is_training else 0,
        collate_fn=collator,
        pin_memory=True,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        drop_last=False,
        prefetch_factor=2 if dset.is_training else None,
    )


def mov20_init_dataloder(cfg):
    from .mov20 import MOV20
    collator_cls = getattr(data_collators, cfg.collator_cls)
    collator_args = cfg.collator_args or {}
    collator = collator_cls(**collator_args)

    tokenizer_cls = getattr(my_tokenizers, cfg.tokenizer_cls)
    tokenizer_args = cfg.tokenizer_args or {}
    tokenizer = tokenizer_cls(**tokenizer_args)

    dset = MOV20(
        phase=cfg.phase, dset_dir=cfg.dset_dir,trans_dir=cfg.trans_dir,
        max_len=cfg.max_len, batch_size=cfg.batch_size, base_frms=cfg.base_frms,
        aug_args=cfg.aug_args, tokenizer=tokenizer
    )
    
    return DataLoader(
        dataset=dset,
        batch_size=1,
        num_workers=4 if dset.is_training else 0,
        collate_fn=collator,
        pin_memory=True,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        drop_last=False,
        prefetch_factor=2 if dset.is_training else None,
    )
