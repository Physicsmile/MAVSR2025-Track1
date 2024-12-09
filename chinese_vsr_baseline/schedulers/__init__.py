import os
import sys
import logging

from .schedulers import TriStageScheduler, ReciprocalScheduler

logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def init_tri_stage_scheduler(optimizer, cfg):
    scheduler = TriStageScheduler(
        optimizer=optimizer,
        warmup_ratio=cfg.warmup_ratio,
        hold_ratio=cfg.hold_ratio,
        decay_ratio=cfg.decay_ratio,
        init_lr=cfg.init_lr,
        peak_lr=cfg.peak_lr,
        final_lr=cfg.final_lr,
        num_epoch=cfg.max_epoch,
        iter_per_epoch=cfg.steps_per_epoch,
    )
    logger.info(
        f"Loading scheduler TriStageScheduler..."
        f"config: {cfg}."
    )
    return scheduler


def init_reciprocal_scheduler(optimizer, cfg):
    if cfg.lr_mode == "single":
        lr = [cfg.lr] if not isinstance(cfg.lr, list) else cfg.lr
    elif cfg.lr_mode == "multiple":
        lr = [l for k, l in cfg.multiple_lrs.items()]
    else:
        raise ValueError(f"Wrong lr mode: {cfg.lr_mode}")
    return ReciprocalScheduler(
        optimizer=optimizer,
        lr=lr,
        max_epoch=cfg.max_epoch,
        steps_per_epoch=cfg.steps_per_epoch,
        warmup_iter=cfg.warmup_iter,
        warmup_ratio=cfg.warmup_ratio,
        warmup_epoch=cfg.warmup_epoch,
        warmup_mode=cfg.warmup_mode,
        layer_wise_lr_decay_eta=getattr(cfg, "layer_wise_lr_decay_eta", None),
        layer_wise_lr_decay_layers=getattr(cfg, "layer_wise_lr_decay_layers", None)
    )
    logging.info(
        f"Loading ReciprocalScheduler..."
        f"config: {cfg}."
    )
