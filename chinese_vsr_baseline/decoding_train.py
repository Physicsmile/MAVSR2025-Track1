import os
import sys
import logging
import time
from pathlib import Path
from argparse import ArgumentParser

import tqdm
import yaml
import editdistance
import numpy as np
import torch
import tensorboardX
from termcolor import colored
from accelerate.utils import set_seed
from accelerate import Accelerator, DistributedDataParallelKwargs

import models as models
import dataloader
import schedulers
import losses
from utils import AttrDict, make_dirs, AverageMeter, print_model_params



logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def get_train_args():
    argparser = ArgumentParser(
        description="""
        Train the baseline model with all samples in the pretraining set and the training set.
        Using ConformerEncoder and using the pretrained visual frontend checkpoint of LRS2.
        """
    )
    argparser.add_argument("cfg_ph", type=str)
    argparser.add_argument("--ngpus", "-n", type=int, required=True)
    argparser.add_argument("--last_epoch", type=int, default=0)
    argparser.add_argument("--log_dir", type=str, default=None)
    argparser.add_argument("--loadmode", type=str, default='.', help="Load ckpt params")
    argparser.add_argument("--find_unused_parameters", action="store_true")
    argparser.add_argument("--print_params", action="store_true")

    args = argparser.parse_args()

    if args.log_dir is None and args.last_epoch == 0:
        or_cfg = yaml.load(open(args.cfg_ph, 'r'), Loader=yaml.SafeLoader)
        cfg = AttrDict(or_cfg)
        cfg.model_tag = '.'.join(os.path.split(args.cfg_ph)[-1].split('.')[:-1])
        make_dirs(cfg)
        yaml.dump(
                or_cfg, open(os.path.join(cfg.model_dir, "config.yaml"), 'w'),
                indent=2, allow_unicode=True
            )
    else:
        cfg = yaml.load(open(os.path.join(args.log_dir, "config.yaml"), 'r'), Loader=yaml.SafeLoader)
        cfg = AttrDict(cfg)
        cfg.model_tag = '.'.join(os.path.split(args.cfg_ph)[-1].split('.')[:-1])
        make_dirs(cfg, path=args.log_dir)
    return cfg, args

        
if __name__ == "__main__":
    if torch.cuda.current_device() == 0:
        logger.info(">>> 💡 Loading args <<<")
    cfg, args = get_train_args()
    train_cfg = cfg.train_cfg

    if torch.cuda.current_device() == 0:
        logger.info(">>> 💡 build accelerator <<<")
    ddp_handler = DistributedDataParallelKwargs(find_unused_parameters=True if args.find_unused_parameters else False)
    accelerator = Accelerator(kwargs_handlers=[ddp_handler])
    if accelerator.state.deepspeed_plugin is not None:
        bs = train_cfg.dataset_cfg.batch_size
        accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = bs
        logger.info(
            f"setting accelerator.state.deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] -> {bs}."
        )

    if accelerator.is_local_main_process:
        logger.info(">>> 💡 set random seeds <<<")
    if accelerator.is_local_main_process:
        logger.info(
            f"Random seed: {train_cfg.random_seed}, "
            f"mixed_precision state: {colored(accelerator.state.mixed_precision, 'red')}"
        )
    set_seed(train_cfg.random_seed)

    if accelerator.is_local_main_process:
        logger.info(">>> 💡 save accelerate state and model cfgs <<<")
        with open(os.path.join(cfg.model_dir, "accelerate.cfg"), 'w') as fp:
            print(accelerator.state, file=fp)

    if accelerator.is_local_main_process:
        logger.info(">>> 💡 load dataloader <<<")
    dataloader_init_fn = getattr(dataloader, cfg.dataloader_init_fn)
    data_loader = dataloader_init_fn(train_cfg.dataset_cfg)

    if accelerator.is_local_main_process:
        logger.info(">>> 💡 load model <<<")
    model = models.init_model(cfg.model_cfg)

    if accelerator.is_local_main_process:
        logger.info(model)

    if accelerator.is_local_main_process:
        logger.info(">>> 💡 load tensorboard writter <<<")
        logger.info(f"Config name: {colored(cfg.config_name, 'red')}")
    writter = tensorboardX.SummaryWriter(log_dir=train_cfg.log_dir)


    if accelerator.is_local_main_process:
        logger.info(">>> 💡 load optimizer <<<")
    train_cfg.opt_cfg.lr *= args.ngpus
    params = [{"params": model.parameters()}]  # or: params = model.parameters()
    optimizer_name = getattr(train_cfg, "optimizer_name", "Adam")
    if accelerator.is_local_main_process:
        logger.info(
            f"Using optimizer: {colored(optimizer_name, 'red')}. "
            f"Lr after scaling: {colored(train_cfg.opt_cfg.lr, 'red')}."
        )
    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(
            params, betas=(0.9, 0.98), eps=1e-8, lr=train_cfg.opt_cfg.lr
        )
    elif optimizer_name == "AdamW":
        # weight_decay=0.04: follow RAVEn
        optimizer = torch.optim.AdamW(
            params, betas=(0.9, 0.98), eps=1e-8, lr=train_cfg.opt_cfg.lr, weight_decay=0.04
        )
    else:
        raise ValueError(f"Not supported optimizer: {optimizer_name}")

    if accelerator.is_local_main_process:
        logger.info(">>> 💡 load scheduler <<<")
    steps_per_epoch = len(data_loader) // args.ngpus
    if len(data_loader) % args.ngpus != 0:
        steps_per_epoch += 1
    if train_cfg.scheduler_name == "one_cycle":
        if args.last_epoch == 0:
            last_epoch_for_one_cycle = -1
        else:
            last_epoch_for_one_cycle = args.last_epoch

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=train_cfg.opt_cfg.lr, epochs=train_cfg.max_epoch,
            steps_per_epoch=steps_per_epoch, pct_start=train_cfg.opt_cfg.pct_start,
            anneal_strategy="cos", last_epoch=last_epoch_for_one_cycle
        )
    elif train_cfg.scheduler_name == "reciprocal":
        train_cfg.opt_cfg.steps_per_epoch = steps_per_epoch
        scheduler = schedulers.init_reciprocal_scheduler(optimizer, train_cfg.opt_cfg)
    else:
        raise NotImplementedError()

    # determine if to load checkpoint
    total_iter = 0
    if accelerator.is_local_main_process:
        logger.info(f">>> 💡 load checkpoint for model, optimizer and scheduler <<<")
    if args.last_epoch > 0:
        ckpts = [str(t) for t in Path(train_cfg.train_dir).rglob("epoch_*.pt")]
        if len(ckpts) > 0 and train_cfg.dynamic_load:
            ckpts = sorted(
                ckpts, key=lambda x: int(os.path.basename(x).split('_')[1]), reverse=True
            )
            if accelerator.is_local_main_process:
                logger.info(
                    # f"Checkpoint candidates: {ckpts}. \n"
                    f"Loading from {ckpts[0]} 🤩."
                )
            args.last_epoch = int(os.path.basename(ckpts[0]).split('_')[1])
            data = torch.load(ckpts[0], map_location="cpu")
            model.load_state_dict(data["model"])
            total_iter = data["step"]
            optimizer.load_state_dict(data["optimizer"])
            scheduler.load_state_dict(data["scheduler"])
            
            
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)

    if accelerator.is_local_main_process:
        logger.info(">>> 💡 💡 💡  start training 💡 💡 💡 <<<")
    accelerator.wait_for_everyone()
    tic = time.time()
    if accelerator.is_local_main_process:
        logger.info(
            f"Start training: {colored(cfg.model_tag, 'red')}. "
            f"Attn loss weight: {cfg.train_cfg.loss_cfg.ce_loss.weight}. "
            f"Last epoch: {args.last_epoch}. "
            f"Last iter: {total_iter}. "
            f"Save_every: {train_cfg.save_every}..."
        )

    ce_crit = losses.LabelSmoothingLoss(**train_cfg.loss_cfg.ce_loss.lsr_args)
    ctc_crit = losses.CTC(**train_cfg.loss_cfg.ctc_loss.ctc_args)

    recorders = {
        k: AverageMeter() for k in ("ce_loss", "ctc_loss", "train_cer")
    }

    for epoch_idx in range(args.last_epoch + 1, train_cfg.max_epoch + 1):  # Indexing from 1
        model.train()  # training mode
        for v in recorders.values():
            v.reset()

        data_loader.dataset.shuffle(shuffle_minibatch=True)
        iterator = accelerator.prepare(data_loader)

        pbar = tqdm.tqdm(
            range(len(iterator)), desc="Training", disable=not accelerator.is_local_main_process
        )
        for i, data in enumerate(iterator, start=0):
            res = model(data["feats"], data["feat_lens"], data["ys_in"])
            
            # CE loss
            logits = res["logits"]  # (B, T, vocab)
            _, preds = logits.max(-1)  # (B, T)

            logits = logits.reshape(-1, logits.size(-1))  # (B, n_cls)
            labels = data["ys_out"].contiguous().reshape(-1)
            ce_loss = ce_crit(logits, labels)
            recorders["ce_loss"].update(ce_loss.item(), 1)

            # CTC loss
            ctc_scores = res["ctc_scores"]
            ctc_loss = ctc_crit(ctc_scores, data["ys_in"][:, 1:], res['input_lens'], data["label_lens"])
            print("ctc_loss:",ctc_loss)
            recorders["ctc_loss"].update(ctc_loss.item(), 1)

            # total loss
            total_loss = ce_loss * train_cfg.loss_cfg.ce_loss.weight + \
                            ctc_loss * train_cfg.loss_cfg.ctc_loss.weight

            # train cer
            preds, gt, y_lens = [t.cpu().detach().numpy() for t in (preds, data["ys_out"], data["label_lens"])]

            tokenizer = data_loader.dataset.tokenizer

            cers = [editdistance.eval(p, g) / len(g) for p, g in zip(preds, gt)]
            cer = np.mean(cers)
            recorders["train_cer"].set(cers, len(cers))

            # backward
            optimizer.zero_grad()
            accelerator.backward(total_loss)
            optimizer.step()
            scheduler.step()

            # write tensorboard
            if accelerator.is_local_main_process:
                writter.add_scalar("lr", optimizer.param_groups[0]["lr"], total_iter)
                if total_iter % train_cfg.log_every == 0:
                    writter.add_scalar("train-cer", cer, total_iter)
                    writter.add_scalar("train-loss", ce_loss, total_iter)

            # refresh pbar
            loss = ce_loss.item()
            pbar.set_postfix({
                "Epoch": epoch_idx,
                "LR": optimizer.param_groups[0]["lr"],
                "Loss": loss,
                "CER": cer
            })
            pbar.update()
            total_iter += 1

        if epoch_idx != train_cfg.max_epoch:
            pbar.reset()

        accelerator.wait_for_everyone()
        if accelerator.is_local_main_process:
            train_cer = recorders["train_cer"].get()
            ce_loss = recorders["ce_loss"].get()

            ckpt_save_path = os.path.join(
                cfg.train_cfg.train_dir,
                f"epoch_{epoch_idx:03d}_traincer_{train_cer:.4f}_celoss_{ce_loss:.4f}.pt"
            )
            unwrapped_model = accelerator.unwrap_model(model)
            torch.save(
                {
                    "model": unwrapped_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "step": total_iter,
                },
                ckpt_save_path
            )
    logging.info(
        f"😀 Training finished. 😀"
        f"Total training time: {(time.time() - tic) / 3600:.2f}h"
    )