
import os
import sys
import logging
import time
import glob
import warnings
import multiprocessing as mp
from pathlib import Path
from argparse import ArgumentParser
from collections import OrderedDict
import re
import yaml


def get_logger(name):
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        stream=sys.stdout,
    )
    logger = logging.getLogger(name)
    return logger


def run(q: mp.Queue, gpu_idx: int, proc_idx: int):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    import tensorboardX
    from termcolor import colored
    import torch
    from accelerate.utils import set_seed

    import dataloader
    import models

    # set_deterministic
    seed = 0
    warnings.warn(f"Seed: {seed}.")
    set_seed(seed)
    torch.cuda.benchmark = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)

    logger = get_logger(f"Worker_{proc_idx}")
    logger.info(f"Start working. Device: {gpu_idx}.")

    while True:
        x = q.get(block=True, timeout=None)
        if x is None:
            logger.info("Finished")
            q.close()
            break
        model_ph, cfg, args = x

        # start inference
        start = time.time()
        # load data_loader
        data_loader_init_fn = getattr(dataloader, cfg.dataloader_init_fn)
        data_loader = data_loader_init_fn(cfg.dataset_cfg)

        # load model
        model = models.init_model(cfg.model_cfg).cuda()
        logger.info(model)
        phase_str = f"{args.phase}_{args.tag}"

       

        state_dict = torch.load(model_ph, map_location=f"cuda:{torch.cuda.current_device()}")
        if "model" in state_dict.keys():
            state_dict = state_dict["model"]    # models trained with ddp
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=True)
        logger.info(
            f"Model path: {colored(model_ph, 'red')}. "
            f"Missing: {missing_keys}, unexpected: {unexpected_keys}"
        )
        save_dir = f"./mov20_{phase_str}.txt"
        test(
            model=model,
            data_loader=data_loader,
            decode_cfg=cfg.decode_cfg,
            save_dir=save_dir,
            phase=phase_str,
            logger=logger,
            proc_idx=proc_idx,
        )


def test(model, data_loader, decode_cfg, save_dir, phase, logger, proc_idx):

    import tqdm
    import editdistance
    import torch

    import models
    from utils import AverageMeter

    # init beam search decoder
    logger.info(f"Initializing beam search decoder...")
    model.eval()
    tokenizer = data_loader.dataset.tokenizer
    beam_search_decoder = models.init_beam_search_decoder(
        decode_cfg, model=model, tokenizer=tokenizer
    )

    pbar = tqdm.tqdm(range(len(data_loader)), desc="Inference", position=proc_idx * 100)
    logger.info(f"phase: {phase}")

    summary = []
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            X, X_lens = data["feats"].cuda(), data["feat_lens"].cuda()
            assert X.size(0) == 1, "Only support batch-size==1!!!"
            pred_sents = beam_search_decoder(X, X_lens)
            for p in pred_sents:
                summary.append(
                    f"{p}"
                )
            pbar.update()


    txt_ph = save_dir

    with open(txt_ph, 'a') as f:
        f.write('\n'.join(summary) + '\n')



def ph2epochidx(model_ph):
    return int(os.path.basename(model_ph).split('_')[1])


if __name__ == "__main__":
    # get args
    mp.set_start_method("spawn")

    import torch
    from termcolor import colored
    from utils import AttrDict, make_dirs

    def get_test_args():
        argparser = ArgumentParser()
        argparser.add_argument("cfg_ph", type=str)
        argparser.add_argument("phase", type=str)
        argparser.add_argument("--epoch_idxs", type=int, default=[], nargs='+')
        argparser.add_argument("--start_epoch", type=int, default=0)
        argparser.add_argument("--end_epoch", type=int, default=1000)
        argparser.add_argument("--model_ph", type=str, default='')
        argparser.add_argument("--check_last", action="store_true")
        argparser.add_argument("--tag", type=str, default='')

        argparser.add_argument("--model_average_max_epoch", "--nmax", type=int, default=0)
        argparser.add_argument("--nbest", type=int, default=0)

        argparser.add_argument("--nprocs", "-n", type=int, default=-1)
        argparser.add_argument("--gpu_idxs", type=str, default="4,5")

        argparser.add_argument("--train_path", type=str, default="")

        args = argparser.parse_args()
        args.gpu_idxs = [int(t) for t in args.gpu_idxs.split(',')]
        if args.nprocs == -1:
            args.nprocs = len(args.gpu_idxs)
        print(args.cfg_ph)
        cfg = yaml.load(open(args.cfg_ph, 'r'), Loader=yaml.SafeLoader)
        cfg = AttrDict(cfg)
        cfg.model_tag = '.'.join(os.path.split(args.cfg_ph)[-1].split('.')[:-1])
        cfg.model_dir = os.path.join(cfg.train_cfg.ckpt_dir, cfg.dset_type, cfg.model_tag)
        make_dirs(cfg, path=args.train_path)
        
        return args, cfg


    logger = get_logger(__name__)
    args, cfg = get_test_args()
    cfg.dataset_cfg.phase = args.phase

    
    # get model paths
    model_phs = []

    # perform model average if necessary
    max_epoch = args.model_average_max_epoch
    if args.nbest is None:
        model_average_nbest = max_epoch // 10           # take 1/10 for model averaging
    else:
        model_average_nbest = args.nbest
    if args.model_average_max_epoch > 0:
        logger.info(
            f"Performing model average. "
            f"nbest: {model_average_nbest}, max_epoch: {max_epoch}"
        )
        avg_model_ph = os.path.join(
            cfg.model_dir, "train",
            f"nmax{args.model_average_max_epoch}_nbest{model_average_nbest}.pt"
        )
        print("avg_model_ph",avg_model_ph)
        if not os.path.exists(avg_model_ph):
            assert len(ckpt_phs) == model_average_nbest
            model_average(ckpt_phs, avg_model_ph)
        model_phs = [avg_model_ph]

    nprocs = min(len(model_phs), args.nprocs)
    q = mp.Queue(maxsize=len(model_phs) + nprocs)

    for h in model_phs:
        q.put((h, cfg, args))
    for _ in range(nprocs):
        q.put(None)

    procs = []
    for proc_idx in range(nprocs):
        gpu_idx = args.gpu_idxs[proc_idx % len(args.gpu_idxs)]
        p = mp.Process(target=run, args=(q, gpu_idx, proc_idx))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    q.close()

