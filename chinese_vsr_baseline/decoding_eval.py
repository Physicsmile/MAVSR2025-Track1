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
            # q.close()
            writter.close()
            break
        model_ph, cfg, args = x

        # start inference
        start = time.time()
        # load data_loader
        data_loader_init_fn = getattr(dataloader, cfg.dataloader_init_fn)
        print("cfg.dataset_cfg.phase",cfg.dataset_cfg.phase)
        data_loader = data_loader_init_fn(cfg.dataset_cfg)
        
        # load writter
        writter = tensorboardX.SummaryWriter(log_dir=cfg.train_cfg.log_dir)
        # load model
        model = models.init_model(cfg.model_cfg).cuda()
        logger.info(model)
        phase_str = f"{args.phase}_{args.tag}"

        if args.model_average_max_epoch > 0:
            epoch_idx = args.model_average_max_epoch
            if args.nbest == 0:
                model_average_nbest = epoch_idx // 10   # take 1/10 for model averaging
            else:
                model_average_nbest = args.nbest
            phase_str += f"_nbest{model_average_nbest}_maxepoch{epoch_idx}"
        else:
            epoch_idx = ph2epochidx(model_ph)

        state_dict = torch.load(model_ph, map_location=f"cuda:{torch.cuda.current_device()}")
        if "model" in state_dict.keys():
            state_dict = state_dict["model"]    # models trained with ddp
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=True)
        logger.info(
            f"Model path: {colored(model_ph, 'red')}. "
            f"Missing: {missing_keys}, unexpected: {unexpected_keys}"
        )
        save_dir = os.path.join(cfg.model_dir, args.phase)
        os.makedirs(save_dir, exist_ok=True)
        logger.info(f"Results saving to {save_dir}...")

        template_path = getattr(cfg.dataset_cfg.gt_template, args.phase)
        gt_template = Path(template_path).read_text().splitlines()
        for i, data in enumerate(data_loader):
            # if i == 30:
            #     break
            X, X_lens = data["feats"].cuda(), data["feat_lens"].cuda()
        assert len(gt_template) == len(data_loader), \
            f"the lengths of template and data_loadre are not equal. " \
            f"data_loader: {len(data_loader)}, template: {len(gt_template)}"

        cer, wer = test(
            model=model,
            data_loader=data_loader,
            gt_template=gt_template,
            decode_cfg=cfg.decode_cfg,
            writter=writter,
            epoch_idx=epoch_idx,
            save_dir=save_dir,
            phase=phase_str,
            logger=logger,
            proc_idx=proc_idx,
        )
        cer_str = colored(f"{cer * 100:.6f}%", "red")
        wer_str = colored(f"{wer * 100:.6f}%", "red")
        logger.info(
            f"Epoch_idx: {epoch_idx}. "
            f"CER: {cer_str}, WER: {wer_str}. "
            f"Duration: {(time.time() - start) / 60:.2f}min."
        )


def test(model, data_loader, gt_template, decode_cfg, writter, epoch_idx, save_dir, phase, logger, proc_idx):

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
    pbar.set_postfix({
        "epoch": epoch_idx,
        "acc": f"{0.:.2f}%"
    })

    res = {k: AverageMeter() for k in ("cer", "wer", "acc")}
    summary = ["gt\tpred\tcer\twer"]
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            X, X_lens = data["feats"].cuda(), data["feat_lens"].cuda()
            assert X.size(0) == 1, "Only support batch-size==1!!!"
            pred_sents = beam_search_decoder(X, X_lens)
            gt_sents = [gt_template[i]]

            print("ys_out",data["ys_out"])
            print("pred_sents",pred_sents)
            print("gt_sents",gt_sents)

            for p, g in zip(pred_sents, gt_sents):
                cer_err = editdistance.eval(p, g)
                res["cer"].set(cer_err, len(g))
                wer_err = editdistance.eval(p.split(), g.split())
                res["wer"].set(wer_err, len(g.split()))
                ulter_cer = 1. * cer_err / len(g)
                ulter_wer = 1. * wer_err / len(g.split())
                summary.append(
                    f"{g}\t{p}\t{ulter_cer:.5f}\t{ulter_wer:.5f}"
                )

            acc = [p == g for p, g in zip(pred_sents, gt_sents)]

            res["acc"].set(sum(acc), len(acc))
            cer = res["cer"].get()
            wer = res["wer"].get()

            pbar.set_postfix({
                "epoch": epoch_idx,
                "cer": f"{res['cer'].get() * 100:.2f}%",
                "wer": f"{res['wer'].get() * 100:.2f}%",
                "acc": f"{res['acc'].get() * 100:.2f}%",
            })
            pbar.update()

        summary.extend([
            f"epoch_idx: {epoch_idx:02d}, "
            f"cer: {res['cer'].get() * 100:.5f}, "
            f"wer: {res['wer'].get() * 100:.5f}, "
            f"acc: {res['acc'].get() * 100:.5f}"
        ])

    txt_ph = os.path.join(
        save_dir,
        f"epoch_{epoch_idx:02d}_{phase}_cer{res['cer'].get() * 100:.3f}_wer{res['wer'].get() * 100:.3f}.txt"
    )
    open(txt_ph, 'w').write('\n'.join(summary))

    writter.add_scalars("model/cer", {phase: res['cer'].get()}, epoch_idx)
    writter.add_scalars("model/wer", {phase: res['wer'].get()}, epoch_idx)
    if "nbest" not in phase:
        writter.add_scalars("model/cer", {phase: res['cer'].get()}, epoch_idx)
        writter.add_scalars("model/wer", {phase: res['wer'].
    get()}, epoch_idx)
    else:
        logger.info(f"Phase: {phase}. Skip logging.")

    return res['cer'].get(), res['wer'].get()


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

    def model_average(ckpt_phs, avg_model_ph):
        avg_model = OrderedDict()
        nbest = len(ckpt_phs)
        for ph in ckpt_phs:
            ckpt = torch.load(ph, map_location="cpu")
            if "model" in ckpt.keys():
                ckpt = ckpt["model"]
            if len(avg_model) == 0:
                for k in ckpt.keys():
                    avg_model[k] = ckpt[k]
            else:
                for k in avg_model:
                    avg_model[k] = avg_model[k] + ckpt[k]
        for k in avg_model:
            if str(avg_model[k].dtype).startswith("torch.int"):
                pass
            else:
                avg_model[k] = avg_model[k] / nbest
        torch.save(avg_model, avg_model_ph)
        return avg_model_ph

    logger = get_logger(__name__)
    args, cfg = get_test_args()
    print('cfg.model_dir!!!!!',cfg.model_dir)
    cfg.dataset_cfg.phase = args.phase

    # get model paths
    model_phs = []
    s, e = args.start_epoch, args.end_epoch
    candidates = [t for t in os.listdir(os.path.join(args.train_path, "train")) if "nbest" not in t]
    candidates = sorted(candidates, key=ph2epochidx)
    for t in candidates:
        if not t.endswith(".pt"):
            logger.info(f"Skip {t}")
            continue
        epoch_idx = ph2epochidx(t)
        if len(args.epoch_idxs) > 0 and epoch_idx not in args.epoch_idxs:  # specify epoch_idx
            logger.info(f"Skip {t}")
            continue
        if (args.check_last and t == candidates[-1]) or (not args.check_last and s <= epoch_idx <= e):
            logger.info(f"To check: {t}")
            model_phs.append(os.path.join(args.train_path, "train", t))
    model_phs = sorted(model_phs, key=lambda x: ph2epochidx(x))
    if len(model_phs) == 0:
        logger.fatal("No model path detected. Exiting...")
        exit()
    if args.check_last:
        model_phs = [model_phs[-1]]
    if args.model_ph != '':
        logger.info(f"Use given model_ph: {args.model_ph}.")
        model_phs = [args.model_ph]
    if len(model_phs) == 0 and (not args.model_average_max_epoch > 0):
        logger.info(f"no ckpt!")
        exit()

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
        # pick the best epochs
        txt_paths = glob.glob(os.path.join(cfg.model_dir, "val", "*.txt"))

        print("cfg.model_dir",cfg.model_dir)
        print("txt_paths",txt_paths)
        epoch_idx_wer = []
        epoch_idx_cer = []
        for path in txt_paths:
            # e.g. epoch_25_acc_0.98.pt -> 0.98
            if "__" not in path:
                continue
            epoch_idx = ph2epochidx(path)
            if epoch_idx > max_epoch:
                continue  # Skip epochs beyond max_epoch
            path = os.path.basename(path)
            wer = float(path[:-4].split('_')[-1][3:])
            cer = float(path[:-4].split('_')[-2][3:])
            epoch_idx = int(path.split('_')[1])
            epoch_idx_wer.append([epoch_idx, wer])
            epoch_idx_cer.append([epoch_idx, cer])
        epoch_idx_wer = sorted(epoch_idx_wer, key=lambda x: x[1], reverse=False)
        epoch_idx_cer = sorted(epoch_idx_cer, key=lambda x: x[1], reverse=False)

        
        # epoch_idxs = [t[0] for t in epoch_idx_wer][:model_average_nbest]

        epoch_idxs = [t[0] for t in epoch_idx_cer][:model_average_nbest]

        ckpt_phs = [t for t in model_phs if ph2epochidx(t) in epoch_idxs]

        logger.info(
            f"epochidx_wer: {epoch_idx_wer}, \n"
            f"epoch_idxs: {epoch_idxs}, \n"
            f"ckpt_phs: {ckpt_phs}."
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

