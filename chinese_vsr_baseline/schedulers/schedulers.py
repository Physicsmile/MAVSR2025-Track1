import os
import sys
import logging
import warnings
import math

import tqdm
import torch
from termcolor import colored

formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
logging.basicConfig(
    format=formatter,
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def g(x):
    if isinstance(x, int):
        x = str(x)
    elif isinstance(x, float):
        x = f"{x:.6f}"
    elif isinstance(x, str):
        pass
    elif isinstance(x, list):
        x = '[' + ', '.join([g(t) for t in x]) + ']'
    else:
        raise ValueError(f"Invalid data type: {type(x)}")
    return colored(x, "red")


class TriStageScheduler(object):
    _attrs_ = (
        "num_epoch",
        "iter_per_epoch",

        "warmup_ratio",
        "hold_ratio",
        "decay_ratio",

        "warmup_steps",
        "hold_steps",
        "decay_steps",

        "init_lr",
        "peak_lr",
        "final_lr",

        "warmup_rate",
        "decay_factor",
        "lr",
        'step_num',
        "total_steps",
    )

    def __init__(
            self, optimizer, warmup_ratio, hold_ratio, decay_ratio,
            init_lr, peak_lr, final_lr,
            num_epoch, iter_per_epoch,
    ):
        """

        References:
            https://github.com/facebookresearch/fairseq/blob/main/fairseq/optim/lr_scheduler/tri_stage_lr_scheduler.py

        Args:
            optimizer:
            warmup_ratio:
            hold_ratio:
            decay_ratio:
            init_lr:
            peak_lr:
            final_lr:
            num_epoch:
            iter_per_epoch:
        """

        self.optimizer = optimizer
        self.num_epoch = num_epoch
        self.iter_per_epoch = iter_per_epoch

        def to_list_with_length_n(x, n):
            assert isinstance(n, int) and n > 0
            if isinstance(x, list):
                assert len(x) == n
                return x
            elif isinstance(x, int) or isinstance(x, float):
                return [x for _ in range(n)]
            else:
                raise ValueError

        # e.g. warmup_ratio = 0.45 -> [0.45, 0.45, 0.45, 0.45, 0.45]
        n = len(optimizer.param_groups)
        attrs = ("init_lr", "peak_lr", "final_lr", "warmup_ratio", "hold_ratio", "decay_ratio")
        for t in attrs:
            setattr(self, t, to_list_with_length_n(eval(t), n))
        attrs = ("warmup_steps", "hold_steps", "decay_steps", "warmup_rate", "decay_factor", "lr")
        for t in attrs:
            setattr(self, t, to_list_with_length_n(-1, n))

        total_steps = num_epoch * iter_per_epoch
        self.total_steps = total_steps
        self.step_num = 0

        for i in range(n):
            w = self.warmup_ratio[i]
            h = self.hold_ratio[i]
            d = self.decay_ratio[i]
            assert w + h + d <= 1.
            if int(w + h + d) == 0:
                logger.warning(f"Ratios per stage: {w:.2f} + {h:.2f} + {d:.2f} = {w + h + d} != 1")
            self.warmup_steps[i] = int(w * total_steps)
            self.hold_steps[i] = int(h * total_steps)
            self.decay_steps[i] = int(d * total_steps)
            assert self.peak_lr[i] > self.init_lr[i] and self.peak_lr[i] > self.final_lr[i]
            if self.warmup_steps[i] != 0:
                self.warmup_rate[i] = (self.peak_lr[i] - self.init_lr[i]) / self.warmup_steps[i]
            else:
                self.warmup_rate[i] = 0
            final_lr_scale = self.final_lr[i] / self.peak_lr[i]
            self.decay_factor[i] = -math.log(final_lr_scale) / self.decay_steps[i]
            self.lr[i] = self.init_lr[i]

        logger.info(
            f"Scheduler: {self.__class__.__name__}, "
            f"number of param groups: {n}, "
            f"init_lr: {self.init_lr}, "
            f"peak_lr: {self.peak_lr}, "
            f"final_lr: {self.final_lr}, "
            f"warmup steps: {self.warmup_steps}, "
            f"holdon steps: {self.hold_steps}, "
            f"decay steps: {self.decay_steps}."
        )

    def step(self):
        self.step_num += 1
        self._update_lr()

    def _get_lr(self):
        lrs = []
        for i in range(len(self.lr)):
            if self.step_num <= self.warmup_steps[i]:
                lrs.append(self.init_lr[i] + self.warmup_rate[i] * self.step_num)
                continue
            offset = self.warmup_steps[i]
            if self.step_num <= offset + self.hold_steps[i]:
                lrs.append(self.peak_lr[i])
                continue
            offset += self.hold_steps[i]
            if self.step_num <= offset + self.decay_steps[i]:
                lrs.append(self.peak_lr[i] * math.exp(-self.decay_factor[i] * (self.step_num - offset)))
                continue
            lrs.append(self.final_lr[i])
        return lrs

    def _update_lr(self):
        lr = self._get_lr()
        self.lr = lr
        assert len(self.lr) == len(self.optimizer.param_groups), \
            f"number of lrs: {len(self.lr)} != number of param groups {len(self.optimizer.param_groups)}"
        for i in range(len(self.optimizer.param_groups)):
            self.optimizer.param_groups[i]["lr"] = self.lr[i]

    def load_state_dict(self, ckpt):
        for attr in self._attrs_:
            if attr not in ckpt:
                raise KeyError(f"{attr} not in scheduler checkpoint {ckpt}")
            setattr(self, attr, ckpt[attr])
        self._update_lr()
        logger.info(
            f"Scheduler ckpt loaded. Lr: {g(self.lr)}. "
            f"Ratios: {self.warmup_ratio, self.hold_ratio, self.decay_ratio}. "
            f"Lrs: {self.init_lr, self.peak_lr, self.final_lr}. "
            f"Step num: {self.step_num}."
        )

    def state_dict(self):
        state_dict = dict()
        for attr in self._attrs_:
            state_dict[attr] = getattr(self, attr)
        return state_dict


class ReciprocalScheduler(object):
    _attrs_ = (
        "warm_up", "step_num", "min_lr", "is_cooling_down", "lr", 'k', "cool_lr_decay",
        "layer_wise_lr_decay_eta", "layer_wise_lr_decay_layers", "lw_decay",
    )

    def __init__(
            self, optimizer, lr, max_epoch, steps_per_epoch,
            warmup_iter, warmup_ratio, warmup_epoch, warmup_mode,
            layer_wise_lr_decay_eta=None, layer_wise_lr_decay_layers=None,
            min_lr=1e-8,
    ):
        if warmup_mode == "ratio":
            assert warmup_ratio > 0
            self.warm_up = max_epoch * steps_per_epoch * warmup_ratio
        elif warmup_mode == "iter":
            assert warmup_iter > 0
            self.warm_up = warmup_iter
        elif warmup_mode == "epoch":
            assert warmup_epoch > 0
            self.warm_up = warmup_epoch * steps_per_epoch
        else:
            raise NotImplementedError

        self.min_lr = min_lr
        self.lr, self.k, self.cool_lr_decay = [], [], []
        for i in range(len(lr)):
            self.lr.append(0.)  # it doesn't matter what we set it
            self.cool_lr_decay.append(None)
            self.k.append(lr[i] / self.warm_up ** -0.5)  # scaling factor

        self.step_num = 0
        self.optimizer = optimizer
        self.is_cooling_down = False

        # layer-wise decay related args
        self.layer_wise_lr_decay_eta = layer_wise_lr_decay_eta
        self.layer_wise_lr_decay_layers = layer_wise_lr_decay_layers
        # layer_wise_lr_decay_layers:
        # # encoder
        # - [0, 1, 1]
        # - [0, 2, 2]
        # - [0, 3, 3]
        # - [0, 4, 4]
        # - [0, 5, 5]
        # - [0, 6, 6]
        # - [0, 7, 7]
        # - [0, 8, 8]
        # - [0, 9, 9]
        # - [0, 10, 10]
        # - [0, 11, 11]
        # # decoder
        # - [12, 13, 1]
        # - [12, 14, 2]
        # - [12, 15, 3]
        # - [12, 16, 4]
        # - [12, 17, 5]、
        if layer_wise_lr_decay_eta is not None:
            self.lw_decay = True
            assert layer_wise_lr_decay_layers is not None
            logger.info(
                f"Enabling layer wise learning rate decay. "
                f"eta: {layer_wise_lr_decay_eta}, layers: {layer_wise_lr_decay_layers}."
            )
            # no repetition of b
            assert len([t[1] for t in layer_wise_lr_decay_layers]) == len(
                set([t[1] for t in layer_wise_lr_decay_layers]))
            # valid B
            assert all([t < len(self.lr) for l in layer_wise_lr_decay_layers for t in l])
        else:
            self.lw_decay = False

    def cool_down(self, epochs, iter_per_epoch):
        # References: Zhai, Xiaohua et al. “Scaling Vision Transformers.” ArXiv abs/2106.04560 (2021): n. pag.
        self.is_cooling_down = True
        for i in range(len(self.optimizer.param_groups)):
            self.cool_lr_decay[i] = (self.lr[i] - self.min_lr) / (epochs * iter_per_epoch)

    def step(self):
        self.step_num += 1
        if not self.lw_decay:
            self._update_lr()
        else:
            self._update_lr_lw_decay()

    def _update_lr(self):
        # update non layer-wise decay learning rates
        for i in range(len(self.optimizer.param_groups)):
            if not self.is_cooling_down:
                self.lr[i] = self.k[i] * min(self.step_num ** (-0.5),
                                             self.step_num * (self.warm_up ** (-1.5)))
            else:
                self.lr[i] -= self.cool_lr_decay[i]
            self.lr[i] = max(self.lr[i], self.min_lr)
            self.optimizer.param_groups[i]["lr"] = self.lr[i]

    def _update_lr_lw_decay(self):
        for i in range(len(self.optimizer.param_groups)):
            if i not in [t[1] for t in self.layer_wise_lr_decay_layers]:  # B
                if not self.is_cooling_down:
                    self.lr[i] = self.k[i] * self.init_lr * min(self.step_num ** (-0.5),
                                                                self.step_num * (self.warm_up ** (-1.5)))
                else:  # b
                    self.lr[i] -= self.cool_lr_decay[i]
            else:
                updated = False
                for B, b, r in self.layer_wise_lr_decay_layers:
                    if b == i:
                        updated = True
                        self.lr[i] = self.lr[B] * (self.layer_wise_lr_decay_eta ** r)
                assert updated
            self.lr[i] = max(self.lr[i], self.min_lr)
            self.optimizer.param_groups[i]["lr"] = self.lr[i]

    def state_dict(self):
        state_dict = dict()
        for attr in self._attrs_:
            state_dict[attr] = getattr(self, attr)
        return state_dict

    def load_state_dict(self, ckpt):
        for attr in self._attrs_:
            setattr(self, attr, ckpt[attr])
        self._update_lr()
        logger.info(
            f"Scheduler ckpt loaded. "
            f"k: {self.k}, lr: {g(self.lr)}, "
            f"step_num: {self.step_num}, warm_up: {self.warm_up}"
        )


if __name__ == "__main__":
    import torch.nn as nn
    import matplotlib.pyplot as plt

    X = torch.randn(2, 10, 88, 88)
    model = torch.nn.Conv2d(10, 1, 3, 2, 1)
    lr = 1e-4
    num_epoch = 10
    num_iter = 1000
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # scheduler = TriStageScheduler(optimizer, 0.35, 0.1, 0.55, 1e-6, lr, 1e-6, num_epoch, num_iter)
    scheduler = ReciprocalScheduler(optimizer, [lr], num_epoch, num_iter, -1, 0.1, -1, "ratio")

    lrs = []
    for _ in range(num_epoch):
        for _ in tqdm.tqdm(range(num_iter)):
            # y = model(X).sum()
            scheduler.step()
            optimizer.step()
            # y.backward()
            lrs.append(optimizer.param_groups[0]["lr"])

    plt.plot(lrs)
    plt.savefig("./a.png")
    plt.clf()