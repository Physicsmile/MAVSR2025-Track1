import os
import sys
import logging

from typing import Union
from copy import deepcopy

import prettytable
from prettytable import PrettyTable

import time

logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class AttrDict(dict):
    """
    Dictionary whose keys can be accessed as attributes.
    """

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        root = kwargs.get("root", deepcopy(self))
        if "root" in self.keys():
            self.pop("root")
        self._normalize(root)

    def _normalize(self, root):
        for item in self.keys():
            v = self[item]
            if isinstance(v, str) and '${' in v:
                v = self._parse(v, root)
                logger.info(f"Normalize: [{item}] {self[item]} -> {v}")
                self[item] = v
                self.__dict__[item] = v
            elif isinstance(v, dict):
                self[item] = AttrDict(self[item], root=root)

    def _parse(self, s, root):
        values = []
        for t in s.split('/'):
            if '$' in t:
                assert t.startswith("${") and t.endswith('}')
                d = root
                for k in t[2:-1].split('.'):  
                    d = d.get(k)
            else:
                d = t
            values.append(str(d))
        if len(values) == 1 and values[0].isnumeric():
            v = values[0]
            if '.' in v:
                return float(v)
            else:
                return int(v)
        else:
            return '/'.join(values)

    def __getattr__(self, item):
        if item not in self:
            raise AttributeError(f"no attribute called {item}!")
        return self[item]

    def __setattr__(self, item, value):
        self.__dict__[item] = value
        self[item] = value


class AverageMeter(object):
    def __init__(self, init_c: float = 0., init_n: int = 0):
        self.c = init_c
        self.n = init_n
        self._recent = 0. 

    def reset(self):
        self.c, self.n, self._recent = 0., 0, 0.

    def set(self, c: Union[list, float], n: Union[list, float]):
        if isinstance(c, list):
            assert len(c) == n
            self.c += sum(c)
            self.n += n
            self._recent = sum(c) / n
        else:
            self.c += c
            self.n += n
            self._recent = c / n

    def update(self, *args, **kwargs):
        return self.set(*args, **kwargs)

    def get(self):
        return self.c / self.n if self.n != 0 else 0.

    @property
    def recent(self):
        return self._recent

    def __repr__(self):
        return f"{self.get():.4f}"

    def __lt__(self, other):
        return self.get() < other

    def __le__(self, other):
        return self.get() <= other

    def __eq__(self, other):
        return self.get() == other

    def __gt__(self, other):
        return self.get() > other

    def __ge__(self, other):
        return self.get() >= other


def make_dirs(cfg: AttrDict, extra_dir: str = None, path: str = None):
    ckpt_dir = os.path.join(cfg.train_cfg.ckpt_dir)
    if path is None:
        train_time = time.strftime("%m-%d-%H-%M-%S", time.localtime())
        cfg.model_dir = os.path.join(ckpt_dir, cfg.dset_type, cfg.train_tag, cfg.model_tag, train_time)
    else:
        cfg.model_dir = path

    dirs = ("log", "train", "test", "val")
    if extra_dir is not None and extra_dir not in dirs:
        dirs = dirs + (extra_dir,)
    for k in dirs:
        setattr(cfg.train_cfg, f"{k}_dir", os.path.join(cfg.model_dir, k))
        dir_ph = getattr(cfg.train_cfg, f"{k}_dir")
        os.makedirs(dir_ph, exist_ok=True)


def print_model_params(model):
    # References: https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model
    table = PrettyTable(
        title="Model Parameter Status",
        field_names=["Modules", "Parameters"],
        header=True,
        header_style="cap",
        border=True,
        hrules=prettytable.FRAME,
    )
    table.set_style(prettytable.DEFAULT)
    table.align["Modules"] = 'l'
    table.align["Parameters"] = 'c'

    keys = ("frontend", "encoder", "decoder")
    param_cls = {k: 0 for k in keys}
    for name, parameter in model.named_parameters():
        params = parameter.numel()
        table.add_row([name, f"{params / 1000:.4f}k"])
        for k in keys:
            if k in name:
                param_cls[k] += params
    for k in keys:
        table.add_row([k, f"{param_cls[k] / 1000000:.4f}m"])

    logger.info(table)
    return table.get_string()
