import os
import sys
import warnings
import pickle
import logging
from pathlib import Path
from typing import List

logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class MappingTokenizer(object):
    def __init__(self, src_path: str, allow_oov: bool = True):
        self.idx2word, self.word2idx = self.load_txt(src_path)
        self.allow_oov = allow_oov
        self.skipped_symbols = []

    @staticmethod
    def load_txt(txt_ph):
        lines = Path(txt_ph).read_text().splitlines()
        skip = [t for t in lines if t.startswith('#')]
        if len(skip) > 0:
            logger.info(f"Skip lines: {skip}")
        lines = [t for t in lines if not t.startswith('#')]  # skip comments
        lines = [t.split(',') for t in lines]  
        word2idx = {k: int(v) for k, v in lines}
        idx2word = {v: k for k, v in word2idx.items()}
        return idx2word, word2idx

    def tokenize(self, chars: str):
        tokens = []
        for c in chars:
            try:
                tokens.append(self.word2idx[c])
            except KeyError as e:
                if not self.allow_oov:
                    raise KeyError(e)
                elif self.allow_oov:
                    if c not in self.skipped_symbols:
                        self.skipped_symbols.append(c)
                        logger.warning(
                            f"Skip OOV symbol: '{c}'. "
                            f"All skipped symbols: {self.skipped_symbols}."
                        )
        return tokens

    def detokenize(self, tokens: List[int]):
        """

        Args:
            tokens:

        Returns:

        """
        trans = []
        for t in tokens:
            if t not in self.idx2word:
                if self.allow_oov and "<UNK>" in self.word2idx:
                    trans.append("<UNK>")
                else:
                    logger.warning(f"token {t} not in `self.idx2word`. Skipped!!!")
            else:
                trans.append(self.idx2word[t])
        return ''.join(trans)

    def __call__(self, *args, **kwargs):
        return self.tokenize(*args, **kwargs)

