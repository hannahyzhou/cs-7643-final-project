import re
from typing import List, Dict

class Vocab:
    def __init__(self, min_freq: int = 1):
        self.min_freq = min_freq
        self.freqs = {}
        self.stoi = {}
        self.itos = []

        self.PAD = "<pad>"
        self.BOS = "<bos>"
        self.EOS = "<eos>"
        self.UNK = "<unk>"
        self.SEP = "<sep>"

        self.special_tokens = [self.PAD, self.BOS, self.EOS, self.UNK, self.SEP]

    def build(self, texts: List[str]):
        for text in texts:
            for tok in self.tokenize(text):
                self.freqs[tok] = self.freqs.get(tok, 0) + 1

        tokens = [
            tok for tok, f in self.freqs.items()
            if f >= self.min_freq
        ]
        tokens = sorted(tokens)

        self.itos = self.special_tokens + tokens
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

    @staticmethod
    def tokenize(text: str) -> List[str]:
        return text.lower().strip().split()

    def numericalize(self, text: str, add_bos_eos: bool = False, max_len: int = None) -> List[int]:
        tokens = self.tokenize(text)
        if add_bos_eos:
            tokens = [self.BOS] + tokens + [self.EOS]
        if max_len is not None:
            tokens = tokens[:max_len]
        return [self.stoi.get(tok, self.stoi[self.UNK]) for tok in tokens]

    def denumericalize(self, ids: List[int]) -> str:
        tokens = [self.itos[i] for i in ids]
        tokens = [t for t in tokens if t not in [self.BOS, self.EOS, self.PAD, self.SEP]]
        return " ".join(tokens)

    @property
    def pad_idx(self):
        return self.stoi[self.PAD]

    @property
    def bos_idx(self):
        return self.stoi[self.BOS]

    @property
    def eos_idx(self):
        return self.stoi[self.EOS]

    @property
    def sep_idx(self):
        return self.stoi[self.SEP]