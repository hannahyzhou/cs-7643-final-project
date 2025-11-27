import re
from typing import List, Dict

class Vocab:
    def __init__(self, min_freq = 1):
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

    def build(self, texts):
        for text in texts:
            for tok in text.lower().strip().split():
                self.freqs[tok] = self.freqs.get(tok, 0) + 1

        tokens = [
            tok for tok, f in self.freqs.items()
            if f >= self.min_freq
        ]
        tokens = sorted(tokens)

        self.itos = self.special_tokens + tokens
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

    def numericalize(self, text, add_bos_eos=False, max_len=None):
        tokens = text.lower().strip().split()
        if add_bos_eos:
            tokens = [self.BOS] + tokens + [self.EOS]
        if max_len is not None:
            tokens = tokens[:max_len]
        return [self.stoi.get(tok, self.stoi[self.UNK]) for tok in tokens]

    def denumericalize(self, ids):
        tokens = [self.itos[i] for i in ids]
        tokens = [t for t in tokens if t not in [self.BOS, self.EOS, self.PAD, self.SEP]]
        return " ".join(tokens)

    def pad_idx(self):
        return self.stoi[self.PAD]

    def bos_idx(self):
        return self.stoi[self.BOS]

    def eos_idx(self):
        return self.stoi[self.EOS]

    def sep_idx(self):
        return self.stoi[self.SEP]