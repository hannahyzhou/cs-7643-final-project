import re
from typing import List, Dict

class Vocab:
    def __init__(self, min_freq = 1):
        self.min_freq = min_freq
        self.freqs = {}
        # Dict of token -> idx
        self.stoi = {}
        # List of all tokens
        self.itos = []

        self.PAD = "<pad>"
        self.BOS = "<bos>"
        self.EOS = "<eos>"
        self.UNK = "<unk>"
        self.SEP = "<sep>"

        self.special_tokens = [self.PAD, self.BOS, self.EOS, self.UNK, self.SEP]

    def build(self, texts):
        """
        Builds both the index to string mapping (itos) and the reverse string to index mapping (stoi).
        """
        for text in texts:
            # Iterate through words in string
            for tok in text.lower().strip().split():
                # Increment frequency
                self.freqs[tok] = self.freqs.get(tok, 0) + 1

        # Get alphabetized list of tokens
        tokens = []
        for tok, f in self.freqs.items():
            if f >= self.min_freq:
                tokens.append(tok)
        tokens = sorted(tokens)

        # Create a list of all tokens with special tokens at the front
        self.itos = self.special_tokens + tokens

        # Create dictionary of token: idx structure
        for i, tok in enumerate(self.itos):
            self.stoi[tok] = i

    def numericalize(self, text, add_bos_eos=False, max_len=None):
        """
        Takes in a string and returns list of token indexes.
        """
        # Get list of tokens from string
        tokens = text.lower().strip().split()
        if add_bos_eos:
            tokens = [self.BOS] + tokens + [self.EOS]
        if max_len is not None:
            # Take only up to max_len number of tokens
            tokens = tokens[:max_len]
        tokenized_str = []
        for tok in tokens:
            # Get either the token's index or UNK token
            tokenized_str.append(self.stoi.get(tok, self.stoi[self.UNK]))
        return tokenized_str

    def denumericalize(self, ids):
        """
        Takes a list of token indexes and returns a list of words, separated by spaces.
        """
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