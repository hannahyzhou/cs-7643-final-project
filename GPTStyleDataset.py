import pandas as pd
import torch
from torch.utils.data import Dataset
from Vocab import Vocab

class GPTStyleDataset(Dataset):
    def __init__(self, df, vocab, max_len=128):
        self.df = df.reset_index(drop=True)
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        instr_text = str(row["instruction"])
        resp_text = str(row["response"])

        instr_ids = self.vocab.numericalize(instr_text, add_bos_eos=False)
        resp_ids = self.vocab.numericalize(resp_text, add_bos_eos=False)

        ids = [self.vocab.bos_idx()] + instr_ids + [self.vocab.sep_idx()] + resp_ids + [self.vocab.eos_idx()]

        ids = ids[:self.max_len]

        return torch.tensor(ids, dtype=torch.long)