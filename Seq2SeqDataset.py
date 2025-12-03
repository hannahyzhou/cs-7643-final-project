import pandas as pd
import torch
from torch.utils.data import Dataset
from Vocab import Vocab

class Seq2SeqDataset(Dataset):
    def __init__(self, df, vocab, max_src_len, max_tgt_len):
        self.df = df.reset_index(drop=True)
        self.vocab = vocab
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        Takes in idx and returns 
        """
        instr = str(self.df.loc[idx, "instruction"])
        resp  = str(self.df.loc[idx, "response"])

        # Turn src and tgt into tokenized strings
        src_ids = self.vocab.numericalize(
            instr, add_bos_eos=False, max_len=self.max_src_len
        )
        tgt_ids = self.vocab.numericalize(
            resp, add_bos_eos=True, max_len=self.max_tgt_len
        )

        # Return src, tgt as tuple pair of torch tensors
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)
