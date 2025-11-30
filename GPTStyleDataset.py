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
        """
        Takes in idx and returns tokenized instruction + response as a torch tensor.
        """
        # Select row # idx
        row = self.df.iloc[idx]
        instr_text = str(row["instruction"])
        resp_text = str(row["response"])

        # Turn instruction and response into tokenized strings
        instr_ids = self.vocab.numericalize(instr_text, add_bos_eos=False)
        resp_ids = self.vocab.numericalize(resp_text, add_bos_eos=False)

        # Combine instrution and response into one long list of tokens: <BOS> + instruction + <SEP> + response + <EOS>
        ids = [self.vocab.bos_idx()] + instr_ids + [self.vocab.sep_idx()] + resp_ids + [self.vocab.eos_idx()]

        # Truncate to max_len
        ids = ids[:self.max_len]

        # Return combined list of tokens as a torch tensor
        return torch.tensor(ids, dtype=torch.long)